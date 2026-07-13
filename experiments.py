"""Experiment entry points for the adaptive predictive-modeling benchmark."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import pandas as pd
import torch

from .data import (
    GenerationConfig,
    adaptation_balance_summary,
    adaptation_set,
    adaptation_subset,
    base_train_set,
    create_adaptation_diagnostics,
    family_test_sets,
    final_test_set,
    load_datasets,
    make_all_datasets,
    summarize_datasets,
    verify_datasets,
    write_datasets,
)
from .models import (
    adapt_moe_step,
    count_parameters,
    evaluate_final_test,
    plot_kmeans_family_clusters,
    routing_summary,
    train_mlp,
    train_moe,
)
from .plotting import color_for_model, linestyle_for_model, marker_for_model
from .uncertainty import (
    adaptive_uncertainty_diagnostics,
    begin_end_uncertainty_diagnostics,
    split_adaptation_for_uq,
    static_uncertainty_diagnostics,
)


def project_paths(project_root: Path) -> dict[str, Path]:

    return {
        "root": project_root,
        "data": project_root / "data",
        "results": project_root / "results",
        "figures": project_root / "results" / "figures",
    }


def run_data_generation(project_root: Path, cfg: GenerationConfig | None = None) -> pd.DataFrame:

    cfg = cfg or GenerationConfig()
    paths = project_paths(project_root)
    datasets = make_all_datasets(cfg)
    write_datasets(datasets, paths["data"])
    paths["results"].mkdir(parents=True, exist_ok=True)

    summary = summarize_datasets(datasets, cfg)
    summary.to_csv(paths["results"] / "data_generation_summary.csv", index=False)
    pd.Series(verify_datasets(datasets, cfg), name="check").to_csv(
        paths["results"] / "data_generation_checks.csv",
        index=False,
    )
    balance = adaptation_balance_summary(datasets["D_adapt"], cfg)
    balance.to_csv(paths["results"] / "adaptation_balance_summary.csv", index=False)
    diagnostics = create_adaptation_diagnostics(datasets, paths["figures"])
    diagnostics.to_csv(paths["results"] / "adaptation_diagnostics.csv", index=False)
    return summary


def create_learning_and_time_plots(metrics: pd.DataFrame, results_dir: Path) -> None:

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = metrics[metrics["family"] == "ALL"].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, group in all_metrics.groupby("model"):
        ax.plot(
            group["n_adapt_used"],
            group["MAE"],
            marker=marker_for_model(model_name),
            color=color_for_model(model_name),
            linestyle=linestyle_for_model(model_name),
            label=model_name,
        )
    ax.set_xlabel("Adaptation samples", fontsize=14)
    ax.set_ylabel(r"MAE", fontsize=14)
    # ax.set_title(r"Learning curve on $D_{test,shifted}$")
    ax.legend(fontsize=14)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(results_dir / "learning_curve_shifted_test.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name in ["Retrained NN", "Retrained MoE", "Adaptive MoE", "Adaptive MoE shuffled"]:
        group = all_metrics[all_metrics["model"] == model_name]
        if not group.empty:
            ax.plot(
                group["n_adapt_used"],
                group["cumulative_training_time_sec"],
                marker=marker_for_model(model_name),
                color=color_for_model(model_name),
                linestyle=linestyle_for_model(model_name),
                label=model_name,
            )
    ax.set_xlabel("Adaptation samples", fontsize=14)
    ax.set_ylabel("Cumulative training/update time (sec)", fontsize=14)
    # ax.set_title("Training-time curve")
    ax.legend(fontsize=14)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(results_dir / "training_time_curve.png", dpi=300)
    plt.close(fig)


def _append_n_adapt(metrics: pd.DataFrame, n_adapt_used: int) -> pd.DataFrame:
    out = metrics.copy()
    out["n_adapt_used"] = n_adapt_used
    return out


def _timed_train_mlp(
    train_df: pd.DataFrame,
    seed: int,
    *,
    tune: bool = False,
    n_trials: int = 12,
    output_dir: Path | None = None,
    model_label: str = "general_nn",
):
    start = time.perf_counter()
    model, scalers = train_mlp(
        train_df,
        seed=seed,
        tune=tune,
        n_trials=n_trials,
        output_dir=output_dir,
        model_label=model_label,
    )
    elapsed = time.perf_counter() - start
    return model, scalers, elapsed


def _timed_train_moe(
    train_df: pd.DataFrame,
    seed: int,
    *,
    tune_experts: bool = False,
    n_trials: int = 8,
    output_dir: Path | None = None,
):
    start = time.perf_counter()
    model, scalers, kmeans = train_moe(
        train_df,
        seed=seed,
        tune_experts=tune_experts,
        n_trials=n_trials,
        output_dir=output_dir,
    )
    elapsed = time.perf_counter() - start
    return model, scalers, kmeans, elapsed


def run_static_models(
    project_root: Path,
    seed: int = 42,
    *,
    tune: bool = True,
    n_trials: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    paths = project_paths(project_root)
    datasets = load_datasets(paths["data"])
    base_df = base_train_set(datasets)
    adapt_df = adaptation_set(datasets)
    test_df = final_test_set(datasets)
    diagnostic_dir = paths["figures"] / "model_diagnostics"
    uncertainty_dir = paths["figures"] / "uncertainty_calibration"

    nn_model, nn_scalers, nn_time = _timed_train_mlp(
        base_df,
        seed,
        tune=tune,
        n_trials=n_trials,
        output_dir=diagnostic_dir,
        model_label="static_general_nn",
    )
    moe_model, moe_scalers, kmeans, moe_time = _timed_train_moe(
        base_df,
        seed,
        tune_experts=tune,
        n_trials=max(4, n_trials),
        output_dir=diagnostic_dir,
    )

    metrics = []
    metrics.append(
        _append_n_adapt(
            evaluate_final_test(
                nn_model,
                test_df,
                nn_scalers,
                model_name="Static NN",
                adapt_pct=0,
                training_time_sec=nn_time,
                cumulative_training_time_sec=nn_time,
                updated_params=count_parameters(nn_model),
            ),
            0,
        )
    )
    metrics.append(
        _append_n_adapt(
            evaluate_final_test(
                moe_model,
                test_df,
                moe_scalers,
                model_name="Static MoE",
                adapt_pct=0,
                training_time_sec=moe_time,
                cumulative_training_time_sec=moe_time,
                updated_params=count_parameters(moe_model),
            ),
            0,
        )
    )
    metrics_df = pd.concat(metrics, ignore_index=True)

    route_rows = []
    for family, df in family_test_sets(datasets).items():
        if family == "ALL":
            continue
        summary = routing_summary(moe_model, df, moe_scalers)
        route_rows.append(summary)
    routes_df = pd.concat(route_rows, ignore_index=True)
    cluster_df = plot_kmeans_family_clusters(
        base_df,
        moe_scalers,
        kmeans,
        diagnostic_dir / "kmeans_family_clusters.png",
    )
    static_uncertainty_diagnostics(
        nn_model=nn_model,
        nn_scalers=nn_scalers,
        moe_model=moe_model,
        moe_scalers=moe_scalers,
        calibration_df=adapt_df,
        test_df=test_df,
        results_dir=paths["results"],
        figures_dir=uncertainty_dir,
    )

    paths["results"].mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(paths["results"] / "static_model_metrics.csv", index=False)
    routes_df.to_csv(paths["results"] / "static_moe_routing_summary.csv", index=False)
    cluster_df.to_csv(paths["results"] / "kmeans_family_cluster_assignments.csv", index=False)
    torch.save(nn_model.state_dict(), paths["results"] / "static_nn_state.pt")
    torch.save(moe_model.state_dict(), paths["results"] / "static_moe_state.pt")
    return metrics_df, routes_df


def run_incremental_comparison(project_root: Path, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:

    cfg = GenerationConfig(seed=seed)
    paths = project_paths(project_root)
    datasets = load_datasets(paths["data"])
    base_df = base_train_set(datasets)
    adapt_df = adaptation_set(datasets)
    test_df = final_test_set(datasets)

    static_nn, static_nn_scalers, static_nn_time = _timed_train_mlp(base_df, seed)
    static_moe, static_moe_scalers, _, static_moe_time = _timed_train_moe(base_df, seed)
    adaptive_moe = copy.deepcopy(static_moe)
    adaptive_moe_shuffled = copy.deepcopy(static_moe)

    static_nn_cum = static_nn_time
    static_moe_cum = static_moe_time
    retrain_nn_cum = 0.0
    retrain_moe_cum = 0.0
    adaptive_cum = static_moe_time
    adaptive_shuffled_cum = static_moe_time

    metric_frames = []
    route_records = []

    for pct in cfg.adapt_percentages:
        seen_adapt = adaptation_subset(adapt_df, pct)
        train_augmented = pd.concat([base_df, seen_adapt], ignore_index=True)
        n_adapt = len(seen_adapt)

        static_nn_step_time = static_nn_time if pct == 0 else 0.0
        static_moe_step_time = static_moe_time if pct == 0 else 0.0
        static_nn_updated = count_parameters(static_nn) if pct == 0 else 0
        static_moe_updated = count_parameters(static_moe) if pct == 0 else 0

        metric_frames.append(
            _append_n_adapt(
                evaluate_final_test(
                    static_nn,
                    test_df,
                    static_nn_scalers,
                    model_name="Static NN",
                    adapt_pct=pct,
                    training_time_sec=static_nn_step_time,
                    cumulative_training_time_sec=static_nn_cum,
                    updated_params=static_nn_updated,
                ),
                n_adapt,
            )
        )
        metric_frames.append(
            _append_n_adapt(
                evaluate_final_test(
                    static_moe,
                    test_df,
                    static_moe_scalers,
                    model_name="Static MoE",
                    adapt_pct=pct,
                    training_time_sec=static_moe_step_time,
                    cumulative_training_time_sec=static_moe_cum,
                    updated_params=static_moe_updated,
                ),
                n_adapt,
            )
        )

        retrained_nn, retrained_nn_scalers, retrained_nn_time = _timed_train_mlp(train_augmented, seed + pct + 10)
        retrain_nn_cum += retrained_nn_time
        metric_frames.append(
            _append_n_adapt(
                evaluate_final_test(
                    retrained_nn,
                    test_df,
                    retrained_nn_scalers,
                    model_name="Retrained NN",
                    adapt_pct=pct,
                    training_time_sec=retrained_nn_time,
                    cumulative_training_time_sec=retrain_nn_cum,
                    updated_params=count_parameters(retrained_nn),
                ),
                n_adapt,
            )
        )

        retrained_moe, retrained_moe_scalers, _, retrained_moe_time = _timed_train_moe(
            train_augmented,
            seed + pct + 20,
        )
        retrain_moe_cum += retrained_moe_time
        metric_frames.append(
            _append_n_adapt(
                evaluate_final_test(
                    retrained_moe,
                    test_df,
                    retrained_moe_scalers,
                    model_name="Retrained MoE",
                    adapt_pct=pct,
                    training_time_sec=retrained_moe_time,
                    cumulative_training_time_sec=retrain_moe_cum,
                    updated_params=count_parameters(retrained_moe),
                ),
                n_adapt,
            )
        )

        if pct == 0:
            adaptive_step_time = static_moe_time
            adaptive_updated_params = count_parameters(adaptive_moe)
            selected_experts: list[int] = []
            shuffled_step_time = static_moe_time
            shuffled_updated_params = count_parameters(adaptive_moe_shuffled)
            shuffled_selected_experts: list[int] = []
        else:
            start = time.perf_counter()
            selected_experts, adaptive_updated_params = adapt_moe_step(
                adaptive_moe,
                static_moe_scalers,
                base_df,
                seen_adapt,
                seed=seed + pct + 100,
            )
            adaptive_step_time = time.perf_counter() - start
            adaptive_cum += adaptive_step_time

            start = time.perf_counter()
            shuffled_selected_experts, shuffled_updated_params = adapt_moe_step(
                adaptive_moe_shuffled,
                static_moe_scalers,
                base_df,
                seen_adapt,
                seed=seed + pct + 700,
                shuffle_targets=True,
            )
            shuffled_step_time = time.perf_counter() - start
            adaptive_shuffled_cum += shuffled_step_time

        metric_frames.append(
            _append_n_adapt(
                evaluate_final_test(
                    adaptive_moe,
                    test_df,
                    static_moe_scalers,
                    model_name="Adaptive MoE",
                    adapt_pct=pct,
                    training_time_sec=adaptive_step_time,
                    cumulative_training_time_sec=adaptive_cum,
                    updated_params=adaptive_updated_params,
                ),
                n_adapt,
            )
        )
        metric_frames.append(
            _append_n_adapt(
                evaluate_final_test(
                    adaptive_moe_shuffled,
                    test_df,
                    static_moe_scalers,
                    model_name="Adaptive MoE shuffled",
                    adapt_pct=pct,
                    training_time_sec=shuffled_step_time,
                    cumulative_training_time_sec=adaptive_shuffled_cum,
                    updated_params=shuffled_updated_params,
                ),
                n_adapt,
            )
        )
        route_records.append(
            {
                "adapt_pct": pct,
                "n_adapt_used": n_adapt,
                "selected_experts": ",".join(map(str, selected_experts)) if selected_experts else "",
                "updated_params": adaptive_updated_params,
                "training_time_sec": adaptive_step_time,
                "cumulative_training_time_sec": adaptive_cum,
                "shuffled_selected_experts": ",".join(map(str, shuffled_selected_experts))
                if shuffled_selected_experts
                else "",
                "shuffled_updated_params": shuffled_updated_params,
                "shuffled_training_time_sec": shuffled_step_time,
                "shuffled_cumulative_training_time_sec": adaptive_shuffled_cum,
            }
        )

    metrics = pd.concat(metric_frames, ignore_index=True)
    routes = pd.DataFrame(route_records)
    paths["results"].mkdir(parents=True, exist_ok=True)
    metrics.to_csv(paths["results"] / "incremental_model_comparison.csv", index=False)
    routes.to_csv(paths["results"] / "adaptive_moe_update_log.csv", index=False)
    create_learning_and_time_plots(metrics, paths["figures"])
    uq_update_df, uq_conformal_df = split_adaptation_for_uq(adapt_df, seed=seed)
    uq_train_augmented = pd.concat([base_df, uq_update_df], ignore_index=True)
    uq_retrained_nn, uq_retrained_nn_scalers, _ = _timed_train_mlp(uq_train_augmented, seed + 901)
    uq_retrained_moe, uq_retrained_moe_scalers, _, _ = _timed_train_moe(uq_train_augmented, seed + 902)
    uq_adaptive_moe = copy.deepcopy(static_moe)
    adapt_moe_step(
        uq_adaptive_moe,
        static_moe_scalers,
        base_df,
        uq_update_df,
        seed=seed + 1901,
    )
    begin_end_uncertainty_diagnostics(
        initial_nn_model=static_nn,
        initial_nn_scalers=static_nn_scalers,
        final_nn_model=uq_retrained_nn,
        final_nn_scalers=uq_retrained_nn_scalers,
        initial_moe_model=static_moe,
        initial_moe_scalers=static_moe_scalers,
        final_retrained_moe_model=uq_retrained_moe,
        final_retrained_moe_scalers=uq_retrained_moe_scalers,
        final_adaptive_moe_model=uq_adaptive_moe,
        adaptive_moe_scalers=static_moe_scalers,
        calibration_df=uq_conformal_df,
        test_df=test_df,
        results_dir=paths["results"],
        figures_dir=paths["figures"] / "uncertainty_calibration",
        n_adapt_used_for_update=len(uq_update_df),
        n_adapt_heldout_for_conformalization=len(uq_conformal_df),
    )
    adaptive_uncertainty_diagnostics(
        base_moe_model=static_moe,
        moe_scalers=static_moe_scalers,
        base_df=base_df,
        adapt_df=adapt_df,
        test_df=test_df,
        results_dir=paths["results"],
        figures_dir=paths["figures"] / "uncertainty_calibration",
        seed=seed,
    )
    torch.save(adaptive_moe.state_dict(), paths["results"] / "adaptive_moe_final_state.pt")
    torch.save(adaptive_moe_shuffled.state_dict(), paths["results"] / "adaptive_moe_shuffled_final_state.pt")
    return metrics, routes


def run_all(project_root: Path, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    summary = run_data_generation(project_root, GenerationConfig(seed=seed))
    metrics, update_log = run_incremental_comparison(project_root, seed=seed)
    return summary, metrics, update_log
