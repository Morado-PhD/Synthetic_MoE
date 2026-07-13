from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mapie.regression import SplitConformalRegressor
from scipy.special import ndtr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split

from .data import TARGET_COL, U_COLS
from .models import DEVICE, MoERegressor, Scalers, adapt_moe_step, inverse_y, to_tensors
from .plotting import category_color, color_for_model, linestyle_for_model, marker_for_model


CALIBRATION_LEVELS: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
NORMAL_95_Z = 1.959963984540054


class TorchRegressorAdapter(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        model: nn.Module,
        scalers: Scalers,
        *,
        expert_id: int | None = None,
    ):
        self.model = model
        self.scalers = scalers
        self.expert_id = expert_id
        self.is_fitted_ = True
        self.fitted_ = True
        self.n_features_in_ = len(U_COLS)

    def fit(self, X, y=None):
        self.is_fitted_ = True
        self.fitted_ = True
        return self

    def predict(self, X) -> np.ndarray:
        x_np = np.asarray(X, dtype=np.float32)
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        x_scaled = self.scalers.x.transform(x_np).astype(np.float32)
        x = torch.tensor(x_scaled, dtype=torch.float32, device=DEVICE)
        self.model.eval()
        with torch.no_grad():
            if self.expert_id is None:
                y_scaled = self.model(x)
            else:
                y_scaled = self.model.experts[self.expert_id](x)
        return inverse_y(y_scaled, self.scalers)


@dataclass
class ConformalResult:


    predictions: pd.DataFrame
    calibration: pd.DataFrame
    conformity: pd.DataFrame


def _features(df: pd.DataFrame) -> np.ndarray:
    return df[U_COLS].to_numpy(dtype=np.float32)


def _targets(df: pd.DataFrame) -> np.ndarray:
    return df[TARGET_COL].to_numpy(dtype=float)


def _assigned_experts(model: MoERegressor, df: pd.DataFrame, scalers: Scalers) -> np.ndarray:
    x, _ = to_tensors(df, scalers)
    model.eval()
    with torch.no_grad():
        return model.gate.weights(x).argmax(dim=1).cpu().numpy().astype(int)


def _minimum_conformalization_samples(confidence_levels: Iterable[float] = CALIBRATION_LEVELS) -> int:


    levels = [float(level) for level in confidence_levels]
    required = max(max(1.0 / level, 1.0 / (1.0 - level)) for level in levels)
    return int(np.ceil(required - 1e-12)) + 1


def _normal_crps(y_true: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray) -> np.ndarray:

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    out = np.abs(y_true - y_pred)
    mask = sigma > 0
    if np.any(mask):
        z = (y_true[mask] - y_pred[mask]) / sigma[mask]
        phi = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
        out[mask] = sigma[mask] * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
    return out


def summarize_uq_metrics(
    predictions: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    uq_scope: str,
    confidence_level: float = 0.95,
) -> pd.DataFrame:

    rows = []
    for (model_name, expert), pred_group in predictions.groupby(["model", "expert"], sort=False):
        cal_group = calibration[(calibration["model"] == model_name) & (calibration["expert"].astype(str) == str(expert))]
        level_group = cal_group[np.isclose(cal_group["confidence_level"], confidence_level)]
        if level_group.empty:
            continue
        sigma = pred_group["width_95"].to_numpy(dtype=float) / (2.0 * NORMAL_95_Z)
        crps = _normal_crps(
            pred_group["y_true"].to_numpy(dtype=float),
            pred_group["y_pred"].to_numpy(dtype=float),
            sigma,
        )
        row = {
            "uq_scope": uq_scope,
            "model": model_name,
            "expert": expert,
            "confidence_level": confidence_level,
            "MACE": float(cal_group["coverage_error"].abs().mean()),
            "PICP": float(pred_group["covered_95"].mean()),
            "MPIW": float(pred_group["width_95"].mean()),
            "CRPS": float(np.mean(crps)),
            "CRPS_method": "normal_approx_from_95pct_MAPIE_interval",
            "n_test": int(len(pred_group)),
            "n_conformalize": int(level_group["n_conformalize"].iloc[0]),
            "coverage_error_95": float(level_group["coverage_error"].iloc[0]),
        }
        for col in ["n_adapt_used_for_update", "n_adapt_heldout_for_conformalization", "updated_params", "selected_experts"]:
            if col in level_group.columns:
                row[col] = level_group[col].iloc[0]
            elif col in pred_group.columns:
                row[col] = pred_group[col].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def write_uq_metric_summary(results_dir: Path) -> pd.DataFrame:

    frames = []
    for prefix in ["static", "adaptive", "begin_end"]:
        pred_path = results_dir / f"{prefix}_prediction_intervals_95.csv"
        cal_path = results_dir / f"{prefix}_interval_calibration.csv"
        if not pred_path.exists() or not cal_path.exists():
            continue
        summary = summarize_uq_metrics(
            pd.read_csv(pred_path),
            pd.read_csv(cal_path),
            uq_scope=prefix,
        )
        summary.to_csv(results_dir / f"{prefix}_uq_metric_summary.csv", index=False)
        frames.append(summary)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined.to_csv(results_dir / "uq_metric_summary.csv", index=False)
    return combined


def _interval_frame(
    *,
    model_name: str,
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    intervals: np.ndarray,
    expert_id: int | None,
    assigned_expert: np.ndarray | None = None,
) -> pd.DataFrame:
    lower = intervals[:, 0, -1]
    upper = intervals[:, 1, -1]
    out = pd.DataFrame(
        {
            "model": model_name,
            "expert": "ensemble" if expert_id is None else int(expert_id),
            "row_id": test_df.index.to_numpy(),
            "family": test_df["family"].to_numpy(),
            "y_true": _targets(test_df),
            "y_pred": y_pred,
            "lower_95": lower,
            "upper_95": upper,
            "width_95": upper - lower,
            "covered_95": (_targets(test_df) >= lower) & (_targets(test_df) <= upper),
        }
    )
    if assigned_expert is not None:
        out["assigned_expert"] = assigned_expert
    return out


def _conformity_frame(
    *,
    model_name: str,
    calibration_df: pd.DataFrame,
    y_pred: np.ndarray,
    expert_id: int | None,
    assigned_expert: np.ndarray | None = None,
) -> pd.DataFrame:
    y_true = _targets(calibration_df)
    out = pd.DataFrame(
        {
            "model": model_name,
            "expert": "ensemble" if expert_id is None else int(expert_id),
            "row_id": calibration_df.index.to_numpy(),
            "family": calibration_df["family"].to_numpy(),
            "y_true": y_true,
            "y_pred": y_pred,
            "abs_residual": np.abs(y_true - y_pred),
        }
    )
    if assigned_expert is not None:
        out["assigned_expert"] = assigned_expert
    return out


def _calibration_frame(
    *,
    model_name: str,
    test_df: pd.DataFrame,
    intervals: np.ndarray,
    confidence_levels: Iterable[float],
    n_conformalize: int,
    expert_id: int | None,
    conformity_scores: np.ndarray,
) -> pd.DataFrame:
    y_true = _targets(test_df)
    conformity_scores = np.asarray(conformity_scores, dtype=float)
    if len(conformity_scores) == 0:
        residual_mean = np.nan
        residual_q95 = np.nan
        residual_max = np.nan
    else:
        residual_mean = float(np.mean(conformity_scores))
        residual_q95 = float(np.quantile(conformity_scores, 0.95))
        residual_max = float(np.max(conformity_scores))
    rows = []
    for level_idx, level in enumerate(confidence_levels):
        lower = intervals[:, 0, level_idx]
        upper = intervals[:, 1, level_idx]
        covered = (y_true >= lower) & (y_true <= upper)
        rows.append(
            {
                "model": model_name,
                "expert": "ensemble" if expert_id is None else int(expert_id),
                "confidence_level": float(level),
                "empirical_coverage": float(np.mean(covered)),
                "coverage_error": float(np.mean(covered) - level),
                "mean_interval_width": float(np.mean(upper - lower)),
                "n_conformalize": int(n_conformalize),
                "n_test": int(len(test_df)),
                "calibration_abs_residual_mean": residual_mean,
                "calibration_abs_residual_q95": residual_q95,
                "calibration_abs_residual_max": residual_max,
            }
        )
    return pd.DataFrame(rows)


def conformalize_prefit_model(
    *,
    model: nn.Module,
    scalers: Scalers,
    calibration_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    expert_id: int | None = None,
    confidence_levels: Iterable[float] = CALIBRATION_LEVELS,
    assigned_expert: np.ndarray | None = None,
    conformity_assigned_expert: np.ndarray | None = None,
) -> ConformalResult:

    confidence_levels = tuple(float(v) for v in confidence_levels)
    adapter = TorchRegressorAdapter(model, scalers, expert_id=expert_id)
    calibration_pred = adapter.predict(_features(calibration_df))
    conformity = _conformity_frame(
        model_name=model_name,
        calibration_df=calibration_df,
        y_pred=calibration_pred,
        expert_id=expert_id,
        assigned_expert=conformity_assigned_expert,
    )
    conformalizer = SplitConformalRegressor(
        estimator=adapter,
        confidence_level=list(confidence_levels),
        conformity_score="absolute",
        prefit=True,
    ).conformalize(_features(calibration_df), _targets(calibration_df))
    y_pred, intervals = conformalizer.predict_interval(_features(test_df))
    return ConformalResult(
        predictions=_interval_frame(
            model_name=model_name,
            test_df=test_df,
            y_pred=y_pred,
            intervals=intervals,
            expert_id=expert_id,
            assigned_expert=assigned_expert,
        ),
        calibration=_calibration_frame(
            model_name=model_name,
            test_df=test_df,
            intervals=intervals,
            confidence_levels=confidence_levels,
            n_conformalize=len(calibration_df),
            expert_id=expert_id,
            conformity_scores=conformity["abs_residual"].to_numpy(),
        ),
        conformity=conformity,
    )


def nn_uncertainty_diagnostics(
    *,
    nn_model: nn.Module,
    nn_scalers: Scalers,
    calibration_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    result = conformalize_prefit_model(
        model=nn_model,
        scalers=nn_scalers,
        calibration_df=calibration_df,
        test_df=test_df,
        model_name=model_name,
    )
    return result.predictions, result.calibration, result.conformity


def moe_uncertainty_diagnostics(
    *,
    moe_model: MoERegressor,
    moe_scalers: Scalers,
    calibration_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    expert_model_prefix: str,
    min_expert_calibration_samples: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    min_samples = max(
        int(min_expert_calibration_samples),
        _minimum_conformalization_samples(CALIBRATION_LEVELS),
    )
    prediction_frames = []
    calibration_frames = []
    conformity_frames = []

    assigned_cal = _assigned_experts(moe_model, calibration_df, moe_scalers)
    assigned_test = _assigned_experts(moe_model, test_df, moe_scalers)
    moe = conformalize_prefit_model(
        model=moe_model,
        scalers=moe_scalers,
        calibration_df=calibration_df,
        test_df=test_df,
        model_name=model_name,
        assigned_expert=assigned_test,
        conformity_assigned_expert=assigned_cal,
    )
    prediction_frames.append(moe.predictions)
    calibration_frames.append(moe.calibration)
    conformity_frames.append(moe.conformity)

    for expert_id in range(moe_model.n_experts):
        cal_mask = assigned_cal == expert_id
        test_mask = assigned_test == expert_id
        if cal_mask.sum() < min_samples or test_mask.sum() == 0:
            continue
        expert = conformalize_prefit_model(
            model=moe_model,
            scalers=moe_scalers,
            calibration_df=calibration_df.loc[cal_mask],
            test_df=test_df.loc[test_mask],
            model_name=f"{expert_model_prefix} {expert_id}",
            expert_id=expert_id,
            assigned_expert=assigned_test[test_mask],
            conformity_assigned_expert=assigned_cal[cal_mask],
        )
        prediction_frames.append(expert.predictions)
        calibration_frames.append(expert.calibration)
        conformity_frames.append(expert.conformity)

    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(calibration_frames, ignore_index=True),
        pd.concat(conformity_frames, ignore_index=True),
    )


def static_uncertainty_diagnostics(
    *,
    nn_model: nn.Module,
    nn_scalers: Scalers,
    moe_model: MoERegressor,
    moe_scalers: Scalers,
    calibration_df: pd.DataFrame,
    test_df: pd.DataFrame,
    results_dir: Path,
    figures_dir: Path,
    min_expert_calibration_samples: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    general = conformalize_prefit_model(
        model=nn_model,
        scalers=nn_scalers,
        calibration_df=calibration_df,
        test_df=test_df,
        model_name="Static NN",
    )
    moe_predictions, moe_calibration, moe_conformity = moe_uncertainty_diagnostics(
        moe_model=moe_model,
        moe_scalers=moe_scalers,
        calibration_df=calibration_df,
        test_df=test_df,
        model_name="Static MoE",
        expert_model_prefix="Static MoE expert",
        min_expert_calibration_samples=min_expert_calibration_samples,
    )

    predictions = pd.concat([general.predictions, moe_predictions], ignore_index=True)
    calibration = pd.concat([general.calibration, moe_calibration], ignore_index=True)
    conformity = pd.concat([general.conformity, moe_conformity], ignore_index=True)
    predictions.to_csv(results_dir / "static_prediction_intervals_95.csv", index=False)
    calibration.to_csv(results_dir / "static_interval_calibration.csv", index=False)
    conformity.to_csv(results_dir / "static_conformity_scores.csv", index=False)
    write_uq_metric_summary(results_dir)
    plot_interval_calibration(calibration, figures_dir / "static_interval_calibration.png")
    plot_prediction_intervals_95(
        predictions[predictions["model"].isin(["Static NN", "Static MoE"])],
        figures_dir / "static_prediction_intervals_95.png",
    )
    plot_prediction_interval_pairplots(
        predictions,
        figures_dir,
        file_prefix="static",
        general_model_name="Static NN",
        moe_model_name="Static MoE",
        expert_model_prefix="Static MoE expert",
    )
    return predictions, calibration


def begin_end_uncertainty_diagnostics(
    *,
    initial_nn_model: nn.Module,
    initial_nn_scalers: Scalers,
    final_nn_model: nn.Module,
    final_nn_scalers: Scalers,
    initial_moe_model: MoERegressor,
    initial_moe_scalers: Scalers,
    final_retrained_moe_model: MoERegressor,
    final_retrained_moe_scalers: Scalers,
    final_adaptive_moe_model: MoERegressor,
    adaptive_moe_scalers: Scalers,
    calibration_df: pd.DataFrame,
    test_df: pd.DataFrame,
    results_dir: Path,
    figures_dir: Path,
    n_adapt_used_for_update: int,
    n_adapt_heldout_for_conformalization: int,
    min_expert_calibration_samples: int = 21,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    prediction_frames = []
    calibration_frames = []
    conformity_frames = []

    for model, scalers, name in [
        (initial_nn_model, initial_nn_scalers, "Initial Static NN"),
        (final_nn_model, final_nn_scalers, "Final Retrained NN"),
    ]:
        predictions, calibration, conformity = nn_uncertainty_diagnostics(
            nn_model=model,
            nn_scalers=scalers,
            calibration_df=calibration_df,
            test_df=test_df,
            model_name=name,
        )
        prediction_frames.append(predictions)
        calibration_frames.append(calibration)
        conformity_frames.append(conformity)

    for model, scalers, model_name, expert_prefix in [
        (initial_moe_model, initial_moe_scalers, "Initial Static MoE", "Initial Static MoE expert"),
        (final_retrained_moe_model, final_retrained_moe_scalers, "Final Retrained MoE", "Final Retrained MoE expert"),
        (initial_moe_model, initial_moe_scalers, "Initial Adaptive MoE", "Initial Adaptive MoE expert"),
        (final_adaptive_moe_model, adaptive_moe_scalers, "Final Adaptive MoE", "Final Adaptive MoE expert"),
    ]:
        predictions, calibration, conformity = moe_uncertainty_diagnostics(
            moe_model=model,
            moe_scalers=scalers,
            calibration_df=calibration_df,
            test_df=test_df,
            model_name=model_name,
            expert_model_prefix=expert_prefix,
            min_expert_calibration_samples=min_expert_calibration_samples,
        )
        prediction_frames.append(predictions)
        calibration_frames.append(calibration)
        conformity_frames.append(conformity)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    calibration = pd.concat(calibration_frames, ignore_index=True)
    conformity = pd.concat(conformity_frames, ignore_index=True)
    for frame in [predictions, calibration, conformity]:
        frame["n_adapt_used_for_update"] = int(n_adapt_used_for_update)
        frame["n_adapt_heldout_for_conformalization"] = int(n_adapt_heldout_for_conformalization)

    predictions.to_csv(results_dir / "begin_end_prediction_intervals_95.csv", index=False)
    calibration.to_csv(results_dir / "begin_end_interval_calibration.csv", index=False)
    conformity.to_csv(results_dir / "begin_end_conformity_scores.csv", index=False)
    write_uq_metric_summary(results_dir)
    plot_interval_calibration(calibration, figures_dir / "begin_end_interval_calibration.png")
    plot_begin_end_pairplot_figures(
        predictions,
        figures_dir,
        file_prefix="begin_end_nn",
        model_names=["Initial Static NN", "Final Retrained NN"],
        color_col="family",
        legend_title="family",
    )
    plot_begin_end_pairplot_figures(
        predictions,
        figures_dir,
        file_prefix="begin_end_retrained_moe",
        model_names=["Initial Static MoE", "Final Retrained MoE"],
        color_col="assigned_expert",
        legend_title="expert",
    )
    plot_begin_end_pairplot_figures(
        predictions,
        figures_dir,
        file_prefix="begin_end_adaptive_moe",
        model_names=["Initial Adaptive MoE", "Final Adaptive MoE"],
        color_col="assigned_expert",
        legend_title="expert",
    )
    return predictions, calibration


def split_adaptation_for_uq(
    adapt_df: pd.DataFrame,
    *,
    calibration_fraction: float = 0.75,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1.")
    n_train = int(round((1.0 - calibration_fraction) * len(adapt_df)))
    n_test = len(adapt_df) - n_train
    if {"family", "target_bin"}.issubset(adapt_df.columns):
        stratify = adapt_df["family"].astype(str) + "_q" + adapt_df["target_bin"].astype(str)
    else:
        stratify = adapt_df["family"]
    if stratify.value_counts().min() < 2 or min(n_train, n_test) < stratify.nunique():
        stratify = adapt_df["family"]
    update_df, conformal_df = train_test_split(
        adapt_df,
        test_size=calibration_fraction,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return update_df.sort_values("adapt_order").copy(), conformal_df.sort_values("adapt_order").copy()


def adaptive_uncertainty_diagnostics(
    *,
    base_moe_model: MoERegressor,
    moe_scalers: Scalers,
    base_df: pd.DataFrame,
    adapt_df: pd.DataFrame,
    test_df: pd.DataFrame,
    results_dir: Path,
    figures_dir: Path,
    seed: int = 42,
    calibration_fraction: float = 0.75,
    min_expert_calibration_samples: int = 21,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    import copy

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    update_df, conformal_df = split_adaptation_for_uq(
        adapt_df,
        calibration_fraction=calibration_fraction,
        seed=seed,
    )
    adaptive_moe = copy.deepcopy(base_moe_model)
    selected_experts, updated_params = adapt_moe_step(
        adaptive_moe,
        moe_scalers,
        base_df,
        update_df,
        seed=seed + 1901,
    )

    predictions, calibration, conformity = moe_uncertainty_diagnostics(
        moe_model=adaptive_moe,
        moe_scalers=moe_scalers,
        calibration_df=conformal_df,
        test_df=test_df,
        model_name="Adaptive MoE",
        expert_model_prefix="Adaptive MoE expert",
        min_expert_calibration_samples=min_expert_calibration_samples,
    )
    predictions["n_adapt_used_for_update"] = len(update_df)
    predictions["n_adapt_heldout_for_conformalization"] = len(conformal_df)
    calibration["n_adapt_used_for_update"] = len(update_df)
    calibration["n_adapt_heldout_for_conformalization"] = len(conformal_df)
    calibration["updated_params"] = int(updated_params)
    calibration["selected_experts"] = ",".join(map(str, selected_experts))
    conformity["n_adapt_used_for_update"] = len(update_df)
    conformity["n_adapt_heldout_for_conformalization"] = len(conformal_df)

    update_df.to_csv(results_dir / "adaptive_uq_update_set.csv", index=False)
    conformal_df.to_csv(results_dir / "adaptive_uq_conformalization_set.csv", index=False)
    predictions.to_csv(results_dir / "adaptive_prediction_intervals_95.csv", index=False)
    calibration.to_csv(results_dir / "adaptive_interval_calibration.csv", index=False)
    conformity.to_csv(results_dir / "adaptive_conformity_scores.csv", index=False)
    write_uq_metric_summary(results_dir)
    torch.save(adaptive_moe.state_dict(), results_dir / "adaptive_moe_uq_state.pt")
    plot_interval_calibration(calibration, figures_dir / "adaptive_interval_calibration.png")
    plot_prediction_interval_pairplots(
        predictions,
        figures_dir,
        file_prefix="adaptive",
        general_model_name=None,
        moe_model_name="Adaptive MoE",
        expert_model_prefix="Adaptive MoE expert",
    )
    return predictions, calibration


def plot_interval_calibration(calibration: pd.DataFrame, output_path: Path) -> None:

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0.45, 1.0], [0.45, 1.0], color="black", linestyle="--", linewidth=1, label="ideal")
    for (model_name, expert), group in calibration.groupby(["model", "expert"], sort=False):
        label = model_name if expert == "ensemble" else f"{model_name}"
        ax.plot(
            group["confidence_level"],
            group["empirical_coverage"],
            marker=marker_for_model(model_name),
            color=color_for_model(model_name),
            linestyle=linestyle_for_model(model_name),
            linewidth=1.8,
            label=label,
        )
    ax.set_xlabel("Nominal confidence level")
    ax.set_ylabel(r"Empirical coverage on $D_{test,shifted}$")
    # ax.set_title("MAPIE interval calibration")
    ax.set_xlim(0.48, 0.97)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_prediction_intervals_95(predictions: pd.DataFrame, output_path: Path) -> None:

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = predictions.sort_values(["model", "y_true"]).reset_index(drop=True)
    fig, axes = plt.subplots(1, plot_df["model"].nunique(), figsize=(12, 4), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    for ax, (model_name, group) in zip(axes, plot_df.groupby("model", sort=False)):
        x = np.arange(len(group))
        ax.errorbar(
            x,
            group["y_pred"],
            yerr=[
                group["y_pred"] - group["lower_95"],
                group["upper_95"] - group["y_pred"],
            ],
            fmt="o",
            markersize=3,
            linewidth=0.8,
            capsize=2,
            alpha=0.75,
            color=color_for_model(model_name),
            label="95% interval",
        )
        ax.scatter(x, group["y_true"], s=12, color="black", label=r"true $T_L$")
        # ax.set_title(model_name)
        ax.set_xlabel(r"$D_{test,shifted}$ samples sorted by true $T_L$")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"$T_L$")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _category_palette(values: Iterable[object]) -> dict[object, str]:

    unique_values = list(pd.Series(list(values)).dropna().unique())
    unique_values = sorted(unique_values, key=lambda value: str(value))
    return {value: category_color(value) for value in unique_values}


def _plot_prediction_interval_pairplot(
    plot_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    color_col: str | None,
    legend_title: str,
) -> None:

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    y_min = float(min(plot_df["y_true"].min(), plot_df["lower_95"].min(), plot_df["y_pred"].min()))
    y_max = float(max(plot_df["y_true"].max(), plot_df["upper_95"].max(), plot_df["y_pred"].max()))
    pad = 0.04 * max(1.0, y_max - y_min)
    ax.plot([y_min - pad, y_max + pad], [y_min - pad, y_max + pad], color="black", linestyle="--", linewidth=1)

    if color_col is None:
        ax.vlines(
            plot_df["y_true"],
            plot_df["lower_95"],
            plot_df["upper_95"],
            color="0.55",
            alpha=0.35,
            linewidth=1.0,
        )
        ax.scatter(
            plot_df["y_true"],
            plot_df["y_pred"],
            s=34,
            color=color_for_model(str(plot_df["model"].iloc[0])) if "model" in plot_df.columns else "#0072B2",
            edgecolors="white",
            linewidths=0.5,
            label="prediction",
            zorder=3,
        )
    else:
        palette = _category_palette(plot_df[color_col])
        for value, group in plot_df.groupby(color_col, sort=True):
            color = palette[value]
            label = f"{legend_title} {int(value)}" if isinstance(value, (int, np.integer, float, np.floating)) else str(value)
            ax.vlines(
                group["y_true"],
                group["lower_95"],
                group["upper_95"],
                color=color,
                alpha=0.32,
                linewidth=1.0,
            )
            ax.scatter(
                group["y_true"],
                group["y_pred"],
                s=36,
                color=color,
                edgecolors="white",
                linewidths=0.5,
                label=label,
                zorder=3,
            )

    ax.set_xlabel(r"True $T_L$ on $D_{test,shifted}$")
    ax.set_ylabel(r"Predicted $T_L$ with 95% MAPIE interval")
    # ax.set_title(title)
    ax.set_xlim(y_min - pad, y_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, title=legend_title if color_col else None)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_begin_end_pairplot_grid(
    predictions: pd.DataFrame,
    output_path: Path,
    *,
    model_names: list[str],
    color_col: str,
    legend_title: str,
    title: str,
) -> None:

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        predictions[(predictions["model"] == model_name) & (predictions["expert"] == "ensemble")].copy()
        for model_name in model_names
    ]
    panels = [panel for panel in panels if not panel.empty]
    if not panels:
        return

    all_df = pd.concat(panels, ignore_index=True)
    y_min = float(min(all_df["y_true"].min(), all_df["lower_95"].min(), all_df["y_pred"].min()))
    y_max = float(max(all_df["y_true"].max(), all_df["upper_95"].max(), all_df["y_pred"].max()))
    pad = 0.04 * max(1.0, y_max - y_min)
    palette = _category_palette(all_df[color_col]) if color_col in all_df.columns else {}

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.7), sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for ax, panel in zip(axes, panels):
        ax.plot([y_min - pad, y_max + pad], [y_min - pad, y_max + pad], color="black", linestyle="--", linewidth=1)
        if color_col in panel.columns:
            for value, group in panel.groupby(color_col, sort=True):
                color = palette[value]
                label = (
                    f"{legend_title} {int(value)}"
                    if isinstance(value, (int, np.integer, float, np.floating))
                    else str(value)
                )
                ax.vlines(group["y_true"], group["lower_95"], group["upper_95"], color=color, alpha=0.32, linewidth=1)
                ax.scatter(
                    group["y_true"],
                    group["y_pred"],
                    s=34,
                    color=color,
                    edgecolors="white",
                    linewidths=0.5,
                    label=label,
                    zorder=3,
                )
        else:
            ax.vlines(panel["y_true"], panel["lower_95"], panel["upper_95"], color="0.55", alpha=0.35, linewidth=1)
            ax.scatter(
                panel["y_true"],
                panel["y_pred"],
                s=34,
                color="tab:blue",
                edgecolors="white",
                linewidths=0.5,
                label="prediction",
                zorder=3,
            )
        # ax.set_title(str(panel["model"].iloc[0]))
        ax.set_xlabel(r"True $T_L$ on $D_{test,shifted}$")
        ax.set_xlim(y_min - pad, y_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(r"Predicted $T_L$ with 95% MAPIE interval")
    axes[-1].legend(fontsize=8, title=legend_title)
    # fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _slug(text: str) -> str:

    return (
        text.lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


def plot_begin_end_pairplot_figures(
    predictions: pd.DataFrame,
    figures_dir: Path,
    *,
    file_prefix: str,
    model_names: list[str],
    color_col: str,
    legend_title: str,
) -> None:

    figures_dir.mkdir(parents=True, exist_ok=True)
    for model_name in model_names:
        panel = predictions[(predictions["model"] == model_name) & (predictions["expert"] == "ensemble")].copy()
        if panel.empty:
            continue
        _plot_prediction_interval_pairplot(
            panel,
            figures_dir / f"{file_prefix}_{_slug(model_name)}_prediction_interval_pairplot.png",
            title=f"{model_name}: prediction interval pairplot",
            color_col=color_col,
            legend_title=legend_title,
        )


def plot_prediction_interval_pairplots(
    predictions: pd.DataFrame,
    figures_dir: Path,
    *,
    file_prefix: str,
    general_model_name: str | None,
    moe_model_name: str,
    expert_model_prefix: str,
) -> None:

    figures_dir.mkdir(parents=True, exist_ok=True)
    general = pd.DataFrame()
    if general_model_name is not None:
        general = predictions[(predictions["model"] == general_model_name) & (predictions["expert"] == "ensemble")]
    moe = predictions[(predictions["model"] == moe_model_name) & (predictions["expert"] == "ensemble")]
    moe_experts = predictions[predictions["model"].str.startswith(expert_model_prefix)]

    if not general.empty:
        _plot_prediction_interval_pairplot(
            general,
            figures_dir / f"{file_prefix}_nn_prediction_interval_pairplot.png",
            title=f"{general_model_name}: prediction interval pairplot",
            color_col="family",
            legend_title="family",
        )
    if not moe.empty:
        _plot_prediction_interval_pairplot(
            moe,
            figures_dir / f"{file_prefix}_moe_prediction_interval_pairplot.png",
            title=f"{moe_model_name}: prediction interval pairplot",
            color_col="assigned_expert",
            legend_title="expert",
        )
    if not moe_experts.empty:
        _plot_prediction_interval_pairplot(
            moe_experts,
            figures_dir / f"{file_prefix}_moe_expert_prediction_interval_pairplot.png",
            title=f"{moe_model_name} routed experts: prediction interval pairplot",
            color_col="assigned_expert",
            legend_title="expert",
        )
