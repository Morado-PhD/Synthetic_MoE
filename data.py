from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

from .plotting import FAMILY_COLORS

Z_COLS = ["zFe", "zCrNi", "zAl", "zNa"]
A_COLS = ["aSi", "aB", "aNa", "aAl"]
X_COLS = ["xFe", "xCrNi", "xSi", "xB", "xNa", "xAl"]
U_COLS = ["W", *Z_COLS, *A_COLS]
TARGET_COL = "T_L"
FAMILIES = ["A", "B", "C"]
SPLIT_COL = "split"


@dataclass(frozen=True)
class GenerationConfig:
    w_min: float = 0.25
    w_max: float = 0.55

    n_per_family: int = 300
    shift_quantile: float = 0.80
    adapt_fraction_of_shift: float = 0.50

    adapt_percentages: tuple[int, ...] = (0, 20, 40, 60, 80, 100)

    shifted_target_bins: int = 3
    composition_clusters: int = 3
    use_composition_clusters: bool = True

    c_crit: float = 0.34
    noise_std: float = 10.0
    seed: int = 42

    family_z_ranges: Mapping[str, Mapping[str, tuple[float, float]]] = field(
        default_factory=lambda: {
            "A": {
                "zFe": (0.45, 0.65),
                "zCrNi": (0.015, 0.07),
                "zAl": (0.10, 0.25),
                "zNa": (0.10, 0.25),
            },
            "B": {
                "zFe": (0.05, 0.18),
                "zCrNi": (0.00, 0.035),
                "zAl": (0.30, 0.50),
                "zNa": (0.35, 0.55),
            },
            "C": {
                "zFe": (0.15, 0.35),
                "zCrNi": (0.18, 0.35),
                "zAl": (0.12, 0.28),
                "zNa": (0.10, 0.28),
            },
        }
    )


def expected_split_counts(cfg: GenerationConfig) -> dict[str, int]:

    n_shift = int(round(cfg.n_per_family * (1.0 - cfg.shift_quantile)))
    n_adapt = int(round(n_shift * cfg.adapt_fraction_of_shift))
    n_test_shifted = n_shift - n_adapt
    n_base = cfg.n_per_family - n_shift
    return {"base": n_base, "adapt": n_adapt, "test_shifted": n_test_shifted, "shifted": n_shift}


def _normalize_rows(values: np.ndarray) -> np.ndarray:

    row_sums = values.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Cannot normalize a row with non-positive sum.")
    return values / row_sums


def sample_waste_spectrum(
    family: str, n: int, rng: np.random.Generator, cfg: GenerationConfig
) -> np.ndarray:

    if family not in cfg.family_z_ranges:
        raise KeyError(f"Unknown family {family!r}; expected one of {FAMILIES}.")
    ranges = cfg.family_z_ranges[family]
    raw = np.column_stack([rng.uniform(*ranges[col], size=n) for col in Z_COLS])
    return _normalize_rows(raw)


def sample_formulation_decisions(
    n: int, rng: np.random.Generator, cfg: GenerationConfig
) -> tuple[np.ndarray, np.ndarray]:

    w = rng.uniform(cfg.w_min, cfg.w_max, size=n)
    a = rng.dirichlet([1.0, 1.0, 1.0, 1.0], size=n)
    return w, a


def glass_composition(w: np.ndarray, z: np.ndarray, a: np.ndarray) -> pd.DataFrame:

    z_fe, z_crni, z_al, z_na = z.T
    a_si, a_b, a_na, a_al = a.T
    one_minus_w = 1.0 - w
    return pd.DataFrame(
        {
            "xFe": w * z_fe,
            "xCrNi": w * z_crni,
            "xSi": one_minus_w * a_si,
            "xB": one_minus_w * a_b,
            "xNa": w * z_na + one_minus_w * a_na,
            "xAl": w * z_al + one_minus_w * a_al,
        }
    )


def crystal_burden(x: pd.DataFrame) -> np.ndarray:
    

    return x["xFe"].to_numpy() + 3.0 * x["xCrNi"].to_numpy() + 0.3 * x["xAl"].to_numpy()


def liquidus_temperature_true(x: pd.DataFrame, cfg: GenerationConfig) -> np.ndarray:
    

    c = crystal_burden(x)
    base = (
        760.0
        + 520.0 * x["xFe"].to_numpy()
        + 1500.0 * x["xCrNi"].to_numpy()
        + 220.0 * x["xAl"].to_numpy()
        - 260.0 * x["xB"].to_numpy()
        + 80.0 * x["xNa"].to_numpy()
    )
    interactions = (
        2400.0 * x["xFe"].to_numpy() * x["xCrNi"].to_numpy()
        + 900.0 * x["xAl"].to_numpy() * x["xCrNi"].to_numpy()
    )
    nonlinear = 2200.0 * np.maximum(0.0, c - cfg.c_crit) ** 2
    return base + interactions + nonlinear


def log_viscosity(x: pd.DataFrame, w: np.ndarray) -> np.ndarray:
    

    return (
        1.3
        + 2.5 * x["xSi"].to_numpy()
        + 1.7 * x["xAl"].to_numpy()
        - 1.4 * x["xNa"].to_numpy()
        - 0.8 * x["xB"].to_numpy()
        + 0.3 * w
    )


def pct_durability(x: pd.DataFrame, w: np.ndarray) -> np.ndarray:

    return (
        0.5
        + 3.0 * x["xNa"].to_numpy()
        + 1.5 * w
        - 2.0 * x["xSi"].to_numpy()
        - 1.2 * x["xAl"].to_numpy()
    )


def make_family_dataset(
    family: str, n: int, rng: np.random.Generator, cfg: GenerationConfig
) -> pd.DataFrame:

    z = sample_waste_spectrum(family, n, rng, cfg)
    w, a = sample_formulation_decisions(n, rng, cfg)
    x = glass_composition(w, z, a)
    y_true = liquidus_temperature_true(x, cfg)
    y_obs = y_true + rng.normal(0.0, cfg.noise_std, size=n)

    df = pd.DataFrame({"W": w})
    for i, col in enumerate(Z_COLS):
        df[col] = z[:, i]
    for i, col in enumerate(A_COLS):
        df[col] = a[:, i]
    df = pd.concat([df, x], axis=1)
    df["C_burden"] = crystal_burden(x)
    df["T_L_true"] = y_true
    df["T_L"] = y_obs
    df["log_eta"] = log_viscosity(x, w)
    df["PCT"] = pct_durability(x, w)
    df["family"] = family
    return df


def _add_shift_strata(df: pd.DataFrame, cfg: GenerationConfig, seed: int) -> pd.DataFrame:
    

    out = df.copy()
    out["target_bin"] = pd.qcut(
        out[TARGET_COL],
        q=min(cfg.shifted_target_bins, len(out)),
        labels=False,
        duplicates="drop",
    ).astype(int)

    if cfg.use_composition_clusters:
        n_clusters = min(cfg.composition_clusters, max(1, len(out) // 8))
        labels = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit_predict(out[X_COLS].to_numpy())
        out["comp_cluster"] = labels.astype(int)
    else:
        out["comp_cluster"] = 0

    out["stratum"] = (
        out["family"].astype(str)
        + "_q"
        + out["target_bin"].astype(str)
        + "_c"
        + out["comp_cluster"].astype(str)
    )
    return out


def _safe_stratified_split(
    df: pd.DataFrame, *, test_size: float, seed: int, stratify_col: str = "stratum"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    

    stratify = df[stratify_col]
    if stratify.value_counts().min() < 2:
        stratify = df["target_bin"]
    left, right = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True, stratify=stratify)
    return left.copy(), right.copy()


def split_family_shifted_tail(
    df: pd.DataFrame, cfg: GenerationConfig, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    

    threshold = float(df[TARGET_COL].quantile(cfg.shift_quantile))
    full = df.copy()
    full["shift_threshold"] = threshold
    full["is_shifted_regime"] = full[TARGET_COL] >= threshold

    base = full[~full["is_shifted_regime"]].copy()
    shifted = full[full["is_shifted_regime"]].copy()
    shifted = _add_shift_strata(shifted, cfg, seed)

    test_size = 1.0 - cfg.adapt_fraction_of_shift
    adapt, shifted_test = _safe_stratified_split(shifted, test_size=test_size, seed=seed)

    base["target_bin"] = pd.qcut(
        base[TARGET_COL],
        q=min(cfg.shifted_target_bins, len(base)),
        labels=False,
        duplicates="drop",
    ).astype(int)
    base["comp_cluster"] = -1
    base["stratum"] = base["family"].astype(str) + "_base"

    base[SPLIT_COL] = "base"
    adapt[SPLIT_COL] = "adapt"
    shifted_test[SPLIT_COL] = "test_shifted"

    return (
        full.reset_index(drop=True),
        base.reset_index(drop=True),
        adapt.reset_index(drop=True),
        shifted_test.reset_index(drop=True),
    )


def assign_adaptation_order(adapt_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    

    rng = np.random.default_rng(seed)
    parts = []
    for _, group in adapt_df.groupby("stratum", sort=False):
        shuffled = group.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).copy()
        shuffled["_within_stratum_order"] = np.arange(len(shuffled))
        shuffled["_stratum_shuffle"] = rng.random(len(shuffled))
        parts.append(shuffled)

    ordered = pd.concat(parts, ignore_index=True).sort_values(
        ["_within_stratum_order", "_stratum_shuffle"],
        kind="mergesort",
    )
    ordered = ordered.drop(columns=["_within_stratum_order", "_stratum_shuffle"]).reset_index(drop=True)
    ordered["adapt_order"] = np.arange(1, len(ordered) + 1)
    ordered["adapt_fraction"] = ordered["adapt_order"] / len(ordered)
    return ordered


def adaptation_subset(adapt_df: pd.DataFrame, percentage: int) -> pd.DataFrame:
    

    if percentage <= 0:
        return adapt_df.iloc[0:0].copy()
    if percentage >= 100:
        return adapt_df.copy()
    n_take = int(round((percentage / 100.0) * len(adapt_df)))
    return adapt_df.sort_values("adapt_order").head(n_take).copy()


def make_all_datasets(cfg: GenerationConfig | None = None) -> Dict[str, pd.DataFrame]:
    

    cfg = cfg or GenerationConfig()
    rng = np.random.default_rng(cfg.seed)
    datasets: dict[str, pd.DataFrame] = {}
    all_rows, base_rows, adapt_rows, test_rows = [], [], [], []

    for offset, family in enumerate(FAMILIES):
        full_raw = make_family_dataset(family, cfg.n_per_family, rng, cfg)
        full, base, adapt, shifted_test = split_family_shifted_tail(full_raw, cfg, cfg.seed + offset * 17)
        all_rows.append(full)
        base_rows.append(base)
        adapt_rows.append(adapt)
        test_rows.append(shifted_test)

        datasets[f"D_{family}_all"] = full
        datasets[f"D_{family}_base"] = base
        datasets[f"D_{family}_adapt"] = adapt
        datasets[f"D_{family}_test_shifted"] = shifted_test

    base_all = pd.concat(base_rows, ignore_index=True)
    adapt_all = assign_adaptation_order(pd.concat(adapt_rows, ignore_index=True), cfg.seed + 101)
    test_shifted_all = pd.concat(test_rows, ignore_index=True)
    all_data = pd.concat(all_rows, ignore_index=True)
    shifted_all = pd.concat([adapt_all, test_shifted_all], ignore_index=True)

    datasets.update(
        {
            "D_all": all_data,
            "D_base": base_all,
            "D_adapt": adapt_all,
            "D_test_shifted": test_shifted_all,
            "D_shifted": shifted_all,
        }
    )
    return datasets


def write_datasets(datasets: Mapping[str, pd.DataFrame], data_dir: Path) -> None:
    

    data_dir.mkdir(parents=True, exist_ok=True)
    for stale_csv in data_dir.glob("*.csv"):
        stale_csv.unlink()
    for name, df in datasets.items():
        df.to_csv(data_dir / f"{name}.csv", index=False)


def load_datasets(data_dir: Path) -> Dict[str, pd.DataFrame]:
    

    return {csv_path.stem: pd.read_csv(csv_path) for csv_path in sorted(data_dir.glob("*.csv"))}


def base_train_set(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    

    return datasets["D_base"].copy()


def adaptation_set(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    

    return datasets["D_adapt"].copy()


def final_test_set(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    

    return datasets["D_test_shifted"].copy()


def family_test_sets(datasets: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    

    test_df = final_test_set(datasets)
    out = {family: test_df[test_df["family"] == family].copy() for family in FAMILIES}
    out["ALL"] = test_df.copy()
    return out


def summarize_datasets(datasets: Mapping[str, pd.DataFrame], cfg: GenerationConfig) -> pd.DataFrame:
    

    rows = []
    names = ["D_all", "D_base", "D_shifted", "D_adapt", "D_test_shifted"]
    names.extend([f"D_{family}_{split}" for family in FAMILIES for split in ["all", "base", "adapt", "test_shifted"]])
    for name in names:
        if name not in datasets:
            continue
        df = datasets[name]
        rows.append(
            {
                "dataset": name,
                "n": len(df),
                "families": ",".join(sorted(df["family"].unique())),
                "T_L_mean": df[TARGET_COL].mean(),
                "T_L_sd": df[TARGET_COL].std(),
                "T_L_min": df[TARGET_COL].min(),
                "T_L_p50": df[TARGET_COL].median(),
                "T_L_max": df[TARGET_COL].max(),
                "C_burden_mean": df["C_burden"].mean(),
                "frac_above_Ccrit": (df["C_burden"] > cfg.c_crit).mean(),
            }
        )
    return pd.DataFrame(rows)


def verify_datasets(datasets: Mapping[str, pd.DataFrame], cfg: GenerationConfig) -> list[str]:

    counts = expected_split_counts(cfg)
    checks: list[str] = []
    expected_counts = {
        "D_all": cfg.n_per_family * len(FAMILIES),
        "D_base": counts["base"] * len(FAMILIES),
        "D_adapt": counts["adapt"] * len(FAMILIES),
        "D_test_shifted": counts["test_shifted"] * len(FAMILIES),
        "D_shifted": counts["shifted"] * len(FAMILIES),
    }
    for family in FAMILIES:
        expected_counts[f"D_{family}_all"] = cfg.n_per_family
        expected_counts[f"D_{family}_base"] = counts["base"]
        expected_counts[f"D_{family}_adapt"] = counts["adapt"]
        expected_counts[f"D_{family}_test_shifted"] = counts["test_shifted"]

    for name, expected_n in expected_counts.items():
        df = datasets[name]
        checks.extend(
            [
                f"{name}: row count = {len(df) == expected_n} ({len(df)} vs {expected_n})",
                f"{name}: z closure = {np.allclose(df[Z_COLS].sum(axis=1), 1.0)}",
                f"{name}: additive closure = {np.allclose(df[A_COLS].sum(axis=1), 1.0)}",
                f"{name}: glass composition closure = {np.allclose(df[X_COLS].sum(axis=1), 1.0)}",
                f"{name}: W bounds = {df['W'].between(cfg.w_min, cfg.w_max).all()}",
            ]
        )

    for family in FAMILIES:
        base = datasets[f"D_{family}_base"]
        adapt = datasets[f"D_{family}_adapt"]
        shifted_test = datasets[f"D_{family}_test_shifted"]
        threshold = float(datasets[f"D_{family}_all"]["shift_threshold"].iloc[0])
        checks.append(f"{family}: base max below q{cfg.shift_quantile:.2f} = {base[TARGET_COL].max() < threshold}")
        checks.append(f"{family}: adapt min in shifted tail = {adapt[TARGET_COL].min() >= threshold}")
        checks.append(f"{family}: shifted test min in shifted tail = {shifted_test[TARGET_COL].min() >= threshold}")

    adapt_df = datasets["D_adapt"]
    for percentage in cfg.adapt_percentages:
        subset = adaptation_subset(adapt_df, percentage)
        family_counts = subset["family"].value_counts().reindex(FAMILIES, fill_value=0).to_dict()
        checks.append(f"D_adapt: {percentage}% prefix has {len(subset)} rows by family {family_counts}")
    return checks


def adaptation_balance_summary(adapt_df: pd.DataFrame, cfg: GenerationConfig) -> pd.DataFrame:
    

    rows = []
    for percentage in cfg.adapt_percentages:
        subset = adaptation_subset(adapt_df, percentage)
        for family in FAMILIES:
            fam_df = subset[subset["family"] == family]
            rows.append(
                {
                    "adapt_pct": percentage,
                    "family": family,
                    "n": len(fam_df),
                    "target_bin_count": int(fam_df["target_bin"].nunique()) if len(fam_df) else 0,
                    "comp_cluster_count": int(fam_df["comp_cluster"].nunique()) if len(fam_df) else 0,
                    "T_L_min": fam_df[TARGET_COL].min() if len(fam_df) else np.nan,
                    "T_L_max": fam_df[TARGET_COL].max() if len(fam_df) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def create_distribution_shift_plot(datasets: Mapping[str, pd.DataFrame], cfg: GenerationConfig, results_dir: Path) -> None:
    

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, family in zip(axes, FAMILIES):
        full = datasets[f"D_{family}_all"]
        threshold = float(full["shift_threshold"].iloc[0])
        ax.hist(full[TARGET_COL], bins=22, color="#9ecae1", edgecolor="white", alpha=0.9)
        ax.axvline(threshold, color="#d62728", linestyle="--", linewidth=1.8, label=f"q0.80 = {threshold:.1f}")
        ax.axvspan(threshold, full[TARGET_COL].max(), color="#d62728", alpha=0.16, label="withheld data")
        ax.set_title(f"Family {family}")
        ax.set_xlabel(r"$T_L$")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Count")
    axes[-1].legend(fontsize=8)
    # fig.suptitle(r"$T_L$ distribution with region removed from base training")
    fig.tight_layout()
    fig.savefig(results_dir / "tl_distribution_withheld_region.png", dpi=200)
    plt.close(fig)


def create_adaptation_diagnostics(datasets: Mapping[str, pd.DataFrame], results_dir: Path) -> pd.DataFrame:
    

    results_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(results_dir.glob("*.png")) + list(results_dir.glob("*.csv")):
        stale.unlink()
    cfg = GenerationConfig()
    base_df = datasets["D_base"].copy()
    adapt_df = datasets["D_adapt"].copy()
    test_df = datasets["D_test_shifted"].copy()
    combined = pd.concat([base_df, adapt_df, test_df], ignore_index=True)

    create_distribution_shift_plot(datasets, cfg, results_dir)

    # PCA coverage: scaler and PCA are fit using base-training data only.
    scaler = StandardScaler().fit(base_df[U_COLS].to_numpy())
    pca = PCA(n_components=2, random_state=0).fit(scaler.transform(base_df[U_COLS].to_numpy()))
    coords = pca.transform(scaler.transform(combined[U_COLS].to_numpy()))
    plot_df = combined[["family", SPLIT_COL, TARGET_COL]].copy()
    plot_df["PC1"] = coords[:, 0]
    plot_df["PC2"] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    markers = {"base": "o", "adapt": "^", "test_shifted": "x"}
    colors = FAMILY_COLORS
    for split, marker in markers.items():
        for family, color in colors.items():
            pts = plot_df[(plot_df[SPLIT_COL] == split) & (plot_df["family"] == family)]
            ax.scatter(pts["PC1"], pts["PC2"], s=28, alpha=0.62, marker=marker, color=color, label=f"{family}-{split}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    # ax.set_title("PCA coverage: base, adaptation, and shifted test")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(results_dir / "pca_coverage.png", dpi=200)
    plt.close(fig)

    # Residual-learnability diagnostic on D_adapt.
    base_model = LinearRegression().fit(base_df[U_COLS].to_numpy(), base_df[TARGET_COL].to_numpy())
    adapt_pred = base_model.predict(adapt_df[U_COLS].to_numpy())
    adapt_residual = adapt_df[TARGET_COL].to_numpy() - adapt_pred
    rf = RandomForestRegressor(n_estimators=120, random_state=0, min_samples_leaf=4)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    residual_pred = cross_val_predict(rf, adapt_df[U_COLS].to_numpy(), adapt_residual, cv=cv)
    residual_r2 = float(r2_score(adapt_residual, residual_pred))

    residual_df = adapt_df[["family", TARGET_COL, "C_burden", "target_bin", "comp_cluster"]].copy()
    residual_df["base_model_pred"] = adapt_pred
    residual_df["base_model_residual"] = adapt_residual
    residual_df["cv_residual_pred"] = residual_pred
    residual_df.to_csv(results_dir / "residual_learnability.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    for family, color in colors.items():
        pts = residual_df[residual_df["family"] == family]
        ax.scatter(pts["C_burden"], pts["base_model_residual"], alpha=0.70, s=30, color=color, label=family)
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Crystal burden C(x)")
    ax.set_ylabel("Adaptation residual: true T_L - base linear prediction")
    # ax.set_title(f"Residual learnability on D_adapt (CV residual R2 = {residual_r2:.2f})")
    ax.legend(title="Family")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(results_dir / "residual_learnability.png", dpi=200)
    plt.close(fig)

    return pd.DataFrame(
        [
            {
                "diagnostic": "tl_distribution_withheld_region",
                "path": str(results_dir / "tl_distribution_withheld_region.png"),
                "value": np.nan,
                "interpretation": "q0.80 high-TL tail removed from base training",
            },
            {
                "diagnostic": "pca_coverage",
                "path": str(results_dir / "pca_coverage.png"),
                "value": float(pca.explained_variance_ratio_.sum()),
                "interpretation": "variance explained by base-fit first two PCs",
            },
            {
                "diagnostic": "residual_learnability",
                "path": str(results_dir / "residual_learnability.png"),
                "value": residual_r2,
                "interpretation": "cross-validated R2 for learning base-model residuals on D_adapt",
            },
        ]
    )
