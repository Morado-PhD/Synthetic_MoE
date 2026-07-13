
from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data import FAMILIES, TARGET_COL, U_COLS
from .plotting import FAMILY_COLORS


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
DEVICE = torch.device("cpu")
DEFAULT_MLP_PARAMS: dict[str, Any] = {
    "num_layers": 2,
    "hidden_dims": [64, 64],
    "activation_name": "GELU",
    "dropout_rate": 0.05,
    "use_batch_norm": False,
    "optimizer_name": "Adam",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "scheduler_name": "plateau",
    "noise_std": 0.0,
}


def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


@dataclass
class Scalers:
    x: StandardScaler
    y: StandardScaler


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:

    if trainable_only:
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return int(sum(p.numel() for p in model.parameters()))


def fit_scalers(train_df: pd.DataFrame) -> Scalers:

    return Scalers(
        x=StandardScaler().fit(train_df[U_COLS].to_numpy()),
        y=StandardScaler().fit(train_df[[TARGET_COL]].to_numpy()),
    )


def to_tensors(df: pd.DataFrame, scalers: Scalers) -> tuple[torch.Tensor, torch.Tensor]:

    x = scalers.x.transform(df[U_COLS].to_numpy()).astype(np.float32)
    y = scalers.y.transform(df[[TARGET_COL]].to_numpy()).astype(np.float32).ravel()
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def inverse_y(y_scaled: np.ndarray | torch.Tensor, scalers: Scalers) -> np.ndarray:

    if isinstance(y_scaled, torch.Tensor):
        y_scaled = y_scaled.detach().cpu().numpy()
    return scalers.y.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()


def activation_from_name(name: str) -> nn.Module:
    registry = {
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
        "ELU": nn.ELU,
        "SELU": nn.SELU,
        "SiLU": nn.SiLU,
        "GELU": nn.GELU,
        "Tanh": nn.Tanh,
    }
    if name not in registry:
        raise ValueError(f"Unknown activation {name!r}.")
    return registry[name]()


def normalize_mlp_params(params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(DEFAULT_MLP_PARAMS)
    if params:
        merged.update(dict(params))
    num_layers = int(merged["num_layers"])
    hidden_dims = list(merged.get("hidden_dims", []))
    if len(hidden_dims) < num_layers:
        hidden_dims.extend([hidden_dims[-1] if hidden_dims else 64] * (num_layers - len(hidden_dims)))
    merged["hidden_dims"] = [int(v) for v in hidden_dims[:num_layers]]
    return merged


def suggest_mlp_params(trial: optuna.Trial) -> dict[str, Any]:
    params: dict[str, Any] = {}
    params["num_layers"] = trial.suggest_int("num_layers", 1, 3)

    hidden_dim_choices = [16, 32, 64, 128]
    params["hidden_dims"] = []
    for layer_idx in range(params["num_layers"]):
        params["hidden_dims"].append(
            trial.suggest_categorical(f"hidden_dim_l{layer_idx + 1}", hidden_dim_choices)
        )

    params["activation_name"] = trial.suggest_categorical(
        "activation_name",
        ["ReLU", "LeakyReLU", "ELU", "SELU", "SiLU", "GELU", "Tanh"],
    )
    params["dropout_rate"] = trial.suggest_float("dropout_rate", 0.0, 0.50)
    params["use_batch_norm"] = trial.suggest_categorical("use_batch_norm", [True, False])
    params["optimizer_name"] = "Adam"
    params["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 5e-2, log=True)
    params["weight_decay"] = trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True)
    params["scheduler_name"] = trial.suggest_categorical("scheduler_name", ["none", "plateau", "cosine"])
    params["noise_std"] = trial.suggest_float("noise_std", 0.0, 0.05)
    return params


class TunableMLP(nn.Module):
    def __init__(
        self,
        d_in: int = 9,
        hidden_dims: list[int] | None = None,
        activation_name: str = "GELU",
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [64, 64]
        layers: list[nn.Module] = []
        prev_dim = d_in
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation_from_name(activation_name))
            if dropout_rate > 0:
                dropout_cls = nn.AlphaDropout if activation_name == "SELU" else nn.Dropout
                layers.append(dropout_cls(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPRegressor(TunableMLP):
    """Static general neural-network surrogate."""


class ExpertMLP(TunableMLP):
    """One tunable local expert inside the MoE."""


class SoftmaxGate(nn.Module):
    """Input-dependent expert weighting network."""

    def __init__(self, d_in: int = 9, d_hidden: int = 32, n_experts: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_experts),
        )

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def weights(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(x), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)


class MoERegressor(nn.Module):
    """Mixture-of-experts surrogate."""

    def __init__(
        self,
        d_in: int = 9,
        n_experts: int = 3,
        expert_params: Mapping[str, Any] | list[Mapping[str, Any]] | None = None,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.gate = SoftmaxGate(d_in=d_in, d_hidden=32, n_experts=n_experts)
        if isinstance(expert_params, list):
            params_by_expert = [normalize_mlp_params(p) for p in expert_params]
        else:
            params_by_expert = [normalize_mlp_params(expert_params) for _ in range(n_experts)]
        self.experts = nn.ModuleList(
            [
                ExpertMLP(
                    d_in=d_in,
                    hidden_dims=params["hidden_dims"],
                    activation_name=params["activation_name"],
                    dropout_rate=params["dropout_rate"],
                    use_batch_norm=params["use_batch_norm"],
                )
                for params in params_by_expert
            ]
        )
        self.expert_params = params_by_expert

    def expert_outputs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([expert(x) for expert in self.experts], dim=-1)

    def forward(self, x: torch.Tensor, return_weights: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        weights = self.gate.weights(x)
        expert_values = self.expert_outputs(x)
        y = (weights * expert_values).sum(dim=-1)
        return (y, weights) if return_weights else y


def _train_val_tensors(
    df: pd.DataFrame,
    scalers: Scalers,
    seed: int,
    val_fraction: float = 0.20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:

    x, y = to_tensors(df, scalers)
    idx = np.arange(len(df))
    train_idx, val_idx = train_test_split(idx, test_size=val_fraction, random_state=seed, shuffle=True)
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx], train_idx, val_idx


def _make_scheduler(optimizer: torch.optim.Optimizer, scheduler_name: str, epochs: int):

    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=12)
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    return None


def train_regression_loop(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    seed: int,
    params: Mapping[str, Any] | None = None,
    lr: float | None = None,
    epochs: int = 260,
    batch_size: int = 64,
    weight_decay: float | None = None,
    patience: int = 45,
    trial: optuna.Trial | None = None,
) -> dict[str, Any]:

    params = normalize_mlp_params(params)
    learning_rate = float(lr if lr is not None else params["learning_rate"])
    wd = float(weight_decay if weight_decay is not None else params["weight_decay"])
    noise_std = float(params.get("noise_std", 0.0))

    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
    scheduler = _make_scheduler(optimizer, params.get("scheduler_name", "none"), epochs)
    rng = torch.Generator().manual_seed(seed)
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale_epochs = 0
    history: dict[str, Any] = {"train_loss": [], "val_loss": [], "best_val_loss": None, "best_epoch": None}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x_train), generator=rng)
        batch_losses = []
        for start in range(0, len(x_train), batch_size):
            idx = perm[start : start + batch_size]
            if len(idx) < 2 and any(isinstance(m, nn.BatchNorm1d) for m in model.modules()):
                continue
            xb = x_train[idx]
            yb = y_train[idx]
            if noise_std > 0:
                xb = xb + torch.randn(xb.shape, generator=rng, dtype=xb.dtype) * noise_std
            optimizer.zero_grad()
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
            val_loss = float(F.mse_loss(model(x_val), y_val).item())
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if scheduler is not None:
            if params.get("scheduler_name") == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    history["best_val_loss"] = best_loss
    history["best_epoch"] = best_epoch
    return history


def tune_mlp_params(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    seed: int,
    n_trials: int = 12,
    tune_epochs: int = 120,
    study_name: str = "mlp_tuning",
) -> tuple[dict[str, Any], pd.DataFrame]:

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_mlp_params(trial)
        set_seed(seed + trial.number)
        model = MLPRegressor(
            d_in=x_train.shape[1],
            hidden_dims=params["hidden_dims"],
            activation_name=params["activation_name"],
            dropout_rate=params["dropout_rate"],
            use_batch_norm=params["use_batch_norm"],
        )
        history = train_regression_loop(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            seed=seed + trial.number,
            params=params,
            epochs=tune_epochs,
            patience=max(18, tune_epochs // 4),
            trial=trial,
        )
        return float(history["best_val_loss"])

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name=study_name)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = normalize_mlp_params(study.best_trial.user_attrs or study.best_params)
    if "hidden_dims" not in best_params or any(k.startswith("hidden_dim_l") for k in study.best_params):
        best_params = dict(study.best_params)
        best_params["hidden_dims"] = [
            best_params[f"hidden_dim_l{i + 1}"] for i in range(int(best_params["num_layers"]))
        ]
        for key in list(best_params):
            if key.startswith("hidden_dim_l"):
                del best_params[key]
        best_params["optimizer_name"] = "Adam"
        best_params = normalize_mlp_params(best_params)
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    return best_params, trials_df


def save_json(path: Path, payload: Mapping[str, Any]) -> None:


    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_loss_history(history: Mapping[str, Any], path: Path, title: str) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.get("train_loss", []), label="train loss", linewidth=1.8)
    ax.plot(history.get("val_loss", []), label="validation loss", linewidth=1.8)
    best_epoch = history.get("best_epoch")
    if best_epoch is not None:
        ax.axvline(int(best_epoch), color="black", linestyle="--", linewidth=1, label="best validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss on scaled T_L")
    # ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def train_mlp(
    train_df: pd.DataFrame,
    seed: int = 42,
    epochs: int = 300,
    *,
    tune: bool = False,
    n_trials: int = 12,
    tune_epochs: int = 120,
    params: Mapping[str, Any] | None = None,
    output_dir: Path | None = None,
    model_label: str = "general_nn",
) -> tuple[MLPRegressor, Scalers]:

    set_seed(seed)
    scalers = fit_scalers(train_df)
    x_tr, y_tr, x_val, y_val, _, _ = _train_val_tensors(train_df, scalers, seed)

    if tune:
        params, trials_df = tune_mlp_params(
            x_tr,
            y_tr,
            x_val,
            y_val,
            seed=seed,
            n_trials=n_trials,
            tune_epochs=tune_epochs,
            study_name=f"{model_label}_optuna",
        )
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            trials_df.to_csv(output_dir / f"{model_label}_optuna_trials.csv", index=False)
    params = normalize_mlp_params(params)
    model = MLPRegressor(
        d_in=x_tr.shape[1],
        hidden_dims=params["hidden_dims"],
        activation_name=params["activation_name"],
        dropout_rate=params["dropout_rate"],
        use_batch_norm=params["use_batch_norm"],
    )
    history = train_regression_loop(model, x_tr, y_tr, x_val, y_val, seed=seed, params=params, epochs=epochs)
    model.training_history_ = history
    model.best_params_ = params

    if output_dir is not None:
        save_json(output_dir / f"{model_label}_best_params.json", params)
        plot_loss_history(history, output_dir / f"{model_label}_loss.png", f"{model_label}: train vs validation loss")
    return model, scalers


def _train_gate(
    gate: SoftmaxGate,
    x_train: torch.Tensor,
    labels_train: torch.Tensor,
    x_val: torch.Tensor,
    labels_val: torch.Tensor,
    *,
    seed: int,
    epochs: int = 120,
) -> None:

    optimizer = torch.optim.Adam(gate.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = torch.Generator().manual_seed(seed)
    best_loss = float("inf")
    best_state = copy.deepcopy(gate.state_dict())

    for _ in range(epochs):
        gate.train()
        perm = torch.randperm(len(x_train), generator=rng)
        for start in range(0, len(x_train), 64):
            idx = perm[start : start + 64]
            optimizer.zero_grad()
            loss = F.cross_entropy(gate(x_train[idx]), labels_train[idx])
            loss.backward()
            optimizer.step()

        gate.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(gate(x_val), labels_val).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(gate.state_dict())

    gate.load_state_dict(best_state)


def train_moe(
    train_df: pd.DataFrame,
    seed: int = 42,
    n_experts: int = 3,
    expert_epochs: int = 180,
    gate_epochs: int = 120,
    joint_epochs: int = 180,
    *,
    tune_experts: bool = False,
    n_trials: int = 8,
    tune_epochs: int = 90,
    expert_params: list[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    output_dir: Path | None = None,
) -> tuple[MoERegressor, Scalers, KMeans]:

    set_seed(seed)
    scalers = fit_scalers(train_df)
    x_all, y_all = to_tensors(train_df, scalers)
    idx = np.arange(len(train_df))
    train_idx, val_idx = train_test_split(idx, test_size=0.20, random_state=seed, shuffle=True)
    x_tr, y_tr = x_all[train_idx], y_all[train_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]

    kmeans = KMeans(n_clusters=n_experts, n_init=20, random_state=seed)
    labels_tr_np = kmeans.fit_predict(x_tr.numpy())
    labels_val_np = kmeans.predict(x_val.numpy())

    tuned_params: list[dict[str, Any]] = []
    if isinstance(expert_params, list):
        tuned_params = [normalize_mlp_params(p) for p in expert_params]
    elif isinstance(expert_params, Mapping):
        tuned_params = [normalize_mlp_params(expert_params) for _ in range(n_experts)]
    else:
        tuned_params = [normalize_mlp_params(None) for _ in range(n_experts)]

    if tune_experts:
        for expert_id in range(n_experts):
            mask_tr = labels_tr_np == expert_id
            if not np.any(mask_tr):
                continue
            mask_val = labels_val_np == expert_id
            val_x = x_val[mask_val] if np.any(mask_val) else x_val
            val_y = y_val[mask_val] if np.any(mask_val) else y_val
            best_params, trials_df = tune_mlp_params(
                x_tr[mask_tr],
                y_tr[mask_tr],
                val_x,
                val_y,
                seed=seed + expert_id,
                n_trials=n_trials,
                tune_epochs=tune_epochs,
                study_name=f"expert_{expert_id}_optuna",
            )
            tuned_params[expert_id] = best_params
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                trials_df.to_csv(output_dir / f"expert_{expert_id}_optuna_trials.csv", index=False)

    model = MoERegressor(n_experts=n_experts, expert_params=tuned_params)
    expert_histories = []
    for expert_id, expert in enumerate(model.experts):
        mask_tr = labels_tr_np == expert_id
        if not np.any(mask_tr):
            expert_histories.append(None)
            continue
        mask_val = labels_val_np == expert_id
        val_x = x_val[mask_val] if np.any(mask_val) else x_val
        val_y = y_val[mask_val] if np.any(mask_val) else y_val
        history = train_regression_loop(
            expert,
            x_tr[mask_tr],
            y_tr[mask_tr],
            val_x,
            val_y,
            seed=seed + expert_id,
            params=tuned_params[expert_id],
            epochs=expert_epochs,
        )
        expert.training_history_ = history
        expert.best_params_ = tuned_params[expert_id]
        expert_histories.append(history)
        if output_dir is not None:
            save_json(output_dir / f"expert_{expert_id}_best_params.json", tuned_params[expert_id])
            plot_loss_history(
                history,
                output_dir / f"expert_{expert_id}_loss.png",
                f"MoE expert {expert_id}: train vs validation loss",
            )

    _train_gate(
        model.gate,
        x_tr,
        torch.tensor(labels_tr_np, dtype=torch.long),
        x_val,
        torch.tensor(labels_val_np, dtype=torch.long),
        seed=seed,
        epochs=gate_epochs,
    )
    joint_history = train_regression_loop(
        model,
        x_tr,
        y_tr,
        x_val,
        y_val,
        seed=seed,
        params=normalize_mlp_params(None),
        lr=5e-4,
        weight_decay=1e-4,
        epochs=joint_epochs,
    )
    model.training_history_ = joint_history
    model.expert_histories_ = expert_histories
    model.expert_best_params_ = tuned_params
    model.kmeans_train_labels_ = labels_tr_np
    model.kmeans_val_labels_ = labels_val_np

    if output_dir is not None:
        save_json(output_dir / "moe_expert_best_params.json", {"experts": tuned_params})
        plot_loss_history(joint_history, output_dir / "moe_joint_loss.png", "MoE joint fine-tuning loss")
    return model, scalers, kmeans


def predict(model: nn.Module, df: pd.DataFrame, scalers: Scalers) -> np.ndarray:

    x, _ = to_tensors(df, scalers)
    model.eval()
    with torch.no_grad():
        y_scaled = model(x)
    return inverse_y(y_scaled, scalers)


def evaluate_final_test(
    model: nn.Module,
    test_df: pd.DataFrame,
    scalers: Scalers,
    *,
    model_name: str,
    adapt_pct: int,
    training_time_sec: float,
    cumulative_training_time_sec: float,
    updated_params: int,
    constraint_violation_rate: float = np.nan,
) -> pd.DataFrame:

    rows = []
    for family in [*FAMILIES, "ALL"]:
        df = test_df if family == "ALL" else test_df[test_df["family"] == family]
        truth = df[TARGET_COL].to_numpy()
        pred = predict(model, df, scalers)
        rows.append(
            {
                "model": model_name,
                "adapt_pct": adapt_pct,
                "n_adapt_used": np.nan,
                "family": family,
                "MAE": float(mean_absolute_error(truth, pred)),
                "RMSE": float(np.sqrt(mean_squared_error(truth, pred))),
                "R2": float(r2_score(truth, pred)),
                "bias": float(np.mean(pred - truth)),
                "training_time_sec": float(training_time_sec),
                "cumulative_training_time_sec": float(cumulative_training_time_sec),
                "updated_params": int(updated_params),
                "constraint_violation_rate": constraint_violation_rate,
            }
        )
    return pd.DataFrame(rows)


def routing_summary(model: MoERegressor, df: pd.DataFrame, scalers: Scalers, group_col: str = "family") -> pd.DataFrame:

    rows = []
    for group_value, group_df in df.groupby(group_col):
        x, _ = to_tensors(group_df, scalers)
        model.eval()
        with torch.no_grad():
            weights = model.gate.weights(x).numpy()
        assigned = weights.argmax(axis=1)
        for expert_id in range(model.n_experts):
            rows.append(
                {
                    group_col: group_value,
                    "expert": expert_id,
                    "count": int(np.sum(assigned == expert_id)),
                    "fraction": float(np.mean(assigned == expert_id)),
                    "mean_weight": float(weights[:, expert_id].mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_kmeans_family_clusters(train_df: pd.DataFrame, scalers: Scalers, kmeans: KMeans, output_path: Path) -> pd.DataFrame:

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_scaled = scalers.x.transform(train_df[U_COLS].to_numpy()).astype(np.float32)
    clusters = kmeans.predict(x_scaled)
    pca = PCA(n_components=2, random_state=0).fit(x_scaled)
    coords = pca.transform(x_scaled)
    plot_df = train_df[["family", TARGET_COL, "zFe", "zCrNi"]].copy()
    plot_df["cluster"] = clusters
    plot_df["PC1"] = coords[:, 0]
    plot_df["PC2"] = coords[:, 1]

    family_colors = FAMILY_COLORS
    cluster_markers = ["o", "s", "^", "D", "P"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for cluster_id in sorted(plot_df["cluster"].unique()):
        for family in FAMILIES:
            pts = plot_df[(plot_df["cluster"] == cluster_id) & (plot_df["family"] == family)]
            marker = cluster_markers[int(cluster_id) % len(cluster_markers)]
            axes[0].scatter(
                pts["zFe"],
                pts["zCrNi"],
                c=family_colors[family],
                marker=marker,
                edgecolors="black",
                linewidths=0.3,
                s=35,
                alpha=0.75,
                label=f"{family}, cluster {cluster_id}",
            )
            axes[1].scatter(
                pts["PC1"],
                pts["PC2"],
                c=family_colors[family],
                marker=marker,
                edgecolors="black",
                linewidths=0.3,
                s=35,
                alpha=0.75,
            )

    axes[0].set_xlabel("zFe")
    axes[0].set_ylabel("zCrNi")
    # axes[0].set_title("Families by color, KMeans clusters by marker")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    # axes[1].set_title("Same clusters in PCA feature space")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8)
    fig.tight_layout(rect=[0, 0, 0.83, 1])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return plot_df


def adapt_moe_step(
    model: MoERegressor,
    scalers: Scalers,
    base_df: pd.DataFrame,
    seen_adapt_df: pd.DataFrame,
    *,
    seed: int,
    replay_size: int = 180,
    epochs: int = 110,
    replay_weight: float = 0.40,
    gate_weight: float = 0.08,
    shuffle_targets: bool = False,
) -> tuple[list[int], int]:

    if len(seen_adapt_df) == 0:
        return [], 0

    adapt_x, adapt_y = to_tensors(seen_adapt_df, scalers)
    if shuffle_targets and len(adapt_y) > 1:
        perm = torch.randperm(len(adapt_y), generator=torch.Generator().manual_seed(seed + 7919))
        adapt_y = adapt_y[perm]
    base_x, base_y = to_tensors(base_df, scalers)
    rng = np.random.default_rng(seed)

    model.eval()
    with torch.no_grad():
        adapt_labels = model.gate.weights(adapt_x).argmax(dim=1)
        base_labels_all = model.gate.weights(base_x).argmax(dim=1)

    selected_experts = sorted(int(v) for v in adapt_labels.unique().tolist())
    for expert_id, expert in enumerate(model.experts):
        for param in expert.parameters():
            param.requires_grad = expert_id in selected_experts
    for param in model.gate.parameters():
        param.requires_grad = True

    updated_params = count_parameters(model, trainable_only=True)
    replay_n = min(replay_size, len(base_df))
    replay_idx = rng.choice(len(base_df), size=replay_n, replace=False)
    replay_x = base_x[replay_idx]
    replay_y = base_y[replay_idx]
    replay_labels = base_labels_all[replay_idx]

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=7e-4, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss_adapt = F.mse_loss(model(adapt_x), adapt_y)
        loss_replay = F.mse_loss(model(replay_x), replay_y)
        gate_adapt = F.cross_entropy(model.gate(adapt_x), adapt_labels)
        gate_replay = F.cross_entropy(model.gate(replay_x), replay_labels)
        loss = loss_adapt + replay_weight * loss_replay + gate_weight * (gate_adapt + gate_replay)
        loss.backward()
        opt.step()

        perm = torch.randperm(len(replay_x), generator=gen)
        replay_x = replay_x[perm]
        replay_y = replay_y[perm]
        replay_labels = replay_labels[perm]

    for param in model.parameters():
        param.requires_grad = True
    return selected_experts, updated_params
