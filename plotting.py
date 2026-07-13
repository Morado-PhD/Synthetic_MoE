
from __future__ import annotations

MODEL_COLORS: dict[str, str] = {
    "Static NN": "#0072B2",
    "Initial Static NN": "#0072B2",
    "Static MoE": "#E69F00",
    "Initial Static MoE": "#E69F00",
    "Retrained NN": "#56B4E9",
    "Final Retrained NN": "#56B4E9",
    "Retrained MoE": "#009E73",
    "Final Retrained MoE": "#009E73",
    "Adaptive MoE": "#D55E00",
    "Initial Adaptive MoE": "#D55E00",
    "Final Adaptive MoE": "#D55E00",
    "Adaptive MoE shuffled": "#CC79A7",
}

EXPERT_COLORS: dict[int, str] = {
    0: "#0072B2",
    1: "#E69F00",
    2: "#009E73",
    3: "#CC79A7",
    4: "#D55E00",
}

FAMILY_COLORS: dict[str, str] = {
    "A": "#0072B2",
    "B": "#E69F00",
    "C": "#009E73",
}


def color_for_model(model_name: str) -> str:

    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]
    for key, color in MODEL_COLORS.items():
        if model_name.startswith(f"{key} expert"):
            return color
    return "#4D4D4D"


def linestyle_for_model(model_name: str) -> str:

    if model_name.startswith("Initial "):
        return "--"
    if model_name.startswith("Adaptive MoE shuffled"):
        return ":"
    return "-"



def marker_for_model(model_name: str) -> str:

    if "NN" in model_name:
        return "o"
    if model_name == "Adaptive MoE":
        return "D"
    if "MoE" in model_name:
        return "s"
    return "o"


def category_color(value: object) -> str:

    if isinstance(value, str) and value in FAMILY_COLORS:
        return FAMILY_COLORS[value]
    try:
        return EXPERT_COLORS[int(value)]
    except (TypeError, ValueError):
        return "#4D4D4D"
