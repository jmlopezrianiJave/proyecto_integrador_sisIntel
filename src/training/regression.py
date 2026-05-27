"""Random Forest regression for future climate window prediction."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REG_TARGETS = {
    "pred_temp_t1": ("temp_media", 1, "°C"),
    "pred_prec_t1": ("prec_media", 1, "mm"),
    "pred_prec_t3": ("prec_media", 3, "mm"),
}


def entrenar_regresion(
    panel_model: pd.DataFrame,
    feature_cols: list,
    cutoff,
    config,
) -> dict:
    """Train one RF regressor per target (temp t+1, prec t+1, prec t+3).

    Adds pred_* columns to panel_model only when R2 > 0.
    Saves scatter plots to docs/graphs/regresion_scatter.png.
    Returns {pred_col: {"model": ..., "mae": ..., "rmse": ..., "r2": ...}}.
    """
    config.PATHS["graphs"].mkdir(parents=True, exist_ok=True)
    results      = {}
    valid_targets = {k: v for k, v in REG_TARGETS.items() if v[0] in panel_model.columns}
    n = len(valid_targets)
    if n == 0:
        return results

    fig, axes = plt.subplots(1, max(n, 1), figsize=(6 * max(n, 1), 5))
    if n == 1:
        axes = [axes]
    plot_idx = 0

    for pred_col, (src_col, horizon, unit) in valid_targets.items():
        df_r = panel_model.copy()
        df_r["_y"] = df_r.groupby("municipio")[src_col].shift(-horizon)
        df_r = df_r.dropna(subset=["_y", src_col])

        tr_r = df_r[df_r["periodo"] <  cutoff]
        te_r = df_r[df_r["periodo"] >= cutoff]

        if len(te_r) < 5:
            print(f"  {pred_col}: test insuficiente ({len(te_r)} filas), saltando")
            continue

        rf_r = RandomForestRegressor(**config.MODEL_PARAMS["rf_reg"])
        rf_r.fit(tr_r[feature_cols].fillna(0), tr_r["_y"])
        yp   = rf_r.predict(te_r[feature_cols].fillna(0))
        mae  = mean_absolute_error(te_r["_y"], yp)
        rmse = float(np.sqrt(mean_squared_error(te_r["_y"], yp)))
        r2   = r2_score(te_r["_y"], yp)
        print(f"  {pred_col}: MAE={mae:.3f} {unit}  RMSE={rmse:.3f}  R2={r2:.3f}")
        results[pred_col] = {"model": rf_r, "mae": mae, "rmse": rmse, "r2": r2}

        if r2 > 0:
            panel_model[pred_col] = rf_r.predict(panel_model[feature_cols].fillna(0))
            print(f"    → '{pred_col}' añadida al panel")
        else:
            print(f"    → '{pred_col}' descartada (R2={r2:.3f} <= 0)")

        lims = [min(te_r["_y"].min(), yp.min()), max(te_r["_y"].max(), yp.max())]
        axes[plot_idx].scatter(te_r["_y"].values, yp, alpha=0.3, color="#3498db", s=10)
        axes[plot_idx].plot(lims, lims, "r--", linewidth=2)
        axes[plot_idx].set_xlabel(f"Real ({unit})")
        axes[plot_idx].set_ylabel(f"Pred ({unit})")
        axes[plot_idx].set_title(f"{pred_col}  R²={r2:.3f}", fontweight="bold")
        plot_idx += 1

    plt.suptitle("Regresión: Ventana Climática Futura", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(config.PATHS["graphs"] / "regresion_scatter.png", dpi=120, bbox_inches="tight")
    plt.close()
    return results
