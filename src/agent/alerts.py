"""Alert level assignment, dashboard generation, and visualization."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def nivel_alerta(score: float, thresholds: list) -> str:
    """Assign alert level from score using an ordered thresholds list."""
    for nivel, thr in thresholds:
        if score >= thr:
            return nivel
    return "SIN ALERTA"


def calcular_score(p_bayesiana: np.ndarray, p_alto_siguiente: np.ndarray, config) -> np.ndarray:
    """Combine Bayesian posterior and Markov high-state probability into alert score."""
    return config.ALERT_SCORE_W_BAYES * p_bayesiana + config.ALERT_SCORE_W_MARKOV * p_alto_siguiente


def generar_dashboard(snap: pd.DataFrame, config) -> pd.DataFrame:
    """Build sorted alert dashboard and save bar chart + score histogram.

    snap must have: municipio, periodo, p_bayesiana, p_alto_sig_mes,
    score_alerta, estado_actual.
    Returns DataFrame sorted by score_alerta descending.
    """
    config.PATHS["graphs"].mkdir(parents=True, exist_ok=True)
    snap = snap.copy()
    snap["alerta"] = snap["score_alerta"].apply(
        lambda s: nivel_alerta(s, config.ALERTA_THRESHOLDS)
    )
    dashboard = snap[
        ["municipio", "p_bayesiana", "p_alto_sig_mes", "score_alerta", "estado_actual", "alerta"]
    ].sort_values("score_alerta", ascending=False)

    cnt_alerta = snap["alerta"].value_counts()
    periodo    = snap["periodo"].iloc[0] if "periodo" in snap.columns else "snapshot"
    print(f"\nDASHBOARD — {periodo}")
    print(dashboard.head(20).to_string(index=False))
    print("\nResumen:")
    for nivel, _ in config.ALERTA_THRESHOLDS:
        print(f"  {nivel:20s}: {cnt_alerta.get(nivel, 0):3d} municipios")

    niveles = [n for n, _ in config.ALERTA_THRESHOLDS]
    vals    = [cnt_alerta.get(n, 0) for n in niveles]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    bars = axes[0].bar(
        niveles, vals,
        color=[config.ALERTA_COLORS[n] for n in niveles],
        edgecolor="black", linewidth=1.2,
    )
    for bar, val in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     str(val), ha="center", fontweight="bold")
    axes[0].set_title(f"Alertas por municipio — {periodo}", fontweight="bold")
    axes[0].set_ylabel("Municipios"); axes[0].tick_params(axis="x", rotation=20)

    axes[1].hist(snap["score_alerta"], bins=30, color="#8e44ad", edgecolor="black", alpha=0.7)
    for nivel, thr in config.ALERTA_THRESHOLDS[:-1]:
        axes[1].axvline(thr, color=config.ALERTA_COLORS[nivel], linestyle="--",
                        linewidth=2, label=f"{nivel} (>{thr:.2f})")
    axes[1].set_title("Distribución del Score de Alerta", fontweight="bold")
    axes[1].set_xlabel("Score (0-1)"); axes[1].set_ylabel("Municipios")
    axes[1].legend(fontsize=9)

    plt.suptitle("Sistema de Alertas Tempranas — Santander", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(config.PATHS["graphs"] / "alertas_dashboard.png", dpi=120, bbox_inches="tight")
    plt.close()
    return dashboard
