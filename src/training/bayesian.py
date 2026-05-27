"""Bayesian sequential update combining classifier probability with rareza and precip anomaly."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, classification_report


def calcular_posterior(
    prior: float,
    p_clf: np.ndarray,
    rareza_z: np.ndarray = None,
    prec_anomaly: np.ndarray = None,
) -> np.ndarray:
    """Sequential Bayesian update via likelihood ratios.

    Evidence chain:
    1. Classifier probability (main signal).
    2. Municipality rareza z-score (+0.15 weight per unit above 0).
    3. Precipitation z-score anomaly (+0.10 weight per unit above 0).
    Returns posterior probabilities clipped to [0, 1].
    """
    eps        = 1e-8
    prior_odds = prior / (1 - prior + eps)
    posteriors = np.empty(len(p_clf))

    for i, p in enumerate(p_clf):
        lr   = (p / (1 - p + eps)) / (prior / (1 - prior + eps))
        odds = lr * prior_odds
        if rareza_z is not None and i < len(rareza_z):
            rz   = float(rareza_z[i]) if not np.isnan(float(rareza_z[i])) else 0.0
            odds *= 1.0 + 0.15 * max(0.0, rz)
        if prec_anomaly is not None and i < len(prec_anomaly):
            pa   = float(prec_anomaly[i]) if not np.isnan(float(prec_anomaly[i])) else 0.0
            odds *= 1.0 + 0.10 * max(0.0, pa)
        posteriors[i] = odds / (1 + odds)

    return np.clip(posteriors, 0.0, 1.0)


def optimizar_threshold(posteriors: np.ndarray, y_true) -> float:
    """Find threshold in [0.01, 0.99] that maximizes F1-score."""
    thrs = np.linspace(0.01, 0.99, 200)
    f1s  = [f1_score(y_true, (posteriors >= t).astype(int), zero_division=0) for t in thrs]
    return float(thrs[int(np.argmax(f1s))])


def evaluar_bayesiano(
    posteriors: np.ndarray,
    y_test,
    best_thr: float,
    config,
) -> dict:
    """Evaluate Bayesian model and save diagnostic plots.

    Saves posterior histogram and F1-vs-threshold chart to docs/graphs/.
    Returns {"f1": float, "ap": float, "auc": float, "threshold": float}.
    """
    config.PATHS["graphs"].mkdir(parents=True, exist_ok=True)
    thrs     = np.linspace(0.01, 0.99, 200)
    f1_curve = [f1_score(y_test, (posteriors >= t).astype(int), zero_division=0) for t in thrs]
    y_pred   = (posteriors >= best_thr).astype(int)
    f1  = f1_score(y_test, y_pred, zero_division=0)
    ap  = average_precision_score(y_test, posteriors) if y_test.sum() > 0 else 0.0
    auc = roc_auc_score(y_test, posteriors) if (y_test.sum()>0 and (y_test==0).sum()>0) else 0.0

    print(f"Bayesiano  tau*={best_thr:.3f}  F1={f1:.4f}  AP={ap:.4f}  AUC={auc:.4f}")
    print(classification_report(y_test, y_pred,
          target_names=["Sin desastre", "Desastre"], zero_division=0))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].hist(posteriors[y_test == 0], bins=40, alpha=0.6, color="#3498db",
                 label="Sin desastre", density=True)
    axes[0].hist(posteriors[y_test == 1], bins=40, alpha=0.7, color="#e74c3c",
                 label="Con desastre", density=True)
    axes[0].axvline(best_thr, color="black", linestyle="--", linewidth=2,
                    label=f"Umbral ({best_thr:.2f})")
    axes[0].set_title("P(desastre | evidencia) — Bayesiano", fontweight="bold")
    axes[0].set_xlabel("Probabilidad posterior"); axes[0].legend()

    axes[1].plot(thrs, f1_curve, color="#2ecc71", linewidth=2)
    axes[1].axvline(best_thr, color="red", linestyle="--", label=f"F1={f1:.3f} @ {best_thr:.2f}")
    axes[1].set_title("F1 vs. Umbral", fontweight="bold")
    axes[1].set_xlabel("Umbral"); axes[1].set_ylabel("F1-score"); axes[1].legend()
    plt.suptitle("Modelo Bayesiano de Alertas", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(config.PATHS["graphs"] / "bayesiano_posterior.png", dpi=120, bbox_inches="tight")
    plt.close()
    return {"f1": f1, "ap": ap, "auc": auc, "threshold": best_thr}
