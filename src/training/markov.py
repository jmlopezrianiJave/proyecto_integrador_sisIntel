"""Markov chain model for risk state transitions between months."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

N_EST   = 3
ESTADOS = {0: "Bajo", 1: "Medio", 2: "Alto"}
COLORS_E = ["#27ae60", "#f39c12", "#e74c3c"]


def calcular_umbrales_empiricos(probs_train: np.ndarray) -> tuple:
    """Return (thr_bajo, thr_alto) from percentiles 33/66 of training probabilities."""
    thr_bajo = float(np.percentile(probs_train, 33))
    thr_alto = float(np.percentile(probs_train, 66))
    print(f"Umbrales Markov (pct 33/66): bajo={thr_bajo:.3f}  alto={thr_alto:.3f}")
    return thr_bajo, thr_alto


def asignar_estados(probs: np.ndarray, thr_bajo: float, thr_alto: float) -> np.ndarray:
    """Map probability array to state indices: 0=Bajo, 1=Medio, 2=Alto."""
    estados = np.full(len(probs), 1)
    estados[probs < thr_bajo]  = 0
    estados[probs >= thr_alto] = 2
    return estados


def calcular_transicion(df_estados: pd.DataFrame) -> tuple:
    """Estimate 3x3 transition matrix from municipio state sequences.

    df_estados must have columns: municipio, periodo, estado.
    Returns (P_matrix, counts_matrix).
    """
    counts = np.zeros((N_EST, N_EST), dtype=float)
    for _, grp in df_estados.sort_values("periodo").groupby("municipio"):
        seq = grp["estado"].values
        for t in range(len(seq) - 1):
            s0, s1 = int(seq[t]), int(seq[t + 1])
            if 0 <= s0 < N_EST and 0 <= s1 < N_EST:
                counts[s0, s1] += 1
    row_s = counts.sum(axis=1, keepdims=True)
    P     = np.where(row_s > 0, counts / row_s, 1.0 / N_EST)
    return P, counts


def distribucion_estacionaria(P: np.ndarray) -> np.ndarray:
    """Return stationary distribution as dominant eigenvector of P^T."""
    evals, evecs = np.linalg.eig(P.T)
    idx          = int(np.argmin(np.abs(evals - 1.0)))
    stat         = np.real(evecs[:, idx])
    return np.abs(stat) / np.abs(stat).sum()


def predecir_siguiente_estado(estado_actual: int, P: np.ndarray, n_steps: int = 1) -> np.ndarray:
    """Return probability vector over states after n_steps given current state."""
    v = np.zeros(N_EST)
    v[estado_actual] = 1.0
    for _ in range(n_steps):
        v = v @ P
    return v


def entrenar_markov(train: pd.DataFrame, probs_train: np.ndarray, config) -> dict:
    """Full Markov training pipeline. Saves heatmap + stationary bar chart.

    Returns {"P": matrix, "thr_bajo": float, "thr_alto": float, "stationary": array}.
    """
    config.PATHS["graphs"].mkdir(parents=True, exist_ok=True)
    thr_bajo, thr_alto = calcular_umbrales_empiricos(probs_train)

    train_mc           = train.copy()
    train_mc["prob_d"] = probs_train
    train_mc["estado"] = asignar_estados(probs_train, thr_bajo, thr_alto)

    P, counts  = calcular_transicion(train_mc)
    stationary = distribucion_estacionaria(P)
    labels     = [ESTADOS[i] for i in range(N_EST)]

    df_P = pd.DataFrame(P, index=[f"De: {l}" for l in labels],
                        columns=[f"A: {l}" for l in labels])
    print("\nMatriz de transición P:")
    print(df_P.round(4).to_string())
    print("\nDistribución estacionaria:")
    for i, p in enumerate(stationary):
        print(f"  {ESTADOS[i]}: {p:.4f} ({p*100:.2f}%)")
    print("\nPredicción de siguiente estado:")
    for s in range(N_EST):
        d = predecir_siguiente_estado(s, P)
        print(f"  HOY={ESTADOS[s]:5s}: " +
              "  ".join([f"P({ESTADOS[j]})={d[j]:.3f}" for j in range(N_EST)]))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(P, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=labels, yticklabels=labels, ax=axes[0],
                linewidths=0.5, vmin=0, vmax=1)
    axes[0].set_title("Matriz de Transición de Riesgo", fontweight="bold", fontsize=13)
    axes[0].set_ylabel("Estado actual"); axes[0].set_xlabel("Estado siguiente")

    bars = axes[1].bar(labels, stationary, color=COLORS_E, edgecolor="black", linewidth=1.2)
    for bar, val in zip(bars, stationary):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", fontweight="bold", fontsize=12)
    axes[1].set_title("Distribución Estacionaria de Riesgo", fontweight="bold", fontsize=13)
    axes[1].set_ylabel("Probabilidad a largo plazo")
    axes[1].set_ylim(0, max(stationary) * 1.3)
    plt.suptitle("Cadenas de Markov — Santander", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(config.PATHS["graphs"] / "markov_transicion.png", dpi=120, bbox_inches="tight")
    plt.close()

    return {"P": P, "thr_bajo": thr_bajo, "thr_alto": thr_alto, "stationary": stationary}
