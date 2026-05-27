"""K-Means clustering of municipality climate profiles. Adds rareza feature."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PERF_COLS_CANDIDATES = [
    "temp_media", "prec_media", "hum_media",
    "temp_std",   "prec_std",   "hum_std",
]


def entrenar_clustering(
    panel_model: pd.DataFrame,
    cutoff,
    config,
) -> tuple:
    """Train K-Means on municipality climate profiles using only training data.

    Chooses optimal K by silhouette score (range 2 to min(10, n//2)).
    Saves silhouette plot to docs/graphs/clustering_silhouette.png.
    Returns (km_model, scaler, all_perfil_df) — all_perfil_df has cluster/dist_centroide/rareza_z.
    """
    config.PATHS["graphs"].mkdir(parents=True, exist_ok=True)
    perf_cols = [c for c in PERF_COLS_CANDIDATES if c in panel_model.columns]
    perfil    = (
        panel_model[panel_model["periodo"] < cutoff]
        .groupby("municipio")[perf_cols].mean().dropna()
    )
    print(f"Municipios con perfil completo: {len(perfil)}")

    if len(perfil) < 4:
        print("Municipios insuficientes para clustering.")
        return None, None, pd.DataFrame()

    scaler = StandardScaler()
    X_cl   = scaler.fit_transform(perfil[perf_cols])
    k_max  = min(10, len(perfil) // 2)
    sil    = []
    for k in range(2, k_max + 1):
        lbl = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_cl)
        sil.append(silhouette_score(X_cl, lbl) if len(np.unique(lbl)) > 1 else -1)

    best_k = 2 + int(np.argmax(sil))
    print(f"K óptimo: {best_k}  (silhouette={max(sil):.3f})")

    km        = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    km.fit(X_cl)
    centroids = km.cluster_centers_

    all_perfil = panel_model.groupby("municipio")[perf_cols].mean().dropna()
    X_all      = scaler.transform(all_perfil[perf_cols])
    clusters   = km.predict(X_all)
    dists      = [
        np.linalg.norm(X_all[i] - centroids[int(clusters[i])])
        for i in range(len(all_perfil))
    ]
    all_perfil          = all_perfil.copy()
    all_perfil["cluster"]        = clusters
    all_perfil["dist_centroide"] = dists
    mu, sd = np.mean(dists), np.std(dists) + 1e-8
    all_perfil["rareza_z"]       = (all_perfil["dist_centroide"] - mu) / sd

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(2, k_max + 1), sil, marker="o", color="#e74c3c", linewidth=2)
    ax.axvline(best_k, color="gray", linestyle="--", label=f"K={best_k}")
    ax.set_title("Selección de K — Silhouette", fontweight="bold")
    ax.set_xlabel("K"); ax.set_ylabel("Silhouette Score"); ax.legend()
    plt.tight_layout()
    plt.savefig(config.PATHS["graphs"] / "clustering_silhouette.png", dpi=120, bbox_inches="tight")
    plt.close()
    return km, scaler, all_perfil


def agregar_rareza(
    panel_model: pd.DataFrame,
    km_model,
    scaler,
    all_perfil: pd.DataFrame,
) -> pd.DataFrame:
    """Merge cluster, dist_centroide, rareza_z into panel_model."""
    if km_model is None or len(all_perfil) == 0:
        return panel_model
    cols = [c for c in ["cluster", "dist_centroide", "rareza_z"] if c in all_perfil.columns]
    panel_model = panel_model.merge(
        all_perfil[cols].reset_index(), on="municipio", how="left"
    )
    print("Clustering: cluster, dist_centroide, rareza_z añadidas al panel")
    return panel_model
