"""Apply trained models to produce per-municipality snapshot probabilities."""

import numpy as np
import pandas as pd
from src.training.bayesian import calcular_posterior
from src.training.markov import asignar_estados, predecir_siguiente_estado, ESTADOS


def inferir_snapshot(
    periodo,
    panel_model: pd.DataFrame,
    best_clf_result: dict,
    scaler,
    feature_cols: list,
    prior: float,
    thr_bajo: float,
    thr_alto: float,
    P_markov: np.ndarray,
) -> pd.DataFrame:
    """Compute alert scores for all municipalities at a given period.

    Returns DataFrame with: municipio, periodo, p_bayesiana, p_alto_sig_mes,
    score_alerta, estado_actual.
    """
    snap = panel_model[panel_model["periodo"] == periodo].copy()
    if len(snap) == 0:
        print(f"Sin datos para el período {periodo}")
        return pd.DataFrame()

    X_sc     = scaler.transform(snap[feature_cols].fillna(0))
    p_clf    = best_clf_result["model"].predict_proba(X_sc)[:, 1]

    rareza_z     = snap["rareza_z"].values          if "rareza_z"          in snap.columns else None
    prec_anomaly = snap["prec_media_zscore"].values  if "prec_media_zscore" in snap.columns else None

    p_bayes     = calcular_posterior(prior, p_clf, rareza_z, prec_anomaly)
    estados     = asignar_estados(p_bayes, thr_bajo, thr_alto)
    p_alto_next = np.array([float(predecir_siguiente_estado(int(e), P_markov)[2]) for e in estados])
    score       = 0.70 * p_bayes + 0.30 * p_alto_next

    snap = snap.copy()
    snap["p_bayesiana"]    = p_bayes
    snap["p_alto_sig_mes"] = p_alto_next
    snap["score_alerta"]   = score
    snap["estado_actual"]  = [ESTADOS[int(e)] for e in estados]
    return snap
