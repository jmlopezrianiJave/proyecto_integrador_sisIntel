import numpy as np
import pandas as pd
import pytest
from src.training.markov import (
    calcular_umbrales_empiricos, asignar_estados,
    calcular_transicion, distribucion_estacionaria, predecir_siguiente_estado,
)


def test_calcular_umbrales_empiricos():
    probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    thr_bajo, thr_alto = calcular_umbrales_empiricos(probs)
    assert thr_bajo < thr_alto
    assert 0 < thr_bajo < 1
    assert 0 < thr_alto < 1


def test_asignar_estados_tres_niveles():
    probs = np.array([0.0, 0.5, 1.0])
    estados = asignar_estados(probs, thr_bajo=0.33, thr_alto=0.66)
    assert estados[0] == 0   # Bajo
    assert estados[1] == 1   # Medio
    assert estados[2] == 2   # Alto


def test_calcular_transicion_filas_suman_1():
    df_est = pd.DataFrame({
        "municipio": ["A", "A", "A", "B", "B", "B"],
        "periodo":   pd.period_range("2021-01", periods=3, freq="M").tolist() * 2,
        "estado":    [0, 1, 2, 0, 0, 1],
    })
    P, counts = calcular_transicion(df_est)
    assert P.shape == (3, 3)
    # Filas con transiciones deben sumar 1
    for i in range(3):
        if counts[i].sum() > 0:
            assert abs(P[i].sum() - 1.0) < 1e-9


def test_distribucion_estacionaria_suma_1():
    P = np.array([[0.7, 0.2, 0.1],
                  [0.3, 0.4, 0.3],
                  [0.1, 0.3, 0.6]])
    stat = distribucion_estacionaria(P)
    assert abs(stat.sum() - 1.0) < 1e-6
    assert all(stat >= 0)


def test_predecir_siguiente_estado_suma_1():
    P = np.array([[0.7, 0.2, 0.1],
                  [0.3, 0.4, 0.3],
                  [0.1, 0.3, 0.6]])
    for estado in [0, 1, 2]:
        dist = predecir_siguiente_estado(estado, P, n_steps=1)
        assert abs(dist.sum() - 1.0) < 1e-9
