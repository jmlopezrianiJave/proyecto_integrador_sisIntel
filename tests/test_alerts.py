import numpy as np
import pandas as pd
import pytest
import config
from src.agent.alerts import nivel_alerta, calcular_score


def test_nivel_alerta_roja():
    assert nivel_alerta(0.60, config.ALERTA_THRESHOLDS) == "ALERTA ROJA"


def test_nivel_alerta_naranja():
    assert nivel_alerta(0.35, config.ALERTA_THRESHOLDS) == "ALERTA NARANJA"


def test_nivel_alerta_amarilla():
    assert nivel_alerta(0.15, config.ALERTA_THRESHOLDS) == "ALERTA AMARILLA"


def test_nivel_alerta_sin_alerta():
    assert nivel_alerta(0.05, config.ALERTA_THRESHOLDS) == "SIN ALERTA"


def test_calcular_score_pesos():
    p_bay  = np.array([1.0, 0.0])
    p_alto = np.array([0.0, 1.0])
    score  = calcular_score(p_bay, p_alto, config)
    assert abs(score[0] - 0.70) < 1e-9
    assert abs(score[1] - 0.30) < 1e-9
