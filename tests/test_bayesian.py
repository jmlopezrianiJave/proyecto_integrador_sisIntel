import numpy as np
import pytest
from src.training.bayesian import calcular_posterior, optimizar_threshold


def test_posterior_sube_con_clf_alto():
    prior  = 0.05
    p_clf  = np.array([0.9, 0.5, 0.1])
    result = calcular_posterior(prior, p_clf)
    # Mayor p_clf → mayor posterior
    assert result[0] > result[1] > result[2]


def test_posterior_sin_evidencia_extra():
    prior  = 0.10
    p_clf  = np.array([prior])
    result = calcular_posterior(prior, p_clf)
    # Cuando p_clf == prior, likelihood ratio = 1, posterior ≈ prior
    assert abs(result[0] - prior) < 0.01


def test_posterior_en_rango():
    prior  = 0.05
    p_clf  = np.random.rand(50)
    result = calcular_posterior(prior, p_clf)
    assert (result >= 0).all() and (result <= 1).all()


def test_posterior_sube_con_rareza():
    prior   = 0.05
    p_clf   = np.array([0.5, 0.5])
    rareza  = np.array([5.0, 0.0])   # primero tiene rareza alta
    result  = calcular_posterior(prior, p_clf, rareza_z=rareza)
    assert result[0] > result[1]


def test_optimizar_threshold_range():
    posteriors = np.random.rand(100)
    y_true     = (posteriors > 0.5).astype(int)
    thr = optimizar_threshold(posteriors, y_true)
    assert 0.0 < thr < 1.0
