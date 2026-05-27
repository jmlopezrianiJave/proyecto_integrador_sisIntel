import pandas as pd
import numpy as np
import pytest
from src.environment.preprocessor import (
    quitar_tildes, haversine, preparar_climatico, detectar_anomalias, agregar_mensual,
)


def test_quitar_tildes_remueve_diacriticos():
    assert quitar_tildes("Bogotá") == "Bogota"
    assert quitar_tildes("BUCARAMANGA") == "BUCARAMANGA"
    assert quitar_tildes("Río Negro") == "Rio Negro"


def test_quitar_tildes_nan():
    result = quitar_tildes(float("nan"))
    assert result != result  # NaN != NaN


def test_haversine_misma_ubicacion():
    assert haversine(0, 0, 0, 0) == pytest.approx(0.0, abs=1e-6)


def test_haversine_distancia_conocida():
    # Bucaramanga (lon=-73.1198, lat=7.1254) a Bogotá (lon=-74.0817, lat=4.7110) ≈ 297 km
    d = haversine(-73.1198, 7.1254, -74.0817, 4.7110)
    assert 250 < d < 350


def _make_clima_df():
    return pd.DataFrame({
        "fechaobservacion": ["2021-01-15", "2021-02-10", None, "2021-03-05"],
        "valorobservado":   ["22.5",        "invalid",   "20.0", "23.1"],
        "municipio":        ["Bogotá",      "Bogotá",    "Bucaramanga", "Bucaramanga"],
        "latitud":          ["4.71",        "4.71",      "7.12",        "7.12"],
        "longitud":         ["-74.08",      "-74.08",    "-73.11",      "-73.11"],
    })


def test_preparar_climatico_elimina_nulos():
    df = _make_clima_df()
    result = preparar_climatico(df, "test")
    # Debe eliminar fila con fechaobservacion None y fila con valorobservado "invalid"
    assert len(result) == 2
    assert result["valorobservado"].dtype == float
    # pandas 3.x uses datetime64[us]; accept any datetime resolution
    assert np.issubdtype(result["fechaobservacion"].dtype, np.datetime64)


def test_preparar_climatico_normaliza_municipio():
    df = _make_clima_df()
    result = preparar_climatico(df, "test")
    assert all(result["municipio"].str.isupper())
    # Bogotá → BOGOTA (sin tilde)
    assert "BOGOTA" in result["municipio"].values


def test_detectar_anomalias_columnas():
    # Need >=11 points so max z-score = (n-1)/sqrt(n) > 3; using n=20 for safety
    df = pd.DataFrame({
        "fechaobservacion": pd.date_range("2021-01-01", periods=20, freq="ME"),
        "valorobservado":   [20.0] * 19 + [500.0],  # último es anomalía extrema
        "municipio":        ["BUCARAMANGA"] * 20,
    })
    result = detectar_anomalias(df, "temp")
    assert "z_score" in result.columns
    assert "es_anomalia" in result.columns
    assert result["es_anomalia"].sum() == 1


def test_agregar_mensual_columnas():
    df = pd.DataFrame({
        "fechaobservacion": pd.date_range("2021-01-01", periods=6, freq="ME"),
        "valorobservado":   [20.0, 21.0, 19.0, 22.0, 18.0, 23.0],
        "municipio":        ["A", "A", "B", "B", "A", "B"],
    })
    muni, dept = agregar_mensual(df, "temp")
    assert "temp_media" in muni.columns
    assert "temp_std"   in muni.columns
    assert "temp_min"   in muni.columns
    assert "temp_max"   in muni.columns
    assert "temp_n"     in muni.columns
    assert "periodo"    in muni.columns
    assert "municipio"  in muni.columns
    assert "temp_media_dept" in dept.columns
    assert dept["temp_media_dept"].notna().all()
