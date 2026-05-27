import pandas as pd
import numpy as np
import pytest
from pandas import Period
from src.training.features import (
    construir_panel_base, agregar_lags_y_rolling, get_feature_cols, split_temporal,
)


def _make_minimal_dfs():
    """Build minimal synthetic DataFrames to test panel construction."""
    meses = pd.period_range("2020-01", periods=12, freq="M")
    munis = ["A", "B"]

    muni_rows = [(m, muni, 22.0, 1.0, 18.0, 26.0, 30) for m in meses for muni in munis]
    muni_df   = pd.DataFrame(muni_rows,
                              columns=["periodo","municipio","temp_media","temp_std",
                                       "temp_min","temp_max","temp_n"])

    dept_df = pd.DataFrame({
        "periodo":          meses,
        "temp_media_dept":  [22.0] * 12,
        "temp_std_dept":    [1.0]  * 12,
        "temp_min_dept":    [18.0] * 12,
        "temp_max_dept":    [26.0] * 12,
    })

    des_df = pd.DataFrame({
        "municipio":           ["A", "A", "B"],
        "fecha_de_ocurrencia": pd.to_datetime(["2020-03-10", "2020-07-05", "2020-11-20"]),
        "tipo_de_evento":      ["INUNDACION", "DESLIZAMIENTO", "INUNDACION"],
    })
    return muni_df, dept_df, des_df


def test_construir_panel_base_dimensiones():
    muni_df, dept_df, des_df = _make_minimal_dfs()
    empty = pd.DataFrame()
    panel = construir_panel_base(des_df, muni_df, empty, empty, dept_df, empty, empty)
    # 2 municipios x 12 meses = 24 filas
    assert len(panel) == 24
    assert "desastre" in panel.columns
    assert panel["desastre"].isin([0, 1]).all()


def test_construir_panel_base_clase1():
    muni_df, dept_df, des_df = _make_minimal_dfs()
    empty = pd.DataFrame()
    panel = construir_panel_base(des_df, muni_df, empty, empty, dept_df, empty, empty)
    # 3 desastres en periodos distintos de municipio A y B → 3 filas con desastre=1
    assert panel["desastre"].sum() == 3


def test_agregar_lags_y_rolling_columnas():
    muni_df, dept_df, des_df = _make_minimal_dfs()
    empty = pd.DataFrame()
    panel = construir_panel_base(des_df, muni_df, empty, empty, dept_df, empty, empty)
    panel = agregar_lags_y_rolling(panel)
    assert "temp_media_lag1" in panel.columns
    assert "temp_media_lag3" in panel.columns
    assert "temp_media_roll3" in panel.columns
    assert "temp_media_roll6" in panel.columns
    assert "temp_media_zscore" in panel.columns
    assert "sin_mes" in panel.columns
    assert "cos_mes" in panel.columns
    assert "desastre_lag1" in panel.columns


def test_get_feature_cols_excluye_identifiers():
    muni_df, dept_df, des_df = _make_minimal_dfs()
    empty = pd.DataFrame()
    panel = construir_panel_base(des_df, muni_df, empty, empty, dept_df, empty, empty)
    panel = agregar_lags_y_rolling(panel)
    cols  = get_feature_cols(panel)
    assert "municipio"   not in cols
    assert "periodo"     not in cols
    assert "desastre"    not in cols
    assert "n_desastres" not in cols
    assert len(cols) > 0


def test_split_temporal_ratio():
    muni_df, dept_df, des_df = _make_minimal_dfs()
    empty = pd.DataFrame()
    panel = construir_panel_base(des_df, muni_df, empty, empty, dept_df, empty, empty)
    panel = agregar_lags_y_rolling(panel)
    train, test, cutoff = split_temporal(panel, ratio=0.80)
    total = len(train) + len(test)
    assert total == len(panel)
    # All train periods < cutoff
    assert (train["periodo"] < cutoff).all()
    assert (test["periodo"]  >= cutoff).all()
