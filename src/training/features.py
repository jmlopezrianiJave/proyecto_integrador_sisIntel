"""Panel construction (municipio × mes), lag/rolling features, temporal split."""

import numpy as np
import pandas as pd

_NON_FEAT = {"municipio", "periodo", "desastre", "n_desastres"}


def _merge_clima_con_fallback(
    panel: pd.DataFrame,
    muni_df: pd.DataFrame,
    dept_df: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """Left-join climate data at municipality level; fill NaN with department average."""
    cols_val = [f"{prefix}_media", f"{prefix}_std", f"{prefix}_min", f"{prefix}_max"]

    if len(muni_df) > 0:
        avail = [c for c in ["periodo", "municipio"] + cols_val + [f"{prefix}_n"]
                 if c in muni_df.columns]
        panel = panel.merge(muni_df[avail], on=["municipio", "periodo"], how="left")

    if len(dept_df) > 0:
        dept_r = dept_df.rename(columns={
            f"{prefix}_media_dept": f"_d_{prefix}_media",
            f"{prefix}_std_dept":   f"_d_{prefix}_std",
            f"{prefix}_min_dept":   f"_d_{prefix}_min",
            f"{prefix}_max_dept":   f"_d_{prefix}_max",
        })
        panel = panel.merge(dept_r, on="periodo", how="left")
        for col in cols_val:
            dcol = f"_d_{col}"
            if col in panel.columns and dcol in panel.columns:
                panel[col] = panel[col].fillna(panel[dcol])
            elif col not in panel.columns and dcol in panel.columns:
                panel[col] = panel[dcol]
        panel.drop(
            columns=[c for c in panel.columns if c.startswith("_d_")],
            inplace=True, errors="ignore",
        )
    return panel


def construir_panel_base(
    df_desastres: pd.DataFrame,
    temp_muni: pd.DataFrame,
    prec_muni: pd.DataFrame,
    hum_muni: pd.DataFrame,
    temp_dept: pd.DataFrame,
    prec_dept: pd.DataFrame,
    hum_dept: pd.DataFrame,
) -> pd.DataFrame:
    """Build the full municipio × mes panel with binary target variable.

    Class 0: no disaster. Class 1: >= 1 disaster in that municipality-month.
    Municipalities without own stations receive departmental fallback values.
    """
    all_periods, all_munis = set(), set()
    for df_m in [temp_muni, prec_muni, hum_muni]:
        if len(df_m) > 0:
            all_periods.update(df_m["periodo"].dropna().unique())
            all_munis.update(df_m["municipio"].dropna().unique())

    if not all_periods:
        raise RuntimeError("Los DataFrames muni están vacíos. Revisa environment/.")

    munis_des = set(df_desastres["municipio"].dropna().unique())
    all_munis.update(munis_des)
    all_periods = sorted(all_periods)
    all_munis   = sorted(all_munis)
    print(f"Panel: {len(all_munis)} municipios × {len(all_periods)} períodos "
          f"= {len(all_munis)*len(all_periods):,}")

    df_des = df_desastres.copy()
    df_des["periodo"] = pd.to_datetime(
        df_des["fecha_de_ocurrencia"], errors="coerce"
    ).dt.to_period("M")
    target = df_des.groupby(["municipio", "periodo"]).size().reset_index(name="n_desastres")
    target["desastre"] = 1

    panel = (
        pd.MultiIndex.from_product([all_munis, all_periods], names=["municipio", "periodo"])
        .to_frame(index=False)
        .merge(target, on=["municipio", "periodo"], how="left")
    )
    panel["desastre"]    = panel["desastre"].fillna(0).astype(int)
    panel["n_desastres"] = panel["n_desastres"].fillna(0).astype(int)

    panel = _merge_clima_con_fallback(panel, temp_muni, temp_dept, "temp")
    panel = _merge_clima_con_fallback(panel, prec_muni, prec_dept, "prec")
    panel = _merge_clima_con_fallback(panel, hum_muni,  hum_dept,  "hum")

    n1 = panel["desastre"].sum()
    n0 = (panel["desastre"] == 0).sum()
    print(f"Clase 1: {n1:,} ({n1/len(panel)*100:.2f}%)  "
          f"Clase 0: {n0:,}  Ratio: {n0/max(n1,1):.1f}:1")
    return panel


def agregar_lags_y_rolling(panel: pd.DataFrame) -> pd.DataFrame:
    """Add lag (1-3), rolling (3-6 months), z-score, percentile, cyclic month, autoregressive features."""
    panel = panel.sort_values(["municipio", "periodo"]).reset_index(drop=True)
    for col in [c for c in ["temp_media", "prec_media", "hum_media"] if c in panel.columns]:
        grp = panel.groupby("municipio")[col]
        for lag in [1, 2, 3]:
            panel[f"{col}_lag{lag}"] = grp.shift(lag)
        for w in [3, 6]:
            panel[f"{col}_roll{w}"] = grp.transform(
                lambda x, w=w: x.shift(1).rolling(w, min_periods=1).mean()
            )
        mu    = grp.transform("mean")
        sigma = grp.transform("std").fillna(1).replace(0, 1)
        panel[f"{col}_zscore"] = (panel[col] - mu) / sigma
        panel[f"{col}_pct"]    = grp.transform(lambda x: x.rank(pct=True))

    panel["mes_num"]            = panel["periodo"].dt.month
    panel["anio"]               = panel["periodo"].dt.year
    panel["sin_mes"]            = np.sin(2 * np.pi * panel["mes_num"] / 12)
    panel["cos_mes"]            = np.cos(2 * np.pi * panel["mes_num"] / 12)
    panel["temporada_lluviosa"] = panel["mes_num"].isin([4, 5, 10, 11]).astype(int)
    panel["desastre_lag1"]      = panel.groupby("municipio")["desastre"].shift(1).fillna(0).astype(int)
    panel["desastre_lag2"]      = panel.groupby("municipio")["desastre"].shift(2).fillna(0).astype(int)
    return panel


def get_feature_cols(panel: pd.DataFrame) -> list:
    """Return list of feature columns excluding identifiers and target."""
    return [c for c in panel.columns if c not in _NON_FEAT]


def split_temporal(panel: pd.DataFrame, ratio: float = 0.80) -> tuple:
    """Chronological train/test split. Returns (train, test, cutoff_period)."""
    sorted_periods = sorted(panel["periodo"].unique())
    cutoff_idx     = int(len(sorted_periods) * ratio)
    cutoff         = sorted_periods[cutoff_idx]
    train = panel[panel["periodo"] <  cutoff].copy()
    test  = panel[panel["periodo"] >= cutoff].copy()
    print(f"Cutoff: {cutoff}  Train: {len(train):,}  Test: {len(test):,}")
    return train, test, cutoff
