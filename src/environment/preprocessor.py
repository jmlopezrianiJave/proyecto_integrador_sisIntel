"""Data cleaning, anomaly detection, multi-level spatial join, monthly aggregation."""

import unicodedata
from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def quitar_tildes(texto) -> str:
    """Remove diacritics from a string using NFKD normalization."""
    if pd.isna(texto):
        return texto
    nfkd = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def haversine(lon1, lat1, lon2, lat2) -> float:
    """Return great-circle distance in km between two geographic points."""
    lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))


def preparar_climatico(df: pd.DataFrame, nombre: str) -> pd.DataFrame:
    """Parse types, strip diacritics from municipio, drop rows missing key fields."""
    print(f"\n--- Preparando {nombre} ---")
    df = df.copy()
    df["fechaobservacion"] = pd.to_datetime(df["fechaobservacion"], errors="coerce")
    df["valorobservado"]   = pd.to_numeric(df["valorobservado"], errors="coerce")
    for col in ["latitud", "longitud"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "municipio" in df.columns:
        df["municipio"] = df["municipio"].apply(quitar_tildes).str.upper().str.strip()
    antes = len(df)
    df = df.dropna(subset=["fechaobservacion", "valorobservado"])
    print(f"  {len(df):,} filas (eliminados {antes - len(df):,} nulos)")
    return df


def detectar_anomalias(df: pd.DataFrame, nombre: str) -> pd.DataFrame:
    """Add z_score and es_anomalia columns computed per municipality."""
    if len(df) == 0 or "municipio" not in df.columns:
        return df
    df = df.copy()
    for col in ["media_muni", "std_muni", "z_score", "es_anomalia"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    df["media_muni"] = df.groupby("municipio")["valorobservado"].transform("mean")
    df["std_muni"]   = df.groupby("municipio")["valorobservado"].transform("std")
    df["z_score"]    = np.where(
        df["std_muni"] > 0,
        (df["valorobservado"] - df["media_muni"]) / df["std_muni"],
        0,
    )
    df["es_anomalia"] = df["z_score"].abs() > 3
    n = df["es_anomalia"].sum()
    print(f"  {nombre}: {n:,} anomalías ({n / len(df) * 100:.2f}%)")
    return df


def agregar_mensual(df: pd.DataFrame, prefix: str) -> tuple:
    """Compute monthly aggregates per municipality and department-wide.

    Returns (muni_df, dept_df) with columns {prefix}_{media,std,min,max,n}.
    """
    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame()
    df = df.copy()
    df["fechaobservacion"] = pd.to_datetime(df["fechaobservacion"], errors="coerce")
    df["valorobservado"]   = pd.to_numeric(df["valorobservado"], errors="coerce")
    df = df.dropna(subset=["fechaobservacion", "valorobservado"])
    if "municipio" in df.columns:
        df["municipio"] = df["municipio"].apply(quitar_tildes).str.upper().str.strip()

    g_muni = df.groupby(
        [df["fechaobservacion"].dt.to_period("M"), "municipio"]
    )["valorobservado"]
    muni = g_muni.agg(
        **{
            f"{prefix}_media": "mean",
            f"{prefix}_std":   "std",
            f"{prefix}_min":   "min",
            f"{prefix}_max":   "max",
            f"{prefix}_n":     "count",
        }
    ).reset_index()
    muni.rename(columns={"fechaobservacion": "periodo"}, inplace=True)

    g_dept = df.groupby(df["fechaobservacion"].dt.to_period("M"))["valorobservado"]
    dept = g_dept.agg(
        **{
            f"{prefix}_media_dept": "mean",
            f"{prefix}_std_dept":   "std",
            f"{prefix}_min_dept":   "min",
            f"{prefix}_max_dept":   "max",
        }
    ).reset_index()
    dept.rename(columns={"fechaobservacion": "periodo"}, inplace=True)
    return muni, dept


def construir_cruce_multinivel(
    df_desastres: pd.DataFrame,
    df_temp: pd.DataFrame,
    df_prec: pd.DataFrame,
    df_hum: pd.DataFrame,
) -> dict:
    """Map each disaster municipality to its nearest climate station.

    Levels: 'directo' (has own station) | 'cercano' (<100 km) | 'departamental'.
    Returns {muni: {"muni_clima": str|None, "distancia_km": float|None, "nivel": str}}.
    """
    munis_des   = set(df_desastres["municipio"].dropna().unique())
    munis_clima = set()
    for df_c in [df_temp, df_prec, df_hum]:
        if "municipio" in df_c.columns:
            munis_clima.update(df_c["municipio"].dropna().unique())

    coords_frames = []
    for df_c in [df_temp, df_prec, df_hum]:
        if len(df_c) > 0 and "latitud" in df_c.columns and "longitud" in df_c.columns:
            coords = (
                df_c.dropna(subset=["latitud", "longitud"])
                .groupby("municipio")
                .agg(lat=("latitud", "mean"), lon=("longitud", "mean"))
                .reset_index()
            )
            coords_frames.append(coords)

    if coords_frames:
        coords_clima = (
            pd.concat(coords_frames)
            .groupby("municipio")
            .agg(lat=("lat", "mean"), lon=("lon", "mean"))
            .reset_index()
        )
    else:
        coords_clima = pd.DataFrame(columns=["municipio", "lat", "lon"])

    coords_dict = dict(zip(coords_clima["municipio"], zip(coords_clima["lat"], coords_clima["lon"])))
    des_tiene_coords = "latitud" in df_desastres.columns and "longitud" in df_desastres.columns

    mapping = {}
    for muni in munis_des & munis_clima:
        mapping[muni] = {"muni_clima": muni, "distancia_km": 0.0, "nivel": "directo"}

    for muni in munis_des - munis_clima:
        lat_des = lon_des = None
        if des_tiene_coords:
            sub = df_desastres[df_desastres["municipio"] == muni][["latitud", "longitud"]].dropna()
            if len(sub) > 0:
                lat_des = sub["latitud"].mean()
                lon_des = sub["longitud"].mean()

        if lat_des is not None and not (np.isnan(lat_des) or np.isnan(lon_des)):
            best_dist, best_muni = float("inf"), None
            for m_c, (lat_c, lon_c) in coords_dict.items():
                try:
                    d = haversine(lon_des, lat_des, lon_c, lat_c)
                    if d < best_dist:
                        best_dist, best_muni = d, m_c
                except Exception:
                    continue
            if best_muni and best_dist < 100:
                mapping[muni] = {"muni_clima": best_muni, "distancia_km": round(best_dist, 1), "nivel": "cercano"}
            else:
                mapping[muni] = {"muni_clima": None, "distancia_km": None, "nivel": "departamental"}
        else:
            mapping[muni] = {"muni_clima": None, "distancia_km": None, "nivel": "departamental"}
    return mapping


def preprocesar_todo(raw_dfs: dict, config) -> dict:
    """Run full preprocessing pipeline. Saves processed parquet files to data/processed/.

    Returns dict with cleaned DataFrames and monthly aggregates.
    """
    config.PATHS["processed"].mkdir(parents=True, exist_ok=True)

    df_des = raw_dfs["desastres"].copy()
    df_des["fecha_de_ocurrencia"] = pd.to_datetime(df_des["fecha_de_ocurrencia"], errors="coerce")
    df_des["municipio"]           = df_des["municipio"].apply(quitar_tildes).str.upper().str.strip()
    df_des["tipo_de_evento"]      = (
        df_des["tipo_de_evento"]
        .str.strip().str.upper().str.replace(r"\s+", " ", regex=True)
    )
    for col in ["familias_afectadas", "viviendas_destruidas", "viviendas_averiadas", "heridos", "fallecidos"]:
        if col in df_des.columns:
            df_des[col] = pd.to_numeric(df_des[col], errors="coerce").fillna(0)
    severidad_cols = {
        "familias_afectadas":  1,
        "viviendas_destruidas": 5,
        "viviendas_averiadas":  2,
        "heridos":             10,
        "fallecidos":          50,
    }
    df_des["severidad"] = sum(
        df_des.get(col, pd.Series(0, index=df_des.index)) * w
        for col, w in severidad_cols.items()
    )

    nombres_clima = ["temperatura", "precipitacion", "humedad"]
    prefixes      = ["temp", "prec", "hum"]
    clima_dfs, muni_dfs, dept_dfs = {}, {}, {}

    for nombre, prefix in zip(nombres_clima, prefixes):
        df_c = preparar_climatico(raw_dfs[nombre], nombre)
        df_c = detectar_anomalias(df_c, nombre)
        clima_dfs[nombre] = df_c
        m, d = agregar_mensual(df_c, prefix)
        muni_dfs[prefix] = m
        dept_dfs[prefix] = d
        if len(m) > 0:
            m.to_parquet(config.PATHS["processed"] / f"{prefix}_muni.parquet", index=False)
        if len(d) > 0:
            d.to_parquet(config.PATHS["processed"] / f"{prefix}_dept.parquet", index=False)

    df_des.to_parquet(config.PATHS["processed"] / "desastres_clean.parquet", index=False)
    print(f"\nDatos procesados guardados en {config.PATHS['processed']}")

    return {
        "desastres":     df_des,
        "temperatura":   clima_dfs["temperatura"],
        "precipitacion": clima_dfs["precipitacion"],
        "humedad":       clima_dfs["humedad"],
        "temp_muni":     muni_dfs["temp"],
        "temp_dept":     dept_dfs["temp"],
        "prec_muni":     muni_dfs["prec"],
        "prec_dept":     dept_dfs["prec"],
        "hum_muni":      muni_dfs["hum"],
        "hum_dept":      dept_dfs["hum"],
    }
