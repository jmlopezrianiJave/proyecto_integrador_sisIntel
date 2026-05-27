# Refactorización Sistema Alertas Climáticas — Plan de Implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactorizar `sistema_alertas_climaticas_santander_completo.py` (2114 líneas, ex-Colab) en un proyecto Python estructurado ejecutable con `python main.py`.

**Architecture:** Tres capas: `environment/` descarga y limpia datos desde Socrata; `training/` corre los 5 tipos de modelos (regresión, clustering, clasificación, bayesiano, Markov); `agent/` aplica los modelos entrenados para generar el dashboard de alertas. `main.py` en la raíz orquesta el pipeline completo. `config.py` centraliza todas las constantes.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn, imbalanced-learn, sodapy, matplotlib, seaborn, pathlib, pytest

---

## Mapa de archivos

| Archivo | Estado | Responsabilidad |
|---------|--------|-----------------|
| `config.py` | Crear | Constantes: IDs Socrata, umbrales, params de modelos, rutas |
| `main.py` | Crear | Orquestador del pipeline completo + gráficos EDA |
| `requirements.txt` | Actualizar | Dependencias extraídas del fuente |
| `README.md` | Crear | Descripción, instalación, uso, estructura |
| `src/__init__.py` | Crear | Vacío, marca el paquete |
| `src/environment/__init__.py` | Crear | Vacío |
| `src/environment/loader.py` | Crear | Descarga desde Socrata + caché CSV |
| `src/environment/preprocessor.py` | Crear | Limpieza, anomalías, cruce multi-nivel, agregados mensuales |
| `src/training/__init__.py` | Crear | Vacío |
| `src/training/features.py` | Crear | Panel muni×mes, lags, rolling, z-scores, split temporal |
| `src/training/regression.py` | Crear | RandomForestRegressor (pred_temp_t1, pred_prec_t1, pred_prec_t3) |
| `src/training/clustering.py` | Crear | K-Means + rareza_z como feature |
| `src/training/classification.py` | Crear | RF/GB/LR + SMOTE + clasificación por tipo |
| `src/training/bayesian.py` | Crear | Actualización bayesiana secuencial por likelihood ratios |
| `src/training/markov.py` | Crear | Cadenas de Markov 3 estados, distribución estacionaria |
| `src/agent/__init__.py` | Crear | Vacío |
| `src/agent/inference.py` | Crear | Inferencia sobre snapshot del último período |
| `src/agent/alerts.py` | Crear | Score integrado, 4 niveles de alerta, dashboard |
| `tests/__init__.py` | Crear | Vacío |
| `tests/test_preprocessor.py` | Crear | Tests de funciones puras de preprocesamiento |
| `tests/test_features.py` | Crear | Tests de construcción de panel y features |
| `tests/test_bayesian.py` | Crear | Tests de actualización bayesiana |
| `tests/test_markov.py` | Crear | Tests de cadenas de Markov |
| `tests/test_alerts.py` | Crear | Tests de asignación de niveles de alerta |
| `data/raw/.gitkeep` | Crear | Marca directorio, ignorado en git |
| `data/processed/.gitkeep` | Crear | Marca directorio |
| `docs/graphs/.gitkeep` | Crear | Marca directorio |

---

## Task 1: Scaffold del proyecto

**Files:**
- Create: `config.py`
- Create: `src/__init__.py`, `src/environment/__init__.py`, `src/training/__init__.py`, `src/agent/__init__.py`
- Create: `tests/__init__.py`
- Create: `data/raw/.gitkeep`, `data/processed/.gitkeep`, `docs/graphs/.gitkeep`

- [ ] **Step 1: Crear directorios y archivos vacíos**

```bash
# Desde la raíz del proyecto
mkdir -p src/environment src/training src/agent tests data/raw data/processed docs/graphs
echo "" > src/__init__.py
echo "" > src/environment/__init__.py
echo "" > src/training/__init__.py
echo "" > src/agent/__init__.py
echo "" > tests/__init__.py
echo "" > data/raw/.gitkeep
echo "" > data/processed/.gitkeep
echo "" > docs/graphs/.gitkeep
```

- [ ] **Step 2: Crear `config.py`**

```python
# config.py
from pathlib import Path

DATASET_IDS = {
    "desastres":     "a4bc-a9tq",
    "temperatura":   "sbwg-7ju4",
    "precipitacion": "s54a-sgyg",
    "humedad":       "uext-mhny",
}

SOCRATA_URL = "www.datos.gov.co"
LIMIT_DES   = 5000
LIMIT_CLIMA = 50000
ANIOS_CLIMA = range(2020, 2025)
AUTORIDADES = ["CAS", "CDMB"]

ALERTA_THRESHOLDS = [
    ("ALERTA ROJA",     0.50),
    ("ALERTA NARANJA",  0.25),
    ("ALERTA AMARILLA", 0.10),
    ("SIN ALERTA",      0.00),
]
ALERTA_COLORS = {
    "ALERTA ROJA":     "#e74c3c",
    "ALERTA NARANJA":  "#e67e22",
    "ALERTA AMARILLA": "#f1c40f",
    "SIN ALERTA":      "#27ae60",
}

ALERT_SCORE_W_BAYES  = 0.70
ALERT_SCORE_W_MARKOV = 0.30

MODEL_PARAMS = {
    "rf_clf": dict(n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1),
    "gb_clf": dict(n_estimators=200, max_depth=4,  learning_rate=0.05, subsample=0.8, random_state=42),
    "lr_clf": dict(class_weight="balanced", max_iter=2000, C=0.5, random_state=42),
    "rf_reg": dict(n_estimators=200, max_depth=8,  min_samples_leaf=5, random_state=42, n_jobs=-1),
}

TRAIN_RATIO      = 0.80
VAL_RATIO        = 0.20
MARKOV_N_ESTADOS = 3
MARKOV_ESTADOS   = {0: "Bajo", 1: "Medio", 2: "Alto"}

PATHS = {
    "raw":       Path("data/raw"),
    "processed": Path("data/processed"),
    "graphs":    Path("docs/graphs"),
}
```

- [ ] **Step 3: Verificar que Python puede importar config**

```bash
python -c "import config; print(config.DATASET_IDS)"
```

Esperado: `{'desastres': 'a4bc-a9tq', 'temperatura': 'sbwg-7ju4', ...}`

- [ ] **Step 4: Commit**

```bash
git add config.py src/ tests/ data/ docs/graphs/.gitkeep
git commit -m "feat: project scaffold — directories, __init__.py files, config.py"
```

---

## Task 2: `src/environment/preprocessor.py` — funciones puras

**Files:**
- Create: `src/environment/preprocessor.py`
- Create: `tests/test_preprocessor.py`

- [ ] **Step 1: Escribir tests que fallan**

```python
# tests/test_preprocessor.py
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
    assert result["fechaobservacion"].dtype == "datetime64[ns]"


def test_preparar_climatico_normaliza_municipio():
    df = _make_clima_df()
    result = preparar_climatico(df, "test")
    assert all(result["municipio"].str.isupper())
    # Bogotá → BOGOTA (sin tilde)
    assert "BOGOTA" in result["municipio"].values


def test_detectar_anomalias_columnas():
    df = pd.DataFrame({
        "fechaobservacion": pd.date_range("2021-01-01", periods=10, freq="ME"),
        "valorobservado":   [20.0] * 9 + [100.0],  # el último es anomalía
        "municipio":        ["BUCARAMANGA"] * 10,
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
```

- [ ] **Step 2: Ejecutar tests — deben fallar**

```bash
python -m pytest tests/test_preprocessor.py -v
```

Esperado: `ImportError` o `ModuleNotFoundError` (preprocessor no existe aún).

- [ ] **Step 3: Crear `src/environment/preprocessor.py` con las funciones puras**

```python
# src/environment/preprocessor.py
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
```

- [ ] **Step 4: Ejecutar tests — deben pasar**

```bash
python -m pytest tests/test_preprocessor.py -v
```

Esperado: `8 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/environment/preprocessor.py tests/test_preprocessor.py
git commit -m "feat: preprocessor pure functions — quitar_tildes, haversine, preparar_climatico, detectar_anomalias, agregar_mensual"
```

---

## Task 3: `preprocessor.py` — cruce multi-nivel y `preprocesar_todo`

**Files:**
- Modify: `src/environment/preprocessor.py` (añadir `construir_cruce_multinivel`, `preprocesar_todo`)

- [ ] **Step 1: Añadir `construir_cruce_multinivel` al final de `preprocessor.py`**

```python
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
```

- [ ] **Step 2: Añadir `preprocesar_todo` al final de `preprocessor.py`**

```python
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
```

- [ ] **Step 3: Verificar que todos los tests siguen pasando**

```bash
python -m pytest tests/test_preprocessor.py -v
```

Esperado: `8 passed`.

- [ ] **Step 4: Commit**

```bash
git add src/environment/preprocessor.py
git commit -m "feat: preprocessor — construir_cruce_multinivel, preprocesar_todo"
```

---

## Task 4: `src/environment/loader.py`

**Files:**
- Create: `src/environment/loader.py`

- [ ] **Step 1: Crear `src/environment/loader.py`**

```python
# src/environment/loader.py
"""Data loading from Socrata API (datos.gov.co) with local CSV cache."""

import pandas as pd
from sodapy import Socrata


def _get_client(config) -> Socrata:
    return Socrata(config.SOCRATA_URL, None, timeout=120)


def cargar_desastres(client: Socrata, config) -> pd.DataFrame:
    """Download disaster dataset filtered to CAS/CDMB authorities.

    Saves to data/raw/desastres.csv. Returns cached file if it exists.
    """
    cache_path = config.PATHS["raw"] / "desastres.csv"
    if cache_path.exists():
        print(f"  desastres: leyendo caché {cache_path}")
        return pd.read_csv(cache_path)

    print("  Descargando desastres desde Socrata...")
    results = client.get(config.DATASET_IDS["desastres"], limit=config.LIMIT_DES)
    df = pd.DataFrame.from_records(results)
    df["autoridad_ambiental"] = df["autoridad_ambiental"].str.upper().str.strip()
    df = df[df["autoridad_ambiental"].isin(config.AUTORIDADES)].copy()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"  desastres: {len(df):,} filas → {cache_path}")
    return df


def cargar_climatico(client: Socrata, nombre: str, dataset_id: str, config) -> pd.DataFrame:
    """Download climate dataset year-by-year (2020-2024) for Santander.

    Saves to data/raw/{nombre}.csv. Returns cached file if it exists.
    """
    cache_path = config.PATHS["raw"] / f"{nombre}.csv"
    if cache_path.exists():
        print(f"  {nombre}: leyendo caché {cache_path}")
        return pd.read_csv(cache_path)

    print(f"  Descargando {nombre} desde Socrata (por año)...")
    frames = []
    for anio in config.ANIOS_CLIMA:
        where = (
            f"departamento = 'SANTANDER' "
            f"AND fechaobservacion >= '{anio}-01-01T00:00:00' "
            f"AND fechaobservacion < '{anio+1}-01-01T00:00:00'"
        )
        try:
            rows = client.get(dataset_id, where=where, limit=config.LIMIT_CLIMA)
            if rows:
                frames.append(pd.DataFrame.from_records(rows))
                print(f"    {anio}: {len(rows):,} registros")
            else:
                print(f"    {anio}: sin datos")
        except Exception as e:
            print(f"    {anio}: error — {e}")

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(df) > 0:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"  {nombre}: {len(df):,} filas → {cache_path}")
    else:
        print(f"  {nombre}: sin datos descargados")
    return df


def cargar_todos(config) -> dict:
    """Load all datasets. Returns {"desastres": df, "temperatura": df, ...}.

    Uses CSV cache if available. Creates data/raw/ if needed.
    """
    config.PATHS["raw"].mkdir(parents=True, exist_ok=True)
    client = _get_client(config)
    result = {"desastres": cargar_desastres(client, config)}
    for nombre in ["temperatura", "precipitacion", "humedad"]:
        result[nombre] = cargar_climatico(client, nombre, config.DATASET_IDS[nombre], config)
    return result
```

- [ ] **Step 2: Verificar importación**

```bash
python -c "from src.environment import loader; print('loader OK')"
```

Esperado: `loader OK`

- [ ] **Step 3: Commit**

```bash
git add src/environment/loader.py
git commit -m "feat: environment/loader — Socrata download with CSV cache"
```

---

## Task 5: `src/training/features.py`

**Files:**
- Create: `src/training/features.py`
- Create: `tests/test_features.py`

- [ ] **Step 1: Escribir tests que fallan**

```python
# tests/test_features.py
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
```

- [ ] **Step 2: Ejecutar tests — deben fallar**

```bash
python -m pytest tests/test_features.py -v
```

Esperado: `ImportError` (features.py no existe).

- [ ] **Step 3: Crear `src/training/features.py`**

```python
# src/training/features.py
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
```

- [ ] **Step 4: Ejecutar tests — deben pasar**

```bash
python -m pytest tests/test_features.py -v
```

Esperado: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/training/features.py tests/test_features.py
git commit -m "feat: training/features — panel construction, lags, rolling, split temporal"
```

---

## Task 6: `src/training/regression.py`

**Files:**
- Create: `src/training/regression.py`

- [ ] **Step 1: Crear `src/training/regression.py`**

```python
# src/training/regression.py
"""Random Forest regression for future climate window prediction."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REG_TARGETS = {
    "pred_temp_t1": ("temp_media", 1, "°C"),
    "pred_prec_t1": ("prec_media", 1, "mm"),
    "pred_prec_t3": ("prec_media", 3, "mm"),
}


def entrenar_regresion(
    panel_model: pd.DataFrame,
    feature_cols: list,
    cutoff,
    config,
) -> dict:
    """Train one RF regressor per target (temp t+1, prec t+1, prec t+3).

    Adds pred_* columns to panel_model only when R2 > 0.
    Saves scatter plots to docs/graphs/regresion_scatter.png.
    Returns {pred_col: {"model": ..., "mae": ..., "rmse": ..., "r2": ...}}.
    """
    config.PATHS["graphs"].mkdir(parents=True, exist_ok=True)
    results      = {}
    valid_targets = {k: v for k, v in REG_TARGETS.items() if v[0] in panel_model.columns}
    n = len(valid_targets)
    if n == 0:
        return results

    fig, axes = plt.subplots(1, max(n, 1), figsize=(6 * max(n, 1), 5))
    if n == 1:
        axes = [axes]
    plot_idx = 0

    for pred_col, (src_col, horizon, unit) in valid_targets.items():
        df_r = panel_model.copy()
        df_r["_y"] = df_r.groupby("municipio")[src_col].shift(-horizon)
        df_r = df_r.dropna(subset=["_y", src_col])

        tr_r = df_r[df_r["periodo"] <  cutoff]
        te_r = df_r[df_r["periodo"] >= cutoff]

        if len(te_r) < 5:
            print(f"  {pred_col}: test insuficiente ({len(te_r)} filas), saltando")
            continue

        rf_r = RandomForestRegressor(**config.MODEL_PARAMS["rf_reg"])
        rf_r.fit(tr_r[feature_cols].fillna(0), tr_r["_y"])
        yp   = rf_r.predict(te_r[feature_cols].fillna(0))
        mae  = mean_absolute_error(te_r["_y"], yp)
        rmse = float(np.sqrt(mean_squared_error(te_r["_y"], yp)))
        r2   = r2_score(te_r["_y"], yp)
        print(f"  {pred_col}: MAE={mae:.3f} {unit}  RMSE={rmse:.3f}  R2={r2:.3f}")
        results[pred_col] = {"model": rf_r, "mae": mae, "rmse": rmse, "r2": r2}

        if r2 > 0:
            panel_model[pred_col] = rf_r.predict(panel_model[feature_cols].fillna(0))
            print(f"    → '{pred_col}' añadida al panel")
        else:
            print(f"    → '{pred_col}' descartada (R2={r2:.3f} <= 0)")

        lims = [min(te_r["_y"].min(), yp.min()), max(te_r["_y"].max(), yp.max())]
        axes[plot_idx].scatter(te_r["_y"].values, yp, alpha=0.3, color="#3498db", s=10)
        axes[plot_idx].plot(lims, lims, "r--", linewidth=2)
        axes[plot_idx].set_xlabel(f"Real ({unit})")
        axes[plot_idx].set_ylabel(f"Pred ({unit})")
        axes[plot_idx].set_title(f"{pred_col}  R²={r2:.3f}", fontweight="bold")
        plot_idx += 1

    plt.suptitle("Regresión: Ventana Climática Futura", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(config.PATHS["graphs"] / "regresion_scatter.png", dpi=120, bbox_inches="tight")
    plt.close()
    return results
```

- [ ] **Step 2: Verificar importación**

```bash
python -c "from src.training import regression; print('regression OK')"
```

Esperado: `regression OK`

- [ ] **Step 3: Commit**

```bash
git add src/training/regression.py
git commit -m "feat: training/regression — RF regressor for climate window prediction"
```

---

## Task 7: `src/training/clustering.py`

**Files:**
- Create: `src/training/clustering.py`

- [ ] **Step 1: Crear `src/training/clustering.py`**

```python
# src/training/clustering.py
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
```

- [ ] **Step 2: Verificar importación**

```bash
python -c "from src.training import clustering; print('clustering OK')"
```

Esperado: `clustering OK`

- [ ] **Step 3: Commit**

```bash
git add src/training/clustering.py
git commit -m "feat: training/clustering — K-Means + rareza_z feature"
```

---

## Task 8: `src/training/classification.py`

**Files:**
- Create: `src/training/classification.py`

- [ ] **Step 1: Crear `src/training/classification.py`**

```python
# src/training/classification.py
"""Binary disaster classification + per-type models. RF, GB, LR + SMOTE."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_recall_curve, roc_auc_score,
    average_precision_score, classification_report, confusion_matrix,
)
from imblearn.over_sampling import SMOTE


def preparar_datos(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list,
) -> tuple:
    """Scale features, apply SMOTE, reserve last 20% of train as validation set.

    Returns (X_tr_sc, y_tr_res, X_te_sc, y_te, X_val_sc, y_val, scaler).
    """
    X_train = train[feature_cols].fillna(0)
    y_train = train["desastre"]
    X_test  = test[feature_cols].fillna(0)
    y_test  = test["desastre"]

    val_n    = max(int(len(X_train) * 0.20), 1)
    X_val_df = X_train.iloc[-val_n:]
    y_val    = y_train.iloc[-val_n:]
    X_tr_df  = X_train.iloc[:-val_n]
    y_tr     = y_train.iloc[:-val_n]

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr_df)
    X_te_sc  = scaler.transform(X_test)
    X_val_sc = scaler.transform(X_val_df)

    min_pos = int(y_tr.sum())
    if min_pos >= 2:
        k_n = min(5, min_pos - 1)
        X_tr_sc, y_tr_res = SMOTE(random_state=42, k_neighbors=k_n).fit_resample(X_tr_sc, y_tr)
        print(f"SMOTE: {len(X_tr_sc):,} muestras "
              f"(cls0={(y_tr_res==0).sum():,}  cls1={(y_tr_res==1).sum():,})")
    else:
        y_tr_res = y_tr
        print("SMOTE omitido (muy pocos positivos)")

    return X_tr_sc, y_tr_res, X_te_sc, y_test, X_val_sc, y_val, scaler


def entrenar_clasificadores(X_tr_sc, y_tr_res, config) -> dict:
    """Train RF, GradientBoosting, and LogisticRegression classifiers. Returns {name: model}."""
    models = {
        "RandomForest":       RandomForestClassifier(**config.MODEL_PARAMS["rf_clf"]),
        "GradientBoosting":   GradientBoostingClassifier(**config.MODEL_PARAMS["gb_clf"]),
        "LogisticRegression": LogisticRegression(**config.MODEL_PARAMS["lr_clf"]),
    }
    for name, model in models.items():
        model.fit(X_tr_sc, y_tr_res)
        print(f"  {name}: entrenado")
    return models


def evaluar_clasificadores(
    models: dict,
    X_te_sc,
    y_test,
    X_val_sc,
    y_val,
    feature_cols: list,
    config,
) -> dict:
    """Evaluate models; find optimal threshold on validation set (no data leakage).

    Saves precision-recall curves, confusion matrix, and feature importance.
    Returns {name: {model, y_pred, y_prob, f1, ap, auc, threshold}}.
    """
    config.PATHS["graphs"].mkdir(parents=True, exist_ok=True)
    results = {}
    thrs    = np.linspace(0.01, 0.99, 200)

    for name, model in models.items():
        y_prob     = model.predict_proba(X_te_sc)[:, 1]
        y_prob_val = model.predict_proba(X_val_sc)[:, 1]
        f1s        = [f1_score(y_val, (y_prob_val >= t).astype(int), zero_division=0)
                      for t in thrs]
        best_thr   = float(thrs[int(np.argmax(f1s))])
        y_pred     = (y_prob >= best_thr).astype(int)
        f1  = f1_score(y_test, y_pred, zero_division=0)
        ap  = average_precision_score(y_test, y_prob)  if y_test.sum() > 0 else 0.0
        auc = roc_auc_score(y_test, y_prob) if (y_test.sum() > 0 and (y_test==0).sum()>0) else 0.0
        results[name] = {
            "model": model, "y_pred": y_pred, "y_prob": y_prob,
            "f1": f1, "ap": ap, "auc": auc, "threshold": best_thr,
        }
        print(f"\n{name}  tau*={best_thr:.3f}  F1={f1:.4f}  AP={ap:.4f}  AUC={auc:.4f}")
        print(classification_report(y_test, y_pred,
              target_names=["Sin desastre", "Desastre"], zero_division=0))

    best_name = max(results, key=lambda k: results[k]["f1"])
    colors = {"RandomForest":"#e74c3c","GradientBoosting":"#3498db","LogisticRegression":"#27ae60"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for name, res in results.items():
        if y_test.sum() > 0:
            prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
            axes[0].plot(rec, prec, label=f"{name}  AP={res['ap']:.3f}",
                         color=colors.get(name), linewidth=2)
    axes[0].axhline(y_test.mean(), color="gray", linestyle="--", alpha=0.7,
                    label=f"Baseline ({y_test.mean():.3f})")
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title("Curvas Precisión-Recall", fontweight="bold")
    axes[0].legend(); axes[0].set_xlim([0,1]); axes[0].set_ylim([0,1])

    cm = confusion_matrix(y_test, results[best_name]["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                xticklabels=["Pred: No", "Pred: Sí"],
                yticklabels=["Real: No", "Real: Sí"])
    axes[1].set_title(f"Confusion Matrix — {best_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(config.PATHS["graphs"] / "clasificacion_precision_recall.png",
                dpi=120, bbox_inches="tight")
    plt.close()

    if "RandomForest" in results:
        imp   = pd.Series(results["RandomForest"]["model"].feature_importances_, index=feature_cols)
        top20 = imp.nlargest(20).sort_values()
        fig, ax = plt.subplots(figsize=(10, 8))
        top20.plot(kind="barh", ax=ax, color="#2ecc71", edgecolor="black")
        ax.set_title("Top-20 Features (RandomForest)", fontweight="bold", fontsize=13)
        ax.set_xlabel("Importancia Gini")
        plt.tight_layout()
        plt.savefig(config.PATHS["graphs"] / "clasificacion_feature_importance.png",
                    dpi=120, bbox_inches="tight")
        plt.close()
    return results


def entrenar_por_tipo(
    panel_model: pd.DataFrame,
    df_des: pd.DataFrame,
    feature_cols: list,
    cutoff,
    config,
) -> dict:
    """Train independent RF for each of the top-5 disaster types.

    Saves F1/AP bar chart to docs/graphs/clasificacion_por_tipo.png.
    Returns {tipo: {"f1": float, "ap": float}}.
    """
    tipo_col = "tipo_de_evento" if "tipo_de_evento" in df_des.columns else None
    if tipo_col is None:
        return {}

    tipo_metrics = []
    for tipo in df_des[tipo_col].value_counts().head(5).index.tolist():
        tipo_df = (
            df_des[df_des[tipo_col] == tipo]
            .groupby(["municipio", "periodo"]).size().reset_index(name="_cnt")
        )
        tipo_df["target_tipo"] = 1
        pm  = panel_model.merge(
            tipo_df[["municipio", "periodo", "target_tipo"]],
            on=["municipio", "periodo"], how="left",
        )
        pm["target_tipo"] = pm["target_tipo"].fillna(0).astype(int)
        tr  = pm[pm["periodo"] < cutoff]
        te  = pm[pm["periodo"] >= cutoff]
        ytr = tr["target_tipo"]
        yte = te["target_tipo"]

        if ytr.sum() < 2 or yte.sum() < 1:
            print(f"  {tipo}: omitido (train_pos={ytr.sum()}, test_pos={yte.sum()})")
            continue

        kn = min(5, int(ytr.sum()) - 1)
        try:
            Xr, yr = SMOTE(random_state=42, k_neighbors=kn).fit_resample(
                tr[feature_cols].fillna(0), ytr
            )
        except Exception:
            Xr, yr = tr[feature_cols].fillna(0), ytr

        rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                    random_state=42, n_jobs=-1)
        rf.fit(Xr, yr)
        yprob  = rf.predict_proba(te[feature_cols].fillna(0))[:, 1]
        thrs_t = np.linspace(0.01, 0.99, 100)
        best_t = float(thrs_t[np.argmax(
            [f1_score(yte, (yprob>=t).astype(int), zero_division=0) for t in thrs_t]
        )])
        yp  = (yprob >= best_t).astype(int)
        f1  = f1_score(yte, yp, zero_division=0)
        ap  = average_precision_score(yte, yprob) if yte.sum() > 0 else 0.0
        print(f"  {tipo[:40]:40s}  F1={f1:.4f}  AP={ap:.4f}")
        tipo_metrics.append({"tipo": tipo, "f1": f1, "ap": ap})

    if tipo_metrics:
        df_tm = pd.DataFrame(tipo_metrics).set_index("tipo")
        fig, ax = plt.subplots(figsize=(12, 5))
        df_tm[["f1", "ap"]].plot(kind="bar", ax=ax,
                                  color=["#e74c3c", "#3498db"], edgecolor="black")
        ax.set_title("F1 y AP por tipo de desastre", fontweight="bold")
        ax.set_ylabel("Métrica")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        ax.legend(["F1-score", "Avg Precision"])
        plt.tight_layout()
        config.PATHS["graphs"].mkdir(parents=True, exist_ok=True)
        plt.savefig(config.PATHS["graphs"] / "clasificacion_por_tipo.png",
                    dpi=120, bbox_inches="tight")
        plt.close()

    return {row["tipo"]: {"f1": row["f1"], "ap": row["ap"]} for row in tipo_metrics}
```

- [ ] **Step 2: Verificar importación**

```bash
python -c "from src.training import classification; print('classification OK')"
```

Esperado: `classification OK`

- [ ] **Step 3: Commit**

```bash
git add src/training/classification.py
git commit -m "feat: training/classification — RF/GB/LR + SMOTE + per-type models"
```

---

## Task 9: `src/training/bayesian.py`

**Files:**
- Create: `src/training/bayesian.py`
- Create: `tests/test_bayesian.py`

- [ ] **Step 1: Escribir tests que fallan**

```python
# tests/test_bayesian.py
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
```

- [ ] **Step 2: Ejecutar tests — deben fallar**

```bash
python -m pytest tests/test_bayesian.py -v
```

Esperado: `ImportError`.

- [ ] **Step 3: Crear `src/training/bayesian.py`**

```python
# src/training/bayesian.py
"""Bayesian sequential update combining classifier probability with rareza and precip anomaly."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, classification_report


def calcular_posterior(
    prior: float,
    p_clf: np.ndarray,
    rareza_z: np.ndarray = None,
    prec_anomaly: np.ndarray = None,
) -> np.ndarray:
    """Sequential Bayesian update via likelihood ratios.

    Evidence chain:
    1. Classifier probability (main signal).
    2. Municipality rareza z-score (+0.15 weight per unit above 0).
    3. Precipitation z-score anomaly (+0.10 weight per unit above 0).
    Returns posterior probabilities clipped to [0, 1].
    """
    eps        = 1e-8
    prior_odds = prior / (1 - prior + eps)
    posteriors = np.empty(len(p_clf))

    for i, p in enumerate(p_clf):
        lr   = (p / (1 - p + eps)) / (prior / (1 - prior + eps))
        odds = lr * prior_odds
        if rareza_z is not None and i < len(rareza_z):
            rz   = float(rareza_z[i]) if not np.isnan(float(rareza_z[i])) else 0.0
            odds *= 1.0 + 0.15 * max(0.0, rz)
        if prec_anomaly is not None and i < len(prec_anomaly):
            pa   = float(prec_anomaly[i]) if not np.isnan(float(prec_anomaly[i])) else 0.0
            odds *= 1.0 + 0.10 * max(0.0, pa)
        posteriors[i] = odds / (1 + odds)

    return np.clip(posteriors, 0.0, 1.0)


def optimizar_threshold(posteriors: np.ndarray, y_true) -> float:
    """Find threshold in [0.01, 0.99] that maximizes F1-score."""
    thrs = np.linspace(0.01, 0.99, 200)
    f1s  = [f1_score(y_true, (posteriors >= t).astype(int), zero_division=0) for t in thrs]
    return float(thrs[int(np.argmax(f1s))])


def evaluar_bayesiano(
    posteriors: np.ndarray,
    y_test,
    best_thr: float,
    config,
) -> dict:
    """Evaluate Bayesian model and save diagnostic plots.

    Saves posterior histogram and F1-vs-threshold chart to docs/graphs/.
    Returns {"f1": float, "ap": float, "auc": float, "threshold": float}.
    """
    config.PATHS["graphs"].mkdir(parents=True, exist_ok=True)
    thrs     = np.linspace(0.01, 0.99, 200)
    f1_curve = [f1_score(y_test, (posteriors >= t).astype(int), zero_division=0) for t in thrs]
    y_pred   = (posteriors >= best_thr).astype(int)
    f1  = f1_score(y_test, y_pred, zero_division=0)
    ap  = average_precision_score(y_test, posteriors) if y_test.sum() > 0 else 0.0
    auc = roc_auc_score(y_test, posteriors) if (y_test.sum()>0 and (y_test==0).sum()>0) else 0.0

    print(f"Bayesiano  tau*={best_thr:.3f}  F1={f1:.4f}  AP={ap:.4f}  AUC={auc:.4f}")
    print(classification_report(y_test, y_pred,
          target_names=["Sin desastre", "Desastre"], zero_division=0))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].hist(posteriors[y_test == 0], bins=40, alpha=0.6, color="#3498db",
                 label="Sin desastre", density=True)
    axes[0].hist(posteriors[y_test == 1], bins=40, alpha=0.7, color="#e74c3c",
                 label="Con desastre", density=True)
    axes[0].axvline(best_thr, color="black", linestyle="--", linewidth=2,
                    label=f"Umbral ({best_thr:.2f})")
    axes[0].set_title("P(desastre | evidencia) — Bayesiano", fontweight="bold")
    axes[0].set_xlabel("Probabilidad posterior"); axes[0].legend()

    axes[1].plot(thrs, f1_curve, color="#2ecc71", linewidth=2)
    axes[1].axvline(best_thr, color="red", linestyle="--", label=f"F1={f1:.3f} @ {best_thr:.2f}")
    axes[1].set_title("F1 vs. Umbral", fontweight="bold")
    axes[1].set_xlabel("Umbral"); axes[1].set_ylabel("F1-score"); axes[1].legend()
    plt.suptitle("Modelo Bayesiano de Alertas", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(config.PATHS["graphs"] / "bayesiano_posterior.png", dpi=120, bbox_inches="tight")
    plt.close()
    return {"f1": f1, "ap": ap, "auc": auc, "threshold": best_thr}
```

- [ ] **Step 4: Ejecutar tests — deben pasar**

```bash
python -m pytest tests/test_bayesian.py -v
```

Esperado: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/training/bayesian.py tests/test_bayesian.py
git commit -m "feat: training/bayesian — sequential likelihood ratio update"
```

---

## Task 10: `src/training/markov.py`

**Files:**
- Create: `src/training/markov.py`
- Create: `tests/test_markov.py`

- [ ] **Step 1: Escribir tests que fallan**

```python
# tests/test_markov.py
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
```

- [ ] **Step 2: Ejecutar tests — deben fallar**

```bash
python -m pytest tests/test_markov.py -v
```

Esperado: `ImportError`.

- [ ] **Step 3: Crear `src/training/markov.py`**

```python
# src/training/markov.py
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
```

- [ ] **Step 4: Ejecutar tests — deben pasar**

```bash
python -m pytest tests/test_markov.py -v
```

Esperado: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/training/markov.py tests/test_markov.py
git commit -m "feat: training/markov — 3-state Markov chains + stationary distribution"
```

---

## Task 11: `src/agent/inference.py` + `src/agent/alerts.py`

**Files:**
- Create: `src/agent/inference.py`
- Create: `src/agent/alerts.py`
- Create: `tests/test_alerts.py`

- [ ] **Step 1: Escribir tests que fallan**

```python
# tests/test_alerts.py
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
```

- [ ] **Step 2: Ejecutar tests — deben fallar**

```bash
python -m pytest tests/test_alerts.py -v
```

Esperado: `ImportError`.

- [ ] **Step 3: Crear `src/agent/inference.py`**

```python
# src/agent/inference.py
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
```

- [ ] **Step 4: Crear `src/agent/alerts.py`**

```python
# src/agent/alerts.py
"""Alert level assignment, dashboard generation, and visualization."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def nivel_alerta(score: float, thresholds: list) -> str:
    """Assign alert level from score using an ordered thresholds list."""
    for nivel, thr in thresholds:
        if score >= thr:
            return nivel
    return "SIN ALERTA"


def calcular_score(p_bayesiana: np.ndarray, p_alto_siguiente: np.ndarray, config) -> np.ndarray:
    """Combine Bayesian posterior and Markov high-state probability into alert score."""
    return config.ALERT_SCORE_W_BAYES * p_bayesiana + config.ALERT_SCORE_W_MARKOV * p_alto_siguiente


def generar_dashboard(snap: pd.DataFrame, config) -> pd.DataFrame:
    """Build sorted alert dashboard and save bar chart + score histogram.

    snap must have: municipio, periodo, p_bayesiana, p_alto_sig_mes,
    score_alerta, estado_actual.
    Returns DataFrame sorted by score_alerta descending.
    """
    config.PATHS["graphs"].mkdir(parents=True, exist_ok=True)
    snap = snap.copy()
    snap["alerta"] = snap["score_alerta"].apply(
        lambda s: nivel_alerta(s, config.ALERTA_THRESHOLDS)
    )
    dashboard = snap[
        ["municipio", "p_bayesiana", "p_alto_sig_mes", "score_alerta", "estado_actual", "alerta"]
    ].sort_values("score_alerta", ascending=False)

    cnt_alerta = snap["alerta"].value_counts()
    periodo    = snap["periodo"].iloc[0] if "periodo" in snap.columns else "snapshot"
    print(f"\nDASHBOARD — {periodo}")
    print(dashboard.head(20).to_string(index=False))
    print("\nResumen:")
    for nivel, _ in config.ALERTA_THRESHOLDS:
        print(f"  {nivel:20s}: {cnt_alerta.get(nivel, 0):3d} municipios")

    niveles = [n for n, _ in config.ALERTA_THRESHOLDS]
    vals    = [cnt_alerta.get(n, 0) for n in niveles]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    bars = axes[0].bar(
        niveles, vals,
        color=[config.ALERTA_COLORS[n] for n in niveles],
        edgecolor="black", linewidth=1.2,
    )
    for bar, val in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     str(val), ha="center", fontweight="bold")
    axes[0].set_title(f"Alertas por municipio — {periodo}", fontweight="bold")
    axes[0].set_ylabel("Municipios"); axes[0].tick_params(axis="x", rotation=20)

    axes[1].hist(snap["score_alerta"], bins=30, color="#8e44ad", edgecolor="black", alpha=0.7)
    for nivel, thr in config.ALERTA_THRESHOLDS[:-1]:
        axes[1].axvline(thr, color=config.ALERTA_COLORS[nivel], linestyle="--",
                        linewidth=2, label=f"{nivel} (>{thr:.2f})")
    axes[1].set_title("Distribución del Score de Alerta", fontweight="bold")
    axes[1].set_xlabel("Score (0-1)"); axes[1].set_ylabel("Municipios")
    axes[1].legend(fontsize=9)

    plt.suptitle("Sistema de Alertas Tempranas — Santander", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(config.PATHS["graphs"] / "alertas_dashboard.png", dpi=120, bbox_inches="tight")
    plt.close()
    return dashboard
```

- [ ] **Step 5: Ejecutar tests — deben pasar**

```bash
python -m pytest tests/test_alerts.py -v
```

Esperado: `5 passed`.

- [ ] **Step 6: Ejecutar suite completa**

```bash
python -m pytest tests/ -v
```

Esperado: todos los tests pasan (23 total aprox.).

- [ ] **Step 7: Commit**

```bash
git add src/agent/inference.py src/agent/alerts.py tests/test_alerts.py
git commit -m "feat: agent/inference + agent/alerts — snapshot inference and 4-level alert dashboard"
```

---

## Task 12: `main.py` con gráficos EDA

**Files:**
- Create: `main.py`

- [ ] **Step 1: Crear `main.py`**

```python
# main.py
"""Orchestrator: environment → training → agent → alert dashboard."""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["font.size"] = 12

import config
from src.environment import loader, preprocessor
from src.training import features, regression, clustering, classification, bayesian, markov
from src.agent import inference, alerts


def _plot_eda(proc_dfs: dict) -> None:
    """Generate EDA plots and save to docs/graphs/."""
    config.PATHS["graphs"].mkdir(parents=True, exist_ok=True)
    df_des  = proc_dfs["desastres"]
    df_temp = proc_dfs["temperatura"]
    df_prec = proc_dfs["precipitacion"]
    df_hum  = proc_dfs["humedad"]

    # 1. Tipos + municipios
    tipo_counts = df_des["tipo_de_evento"].value_counts()
    muni_counts = df_des["municipio"].value_counts().head(15)
    fig, axes   = plt.subplots(1, 2, figsize=(18, 7))
    tipo_counts.plot(kind="barh", ax=axes[0],
                     color=sns.color_palette("Reds_r", len(tipo_counts)))
    axes[0].set_title("Tipos de desastres en Santander", fontweight="bold")
    axes[0].set_xlabel("Cantidad"); axes[0].invert_yaxis()
    muni_counts.plot(kind="barh", ax=axes[1],
                     color=sns.color_palette("YlOrRd", len(muni_counts)))
    axes[1].set_title("Top 15 municipios con más desastres", fontweight="bold")
    axes[1].set_xlabel("Cantidad"); axes[1].invert_yaxis()
    plt.suptitle("Desastres Naturales y Antrópicos — Santander", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(config.PATHS["graphs"] / "eda_desastres_tipos_municipios.png",
                dpi=120, bbox_inches="tight")
    plt.close()

    # 2. Estacionalidad y tendencia anual
    df_des = df_des.copy()
    df_des["anio"] = df_des["fecha_de_ocurrencia"].dt.year
    df_des["mes"]  = df_des["fecha_de_ocurrencia"].dt.month
    meses          = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    fig, axes      = plt.subplots(1, 2, figsize=(18, 6))
    mes_counts     = df_des.groupby("mes").size()
    mes_counts.index = [meses[i - 1] for i in mes_counts.index]
    mes_counts.plot(kind="bar", ax=axes[0], color="#e67e22", alpha=0.8, edgecolor="black")
    axes[0].set_title("Desastres por mes (estacionalidad)", fontweight="bold")
    axes[0].set_ylabel("Cantidad")
    df_des.groupby("anio").size().plot(ax=axes[1], marker="o", color="#e67e22", linewidth=2)
    axes[1].set_title("Desastres por año", fontweight="bold")
    axes[1].set_ylabel("Cantidad")
    plt.tight_layout()
    plt.savefig(config.PATHS["graphs"] / "eda_desastres_temporal.png",
                dpi=120, bbox_inches="tight")
    plt.close()

    # 3. Top-5 tipos por mes
    top5_tipos = df_des["tipo_de_evento"].value_counts().head(5).index.tolist()
    tipo_mes   = pd.crosstab(df_des["mes"], df_des["tipo_de_evento"])
    tipo_mes_top = tipo_mes[[t for t in top5_tipos if t in tipo_mes.columns]]
    if len(tipo_mes_top.columns) > 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        tipo_mes_top.plot(kind="bar", ax=ax, stacked=True, colormap="Set2",
                          edgecolor="black", alpha=0.85)
        ax.set_xticklabels(
            [meses[int(m) - 1] if isinstance(m, (int, float)) else str(m)
             for m in tipo_mes_top.index],
            rotation=45,
        )
        ax.set_title("Tipo de desastre por mes — Top 5", fontweight="bold", fontsize=14)
        ax.set_xlabel("Mes"); ax.set_ylabel("Cantidad")
        ax.legend(title="Tipo de evento", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(config.PATHS["graphs"] / "eda_desastres_tipo_por_mes.png",
                    dpi=120, bbox_inches="tight")
        plt.close()

    # 4. Distribuciones climáticas
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, df_c, color, label in zip(
        axes,
        [df_temp, df_prec, df_hum],
        ["#e74c3c", "#3498db", "#27ae60"],
        ["Temperatura (°C)", "Precipitación (mm)", "Humedad (%)"],
    ):
        if len(df_c) > 0:
            df_c["valorobservado"].dropna().hist(bins=50, ax=ax, color=color,
                                                  alpha=0.7, edgecolor="black")
            ax.set_title(f"{label} — Santander", fontweight="bold")
    plt.suptitle("Distribuciones climáticas — Santander", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(config.PATHS["graphs"] / "eda_distribuciones_climaticas.png",
                dpi=120, bbox_inches="tight")
    plt.close()

    # 5. Correlación mensual + precipitación vs. desastres
    monthly = {}
    for df_c, nombre in [
        (df_temp, "Temperatura"), (df_prec, "Precipitación"), (df_hum, "Humedad")
    ]:
        if len(df_c) > 0:
            monthly[nombre] = df_c.groupby(
                df_c["fechaobservacion"].dt.to_period("M")
            )["valorobservado"].mean()
    if len(monthly) >= 2:
        df_monthly = pd.DataFrame(monthly).dropna()
        des_monthly = df_des.groupby(
            df_des["fecha_de_ocurrencia"].dt.to_period("M")
        ).size()
        des_monthly.name = "N_Desastres"
        df_monthly = df_monthly.join(des_monthly, how="left").fillna(0)
        fig, axes  = plt.subplots(1, 2, figsize=(16, 6))
        sns.heatmap(df_monthly.corr(), annot=True, cmap="RdYlBu_r", center=0,
                    fmt=".3f", ax=axes[0], square=True, linewidths=1)
        axes[0].set_title("Correlación mensual — Santander", fontweight="bold")
        if "Precipitación" in df_monthly.columns:
            idx = df_monthly.index.to_timestamp()
            axes[1].bar(idx, df_monthly["Precipitación"], alpha=0.4,
                        color="#3498db", label="Precipitación (mm)")
            ax2 = axes[1].twinx()
            ax2.plot(idx, df_monthly["N_Desastres"], color="#e74c3c",
                     marker="o", linewidth=2, label="Nº Desastres")
            ax2.set_ylabel("Desastres", color="#e74c3c")
            axes[1].set_title("Precipitación vs. Desastres (mensual)", fontweight="bold")
            axes[1].legend(loc="upper left"); ax2.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(config.PATHS["graphs"] / "eda_correlacion_mensual.png",
                    dpi=120, bbox_inches="tight")
        plt.close()


def main():
    print("=" * 60)
    print("SISTEMA DE ALERTAS CLIMÁTICAS — SANTANDER")
    print("=" * 60)

    # ── 1. Environment ────────────────────────────────────────────
    print("\n[1/5] Cargando datos desde Socrata (con caché)...")
    raw_dfs  = loader.cargar_todos(config)

    print("\n[2/5] Preprocesando...")
    proc_dfs = preprocessor.preprocesar_todo(raw_dfs, config)
    _plot_eda(proc_dfs)

    # ── 2. Feature engineering ────────────────────────────────────
    print("\n[3/5] Construyendo panel y features...")
    panel = features.construir_panel_base(
        proc_dfs["desastres"],
        proc_dfs["temp_muni"], proc_dfs["prec_muni"], proc_dfs["hum_muni"],
        proc_dfs["temp_dept"], proc_dfs["prec_dept"], proc_dfs["hum_dept"],
    )
    panel       = features.agregar_lags_y_rolling(panel)
    panel_model = panel.dropna(subset=["temp_media"]).copy().reset_index(drop=True)
    _, _, cutoff = features.split_temporal(panel_model, config.TRAIN_RATIO)
    feat_base    = features.get_feature_cols(panel_model)

    # ── 3. Training ───────────────────────────────────────────────
    print("\n[4/5] Entrenando modelos...")

    # 3a. Regression
    regression.entrenar_regresion(panel_model, feat_base, cutoff, config)

    # 3b. Clustering
    km_model, sc_cl, all_perfil = clustering.entrenar_clustering(panel_model, cutoff, config)
    panel_model = clustering.agregar_rareza(panel_model, km_model, sc_cl, all_perfil)

    # Final feature list + final split
    feature_cols = features.get_feature_cols(panel_model)
    train, test, cutoff = features.split_temporal(panel_model, config.TRAIN_RATIO)

    # 3c. Classification
    X_tr, y_tr, X_te, y_te, X_val, y_val, scaler = classification.preparar_datos(
        train, test, feature_cols
    )
    clf_models  = classification.entrenar_clasificadores(X_tr, y_tr, config)
    clf_results = classification.evaluar_clasificadores(
        clf_models, X_te, y_te, X_val, y_val, feature_cols, config
    )
    classification.entrenar_por_tipo(
        panel_model, proc_dfs["desastres"], feature_cols, cutoff, config
    )

    best_clf_name = max(clf_results, key=lambda k: clf_results[k]["f1"])
    print(f"\n>>> Mejor clasificador: {best_clf_name}  "
          f"F1={clf_results[best_clf_name]['f1']:.4f}")

    # 3d. Bayesian
    prior      = float(y_tr.mean()) if hasattr(y_tr, "mean") else float(np.array(y_tr).mean())
    p_clf_te   = clf_results[best_clf_name]["y_prob"]
    rareza_te  = test["rareza_z"].values          if "rareza_z"          in test.columns else None
    prec_an_te = test["prec_media_zscore"].values  if "prec_media_zscore" in test.columns else None
    p_bayes_te = bayesian.calcular_posterior(prior, p_clf_te, rareza_te, prec_an_te)
    best_thr   = bayesian.optimizar_threshold(p_bayes_te, y_te)
    bayesian.evaluar_bayesiano(p_bayes_te, y_te, best_thr, config)

    # 3e. Markov
    p_clf_train  = clf_results[best_clf_name]["model"].predict_proba(
        scaler.transform(train[feature_cols].fillna(0))
    )[:, 1]
    markov_result = markov.entrenar_markov(train, p_clf_train, config)
    P_markov  = markov_result["P"]
    thr_bajo  = markov_result["thr_bajo"]
    thr_alto  = markov_result["thr_alto"]

    # Summary table
    rows = [
        {
            "Modelo": name, "Tipo": "Clasificacion",
            "F1": res["f1"], "Avg-Precision": res["ap"], "AUC-ROC": res["auc"],
        }
        for name, res in clf_results.items()
    ]
    print("\n--- Resumen de modelos ---")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # ── 4. Agent ──────────────────────────────────────────────────
    print("\n[5/5] Generando dashboard de alertas...")
    ultimo_periodo = test["periodo"].max()
    snap = inference.inferir_snapshot(
        ultimo_periodo, panel_model,
        clf_results[best_clf_name], scaler, feature_cols,
        prior, thr_bajo, thr_alto, P_markov,
    )
    if len(snap) > 0:
        alerts.generar_dashboard(snap, config)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETO.")
    print("Modelos: Clasificación (RF/GB/LR) | Regresión | Clustering | Bayesiano | Markov")
    print("Sistema: 4 niveles de alerta (Roja / Naranja / Amarilla / Sin alerta)")
    print(f"Gráficos en: {config.PATHS['graphs']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verificar que main.py puede importar todos los módulos**

```bash
python -c "import main; print('main OK')"
```

Esperado: `main OK` (no ejecuta el pipeline, solo importa).

- [ ] **Step 3: Ejecutar suite completa de tests**

```bash
python -m pytest tests/ -v
```

Esperado: todos los tests pasan.

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: main.py — full pipeline orchestrator with EDA plots"
```

---

## Task 13: `requirements.txt` y `README.md`

**Files:**
- Modify: `requirements.txt`
- Create: `README.md`

- [ ] **Step 1: Actualizar `requirements.txt`**

```
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
sodapy>=2.2
scikit-learn>=1.2
imbalanced-learn>=0.10
pyarrow>=11.0
pytest>=7.0
```

- [ ] **Step 2: Crear `README.md`**

```markdown
# Sistema Inteligente de Alertas Climáticas Tempranas — Santander

Sistema de monitoreo ambiental e inteligencia artificial para la predicción de desastres naturales en el departamento de Santander, Colombia. Cruza datos climáticos del IDEAM (temperatura, precipitación, humedad) con registros históricos de desastres de las corporaciones CAS y CDMB.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
python main.py
```

La primera ejecución descarga los datos desde [datos.gov.co](https://datos.gov.co) y los guarda en `data/raw/`. Las ejecuciones siguientes usan el caché local.

Los gráficos se guardan automáticamente en `docs/graphs/`.

## Estructura del proyecto

```
proyecto_integrador_sisIntel/
├── main.py                    # Ejecutar el pipeline completo
├── config.py                  # Constantes: IDs Socrata, umbrales, parámetros de modelos
├── requirements.txt
├── src/
│   ├── environment/
│   │   ├── loader.py          # Descarga y caché de datos desde Socrata API
│   │   └── preprocessor.py    # Limpieza, anomalías, cruce multi-nivel espacio-temporal
│   ├── training/
│   │   ├── features.py        # Panel municipio × mes, lags, rolling, z-scores
│   │   ├── regression.py      # RF Regressor — predicción climática futura (t+1, t+3)
│   │   ├── clustering.py      # K-Means — perfil climático municipal + rareza
│   │   ├── classification.py  # RF / GB / LR + SMOTE + clasificación por tipo
│   │   ├── bayesian.py        # Actualización bayesiana secuencial por likelihood ratios
│   │   └── markov.py          # Cadenas de Markov 3 estados (Bajo / Medio / Alto)
│   └── agent/
│       ├── inference.py       # Inferencia sobre snapshot del último período
│       └── alerts.py          # Score integrado, 4 niveles de alerta, dashboard
├── data/
│   ├── raw/                   # CSVs crudos descargados (generado automáticamente)
│   └── processed/             # Parquet procesados (generado automáticamente)
├── docs/
│   ├── graphs/                # Gráficos PNG (generados automáticamente)
│   ├── report.pdf
│   └── ethics.md
└── tests/                     # Tests unitarios (pytest)
```

## Pipeline de modelos

1. **Regresión** — predice temperatura y precipitación a t+1 y t+3 meses
2. **Clustering** — agrupa municipios por perfil climático; la distancia al centroide se usa como feature de rareza
3. **Clasificación supervisada** — RandomForest, GradientBoosting y LogisticRegression con SMOTE; umbral óptimo por F1 en conjunto de validación
4. **Modelo Bayesiano** — combina la probabilidad del clasificador, la rareza del municipio y la anomalía de precipitación mediante likelihood ratios secuenciales
5. **Cadenas de Markov** — modela transiciones entre estados de riesgo Bajo / Medio / Alto

## Sistema de alertas

| Nivel | Score |
|-------|-------|
| 🔴 ALERTA ROJA | ≥ 0.50 |
| 🟠 ALERTA NARANJA | ≥ 0.25 |
| 🟡 ALERTA AMARILLA | ≥ 0.10 |
| 🟢 SIN ALERTA | < 0.10 |

Score = 0.70 × P_bayesiana + 0.30 × P_Markov(Alto en t+1)

## Tests

```bash
pytest tests/ -v
```
```

- [ ] **Step 3: Commit final**

```bash
git add requirements.txt README.md
git commit -m "docs: requirements.txt and README.md"
```

---

## Verificación final

- [ ] **Ejecutar suite completa de tests**

```bash
python -m pytest tests/ -v
```

Esperado: todos los tests pasan.

- [ ] **Verificar que todos los módulos importan correctamente**

```bash
python -c "
import config
from src.environment import loader, preprocessor
from src.training import features, regression, clustering, classification, bayesian, markov
from src.agent import inference, alerts
print('Todos los módulos importados correctamente.')
"
```

Esperado: `Todos los módulos importados correctamente.`

- [ ] **Verificar estructura de directorios**

```bash
# Windows PowerShell
Get-ChildItem -Recurse -Include "*.py" src/ | Select-Object FullName
```

Esperado: 11 archivos `.py` en `src/` (sin contar `__init__.py`).
