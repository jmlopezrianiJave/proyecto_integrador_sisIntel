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
