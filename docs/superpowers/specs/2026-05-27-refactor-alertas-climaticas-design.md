# Diseño: Refactorización del Sistema de Alertas Climáticas — Santander

**Fecha:** 2026-05-27  
**Archivo fuente:** `sistema_alertas_climaticas_santander_completo.py` (2114 líneas, convertido de Colab)  
**Objetivo:** Convertir el notebook monolítico en un proyecto Python estructurado, ejecutable con `python main.py`.

---

## 1. Contexto del archivo fuente

El archivo tiene dos partes:

| Parte | Líneas | Contenido |
|-------|--------|-----------|
| EDA | 1–1090 | Carga Socrata, preprocesamiento, cruce multi-nivel espacio-temporal, 15+ visualizaciones |
| Pipeline ML | 1091–2114 | Feature engineering, regresión, clustering, clasificación (RF/GB/LR + SMOTE), bayesiano, Markov, sistema de alertas |

**Problemas identificados en el fuente:**
- `!pip install` y `display()` son artefactos de Colab
- Funciones duplicadas (`quitar_tildes`/`_quitar_tildes`, `agregar_mensual`/`_agregar_mensual`)
- Todo corre como script suelto sin funciones orquestadoras
- `plt.show()` en lugar de `plt.savefig()`
- Sin rutas estructuradas (todo en memoria global)

---

## 2. Estructura de archivos objetivo

```
proyecto_integrador_sisIntel/
├── main.py                          # orquestador: environment → training → agent
├── config.py                        # constantes globales
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── loader.py                # carga API Socrata → data/raw/
│   │   └── preprocessor.py         # limpieza, tildes, anomalías → data/processed/
│   ├── training/
│   │   ├── __init__.py
│   │   ├── features.py              # panel municipio×mes, lags, rolling, z-scores
│   │   ├── regression.py            # RF regressor (pred_temp_t1, prec_t1, prec_t3)
│   │   ├── classification.py        # RF/GB/LR + SMOTE + clasificación por tipo
│   │   ├── clustering.py            # K-Means + rareza_z + dist_centroide
│   │   ├── bayesian.py              # actualización bayesiana secuencial
│   │   └── markov.py                # cadenas de Markov, 3 estados
│   └── agent/
│       ├── __init__.py
│       ├── inference.py             # aplica modelos entrenados a snapshot
│       └── alerts.py                # score = 0.7×P_bay + 0.3×P_markov, 4 niveles
├── data/
│   ├── raw/                         # CSVs crudos desde Socrata
│   └── processed/                   # panel_base.parquet, aggregados mensuales
├── docs/
│   ├── graphs/                      # todos los PNG generados (15+ gráficos)
│   ├── report.pdf
│   └── ethics.md                    # se escribe DESPUÉS de ejecutar el proyecto
└── video/
```

---

## 3. config.py — Constantes globales

```python
from pathlib import Path

DATASET_IDS = {
    "desastres":    "a4bc-a9tq",
    "temperatura":  "sbwg-7ju4",
    "precipitacion":"s54a-sgyg",
    "humedad":      "uext-mhny",
}

SOCRATA_URL   = "www.datos.gov.co"
LIMIT_DES     = 5000
LIMIT_CLIMA   = 50000
ANIOS_CLIMA   = range(2020, 2025)

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
    "rf_clf": {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 5, "random_state": 42},
    "gb_clf": {"n_estimators": 200, "max_depth": 4,  "learning_rate": 0.05, "subsample": 0.8, "random_state": 42},
    "lr_clf": {"class_weight": "balanced", "max_iter": 2000, "C": 0.5, "random_state": 42},
    "rf_reg": {"n_estimators": 200, "max_depth": 8,  "min_samples_leaf": 5, "random_state": 42},
}

TRAIN_RATIO      = 0.80
VAL_RATIO        = 0.20   # fracción del train para búsqueda de threshold
MARKOV_N_ESTADOS = 3

PATHS = {
    "raw":       Path("data/raw"),
    "processed": Path("data/processed"),
    "graphs":    Path("docs/graphs"),
}
```

---

## 4. Módulo `src/environment/`

### `loader.py`

**Responsabilidad:** Descargar datos crudos desde Socrata y guardarlos en `data/raw/`.

```python
def cargar_desastres(client, config) -> pd.DataFrame
    """Descarga dataset de desastres, filtra CAS/CDMB, guarda data/raw/desastres.csv."""

def cargar_climatico(client, nombre, dataset_id, config) -> pd.DataFrame
    """Descarga datos climáticos por año (2020-2024), guarda data/raw/{nombre}.csv."""

def cargar_todos(config) -> dict[str, pd.DataFrame]
    """Punto de entrada: retorna {"desastres": df, "temperatura": df, ...}."""
```

Notas:
- Socrata client se instancia aquí con `timeout=120`
- Si `data/raw/{nombre}.csv` ya existe, se lee del disco (evita descargas repetidas)
- Sin app_token (acceso anónimo)

### `preprocessor.py`

**Responsabilidad:** Limpiar datos crudos y construir el cruce multi-nivel.

```python
def quitar_tildes(texto: str) -> str
    """Elimina diacríticos (unicodedata NFKD). Función pura, usada en todo el proyecto."""

def preparar_climatico(df, nombre) -> pd.DataFrame
    """Conversión de tipos, eliminación de nulos, estandarización de municipios."""

def detectar_anomalias(df, nombre) -> pd.DataFrame
    """Z-score por municipio (std local). Añade columnas z_score, es_anomalia."""

def agregar_mensual(df, prefix) -> tuple[pd.DataFrame, pd.DataFrame]
    """Retorna (muni_df, dept_df) con media/std/min/max/n por período."""

def construir_cruce_multinivel(df_desastres, df_temp, df_prec, df_hum) -> dict
    """Mapeo municipio_desastre → municipio_clima_más_cercano (haversine).
    Niveles: 'directo' | 'cercano' (<100km) | 'departamental'."""

def preprocesar_todo(raw_dfs, config) -> dict[str, pd.DataFrame]
    """Orquesta todos los pasos; guarda data/processed/. Retorna dfs procesados."""
```

Outputs a `data/processed/`:
- `desastres_clean.parquet`
- `temperatura_mensual_muni.parquet`, `temperatura_mensual_dept.parquet`
- `precipitacion_mensual_muni.parquet`, `precipitacion_mensual_dept.parquet`
- `humedad_mensual_muni.parquet`, `humedad_mensual_dept.parquet`

---

## 5. Módulo `src/training/`

### `features.py`

**Responsabilidad:** Construir el panel `municipio × mes` con variable objetivo y features de rezago.

```python
def construir_panel_base(df_desastres, temp_muni, prec_muni, hum_muni,
                          temp_dept, prec_dept, hum_dept) -> pd.DataFrame
    """Producto cartesiano municipio×período; merge climático con fallback departamental.
    Variable objetivo: desastre=1 si ≥1 evento en ese municipio-mes."""

def agregar_lags_y_rolling(panel) -> pd.DataFrame
    """Lags 1-3, rolling 3-6 meses (shift(1) para evitar leakage), z-score, percentil.
    Codificación cíclica sin/cos del mes. Lag autoregresivo del target."""

def get_feature_cols(panel) -> list[str]
    """Lista de columnas de features (excluye municipio, periodo, desastre, n_desastres)."""

def split_temporal(panel, ratio=0.80) -> tuple[pd.DataFrame, pd.DataFrame, object]
    """Split cronológico 80/20. Retorna (train, test, cutoff_period)."""
```

### `regression.py`

**Responsabilidad:** Predecir variables climáticas futuras (t+1, t+3).

```python
def entrenar_regresion(panel_model, feature_cols, cutoff, config) -> dict
    """Entrena RF regressor para pred_temp_t1, pred_prec_t1, pred_prec_t3.
    Solo añade la predicción al panel si R2 > 0 (descarta si peor que la media).
    Guarda gráfico scatter real vs. predicho en docs/graphs/regresion_*.png."""

def predecir_ventana_futura(models, panel_model, feature_cols) -> pd.DataFrame
    """Aplica modelos entrenados; añade columnas pred_* al panel_model."""
```

### `clustering.py`

**Responsabilidad:** Perfil climático municipal → rareza como feature.

```python
def entrenar_clustering(panel_model, cutoff, perf_cols, config) -> tuple
    """K-Means con K óptimo por silhouette (rango 2–min(10, n//2)).
    Entrenado solo sobre datos de train. Guarda gráfico silhouette.
    Retorna (km_model, scaler, perfil_df)."""

def añadir_rareza(panel_model, km_model, scaler, perfil_df) -> pd.DataFrame
    """Añade cluster, dist_centroide, rareza_z al panel_model completo."""
```

### `classification.py`

**Responsabilidad:** Clasificación binaria (¿habrá desastre este mes?) + por tipo.

```python
def preparar_datos(train, test, feature_cols) -> tuple
    """Escala con StandardScaler, aplica SMOTE al train.
    Reserva 20% final del train como validation set para búsqueda de threshold."""

def entrenar_clasificadores(X_tr_sc, y_tr_res, config) -> dict
    """Entrena RF, GB, LR. Retorna dict con modelos."""

def evaluar_clasificadores(models, X_te_sc, y_test, X_val_sc, y_val, feature_cols) -> dict
    """Calcula F1, AP, AUC-ROC. Threshold óptimo sobre validation (sin data leakage).
    Guarda curvas PR y confusion matrix en docs/graphs/clasificacion_*.png."""

def entrenar_por_tipo(panel_model, df_des, feature_cols, cutoff, config) -> dict
    """RF independiente por cada uno de los top-5 tipos de desastre.
    Guarda gráfico F1/AP por tipo en docs/graphs/clasificacion_por_tipo.png."""
```

### `bayesian.py`

**Responsabilidad:** Combinar evidencias de forma probabilística.

```python
def calcular_posterior(prior, p_clf, rareza_z=None, prec_anomaly=None) -> np.ndarray
    """Actualización secuencial vía likelihood ratios:
    1. P_clf  2. Rareza del municipio (±0.15)  3. Anomalía precipitación (±0.10).
    Guarda histograma de distribución posterior en docs/graphs/bayesiano_*.png."""

def optimizar_threshold(posteriors, y_true) -> float
    """Búsqueda en linspace(0.01, 0.99, 200); maximiza F1."""
```

### `markov.py`

**Responsabilidad:** Modelar transiciones de riesgo entre meses.

```python
def calcular_umbrales_empiricos(probs_train) -> tuple[float, float]
    """Percentiles 33/66 sobre probabilidades del train (no fijos 0.2/0.5)."""

def asignar_estados(probs, thr_bajo, thr_alto) -> np.ndarray
    """0=Bajo, 1=Medio, 2=Alto."""

def calcular_transicion(df_estados) -> tuple[np.ndarray, np.ndarray]
    """Matriz de transición 3×3 normalizada por filas. Retorna (P, counts)."""

def distribucion_estacionaria(P) -> np.ndarray
    """Autovector dominante de P^T."""

def predecir_siguiente_estado(estado_actual, P, n_steps=1) -> np.ndarray
    """Distribución de probabilidad sobre estados después de n_steps."""
```

Guarda heatmap de matriz y distribución estacionaria en `docs/graphs/markov_*.png`.

---

## 6. Módulo `src/agent/`

### `inference.py`

**Responsabilidad:** Aplicar los modelos entrenados a un snapshot del último período.

```python
def inferir_snapshot(periodo, panel_model, clf_results, scaler, feature_cols,
                     prior, thr_bajo, thr_alto, P_markov, config) -> pd.DataFrame
    """Para cada municipio en el período dado:
    1. Probabilidad P_bayesiana
    2. Estado actual → P(Alto en t+1)
    3. Score integrado."""
```

### `alerts.py`

**Responsabilidad:** Clasificar en niveles de alerta y generar dashboard.

```python
def calcular_score(p_bayesiana, p_alto_siguiente, config) -> np.ndarray
    """score = 0.70 × P_bayesiana + 0.30 × P_markov(Alto)."""

def nivel_alerta(score, thresholds) -> str
    """ALERTA ROJA / NARANJA / AMARILLA / SIN ALERTA."""

def generar_dashboard(snap, config) -> pd.DataFrame
    """DataFrame ordenado por score descendente.
    Guarda gráficos de barras y distribución en docs/graphs/alertas_*.png."""
```

---

## 7. `main.py` — Orquestador

```python
from pathlib import Path
import config
from src.environment import loader, preprocessor
from src.training import features, regression, clustering, classification, bayesian, markov
from src.agent import inference, alerts

def main():
    # 1. Environment
    raw_dfs   = loader.cargar_todos(config)
    proc_dfs  = preprocessor.preprocesar_todo(raw_dfs, config)

    # 2. Feature engineering + split
    panel        = features.construir_panel_base(...)
    panel        = features.agregar_lags_y_rolling(panel)
    feature_cols = features.get_feature_cols(panel)
    train, test, cutoff = features.split_temporal(panel)

    # 3. Regresión (añade pred_* al panel)
    reg_models   = regression.entrenar_regresion(panel, feature_cols, cutoff, config)
    panel        = regression.predecir_ventana_futura(reg_models, panel, feature_cols)
    feature_cols = features.get_feature_cols(panel)          # actualizar tras pred_*

    # 4. Clustering (añade rareza_z al panel)
    km, sc_cl, perfil = clustering.entrenar_clustering(train, cutoff, config)
    panel             = clustering.añadir_rareza(panel, km, sc_cl, perfil)
    feature_cols      = features.get_feature_cols(panel)     # actualizar tras rareza_z

    # Re-split con feature_cols final
    train, test, cutoff = features.split_temporal(panel)

    # 5. Clasificación
    X_tr, y_tr, X_te, y_te, X_val, y_val, scaler = classification.preparar_datos(train, test, feature_cols)
    clf_models   = classification.entrenar_clasificadores(X_tr, y_tr, config)
    clf_results  = classification.evaluar_clasificadores(clf_models, X_te, y_te, X_val, y_val, feature_cols)
    classification.entrenar_por_tipo(panel, proc_dfs["desastres"], feature_cols, cutoff, config)

    # 6. Bayesiano
    best_clf    = max(clf_results, key=lambda k: clf_results[k]["f1"])
    prior       = float(y_tr.mean()) if hasattr(y_tr, "mean") else ...
    p_clf_te    = clf_results[best_clf]["y_prob"]
    p_bayes_te  = bayesian.calcular_posterior(prior, p_clf_te, ...)
    thr_bayes   = bayesian.optimizar_threshold(p_bayes_te, y_te)

    # 7. Markov
    thr_bajo, thr_alto = markov.calcular_umbrales_empiricos(...)
    estados_train      = markov.asignar_estados(...)
    P_markov, _        = markov.calcular_transicion(...)

    # 8. Agent — alertas
    snap      = inference.inferir_snapshot(...)
    dashboard = alerts.generar_dashboard(snap, config)
    print(dashboard.head(20).to_string())

if __name__ == "__main__":
    main()
```

---

## 8. Gráficos generados → `docs/graphs/`

| Archivo | Contenido | Módulo origen |
|---------|-----------|---------------|
| `eda_desastres_tipos_municipios.png` | Tipos de desastre + top municipios | preprocessor |
| `eda_desastres_temporal.png` | Estacionalidad + tendencia anual | preprocessor |
| `eda_desastres_tipo_por_mes.png` | Top-5 tipos por mes | preprocessor |
| `eda_distribuciones_climaticas.png` | Histogramas temp/prec/hum | preprocessor |
| `eda_desbalanceo_clases.png` | Clases 0 vs 1 | features |
| `eda_correlacion_mensual.png` | Heatmap + prec vs desastres | preprocessor |
| `regresion_scatter_*.png` | Real vs. predicho por variable | regression |
| `clustering_silhouette.png` | Selección de K | clustering |
| `clasificacion_precision_recall.png` | Curvas PR para los 3 modelos | classification |
| `clasificacion_confusion_matrix.png` | Confusion matrix del mejor modelo | classification |
| `clasificacion_feature_importance.png` | Top-20 features RF | classification |
| `clasificacion_por_tipo.png` | F1/AP por tipo de desastre | classification |
| `bayesiano_posterior.png` | Distribución posterior | bayesian |
| `bayesiano_f1_vs_umbral.png` | F1 vs. threshold | bayesian |
| `markov_transicion.png` | Heatmap de matriz de transición | markov |
| `markov_estacionaria.png` | Distribución estacionaria | markov |
| `alertas_dashboard.png` | Barras por nivel de alerta | alerts |
| `alertas_score_distribucion.png` | Histograma de scores | alerts |

---

## 9. `requirements.txt`

```
pandas
numpy
matplotlib
seaborn
sodapy
scikit-learn
imbalanced-learn
pathlib  # stdlib Python 3.4+, listado para documentación
```

---

## 10. Decisiones de diseño

| Decisión | Razón |
|----------|-------|
| `display()` → `print(df.to_string())` | Eliminar dependencia de IPython fuera de Colab |
| `plt.show()` → `plt.savefig()` | Los gráficos deben persistir en `docs/graphs/` |
| `quitar_tildes` unificada en `preprocessor.py` | Eliminar duplicados del fuente |
| `agregar_mensual` unificada en `preprocessor.py` | Ídem |
| Config centralizado en `config.py` | Todos los IDs, umbrales y params en un lugar |
| Regresión descarta modelos con R2 ≤ 0 | Lógica del fuente original preservada |
| Fallback departamental en el merge climático | Preserva la cobertura ~100% del fuente original |
| `data/raw/` caché de CSV | Evitar descargas repetidas durante desarrollo |
| `ethics.md` creado después de ejecutar | El usuario lo escribe con resultados reales; el proyecto solo crea `docs/graphs/` |

---

## 11. Fuera de scope

- Ninguna funcionalidad del notebook original se elimina ni modifica
- No se añade testing, CI, docker ni logging avanzado
- `ethics.md` y `report.pdf` no son generados por el código
- El `video/` queda como directorio vacío
