# Sistema Inteligente de Alertas Climáticas Tempranas — Santander

Sistema de monitoreo ambiental e inteligencia artificial para la predicción de desastres naturales en el departamento de Santander, Colombia. Cruza datos climáticos del IDEAM (temperatura, precipitación, humedad) con registros históricos de desastres de las corporaciones CAS y CDMB.

Elaborado por Jose M. Lopez, Maria C. Caicedo, Kevin Perez y Juan D. Valencia.

## Paso a paso para ejecutar el proyecto

**Requisitos previos:** Python 3.10 o superior, conexión a internet (solo la primera vez).

**1. Clonar el repositorio**

```bash
git clone https://github.com/jmlopezrianiJave/proyecto_integrador_sisIntel.git
cd proyecto_integrador_sisIntel
```

**2. (Opcional) Crear un entorno virtual**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Instalar dependencias**

```bash
pip install -r requirements.txt
```

**4. Ejecutar el pipeline completo**

```bash
python main.py
```

La primera ejecución descarga los datos desde [datos.gov.co](https://datos.gov.co) y los guarda en `data/raw/` (puede tomar varios minutos). Las ejecuciones siguientes usan el caché local y son mucho más rápidas.

Al terminar encontrarás los gráficos en `docs/graphs/` y los datos procesados en `data/processed/`.

**5. (Opcional) Verificar los tests**

```bash
pytest tests/ -v
```

Esperado: 28 tests pasando.

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
| ALERTA ROJA | >= 0.50 |
| ALERTA NARANJA | >= 0.25 |
| ALERTA AMARILLA | >= 0.10 |
| SIN ALERTA | < 0.10 |

Score = 0.70 × P_bayesiana + 0.30 × P_Markov(Alto en t+1)

## Tests

```bash
pytest tests/ -v
```
