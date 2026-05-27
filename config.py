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
