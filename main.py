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
