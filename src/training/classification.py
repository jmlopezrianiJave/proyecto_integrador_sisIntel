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
