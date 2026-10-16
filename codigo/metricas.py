"""
metricas.py — Métricas del Cuadro 3.x y comparación con la línea base (Sección 4.6.2).
"""

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_squared_error, r2_score)

CUADRANTES = ["Q1", "Q2", "Q3", "Q4"]


def metricas_regresion(v, a, v_pred, a_pred) -> dict:
    """MSE, Pearson y R^2 por dimensión, más el MSE promedio (criterio de selección)."""
    r = {}
    for nombre, y, yp in (("valencia", v, v_pred), ("activacion", a, a_pred)):
        y, yp = np.asarray(y, float), np.asarray(yp, float)
        r[f"mse_{nombre}"] = mean_squared_error(y, yp)
        r[f"pearson_{nombre}"] = pearsonr(y, yp)[0] if np.std(yp) > 0 else float("nan")
        r[f"r2_{nombre}"] = r2_score(y, yp)
    r["mse_promedio"] = (r["mse_valencia"] + r["mse_activacion"]) / 2
    return r


def cuadrante_desde_va(v, a) -> np.ndarray:
    """Q1: v>=0,a>=0 | Q2: v<0,a>=0 | Q3: v<0,a<0 | Q4: v>=0,a<0."""
    v, a = np.asarray(v), np.asarray(a)
    return np.where(a >= 0, np.where(v >= 0, "Q1", "Q2"),
                            np.where(v >= 0, "Q4", "Q3"))


def metricas_cuadrante(cuadrante_real, v_pred, a_pred) -> dict:
    q_pred = cuadrante_desde_va(v_pred, a_pred)
    return {"exactitud_cuadrante": accuracy_score(cuadrante_real, q_pred),
            "f1_macro_cuadrante": f1_score(cuadrante_real, q_pred, average="macro",
                                           labels=CUADRANTES, zero_division=0),
            "matriz_confusion": confusion_matrix(cuadrante_real, q_pred, labels=CUADRANTES)}


def metricas_clasificacion(y, probs, tau=0.5) -> dict:
    """Exactitud (coincidencia exacta del conjunto de etiquetas), F1 macro y micro."""
    y = np.asarray(y)
    pred = (np.asarray(probs) >= tau).astype(int)
    return {"exactitud": accuracy_score(y, pred),
            "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
            "f1_micro": f1_score(y, pred, average="micro", zero_division=0)}


def ajustar_umbral(y_val, probs_val, candidatos=np.arange(0.1, 0.91, 0.05)) -> float:
    """Umbral tau que maximiza F1 macro en la validación de BRIGHTER."""
    mejores = [(metricas_clasificacion(y_val, probs_val, t)["f1_macro"], t) for t in candidatos]
    return float(max(mejores)[1])


# Línea base publicada (letras, MERGE) — Comparación 2
LINEA_BASE_MERGE = {"f1_cuadrante": 0.7140, "r2_activacion": 0.366, "r2_valencia": 0.569}
