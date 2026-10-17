"""
Experimento reproducible: TabPFN frente a modelos clásicos en detección de fraude
(Bank Account Fraud dataset, variante Base: 1,000,000 filas, ~1.1 % fraude).

Protocolo:
  - Partición temporal por 'month': entrenamiento = meses 0-4, validación = mes 5,
    prueba = meses 6-7. Validación (20,000 filas) y prueba (50,000 filas) son submuestras
    estratificadas fijas, idénticas para todos los modelos y repeticiones.
  - Desbalance: submuestreo aleatorio de la clase mayoritaria SOLO en entrenamiento
    (hasta 10 % de fraude), igual para todos los modelos, sin pesos por clase.
  - Preprocesamiento ajustado solo con entrenamiento: -1 (y montos negativos) -> faltante
    con indicador, imputación por mediana, estandarización, one-hot para categóricas.
    Sin selección de variables: todos los modelos usan las 29 predictoras (56 columnas).
  - Métricas en prueba: ROC-AUC, PR-AUC, recall a 5 % de FPR; con el umbral fijado en
    validación (FPR = 5 %): precisión, recall, F1 y matriz de confusión. La exactitud con el
    umbral por defecto se reporta solo como referencia.
  - Curva de aprendizaje: tamaños de entrenamiento 1,000 / 2,500 / 5,000 / 10,000 filas y el
    entrenamiento balanceado completo (67,400; TabPFN solo hasta 10,000).
  - Variantes de los modelos clásicos: "defecto" (hiperparámetros fijos) y "ajustado"
    (búsqueda aleatoria de N configuraciones elegida por PR-AUC en validación, dentro de
    cada repetición). TabPFN se usa siempre con su configuración por defecto.
  - 10 repeticiones por (tamaño, variante); prueba de Friedman + Wilcoxon pareado con corrección de Holm.
  - Importancia de variables por permutación (--importancia): las 29 variables originales,
    modelos por defecto con 10,000 filas (repetición 0), sobre una submuestra de prueba.

Uso (desde esta carpeta, con el entorno de requirements-clasicos-v2.txt):
  python experimento_fraude.py --datos data/Base.csv
  ... --prueba-rapida      corrida corta para verificar que todo funciona
  ... --importancia        calcula la importancia de variables por permutación
  ... --solo-analisis      regenera tablas, figuras y pruebas a partir de los CSV
                           (incluye TabPFN-3.5 si existe resultados_colab/resultados_tabpfn35.csv)
  ... --reiniciar          borra los resultados previos y empieza de cero

Los resultados se guardan fila por fila en <salida>/resultados_crudos.csv (e importancia.csv);
si el script se interrumpe, al relanzarlo continúa donde se quedó.
"""
import argparse
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent
OBJETIVO = "fraud_bool"
CATEGORICAS = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
# En esta base, -1 codifica un valor faltante en estas columnas
FALTANTE_MENOS_UNO = ["prev_address_months_count", "current_address_months_count",
                      "bank_months_count", "session_length_in_minutes",
                      "device_distinct_emails_8w"]
# intended_balcon_amount: valores negativos = faltante (datasheet de la base)
FALTANTE_NEGATIVO = ["intended_balcon_amount"]
EXCLUIR = [OBJETIVO, "month", "device_fraud_count"]  # device_fraud_count es constante (0)
FPR_OBJETIVO = 0.05
TAMANO_MAX_TABPFN = 10000
# La copia de la base con la que se hizo el estudio (Base.xlsx) trae dos encabezados
# alterados; internamente se conservan esos nombres (las huellas de particiones.npz y
# huellas.json dependen de ellos) y en tablas y figuras se usan los originales de BAF
NOMBRES_VARIABLE = {"1nk_branch_count_8w": "bank_branch_count_8w",
                    "has_others_cards": "has_other_cards"}
# Códigos enteros de las variables categóricas en esa copia; se aplican al leer Base.csv
# de Kaggle para obtener exactamente los mismos valores
CODIGOS_CATEGORICAS = {
    "payment_type": {"AA": 1, "AB": 2, "AC": 3, "AD": 4, "AE": 5},
    "employment_status": {"CA": 1, "CB": 2, "CC": 3, "CD": 4, "CE": 5, "CF": 6, "CG": 7},
    "housing_status": {"BA": 1, "BB": 2, "BC": 3, "BD": 4, "BE": 5, "BF": 6, "BG": 7},
    "source": {"TELEAPP": 0, "INTERNET": 1},
    "device_os": {"linux": 1, "macintosh": 2, "x11": 3, "windows": 4, "other": 5},
}

MODELOS = ["TabPFN", "SVM lineal", "Logistic L1", "Ridge", "Random Forest", "XGBoost",
           "MLP sklearn", "Red Keras"]
CLASICOS = MODELOS[1:]
# TabPFN-3.5 se evaluó aparte en Google Colab (tabpfn35_colab.py) con las mismas particiones;
# sus resultados se incorporan solo en el análisis
CSV_TABPFN35 = BASE_DIR / "resultados_colab" / "resultados_tabpfn35.csv"
MODELOS_ANALISIS = ["TabPFN-3.5"] + MODELOS
FUNDACIONALES = ["TabPFN-3.5", "TabPFN"]
# Claves del protocolo: si cambian, los resultados guardados dejan de ser comparables
CLAVES_PROTOCOLO = ["n_val", "n_test", "proporcion_fraude", "tabpfn_estimadores",
                    "tabpfn_lote", "keras_epocas", "keras_lote"]


# ----------------------------------------------------------------------------- utilidades
class Registro:
    def __init__(self, ruta):
        self.f = open(ruta, "a", encoding="utf-8")

    def __call__(self, *msg):
        linea = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(m) for m in msg)
        print(linea, flush=True)
        self.f.write(linea + "\n")
        self.f.flush()


def cargar_datos(ruta_datos, ruta_cache, log):
    """Lee la base BAF (variante Base): Base.csv de Kaggle o la copia Base.xlsx del estudio."""
    if ruta_cache.exists():
        log("Cargando caché", ruta_cache.name)
        return pd.read_pickle(ruta_cache)
    log(f"Leyendo {ruta_datos.name} (la primera vez puede tardar varios minutos)...")
    t = time.perf_counter()
    if ruta_datos.suffix.lower() == ".csv":
        df = pd.read_csv(ruta_datos)
        df = df.rename(columns={v: k for k, v in NOMBRES_VARIABLE.items()})
        for c, codigos in CODIGOS_CATEGORICAS.items():
            df[c] = df[c].map(codigos)
            if df[c].isna().any():
                raise ValueError(f"Valores no reconocidos en {c}")
            df[c] = df[c].astype("int64")
        # La copia del estudio tiene las mismas filas en otro orden; como las particiones se
        # definen por posición, se restaura ese orden para que las semillas den exactamente
        # las mismas particiones (orden[i] = fila de Base.csv que ocupa la posición i)
        ruta_orden = BASE_DIR / "particiones" / "orden_filas.npz"
        if ruta_orden.exists():
            orden = np.load(ruta_orden)["orden"]
            if len(orden) != len(df):
                raise ValueError(f"{ruta_datos.name} no tiene las {len(orden):,} filas de BAF Base")
            df = df.iloc[orden].reset_index(drop=True)
    else:
        df = pd.read_excel(ruta_datos, engine="openpyxl")
    log(f"Leído en {time.perf_counter() - t:.0f} s, forma {df.shape}")
    df.to_pickle(ruta_cache)
    return df


def limpiar(df):
    df = df.copy()
    for c in FALTANTE_MENOS_UNO + FALTANTE_NEGATIVO:
        df.loc[df[c] < 0, c] = np.nan
    return df


def preprocesador(columnas):
    numericas = [c for c in columnas if c not in CATEGORICAS]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("esc", StandardScaler())]), numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
    ])


def submuestra_estratificada(X, y, n, semilla):
    if n >= len(y):
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=semilla)
    return Xs, ys


def submuestreo_mayoritaria(X, y, proporcion_fraude, semilla):
    """Conserva todos los fraudes y una muestra aleatoria de legítimos hasta la proporción dada."""
    pos = np.flatnonzero(y.to_numpy() == 1)
    neg = np.flatnonzero(y.to_numpy() == 0)
    n_neg = min(len(neg), int(round(len(pos) * (1 - proporcion_fraude) / proporcion_fraude)))
    rng = np.random.default_rng(semilla)
    idx = np.sort(np.concatenate([pos, rng.choice(neg, n_neg, replace=False)]))
    return X.iloc[idx], y.iloc[idx]


def puntajes(modelo, X, lote=None):
    if lote:
        return np.concatenate([puntajes(modelo, X[i:i + lote]) for i in range(0, len(X), lote)])
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return modelo.decision_function(X)


def agregar_fila(ruta, fila):
    """Agrega una fila al CSV respetando el orden de columnas del encabezado existente."""
    df = pd.DataFrame([fila])
    if ruta.exists():
        columnas = pd.read_csv(ruta, nrows=0).columns
        faltan = set(columnas) ^ set(df.columns)
        if faltan:
            raise ValueError(f"Columnas incompatibles con {ruta.name}: {faltan}")
        df[list(columnas)].to_csv(ruta, mode="a", header=False, index=False)
    else:
        df.to_csv(ruta, index=False)


def sincronizar_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def entrenamiento(entren, columnas, tamano, semilla, args):
    """Conjunto de entrenamiento de una repetición: submuestreo de la mayoritaria y, si
    el tamaño no es 'completo', submuestra estratificada de ese tamaño."""
    Xtr, ytr = submuestreo_mayoritaria(entren[columnas], entren[OBJETIVO],
                                       args.proporcion_fraude, semilla)
    if tamano != "completo":
        Xtr, ytr = submuestra_estratificada(Xtr, ytr, int(tamano), semilla)
    return Xtr, ytr


# ----------------------------------------------------------------------------- modelos
class RedKeras:
    """Red densa (por defecto 64-32-1)."""

    def __init__(self, semilla, epocas, lote, capas=(64, 32), lr=1e-3, dropout=0.0):
        self.semilla, self.epocas, self.lote = semilla, epocas, lote
        self.capas, self.lr, self.dropout = capas, lr, dropout

    def fit(self, X, y):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(self.semilla)
        tf.keras.backend.clear_session()
        capas = [tf.keras.Input(shape=(X.shape[1],))]  # válido en Keras 2 y Keras 3
        for u in self.capas:
            capas.append(tf.keras.layers.Dense(u, activation="relu"))
            if self.dropout > 0:
                capas.append(tf.keras.layers.Dropout(self.dropout))
        capas.append(tf.keras.layers.Dense(1, activation="sigmoid"))
        self.m = tf.keras.Sequential(capas)
        self.m.compile(optimizer=tf.keras.optimizers.Adam(self.lr), loss="binary_crossentropy")
        self.m.fit(X, y, epochs=self.epocas, batch_size=self.lote, verbose=0)
        return self

    def predict_proba(self, X):
        p = self.m.predict(X, batch_size=8192, verbose=0).ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def crear_modelo(nombre, semilla, args, p=None):
    """p = hiperparámetros de la búsqueda; None = configuración por defecto."""
    p = p or {}
    if nombre == "SVM lineal":
        return LinearSVC(C=p.get("C", 1.0), dual=False, max_iter=5000, random_state=semilla)
    if nombre == "Logistic L1":
        return LogisticRegression(penalty="l1", solver="liblinear", C=p.get("C", 1.0),
                                  max_iter=1000, random_state=semilla)
    if nombre == "Ridge":
        return RidgeClassifier(alpha=p.get("alpha", 1.0))
    if nombre == "Random Forest":
        return RandomForestClassifier(n_estimators=p.get("n_estimators", 100),
                                      max_depth=p.get("max_depth"),
                                      min_samples_leaf=p.get("min_samples_leaf", 5),
                                      max_features=p.get("max_features", "sqrt"),
                                      n_jobs=-1, random_state=semilla)
    if nombre == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=p.get("n_estimators", 300),
                             learning_rate=p.get("learning_rate", 0.1),
                             max_depth=p.get("max_depth", 6),
                             subsample=p.get("subsample", 1.0),
                             colsample_bytree=p.get("colsample_bytree", 1.0),
                             min_child_weight=p.get("min_child_weight", 1.0),
                             reg_lambda=p.get("reg_lambda", 1.0),
                             tree_method="hist", eval_metric="aucpr", n_jobs=-1,
                             random_state=semilla)
    if nombre == "MLP sklearn":
        return MLPClassifier(hidden_layer_sizes=p.get("hidden_layer_sizes", (50,)),
                             alpha=p.get("alpha", 1e-4),
                             learning_rate_init=p.get("learning_rate_init", 1e-3),
                             max_iter=200, early_stopping=True, random_state=semilla)
    if nombre == "Red Keras":
        return RedKeras(semilla, epocas=p.get("epocas", args.keras_epocas), lote=args.keras_lote,
                        capas=p.get("capas", (64, 32)), lr=p.get("lr", 1e-3),
                        dropout=p.get("dropout", 0.0))
    if nombre == "TabPFN":
        import torch
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(device="cuda" if torch.cuda.is_available() else "cpu",
                                n_estimators=args.tabpfn_estimadores, random_state=semilla)
    if nombre == "TabPFN-3.5":  # requiere tabpfn >= 9 (solo se usó en Colab)
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier.create_default_for_version("v3.5", device="cuda",
                                                          random_state=semilla)
    raise ValueError(nombre)


def muestrear_params(nombre, rng):
    """Espacios de búsqueda aleatoria (log-uniforme para escalas)."""
    def logu(a, b):
        return float(10 ** rng.uniform(np.log10(a), np.log10(b)))

    def elegir(opciones):
        return opciones[int(rng.integers(len(opciones)))]

    if nombre in ("SVM lineal", "Logistic L1"):
        return {"C": logu(1e-3, 1e2)}
    if nombre == "Ridge":
        return {"alpha": logu(1e-3, 1e3)}
    if nombre == "Random Forest":
        return {"n_estimators": elegir([100, 200, 400]), "max_depth": elegir([None, 8, 16, 32]),
                "min_samples_leaf": elegir([1, 2, 5, 10, 20]),
                "max_features": elegir(["sqrt", 0.3, 0.5])}
    if nombre == "XGBoost":
        return {"n_estimators": int(rng.integers(100, 801)), "learning_rate": logu(0.01, 0.3),
                "max_depth": int(rng.integers(3, 11)), "subsample": float(rng.uniform(0.6, 1.0)),
                "colsample_bytree": float(rng.uniform(0.5, 1.0)),
                "min_child_weight": logu(1, 20), "reg_lambda": logu(1e-2, 10)}
    if nombre == "MLP sklearn":
        return {"hidden_layer_sizes": elegir([(50,), (100,), (64, 32), (128, 64)]),
                "alpha": logu(1e-5, 1e-1), "learning_rate_init": logu(1e-4, 1e-2)}
    if nombre == "Red Keras":
        return {"capas": elegir([(32, 16), (64, 32), (128, 64)]), "lr": logu(1e-4, 1e-2),
                "epocas": elegir([10, 20]), "dropout": elegir([0.0, 0.2])}
    raise ValueError(nombre)


# ----------------------------------------------------------------------------- evaluación
def metricas(y_val, s_val, y_te, s_te, pred_def):
    # Umbral elegido en validación para FPR = 5 %, aplicado sin cambios a la prueba
    umbral = np.quantile(s_val[y_val == 0], 1 - FPR_OBJETIVO)
    pred = (s_te > umbral).astype(int)
    tp = int(((pred == 1) & (y_te == 1)).sum())
    fp = int(((pred == 1) & (y_te == 0)).sum())
    fn = int(((pred == 0) & (y_te == 1)).sum())
    tn = int(((pred == 0) & (y_te == 0)).sum())
    fpr, tpr, _ = roc_curve(y_te, s_te)
    return {
        "roc_auc": roc_auc_score(y_te, s_te),
        "pr_auc": average_precision_score(y_te, s_te),
        "recall_fpr5_test": float(np.interp(FPR_OBJETIVO, fpr, tpr)),
        "recall_umbral_val": recall_score(y_te, pred, zero_division=0),
        "precision_umbral_val": precision_score(y_te, pred, zero_division=0),
        "f1_umbral_val": f1_score(y_te, pred, zero_division=0),
        "fpr_umbral_val": fp / (fp + tn),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy_defecto": accuracy_score(y_te, pred_def),
        "f1_defecto": f1_score(y_te, pred_def, zero_division=0),
        "fraudes_predichos_defecto": int(pred_def.sum()),
    }


def prediccion_defecto(nombre, modelo, X, s):
    if nombre in FUNDACIONALES or nombre == "Red Keras":
        return (s >= 0.5).astype(int)
    return modelo.predict(X)


def correr(nombre, variante, semilla, Ptr, ytr, Pv, yv, Pt, yt, args, log):
    lote = args.tabpfn_lote if nombre in FUNDACIONALES else None
    t_busqueda, params = np.nan, ""
    if variante == "defecto":
        modelo = crear_modelo(nombre, semilla, args)
        sincronizar_gpu()
        t0 = time.perf_counter()
        modelo.fit(Ptr, ytr)
        sincronizar_gpu()
        t_ajuste = time.perf_counter() - t0
        s_val = puntajes(modelo, Pv, lote)
    else:
        # Búsqueda aleatoria: se elige la configuración con mejor PR-AUC en validación
        rng = np.random.default_rng(semilla * 100 + MODELOS.index(nombre))
        mejor = None
        t_inicio = time.perf_counter()
        for _ in range(args.ensayos):
            p = muestrear_params(nombre, rng)
            m = crear_modelo(nombre, semilla, args, p)
            t0 = time.perf_counter()
            m.fit(Ptr, ytr)
            t_fit = time.perf_counter() - t0
            s = puntajes(m, Pv)
            ap = average_precision_score(yv, s)
            if mejor is None or ap > mejor[0]:
                mejor = (ap, m, p, t_fit, s)
        t_busqueda = time.perf_counter() - t_inicio
        _, modelo, p, t_ajuste, s_val = mejor
        params = json.dumps(p, default=str)
    t0 = time.perf_counter()
    s_te = puntajes(modelo, Pt, lote)
    sincronizar_gpu()
    t_pred = time.perf_counter() - t0
    r = metricas(yv, s_val, yt, s_te, prediccion_defecto(nombre, modelo, Pt, s_te))
    r.update({"t_ajuste_s": t_ajuste, "t_prediccion_s": t_pred, "t_busqueda_s": t_busqueda,
              "hiperparametros": params})
    return r


# ----------------------------------------------------------------------------- importancia
def importancia(args, datos, dir_salida, log, modelos=None):
    """Importancia por permutación de las variables originales (caída de ROC-AUC y PR-AUC)."""
    entren, columnas, Xv, yv, Xt, yt = datos
    ruta = dir_salida / "importancia.csv"
    hechos = set()
    if ruta.exists():
        prev = pd.read_csv(ruta)
        hechos = set(zip(prev.modelo, prev.variable))
    semilla = 1000  # repetición 0
    Xtr, ytr = entrenamiento(entren, columnas, args.imp_tamano, semilla, args)
    prep = preprocesador(columnas).fit(Xtr)
    Ptr = prep.transform(Xtr).astype(np.float32)
    Xi, yi = submuestra_estratificada(Xt, pd.Series(yt, index=Xt.index), args.imp_n, 0)
    yi = yi.to_numpy()
    log(f"Importancia: entrenamiento {args.imp_tamano} filas, evaluación {len(yi):,} filas "
        f"({int(yi.sum())} fraudes), {args.imp_repeticiones} permutaciones por variable")
    for nombre in modelos or MODELOS:
        pendientes = [c for c in columnas if (nombre, c) not in hechos]
        if not pendientes:
            continue
        lote = args.tabpfn_lote if nombre in FUNDACIONALES else None
        modelo = crear_modelo(nombre, semilla, args).fit(Ptr, ytr.to_numpy())
        s0 = puntajes(modelo, prep.transform(Xi).astype(np.float32), lote)
        base_roc, base_pr = roc_auc_score(yi, s0), average_precision_score(yi, s0)
        log(f"   {nombre}: base ROC-AUC {base_roc:.4f}, PR-AUC {base_pr:.4f}")
        rng = np.random.default_rng(0)
        for c in pendientes:
            caidas_roc, caidas_pr = [], []
            for _ in range(args.imp_repeticiones):
                Xp = Xi.copy()
                Xp[c] = rng.permutation(Xp[c].to_numpy())
                s = puntajes(modelo, prep.transform(Xp).astype(np.float32), lote)
                caidas_roc.append(base_roc - roc_auc_score(yi, s))
                caidas_pr.append(base_pr - average_precision_score(yi, s))
            fila = {"modelo": nombre, "variable": c, "base_roc_auc": base_roc,
                    "base_pr_auc": base_pr, "caida_roc_auc": np.mean(caidas_roc),
                    "caida_roc_auc_de": np.std(caidas_roc, ddof=1),
                    "caida_pr_auc": np.mean(caidas_pr), "caida_pr_auc_de": np.std(caidas_pr, ddof=1)}
            agregar_fila(ruta, fila)
        log(f"   {nombre}: listo")


# ----------------------------------------------------------------------------- análisis
NOMBRES_EN = {"TabPFN-3.5": "TabPFN-3.5", "TabPFN": "TabPFN v2", "SVM lineal": "Linear SVM",
              "Logistic L1": "Logistic (L1)", "Ridge": "Ridge", "Random Forest": "Random Forest",
              "XGBoost": "XGBoost", "MLP sklearn": "MLP (scikit-learn)", "Red Keras": "Keras NN"}
# Paleta categórica de referencia (orden validado); TabPFN en naranja, TabPFN-3.5 en marrón
COLOR_MODELO = {"SVM lineal": "#2a78d6", "TabPFN": "#eb6834", "Logistic L1": "#1baf7a",
                "Ridge": "#eda100", "Random Forest": "#e87ba4", "XGBoost": "#008300",
                "MLP sklearn": "#4a3aa7", "Red Keras": "#e34948", "TabPFN-3.5": "#8a3b12"}
MARCADOR = {"TabPFN": "o", "SVM lineal": "s", "Logistic L1": "^", "Ridge": "v",
            "Random Forest": "D", "XGBoost": "P", "MLP sklearn": "X", "Red Keras": "*",
            "TabPFN-3.5": "h"}
AZUL, NARANJA, AQUA, GRIS, TINTA = "#2a78d6", "#eb6834", "#1baf7a", "#d9d8d4", "#52514e"


REFERENCIA_PAREADA = "TabPFN-3.5"
# Métodos de referencia para las comparaciones pareadas y sufijo de sus archivos
REFERENCIAS_PAREADAS = {"TabPFN-3.5": "", "TabPFN": "_v2"}


def wilcoxon_holm(tabla, referencia):
    """Prueba de rangos con signo de Wilcoxon (pareada por repetición, bilateral) de la
    referencia frente a cada otro método, con corrección de Holm para el conjunto de
    comparaciones (Benavoli et al., 2016)."""
    from scipy.stats import wilcoxon
    filas = []
    for met in tabla.columns.drop(referencia):
        dif = tabla[referencia] - tabla[met]
        p = 1.0 if np.allclose(dif, 0) else wilcoxon(tabla[referencia], tabla[met]).pvalue
        filas.append({"metodo": met, "n": len(dif), "victorias": int((dif > 0).sum()),
                      "diferencia_media": dif.mean(), "p": p})
    df = pd.DataFrame(filas).sort_values("p").reset_index(drop=True)
    k = len(df)
    df["p_holm"] = np.minimum(1, np.maximum.accumulate((k - df.index) * df["p"]))
    return df


def cargar_resultados(dir_salida):
    res = pd.read_csv(dir_salida / "resultados_crudos.csv", dtype={"escenario": str})
    if CSV_TABPFN35.exists() and "TabPFN-3.5" not in set(res.modelo):
        extra = pd.read_csv(CSV_TABPFN35, dtype={"escenario": str})
        res = pd.concat([res, extra[res.columns]], ignore_index=True)
    return res


def etiqueta_escenario(esc, n):
    return (f"Undersampled training set, {n:,} rows (full)" if esc == "completo"
            else f"{int(esc):,} training rows")


def ic95(x):
    from scipy import stats
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def orden_escenarios(valores):
    return sorted(set(valores), key=lambda e: float("inf") if e == "completo" else int(e))


def metodo(fila):
    return fila["modelo"] + (" (ajustado)" if fila["variante"] == "ajustado" else "")


def analizar(dir_salida, log):
    from scipy.stats import friedmanchisquare
    res = cargar_resultados(dir_salida)
    res["metodo"] = res.apply(metodo, axis=1)
    metr = ["roc_auc", "pr_auc", "recall_fpr5_test", "f1_umbral_val", "precision_umbral_val",
            "recall_umbral_val", "tp", "fp", "fn", "tn", "accuracy_defecto", "f1_defecto",
            "t_ajuste_s", "t_prediccion_s", "t_busqueda_s"]
    filas = []
    for (esc, met), g in res.groupby(["escenario", "metodo"], sort=False):
        fila = {"escenario": esc, "metodo": met, "modelo": g.modelo.iloc[0],
                "variante": g.variante.iloc[0], "n_entrenamiento": int(g.n_entrenamiento.mean()),
                "repeticiones": len(g)}
        for m in metr:
            fila[m + "_media"] = g[m].mean()
            fila[m + "_de"] = g[m].std(ddof=1) if len(g) > 1 else 0.0
            fila[m + "_ic95"] = ic95(g[m].dropna()) if g[m].notna().any() else np.nan
        filas.append(fila)
    resumen = pd.DataFrame(filas)
    resumen.to_csv(dir_salida / "resumen.csv", index=False)

    lineas = ["# Resumen de resultados (media ± IC 95 %, 10 repeticiones)\n"]
    for esc in orden_escenarios(resumen.escenario):
        g = resumen[resumen.escenario == esc].sort_values("pr_auc_media", ascending=False)
        lineas.append(f"\n## Entrenamiento: {etiqueta_escenario(esc, g.n_entrenamiento.iloc[0])}\n")
        lineas.append("| Método | ROC-AUC | PR-AUC | Recall@5%FPR | Precisión (umbral val.) | "
                      "F1 (umbral val.) | TP | FN | FP | TN | Ajuste (s) | Búsqueda (s) | "
                      "Predicción (s) |")
        lineas.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in g.iterrows():
            def f(m, d=4):
                return f"{r[m + '_media']:.{d}f} ± {r[m + '_ic95']:.{d}f}"
            busq = "—" if np.isnan(r["t_busqueda_s_media"]) else f"{r['t_busqueda_s_media']:.1f}"
            lineas.append(
                f"| {r['metodo']} | {f('roc_auc')} | {f('pr_auc')} | {f('recall_fpr5_test')} | "
                f"{f('precision_umbral_val')} | {f('f1_umbral_val')} | {r['tp_media']:.0f} | "
                f"{r['fn_media']:.0f} | {r['fp_media']:.0f} | {r['tn_media']:.0f} | "
                f"{r['t_ajuste_s_media']:.2f} | {busq} | {r['t_prediccion_s_media']:.2f} |")
    lineas.append("\nTP/FN/FP/TN: matriz de confusión media en prueba con el umbral fijado en "
                  "validación (FPR = 5 %).")
    (dir_salida / "resumen.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")

    # Friedman por tamaño de entrenamiento (bloques = repeticiones) y, como post hoc, Wilcoxon
    # pareado + Holm de cada TabPFN frente a los demás métodos (Benavoli et al., 2016)
    for viejo in list(dir_salida.glob("nemenyi_*.csv")) + list(dir_salida.glob("wilcoxon_holm_*.csv")):
        viejo.unlink()
    texto = []
    for esc in orden_escenarios(res.escenario):
        g = res[res.escenario == esc]
        for m in ["pr_auc", "roc_auc", "recall_fpr5_test"]:
            tabla = g.pivot(index="rep", columns="metodo", values=m).dropna()
            if tabla.shape[0] < 2 or tabla.shape[1] < 3:
                texto.append(f"[{esc}] {m}: repeticiones insuficientes para Friedman")
                continue
            est, p = friedmanchisquare(*[tabla[c] for c in tabla.columns])
            rangos = tabla.rank(axis=1, ascending=False).mean().sort_values()
            texto.append(f"[{esc}] {m}: Friedman chi2 = {est:.3f}, p = {p:.3g}, "
                         f"bloques = {tabla.shape[0]}, métodos = {tabla.shape[1]}")
            texto.append("   rango promedio (1 = mejor): " +
                         ", ".join(f"{k} {v:.2f}" for k, v in rangos.items()))
            for ref, sufijo in REFERENCIAS_PAREADAS.items():
                if ref not in tabla.columns:
                    continue
                pareadas = wilcoxon_holm(tabla, ref)
                pareadas.to_csv(dir_salida / f"wilcoxon_holm{sufijo}_{esc}_{m}.csv", index=False)
                texto.append(f"   Wilcoxon pareado + Holm, {ref} frente a cada método: " +
                             "; ".join(f"{r.metodo} gana {r.victorias}/{r.n}, p = {r.p:.4f}, "
                                       f"p Holm = {r.p_holm:.4f}" for r in pareadas.itertuples()))
    (dir_salida / "pruebas_estadisticas.txt").write_text("\n".join(texto) + "\n", encoding="utf-8")

    imp = None
    if (dir_salida / "importancia.csv").exists():
        imp = pd.read_csv(dir_salida / "importancia.csv")
        resumen_importancia(imp, dir_salida)
    graficar(res, imp, dir_salida)
    log("Análisis guardado en", dir_salida)
    print((dir_salida / "resumen.md").read_text(encoding="utf-8"))
    print((dir_salida / "pruebas_estadisticas.txt").read_text(encoding="utf-8"))


def resumen_importancia(imp, dir_salida):
    tabla = imp.pivot(index="variable", columns="modelo", values="caida_roc_auc")
    tabla = tabla[[m for m in MODELOS_ANALISIS if m in tabla.columns]].rename(index=NOMBRES_VARIABLE)
    rangos = tabla.rank(ascending=False)
    tabla["rango_medio"] = rangos.mean(axis=1)
    tabla.sort_values("rango_medio").to_csv(dir_salida / "importancia_resumen.csv")
    rangos.corr(method="spearman").to_csv(dir_salida / "importancia_concordancia_spearman.csv")


def graficar(res, imp, dir_salida):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": TINTA, "axes.labelcolor": TINTA,
                         "xtick.color": TINTA, "ytick.color": TINTA})
    # fig_correlacion.png se genera aparte (no depende de los resultados): no se borra
    for viejo in dir_salida.glob("fig_*.png"):
        if viejo.name != "fig_correlacion.png":
            viejo.unlink()
    prev = res["prevalencia_test"].iloc[0]
    n_prueba = int(res["n_prueba"].iloc[0])
    n_reps = int(res.groupby(["escenario", "variante", "modelo"]).size().max())
    n_completo = int(res.loc[res.escenario == "completo", "n_entrenamiento"].max()) \
        if (res.escenario == "completo").any() else 0

    def cajas(ax, datos, pos, colores, ancho):
        bp = ax.boxplot(datos, positions=pos, widths=ancho, patch_artist=True, showmeans=True,
                        medianprops=dict(color=TINTA, lw=1.5),
                        meanprops=dict(marker="D", markersize=3.5, markerfacecolor="white",
                                       markeredgecolor=TINTA),
                        flierprops=dict(marker="o", markersize=3.5, markeredgecolor=TINTA),
                        whiskerprops=dict(color=TINTA), capprops=dict(color=TINTA))
        for caja, c in zip(bp["boxes"], colores):
            caja.set(facecolor=c, edgecolor=TINTA)

    # --- Figura 1: desempeño con 10,000 filas y con el entrenamiento completo
    escs = [e for e in ["10000", "completo"] if e in set(res.escenario)]
    metricas_fig = [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"),
                    ("recall_fpr5_test", "Recall at 5% FPR")]
    ref = res[(res.escenario == "10000") & (res.modelo == "TabPFN")]
    if escs:
        fig, ejes = plt.subplots(3, len(escs), figsize=(max(11, 7 * len(escs)), 11), sharey="row",
                                 squeeze=False)
        for j, esc in enumerate(escs):
            g = res[res.escenario == esc]
            modelos = [m for m in MODELOS_ANALISIS if m in set(g.modelo)]
            base = np.arange(1, len(modelos) + 1)
            for i, (m, titulo) in enumerate(metricas_fig):
                ax = ejes[i, j]
                for var, desp, color_def in [("defecto", -0.19, GRIS), ("ajustado", 0.19, AZUL)]:
                    gv = g[g.variante == var]
                    ms = [mo for mo in modelos if mo in set(gv.modelo)]
                    if not ms:
                        continue
                    pos = [base[modelos.index(mo)] + (0 if mo in FUNDACIONALES else desp) for mo in ms]
                    cols = [COLOR_MODELO[mo] if mo in FUNDACIONALES else color_def for mo in ms]
                    cajas(ax, [gv.loc[gv.modelo == mo, m].values for mo in ms], pos, cols, 0.34)
                if esc == "completo" and len(ref):
                    ax.axhline(ref[m].mean(), ls="--", lw=1.5, color=NARANJA)
                ax.set_xticks(base)
                ax.set_xticklabels([NOMBRES_EN[mo] for mo in modelos], rotation=35, ha="right")
                ax.grid(axis="y", alpha=0.3)
                if j == 0:
                    ax.set_ylabel(titulo)
                if i == 0:
                    n = int(g.n_entrenamiento.max())
                    ax.set_title(etiqueta_escenario(esc, n), fontsize=10)
                if i < 2:
                    ax.tick_params(labelbottom=False)
        leyenda = [Patch(facecolor=COLOR_MODELO["TabPFN-3.5"], edgecolor=TINTA, label="TabPFN-3.5 (default)"),
                   Patch(facecolor=NARANJA, edgecolor=TINTA, label="TabPFN v2 (default)"),
                   Patch(facecolor=GRIS, edgecolor=TINTA, label="Classical, default hyperparameters"),
                   Patch(facecolor=AZUL, edgecolor=TINTA, label="Classical, tuned hyperparameters")]
        if "completo" in escs and len(ref):
            leyenda.append(Line2D([], [], ls="--", color=NARANJA,
                                  label="TabPFN v2 mean with 10,000 rows"))
        fig.legend(handles=leyenda, loc="upper center", ncol=3 if len(escs) > 1 else 2,
                   frameon=False,
                   fontsize=9, bbox_to_anchor=(0.5, 1.0))
        fig.text(0.01, 0.005, f"Test set: months 6-7, {n_prueba:,} rows, fraud prevalence "
                 f"{prev:.2%} (PR-AUC of a random classifier = {prev:.3f}).\nBoxes: {n_reps} repetitions; "
                 "line = median, diamond = mean.", fontsize=8, color=TINTA)
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        fig.savefig(dir_salida / "fig_desempeno.png", dpi=300)
        plt.close(fig)

    # --- Figura 2: curva de aprendizaje (media ± IC 95 %)
    variantes = [v for v in ["defecto", "ajustado"] if v in set(res.variante)]
    if res.escenario.nunique() > 1:
        fig, ejes = plt.subplots(2, len(variantes), figsize=(7 * len(variantes), 9),
                                 sharey="row", squeeze=False)
        for j, var in enumerate(variantes):
            for i, (m, titulo) in enumerate([("pr_auc", "PR-AUC"),
                                             ("recall_fpr5_test", "Recall at 5% FPR")]):
                ax = ejes[i, j]
                for mo in MODELOS_ANALISIS:
                    g = res[(res.modelo == mo) & ((res.variante == var) | res.modelo.isin(FUNDACIONALES))]
                    if g.empty:
                        continue
                    agg = g.groupby("escenario").agg(n=("n_entrenamiento", "mean"),
                                                     media=(m, "mean"),
                                                     ic=(m, ic95)).sort_values("n")
                    ax.plot(agg.n, agg.media, marker=MARCADOR[mo], color=COLOR_MODELO[mo],
                            lw=2.5 if mo in FUNDACIONALES else 1.6, markersize=6,
                            label=NOMBRES_EN[mo], zorder=3 if mo in FUNDACIONALES else 2)
                    ax.fill_between(agg.n, agg.media - agg.ic, agg.media + agg.ic,
                                    color=COLOR_MODELO[mo], alpha=0.12, lw=0)
                ax.set_xscale("log")
                ns = sorted(res.n_entrenamiento.unique())
                ax.set_xticks(ns)
                ax.set_xticklabels([f"{int(n):,}" for n in ns], rotation=35, ha="right")
                ax.minorticks_off()
                ax.grid(alpha=0.3)
                if j == 0:
                    ax.set_ylabel(titulo)
                if i == 0:
                    ax.set_title("Classical models: " + ("default hyperparameters" if var == "defecto"
                                                         else "tuned hyperparameters")
                                 + " (TabPFN always default)", fontsize=10)
                if i == 1:
                    ax.set_xlabel("Training rows (log scale; 10% fraud after undersampling)")
        h, l = ejes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=9,
                   bbox_to_anchor=(0.5, 1.0))
        fig.text(0.01, 0.005, f"Lines: mean of {n_reps} repetitions; bands: 95% CI. TabPFN v2 is limited to "
                 f"{TAMANO_MAX_TABPFN:,} training rows.", fontsize=8, color=TINTA)
        fig.tight_layout(rect=(0, 0.02, 1, 0.94))
        fig.savefig(dir_salida / "fig_curva_aprendizaje.png", dpi=300)
        plt.close(fig)

    # --- Figura 3: costo computacional (log), una figura por escenario para que las cajas se lean
    # bien; misma escala vertical en ambas para poder compararlas
    series = [("defecto", "t_ajuste_s", -0.28, GRIS, "Fit time (default)"),
              ("ajustado", "t_busqueda_s", 0.0, AZUL, "Total tuning time (random search)"),
              ("defecto", "t_prediccion_s", 0.28, AQUA,
               f"Inference time, {n_prueba:,} test rows (default)")]
    tiempos = res[res.escenario.isin(escs)][[s[1] for s in series]].clip(lower=1e-3)
    limites_y = (tiempos.min().min() / 2, tiempos.max().max() * 2)
    for esc in escs:
        fig, ax = plt.subplots(figsize=(11, 6.2))
        g = res[res.escenario == esc]
        modelos = [m for m in MODELOS_ANALISIS if m in set(g.modelo)]
        base = np.arange(1, len(modelos) + 1)
        for var, col, desp, color, _ in series:
            gv = g[(g.variante == var) & g[col].notna()]
            ms = [mo for mo in modelos if mo in set(gv.modelo)]
            if not ms:
                continue
            datos = [gv.loc[gv.modelo == mo, col].clip(lower=1e-3).values for mo in ms]
            pos = [base[modelos.index(mo)] + desp for mo in ms]
            cajas(ax, datos, pos, [color] * len(ms), 0.26)
            # Con 10 repeticiones casi idénticas la caja se reduce a una línea: la media se dibuja
            # con el color de la serie para que siempre se distinga
            ax.scatter(pos, [np.mean(d) for d in datos], marker="D", s=34, color=color,
                       edgecolor=TINTA, linewidth=0.8, zorder=5)
        ax.set_yscale("log")
        ax.set_ylim(*limites_y)
        ax.set_xticks(base)
        ax.set_xticklabels([NOMBRES_EN[mo] for mo in modelos], rotation=30, ha="right", fontsize=10)
        for x in base[:-1] + 0.5:
            ax.axvline(x, color=TINTA, alpha=0.12, lw=0.8)
        ax.set_xlim(base[0] - 0.5, base[-1] + 0.5)
        ax.set_ylabel("Seconds (log scale; values < 1 ms shown at 1 ms)", fontsize=10)
        ax.set_title(etiqueta_escenario(esc, int(g.n_entrenamiento.max())), fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(handles=[Patch(facecolor=c, edgecolor=TINTA, label=et) for *_, c, et in series],
                  loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False,
                  fontsize=9.5)
        fig.tight_layout()
        fig.savefig(dir_salida / f"fig_costo_{esc}.png", dpi=300)
        plt.close(fig)

    # --- Figura 4: importancia de variables (caída de ROC-AUC, escalada al máximo de cada modelo)
    if imp is not None and not imp.empty:
        tabla = imp.pivot(index="variable", columns="modelo", values="caida_roc_auc")
        tabla = tabla[[m for m in MODELOS_ANALISIS if m in tabla.columns]].rename(index=NOMBRES_VARIABLE)
        rel = tabla.clip(lower=0) / tabla.clip(lower=0).max()
        orden = tabla.rank(ascending=False).mean(axis=1).sort_values().index
        rel = rel.loc[orden]
        fig, ax = plt.subplots(figsize=(1.1 * rel.shape[1] + 4, 0.32 * rel.shape[0] + 1.8))
        im = ax.imshow(rel.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(rel.shape[1]))
        ax.set_xticklabels([NOMBRES_EN[m] for m in rel.columns], rotation=35, ha="right")
        ax.set_yticks(range(rel.shape[0]))
        ax.set_yticklabels(rel.index, fontsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.04)
        cb.set_label("Permutation importance\n(ROC-AUC drop, relative to each model's maximum)")
        ax.set_title("Permutation importance of the original variables (sorted by mean rank)",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(dir_salida / "fig_importancia.png", dpi=300)
        plt.close(fig)


# ----------------------------------------------------------------------------- principal
def guardar_entorno(ruta, args):
    info = {"python": sys.version, "plataforma": platform.platform(), "argumentos": vars(args)}
    for mod in ["numpy", "pandas", "sklearn", "scipy", "xgboost", "torch", "tabpfn", "tensorflow"]:
        try:
            info[mod] = __import__(mod).__version__
        except Exception as e:
            info[mod] = f"no disponible ({e})"
    try:
        import torch
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "ninguna"
    except Exception:
        pass
    ruta.write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str), encoding="utf-8")



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", default=str(BASE_DIR / "data" / "Base.csv"))
    ap.add_argument("--salida", default=str(BASE_DIR / "resultados_experimento"))
    ap.add_argument("--tamanos", default="1000,2500,5000,10000,completo")
    ap.add_argument("--variantes", default="defecto,ajustado")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--ensayos", type=int, default=20, help="configuraciones por búsqueda aleatoria")
    ap.add_argument("--proporcion-fraude", type=float, default=0.10)
    ap.add_argument("--n-val", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=50000)
    ap.add_argument("--tabpfn-estimadores", type=int, default=8)
    ap.add_argument("--tabpfn-lote", type=int, default=5000)
    ap.add_argument("--keras-epocas", type=int, default=10)
    ap.add_argument("--keras-lote", type=int, default=256)
    ap.add_argument("--modelos", default="", help="lista separada por comas para limitar modelos")
    ap.add_argument("--importancia", action="store_true")
    ap.add_argument("--imp-tamano", default="10000")
    ap.add_argument("--imp-n", type=int, default=5000)
    ap.add_argument("--imp-repeticiones", type=int, default=3)
    ap.add_argument("--prueba-rapida", action="store_true")
    ap.add_argument("--solo-analisis", action="store_true")
    ap.add_argument("--reiniciar", action="store_true")
    args = ap.parse_args()

    if args.prueba_rapida:
        args.tamanos, args.reps, args.ensayos = "500,1000,completo", 2, 3
        args.n_val, args.n_test, args.tabpfn_estimadores, args.keras_epocas = 5000, 10000, 2, 2
        args.imp_tamano, args.imp_n, args.imp_repeticiones = "1000", 1000, 1
        args.salida = str(Path(args.salida).with_name(Path(args.salida).name + "_prueba"))

    dir_salida = Path(args.salida)
    dir_salida.mkdir(parents=True, exist_ok=True)
    log = Registro(dir_salida / "experimento.log")
    ruta_csv = dir_salida / "resultados_crudos.csv"
    ruta_cfg = dir_salida / "configuracion.json"

    if args.solo_analisis:
        analizar(dir_salida, log)
        return

    if args.reiniciar:
        for r in (ruta_csv, ruta_cfg, dir_salida / "importancia.csv"):
            r.unlink(missing_ok=True)
    protocolo = {k: getattr(args, k) for k in CLAVES_PROTOCOLO}
    if ruta_cfg.exists():
        previo = json.loads(ruta_cfg.read_text(encoding="utf-8"))
        distintos = {k: (previo.get(k), v) for k, v in protocolo.items() if previo.get(k) != v}
        if distintos:
            sys.exit(f"El protocolo cambió respecto a los resultados guardados: {distintos}. "
                     "Usa --reiniciar u otra carpeta con --salida.")
    ruta_cfg.write_text(json.dumps(protocolo, indent=2), encoding="utf-8")
    guardar_entorno(dir_salida / "entorno.json", args)

    df = limpiar(cargar_datos(Path(args.datos), BASE_DIR / "data" / "Base_cache.pkl", log))
    columnas = [c for c in df.columns if c not in EXCLUIR]
    entren = df[df.month <= 4]
    valid = df[df.month == 5]
    prueba = df[df.month >= 6]
    log(f"Entrenamiento (meses 0-4): {len(entren):,} filas, fraude {entren[OBJETIVO].mean():.4%}")
    log(f"Validación (mes 5): {len(valid):,} filas, fraude {valid[OBJETIVO].mean():.4%}")
    log(f"Prueba (meses 6-7): {len(prueba):,} filas, fraude {prueba[OBJETIVO].mean():.4%}")
    Xv, yv = submuestra_estratificada(valid[columnas], valid[OBJETIVO], args.n_val, 0)
    Xt, yt = submuestra_estratificada(prueba[columnas], prueba[OBJETIVO], args.n_test, 0)
    yv, yt = yv.to_numpy(), yt.to_numpy()

    if args.importancia:
        importancia(args, (entren, columnas, Xv, yv, Xt, yt), dir_salida, log)
        analizar(dir_salida, log)
        return

    hechos = set()
    if ruta_csv.exists():
        prev = pd.read_csv(ruta_csv, dtype={"escenario": str})
        hechos = set(zip(prev.escenario, prev.variante, prev.rep, prev.modelo))
        log(f"Reanudando: {len(hechos)} resultados ya guardados")

    filtro = set(m.strip() for m in args.modelos.split(",") if m.strip())
    tamanos = [t.strip() for t in args.tamanos.split(",")]
    plan = []
    # Primero todas las configuraciones por defecto (incluye TabPFN), luego las ajustadas
    for var in [v.strip() for v in args.variantes.split(",")]:
        for tam in tamanos:
            for rep in range(args.reps):
                for mod in MODELOS:
                    if mod == "TabPFN" and (var != "defecto" or tam == "completo"
                                            or int(tam) > TAMANO_MAX_TABPFN):
                        continue
                    if (not filtro or mod in filtro) and (tam, var, rep, mod) not in hechos:
                        plan.append((tam, var, rep, mod))
    log(f"Pendientes: {len(plan)} ajustes")

    cache = {}
    for i, (tam, var, rep, mod) in enumerate(plan, 1):
        semilla = 1000 + rep
        if (tam, rep) not in cache:
            Xtr, ytr = entrenamiento(entren, columnas, tam, semilla, args)
            prep = preprocesador(columnas).fit(Xtr)
            cache = {(tam, rep): (prep.transform(Xtr).astype(np.float32), ytr.to_numpy(),
                                  prep.transform(Xv).astype(np.float32),
                                  prep.transform(Xt).astype(np.float32))}
        Ptr, ytr_np, Pv, Pt = cache[(tam, rep)]
        log(f"[{i}/{len(plan)}] n={tam}, {var}, rep {rep}, {mod} "
            f"({len(ytr_np):,} filas, {int(ytr_np.sum())} fraudes)...")
        try:
            r = correr(mod, var, semilla, Ptr, ytr_np, Pv, yv, Pt, yt, args, log)
        except Exception as e:
            log(f"   ERROR en {mod}: {type(e).__name__}: {e}")
            continue
        fila = {"escenario": tam, "variante": var, "rep": rep, "modelo": mod, "semilla": semilla,
                "n_entrenamiento": len(ytr_np), "fraudes_entrenamiento": int(ytr_np.sum()),
                "n_prueba": len(yt), "prevalencia_test": yt.mean(), **r}
        agregar_fila(ruta_csv, fila)
        busq = "" if np.isnan(r["t_busqueda_s"]) else f" | búsqueda {r['t_busqueda_s']:.1f} s"
        log(f"   ROC-AUC {r['roc_auc']:.4f} | PR-AUC {r['pr_auc']:.4f} | "
            f"Recall@5%FPR {r['recall_fpr5_test']:.4f} | TP {r['tp']} FN {r['fn']} "
            f"FP {r['fp']} | ajuste {r['t_ajuste_s']:.1f} s{busq} | pred {r['t_prediccion_s']:.1f} s")

    if ruta_csv.exists():
        analizar(dir_salida, log)


if __name__ == "__main__":
    main()
