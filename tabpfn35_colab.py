"""Evaluación de TabPFN-3.5 con el mismo protocolo del experimento (para Google Colab).

Usa las particiones exactas exportadas por exportar_para_colab.py (mismas filas de
entrenamiento, validación y prueba que los demás modelos) y reproduce la limpieza, el
preprocesamiento y las métricas de experimento_fraude.py. Antes de cada corrida verifica,
con una huella SHA-256, que la matriz de prueba preprocesada sea idéntica a la original.

Los resultados se guardan fila por fila en <carpeta>/resultados_tabpfn35.csv; si la sesión
de Colab se corta, al volver a ejecutar el script continúa donde se quedó.

Uso (en una celda de Colab, con la GPU activada y TABPFN_TOKEN definido):
  !python tabpfn35_colab.py --solo-verificar      # comprueba datos y particiones (sin GPU)
  !python tabpfn35_colab.py --prueba-rapida       # 1 repetición de 1,000 y de 10,000 filas
  !python tabpfn35_colab.py                       # corrida completa (5 tamaños x 10 reps)
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("TABPFN_NO_BROWSER", "1")

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Idéntico a experimento_fraude.py --------------------------------------------------
OBJETIVO = "fraud_bool"
CATEGORICAS = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
FALTANTE_MENOS_UNO = ["prev_address_months_count", "current_address_months_count",
                      "bank_months_count", "session_length_in_minutes",
                      "device_distinct_emails_8w"]
FALTANTE_NEGATIVO = ["intended_balcon_amount"]
FPR_OBJETIVO = 0.05
TAMANOS = ["1000", "2500", "5000", "10000", "completo"]
NOMBRE_MODELO = "TabPFN-3.5"


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
# ---------------------------------------------------------------------------------------


def huella(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def registro(carpeta):
    f = open(carpeta / "corrida_tabpfn35.log", "a", encoding="utf-8")

    def log(*msg):
        linea = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(m) for m in msg)
        print(linea, flush=True)
        f.write(linea + "\n")
        f.flush()
    return log


def sincronizar():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def puntajes(modelo, X, lote):
    return np.concatenate([modelo.predict_proba(X[i:i + lote])[:, 1]
                           for i in range(0, len(X), lote)])


def agregar_fila(ruta, fila):
    df = pd.DataFrame([fila])
    if ruta.exists():
        columnas = pd.read_csv(ruta, nrows=0).columns
        df[list(columnas)].to_csv(ruta, mode="a", header=False, index=False)
    else:
        df.to_csv(ruta, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--carpeta", default="/content/drive/MyDrive/paquete_colab")
    ap.add_argument("--version", default="v3.5")
    ap.add_argument("--tamanos", default=",".join(TAMANOS))
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--lote", type=int, default=5000, help="filas de prueba por llamada")
    ap.add_argument("--solo-verificar", action="store_true")
    ap.add_argument("--prueba-rapida", action="store_true")
    args = ap.parse_args()
    if args.prueba_rapida:
        args.tamanos, args.reps = "1000,10000", 1

    carpeta = Path(args.carpeta)
    log = registro(carpeta)
    datos = np.load(carpeta / "datos_baf.npz", allow_pickle=False)
    df = limpiar(pd.DataFrame(datos["valores"], columns=[str(c) for c in datos["columnas"]]))
    part = np.load(carpeta / "particiones.npz", allow_pickle=False)
    ref = json.loads((carpeta / "huellas.json").read_text(encoding="utf-8"))
    columnas = ref["columnas_predictoras"]

    Xv = df.loc[part["validacion"], columnas]
    yv = df.loc[part["validacion"], OBJETIVO].to_numpy().astype(int)
    Xt = df.loc[part["prueba"], columnas]
    yt = df.loc[part["prueba"], OBJETIVO].to_numpy().astype(int)
    assert len(yv) == ref["n_val"] and len(yt) == ref["n_test"]
    assert int(yt.sum()) == ref["fraudes_prueba"]
    log(f"Datos: {df.shape}; validación {len(yv):,} ({yv.sum()} fraudes); "
        f"prueba {len(yt):,} ({yt.sum()} fraudes)")

    ruta_csv = carpeta / ("resultados_tabpfn35_prueba.csv" if args.prueba_rapida
                          else "resultados_tabpfn35.csv")
    hechos = set()
    if ruta_csv.exists():
        prev = pd.read_csv(ruta_csv, dtype={"escenario": str})
        hechos = set(zip(prev.escenario, prev.rep))
        log(f"Reanudando: {len(hechos)} corridas ya guardadas en {ruta_csv.name}")

    if not args.solo_verificar:
        import torch
        from importlib.metadata import version as pkg_version
        from tabpfn import TabPFNClassifier
        if not torch.cuda.is_available():
            sys.exit("No hay GPU. En Colab: Entorno de ejecución > Cambiar tipo de entorno > GPU.")
        if not (os.environ.get("TABPFN_TOKEN")
                or (Path.home() / ".cache" / "tabpfn" / "auth_token").exists()):
            sys.exit("Falta TABPFN_TOKEN (ver las instrucciones del cuaderno).")
        entorno = {"python": sys.version, "plataforma": platform.platform(),
                   "tabpfn": pkg_version("tabpfn"), "torch": torch.__version__,
                   "sklearn": pkg_version("scikit-learn"), "numpy": np.__version__,
                   "pandas": pd.__version__, "gpu": torch.cuda.get_device_name(0),
                   "memoria_gpu_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
                   "version_modelo": args.version, "lote": args.lote}
        (carpeta / "entorno_colab.json").write_text(json.dumps(entorno, indent=2), encoding="utf-8")
        log("Entorno:", entorno)
        (carpeta / "puntajes").mkdir(exist_ok=True)

    tamanos = [t.strip() for t in args.tamanos.split(",")]
    # Una repetición de cada tamaño antes de pasar a la siguiente: así el tamaño completo
    # se prueba desde el principio y cualquier problema aparece pronto.
    plan = [(t, r) for r in range(args.reps) for t in tamanos]
    t_inicio = time.perf_counter()
    for k, (tam, rep) in enumerate(plan, 1):
        clave = f"{tam}_rep{rep}"
        info = ref["entrenamiento"][clave]
        Xtr = df.loc[part["entrenamiento_" + clave], columnas]
        ytr = df.loc[part["entrenamiento_" + clave], OBJETIVO].to_numpy().astype(int)
        assert len(ytr) == info["n"] and int(ytr.sum()) == info["fraudes"], clave
        prep = preprocesador(columnas).fit(Xtr)
        Pt64 = prep.transform(Xt).astype(np.float64)
        h = huella(np.round(Pt64, 5))
        if h != info["huella_prueba"]:
            sys.exit(f"[{clave}] La matriz de prueba preprocesada NO coincide con la original "
                     f"({h} != {info['huella_prueba']}). No se ejecuta nada más.")
        if args.solo_verificar:
            log(f"[{k}/{len(plan)}] {clave}: datos y preprocesamiento idénticos ({h})")
            continue
        if (tam, rep) in hechos:
            continue

        Ptr = prep.transform(Xtr).astype(np.float32)
        Pv = prep.transform(Xv).astype(np.float32)
        Pt = Pt64.astype(np.float32)
        semilla = info["semilla"]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            modelo = TabPFNClassifier.create_default_for_version(
                args.version, device="cuda", random_state=semilla)
            sincronizar()
            t0 = time.perf_counter()
            modelo.fit(Ptr, ytr)
            sincronizar()
            t_ajuste = time.perf_counter() - t0
            s_val = puntajes(modelo, Pv, args.lote)
            sincronizar()
            t0 = time.perf_counter()
            s_te = puntajes(modelo, Pt, args.lote)
            sincronizar()
            t_pred = time.perf_counter() - t0
        except Exception as e:
            log(f"[{clave}] ERROR {type(e).__name__}: {e}")
            (carpeta / "errores_tabpfn35.log").open("a", encoding="utf-8").write(
                f"{time.ctime()} {clave}\n{traceback.format_exc()}\n")
            continue

        r = metricas(yv, s_val, yt, s_te, (s_te >= 0.5).astype(int))
        hiper = {"version_modelo": args.version, "n_estimators": str(modelo.n_estimators),
                 "lote": args.lote}
        fila = {"escenario": tam, "variante": "defecto", "rep": rep, "modelo": NOMBRE_MODELO,
                "semilla": semilla, "n_entrenamiento": len(ytr),
                "fraudes_entrenamiento": int(ytr.sum()), "n_prueba": len(yt),
                "prevalencia_test": float(yt.mean()), **r,
                "t_ajuste_s": t_ajuste, "t_prediccion_s": t_pred, "t_busqueda_s": np.nan,
                "hiperparametros": json.dumps(hiper),
                "memoria_gpu_pico_gb": torch.cuda.max_memory_allocated() / 1e9,
                "huella_prueba": h}
        agregar_fila(ruta_csv, fila)
        if not args.prueba_rapida:
            np.savez_compressed(carpeta / "puntajes" / f"{clave}.npz", s_val=s_val, s_te=s_te)
        transcurrido = (time.perf_counter() - t_inicio) / 60
        log(f"[{k}/{len(plan)}] {clave}: PR-AUC {r['pr_auc']:.4f}, ROC-AUC {r['roc_auc']:.4f}, "
            f"recall@5%FPR {r['recall_fpr5_test']:.4f} | ajuste {t_ajuste:.1f} s, "
            f"predicción {t_pred:.1f} s, GPU {fila['memoria_gpu_pico_gb']:.1f} GB | "
            f"{transcurrido:.0f} min transcurridos")
    log("Terminado.")


if __name__ == "__main__":
    main()
