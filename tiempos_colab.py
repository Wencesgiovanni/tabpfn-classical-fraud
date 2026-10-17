"""Corrida de los modelos clásicos y TabPFN v2 en Google Colab (A100), con tiempos.

Aplica el protocolo de experimento_fraude.py (mismo código de modelos, búsqueda aleatoria,
preprocesamiento y métricas) sobre las particiones exactas exportadas por
exportar_para_colab.py, de modo que el desempeño y los tiempos de ajuste, búsqueda e
inferencia de todos los modelos se miden en el mismo hardware: modelos clásicos (incluida la
red Keras) en CPU y TabPFN en GPU.

Grupos (cada uno necesita su versión de tabpfn, ver el cuaderno):
  principal  7 modelos clásicos (por defecto y ajustados) + TabPFN v2   (tabpfn 2.1.3)
  v35        TabPFN-3.5, repetición 0, como control de tiempos en otra sesión (tabpfn 9.1.0)
  importancia    importancia por permutación (10,000 filas, rep. 0): clásicos + TabPFN v2 (tabpfn 2.1.3)
  importancia35  importancia por permutación de TabPFN-3.5                     (tabpfn 9.1.0)

Cada resultado se compara con el publicado (resultados_publicados.csv, si existe) para
confirmar que el desempeño se reproduce. Los resultados se
guardan fila por fila en <carpeta>/resultados_tiempos_colab.csv; si la sesión se corta, al
volver a ejecutar el script continúa donde se quedó.

Uso (en Colab):
  !python tiempos_colab.py --grupo principal --prueba-rapida
  !python tiempos_colab.py --grupo principal
  !python tiempos_colab.py --grupo v35
  !python tiempos_colab.py --grupo importancia35
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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TABPFN_NO_BROWSER", "1")

import experimento_fraude as ef  # noqa: E402

TAMANOS = ["1000", "2500", "5000", "10000", "completo"]


class Args:
    """Valores del protocolo del artículo."""
    ensayos = 20
    tabpfn_estimadores = 8
    tabpfn_lote = 5000
    keras_epocas = 10
    keras_lote = 256
    proporcion_fraude = 0.10
    imp_tamano = "10000"
    imp_n = 5000
    imp_repeticiones = 3


def huella(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def modelo_cpu():
    try:
        for linea in Path("/proc/cpuinfo").read_text().splitlines():
            if linea.startswith("model name"):
                return linea.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--carpeta", default="/content/drive/MyDrive/paquete_colab")
    ap.add_argument("--grupo", choices=["principal", "v35", "importancia", "importancia35"], required=True)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--prueba-rapida", action="store_true")
    args = ap.parse_args()

    carpeta = Path(args.carpeta)
    archivo_log = carpeta / "corrida_tiempos.log"
    f_log = open(archivo_log, "a", encoding="utf-8")

    def log(*msg):
        linea = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(m) for m in msg)
        print(linea, flush=True)
        f_log.write(linea + "\n")
        f_log.flush()

    # --- Dispositivos (protocolo del artículo): la red Keras en CPU
    if args.grupo in ("principal", "importancia"):
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
    import torch
    from importlib.metadata import version as pkg_version
    if not torch.cuda.is_available():
        sys.exit("No hay GPU. En Colab: Entorno de ejecución > Cambiar tipo de entorno > GPU.")
    v_tabpfn = pkg_version("tabpfn")
    esperada = "2." if args.grupo in ("principal", "importancia") else "9."
    if not v_tabpfn.startswith(esperada):
        sys.exit(f"El grupo '{args.grupo}' necesita tabpfn {esperada}x y está instalada la "
                 f"{v_tabpfn}. Ejecuta la celda de instalación correspondiente.")
    if args.grupo in ("v35", "importancia35") and not (os.environ.get("TABPFN_TOKEN") or
                                    (Path.home() / ".cache" / "tabpfn" / "auth_token").exists()):
        sys.exit("Falta TABPFN_TOKEN (ver las instrucciones del cuaderno).")

    entorno = {"grupo": args.grupo, "python": sys.version, "plataforma": platform.platform(),
               "cpu": modelo_cpu(), "nucleos_logicos": os.cpu_count(),
               "gpu": torch.cuda.get_device_name(0),
               "memoria_gpu_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)}
    for paq in ["tabpfn", "torch", "scikit-learn", "xgboost", "tensorflow", "numpy", "pandas",
                "scipy"]:
        try:
            entorno[paq] = pkg_version(paq)
        except Exception:
            entorno[paq] = "no instalado"
    (carpeta / f"entorno_tiempos_{args.grupo}.json").write_text(json.dumps(entorno, indent=2),
                                                                encoding="utf-8")
    log("Entorno:", entorno)

    # --- Datos y particiones (idénticos al experimento; se verifica con huellas)
    datos = np.load(carpeta / "datos_baf.npz", allow_pickle=False)
    df = ef.limpiar(pd.DataFrame(datos["valores"], columns=[str(c) for c in datos["columnas"]]))
    part = np.load(carpeta / "particiones.npz", allow_pickle=False)
    ref = json.loads((carpeta / "huellas.json").read_text(encoding="utf-8"))
    columnas = ref["columnas_predictoras"]
    Xv = df.loc[part["validacion"], columnas]
    yv = df.loc[part["validacion"], ef.OBJETIVO].to_numpy().astype(int)
    Xt = df.loc[part["prueba"], columnas]
    yt = df.loc[part["prueba"], ef.OBJETIVO].to_numpy().astype(int)
    assert len(yv) == ref["n_val"] and len(yt) == ref["n_test"]
    assert int(yt.sum()) == ref["fraudes_prueba"]

    if args.grupo.startswith("importancia"):
        # Mismo procedimiento que experimento_fraude.py --importancia; continúa si se corta
        dir_imp = carpeta / "importancia_colab"
        dir_imp.mkdir(exist_ok=True)
        modelos = ef.MODELOS if args.grupo == "importancia" else ["TabPFN-3.5"]
        entren = df[df.month <= 4]
        t0 = time.perf_counter()
        ef.importancia(Args, (entren, columnas, Xv, yv, Xt, yt), dir_imp, log, modelos)
        log(f"Importancia ({args.grupo}) terminada en {(time.perf_counter() - t0) / 60:.0f} min.")
        return

    # --- Resultados publicados, para comprobar que el desempeño se reproduce
    pr_previo = {}
    if (carpeta / "resultados_publicados.csv").exists():
        previos = pd.read_csv(carpeta / "resultados_publicados.csv", dtype={"escenario": str})
        pr_previo = {(r.escenario, r.variante, int(r.rep), r.modelo): r.pr_auc
                     for r in previos.itertuples()}

    ruta_csv = carpeta / ("resultados_tiempos_colab_prueba.csv" if args.prueba_rapida
                          else "resultados_tiempos_colab.csv")
    hechos = set()
    if ruta_csv.exists():
        prev = pd.read_csv(ruta_csv, dtype={"escenario": str})
        hechos = set(zip(prev.escenario, prev.variante, prev.rep, prev.modelo))
        log(f"Reanudando: {len(hechos)} resultados ya guardados en {ruta_csv.name}")

    # --- Plan: una repetición completa (todos los tamaños y modelos) antes de la siguiente,
    # para que, si se acaba el tiempo de Colab, lo medido cubra todos los modelos por igual
    if args.grupo == "principal":
        reps = args.reps or 10
        tamanos = TAMANOS
        if args.prueba_rapida:
            reps, tamanos = 1, ["1000"]
        plan = []
        for rep in range(reps):
            for tam in tamanos:
                if tam != "completo":
                    plan.append((tam, "defecto", rep, "TabPFN"))
                for var in ["defecto", "ajustado"]:
                    plan += [(tam, var, rep, m) for m in ef.CLASICOS]
    else:
        reps = args.reps or 1
        plan = [(tam, "defecto", rep, "TabPFN-3.5") for rep in range(reps) for tam in TAMANOS]
    plan = [p for p in plan if p not in hechos]
    log(f"Pendientes: {len(plan)} ajustes ({args.grupo})")

    cache = {}
    t_inicio = time.perf_counter()
    for k, (tam, var, rep, mod) in enumerate(plan, 1):
        clave = f"{tam}_rep{rep}"
        info = ref["entrenamiento"][clave]
        if clave not in cache:
            Xtr = df.loc[part["entrenamiento_" + clave], columnas]
            ytr = df.loc[part["entrenamiento_" + clave], ef.OBJETIVO].to_numpy().astype(int)
            assert len(ytr) == info["n"] and int(ytr.sum()) == info["fraudes"], clave
            prep = ef.preprocesador(columnas).fit(Xtr)
            Pt64 = prep.transform(Xt).astype(np.float64)
            h = huella(np.round(Pt64, 5))
            if h != info["huella_prueba"]:
                sys.exit(f"[{clave}] La matriz de prueba preprocesada NO coincide con la "
                         f"de referencia ({h} != {info['huella_prueba']}). No se ejecuta nada más.")
            cache = {clave: (prep.transform(Xtr).astype(np.float32), ytr,
                             prep.transform(Xv).astype(np.float32), Pt64.astype(np.float32), h)}
        Ptr, ytr, Pv, Pt, h = cache[clave]
        if mod in ef.FUNDACIONALES:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            r = ef.correr(mod, var, info["semilla"], Ptr, ytr, Pv, yv, Pt, yt, Args, log)
        except Exception as e:
            log(f"[{k}/{len(plan)}] {clave} {var} {mod}: ERROR {type(e).__name__}: {e}")
            (carpeta / "errores_tiempos.log").open("a", encoding="utf-8").write(
                f"{time.ctime()} {clave} {var} {mod}\n{traceback.format_exc()}\n")
            continue
        previo = pr_previo.get((tam, var, rep, mod), np.nan)
        fila = {"escenario": tam, "variante": var, "rep": rep, "modelo": mod,
                "semilla": info["semilla"], "n_entrenamiento": len(ytr),
                "fraudes_entrenamiento": int(ytr.sum()), "n_prueba": len(yt),
                "prevalencia_test": float(yt.mean()), **r,
                "pr_auc_publicado": previo,
                "memoria_gpu_pico_gb": (torch.cuda.max_memory_allocated() / 1e9
                                        if mod in ef.FUNDACIONALES else np.nan),
                "huella_prueba": h, "grupo": args.grupo}
        ef.agregar_fila(ruta_csv, fila)
        busq = "" if np.isnan(r["t_busqueda_s"]) else f", búsqueda {r['t_busqueda_s']:.1f} s"
        log(f"[{k}/{len(plan)}] {clave} {var} {mod}: PR-AUC {r['pr_auc']:.4f} "
            f"(publicado {previo:.4f}) | ajuste {r['t_ajuste_s']:.2f} s{busq}, "
            f"predicción {r['t_prediccion_s']:.2f} s | "
            f"{(time.perf_counter() - t_inicio) / 60:.0f} min transcurridos")
    log("Terminado.")


if __name__ == "__main__":
    main()
