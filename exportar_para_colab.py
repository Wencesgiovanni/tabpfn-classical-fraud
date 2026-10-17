"""Prepara la carpeta paquete_colab/ para las corridas en Google Colab.

Lee la base BAF (data/Base.csv de Kaggle) y genera:
  - datos_baf.npz       todas las filas y columnas de la base (valores exactos)
  - particiones.npz     índices de validación, prueba y de cada conjunto de entrenamiento
                        (5 tamaños x 10 repeticiones)
  - huellas.json        huella SHA-256 de la matriz de prueba preprocesada de cada
                        (tamaño, repetición); los scripts de Colab la verifican antes de correr
  - los scripts de Colab y resultados_publicados.csv (resultados del artículo, para
    comparar cada corrida con el valor publicado)

Además compara las particiones obtenidas con las del estudio (particiones/particiones.npz) y
las huellas con las de referencia (particiones/huellas.json), para confirmar que los datos
descargados dan exactamente las mismas particiones y matrices.

Uso (desde esta carpeta, con el entorno de requirements-clasicos-v2.txt):
  python exportar_para_colab.py --datos data/Base.csv
"""
import argparse
import hashlib
import json
import shutil
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import experimento_fraude as ef

TAMANOS = ["1000", "2500", "5000", "10000", "completo"]
REPS = 10
N_VAL, N_TEST = 20000, 50000
SALIDA = ef.BASE_DIR / "paquete_colab"
REFERENCIA = ef.BASE_DIR / "particiones"


class Args:
    proporcion_fraude = 0.10


def huella(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def comparar_con_estudio(indices, huellas):
    if not (REFERENCIA / "particiones.npz").exists():
        print("AVISO: no se encontró particiones/ para comparar con el estudio")
        return
    ref = np.load(REFERENCIA / "particiones.npz")
    ref_h = json.loads((REFERENCIA / "huellas.json").read_text(encoding="utf-8"))
    malas_idx = [k for k in ref.files if not np.array_equal(ref[k], indices[k])]
    # huellas.json se obtuvo con Base.csv de Kaggle; las del estudio (huellas_copia_estudio.json)
    # difieren porque la copia usada redondeaba algunos valores a unas 10 cifras significativas
    malas_h = [k for k, v in ref_h["entrenamiento"].items()
               if v["huella_prueba"] != huellas["entrenamiento"][k]["huella_prueba"]]
    print(f"Particiones idénticas a las del estudio: {len(ref.files) - len(malas_idx)}/{len(ref.files)}")
    print(f"Matrices de prueba preprocesadas idénticas a las de referencia (Base.csv de Kaggle): "
          f"{len(ref_h['entrenamiento']) - len(malas_h)}/{len(ref_h['entrenamiento'])}")
    if malas_idx or malas_h:
        print("AVISO: los datos no reproducen exactamente las particiones o matrices de "
              "referencia; revisa que data/Base.csv sea el archivo Base.csv original de BAF.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", default=str(ef.BASE_DIR / "data" / "Base.csv"))
    args = ap.parse_args()
    SALIDA.mkdir(exist_ok=True)
    crudo = ef.cargar_datos(Path(args.datos), ef.BASE_DIR / "data" / "Base_cache.pkl", print)
    assert crudo.shape == (1000000, 32), crudo.shape
    np.savez_compressed(SALIDA / "datos_baf.npz",
                        valores=crudo.to_numpy(dtype=np.float64),
                        columnas=np.array(crudo.columns, dtype=str))
    print("datos_baf.npz:", crudo.shape)

    df = ef.limpiar(crudo)
    columnas = [c for c in df.columns if c not in ef.EXCLUIR]
    entren, valid, prueba = df[df.month <= 4], df[df.month == 5], df[df.month >= 6]
    Xv, _ = ef.submuestra_estratificada(valid[columnas], valid[ef.OBJETIVO], N_VAL, 0)
    Xt, yt = ef.submuestra_estratificada(prueba[columnas], prueba[ef.OBJETIVO], N_TEST, 0)
    indices = {"validacion": Xv.index.to_numpy(), "prueba": Xt.index.to_numpy()}
    huellas = {"n_val": N_VAL, "n_test": N_TEST, "fraudes_prueba": int(yt.sum()),
               "columnas_predictoras": columnas, "entrenamiento": {}}
    for tam in TAMANOS:
        for rep in range(REPS):
            semilla = 1000 + rep
            Xtr, ytr = ef.entrenamiento(entren, columnas, tam, semilla, Args)
            prep = ef.preprocesador(columnas).fit(Xtr)
            # Redondeo a 5 decimales: tolera diferencias de punto flotante entre versiones
            Pt = np.round(prep.transform(Xt).astype(np.float64), 5)
            clave = f"{tam}_rep{rep}"
            indices["entrenamiento_" + clave] = Xtr.index.to_numpy()
            huellas["entrenamiento"][clave] = {"semilla": semilla, "n": int(len(ytr)),
                                               "fraudes": int(ytr.sum()),
                                               "columnas_preprocesadas": int(Pt.shape[1]),
                                               "huella_prueba": huella(Pt)}
        print("particiones listas:", tam)
    np.savez_compressed(SALIDA / "particiones.npz", **indices)
    (SALIDA / "huellas.json").write_text(json.dumps(huellas, indent=1), encoding="utf-8")
    comparar_con_estudio(indices, huellas)

    for nombre in ["experimento_fraude.py", "tabpfn35_colab.py", "tiempos_colab.py"]:
        shutil.copy(ef.BASE_DIR / nombre, SALIDA / nombre)
    publicados = ef.BASE_DIR / "resultados_finales" / "resultados_crudos.csv"
    if publicados.exists():
        shutil.copy(publicados, SALIDA / "resultados_publicados.csv")
    for f in sorted(SALIDA.iterdir()):
        print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
