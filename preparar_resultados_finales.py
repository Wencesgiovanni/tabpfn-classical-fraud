"""Reúne en resultados_finales/ los resultados obtenidos en Google Colab (A100) y genera
el resumen, las figuras y las pruebas estadísticas del artículo.

Fuentes (todas de Colab, carpeta resultados_colab/):
  - resultados_tiempos_colab.csv   modelos clásicos (por defecto y ajustados) + TabPFN v2
                                   (grupo "principal"; las filas del grupo "v35" son solo
                                   un control de tiempos y no se usan aquí)
  - resultados_tabpfn35.csv        TabPFN-3.5, 10 repeticiones
  - importancia_colab/importancia.csv   importancia por permutación (todos los modelos)

Uso (con el entorno de requirements-clasicos-v2.txt):
  python preparar_resultados_finales.py
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

import experimento_fraude as ef

COLAB = ef.BASE_DIR / "resultados_colab"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--principal", default=str(COLAB / "resultados_tiempos_colab.csv"))
    ap.add_argument("--importancia", default=str(COLAB / "importancia_colab" / "importancia.csv"))
    ap.add_argument("--salida", default=str(ef.BASE_DIR / "resultados_finales"))
    args = ap.parse_args()

    salida = Path(args.salida)
    salida.mkdir(exist_ok=True)
    principal = pd.read_csv(args.principal, dtype={"escenario": str})
    if "grupo" in principal.columns:
        principal = principal[principal.grupo == "principal"]
    v35 = pd.read_csv(ef.CSV_TABPFN35, dtype={"escenario": str})
    columnas = [c for c in pd.read_csv(ef.CSV_TABPFN35, nrows=0).columns
                if c not in ("memoria_gpu_pico_gb", "huella_prueba")]
    res = pd.concat([principal[columnas], v35[columnas]], ignore_index=True)

    # Comprobaciones: 10 repeticiones por combinación y el mismo conjunto de prueba
    conteo = res.groupby(["escenario", "variante", "modelo"]).size()
    incompletas = conteo[conteo != 10]
    if len(incompletas):
        print("AVISO: combinaciones sin 10 repeticiones:\n", incompletas)
    assert (res.tp + res.fn).eq(702).all() and res.n_prueba.eq(50000).all()
    res.to_csv(salida / "resultados_crudos.csv", index=False)
    print(f"resultados_crudos.csv: {len(res)} filas, {res.modelo.nunique()} modelos")

    if Path(args.importancia).exists():
        shutil.copy(args.importancia, salida / "importancia.csv")
    else:
        print("AVISO: falta la importancia por permutación de Colab")
    # La figura de correlaciones solo depende de los datos: figura_correlacion.py
    ef.analizar(salida, print)


if __name__ == "__main__":
    main()
