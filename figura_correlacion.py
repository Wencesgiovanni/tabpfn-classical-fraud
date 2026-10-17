"""Figura 1 del artículo: matriz de correlación de Pearson de las variables de BAF Base.

Los valores centinela se tratan como faltantes (correlaciones con observaciones completas por
pares) y se omite device_fraud_count, que es constante. Escribe
resultados_finales/fig_correlacion.png.

Uso (con el entorno de requirements-clasicos-v2.txt):
  python figura_correlacion.py --datos data/Base.csv
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import experimento_fraude as ef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default=str(ef.BASE_DIR / "data" / "Base.csv"))
    args = ap.parse_args()
    df = ef.cargar_datos(Path(args.datos), ef.BASE_DIR / "data" / "Base_cache.pkl", print)
    df = ef.limpiar(df).drop(columns=["device_fraud_count"]).rename(columns=ef.NOMBRES_VARIABLE)
    c = df.corr()
    print("Mayores |r| con fraud_bool:",
          c["fraud_bool"].drop("fraud_bool").abs().sort_values().tail(5).round(3).to_dict())
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)))
    ax.set_xticklabels(c.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(c)))
    ax.set_yticklabels(c.columns, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Pearson correlation (pairwise complete observations)")
    fig.tight_layout()
    salida = ef.BASE_DIR / "resultados_finales"
    salida.mkdir(exist_ok=True)
    fig.savefig(salida / "fig_correlacion.png", dpi=300)
    print("escrito", salida / "fig_correlacion.png")


if __name__ == "__main__":
    main()
