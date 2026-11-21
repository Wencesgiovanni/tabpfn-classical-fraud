"""Diagnóstico de x1 y x2 en las variantes III y V de BAF.

Para cada variante y cada una de las dos columnas calcula, con todas las solicitudes:
  - el ROC-AUC univariado de la columna para el fraude en los meses de entrenamiento (0-4),
    validación (5) y prueba (6-7);
  - el estadístico de Kolmogorov-Smirnov entre su distribución en los meses 0-4 y 6-7.
En la variante V el sesgo de separabilidad solo existe en los primeros seis meses (Jesus et al.,
2022), así que x1 y x2 predicen el fraude en entrenamiento y validación pero no en prueba.

Uso (desde esta carpeta, con data/Variant III.csv y data/Variant V.csv):
  python diagnostico_x1_x2.py
Escribe resultados_finales/robustez/diagnostico_x1_x2.csv.
"""
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

import experimento_fraude as ef

PERIODOS = {"entrenamiento (0-4)": lambda m: m <= 4, "validacion (5)": lambda m: m == 5,
            "prueba (6-7)": lambda m: m >= 6}


def main():
    filas = []
    for nombre in ["Variant III", "Variant V"]:
        ruta = ef.BASE_DIR / "data" / f"{nombre}.csv"
        df = ef.cargar_datos(ruta, ef.BASE_DIR / "data" / f"{nombre}_cache.pkl", print)
        for col in ["x1", "x2"]:
            fila = {"conjunto": nombre, "columna": col}
            for etiqueta, cond in PERIODOS.items():
                g = df[cond(df.month)]
                fila[f"roc_auc_{etiqueta}"] = roc_auc_score(g[ef.OBJETIVO], g[col])
            fila["ks_entrenamiento_vs_prueba"] = ks_2samp(df.loc[df.month <= 4, col],
                                                          df.loc[df.month >= 6, col]).statistic
            filas.append(fila)
    res = pd.DataFrame(filas)
    salida = ef.BASE_DIR / "resultados_finales" / "robustez"
    salida.mkdir(parents=True, exist_ok=True)
    res.to_csv(salida / "diagnostico_x1_x2.csv", index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
