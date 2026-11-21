"""Análisis de robustez: Base y las variantes I-V de BAF (y la ablación de V sin x1 ni x2).

Fuentes (todas de Colab A100):
  - resultados_finales/resultados_crudos.csv            Base (estudio principal)
  - resultados_colab/variantes/<v>/resultados_tiempos_colab.csv   TabPFN v2 y clásicos
  - resultados_colab/variantes/<v>/resultados_tabpfn35.csv        TabPFN-3.5
  con <v> = v1 ... v5 y, si existe, v5sx (variante V sin x1 ni x2, solo 10,000 filas).

Métodos comunes a todos los conjuntos: TabPFN-3.5, TabPFN v2 (solo 10,000 filas), XGBoost por
defecto y ajustado, Random Forest y Logistic L1 por defecto. En Base se toman solo esos métodos
y los tamaños de 10,000 filas y completo, para que todos los conjuntos se analicen igual.

Para cada conjunto, tamaño y métrica (PR-AUC, ROC-AUC, recall al 5 % de FPR): Friedman
(bloques = repeticiones) y Wilcoxon pareado bilateral con corrección de Holm, con cada TabPFN
como referencia (el mismo procedimiento que en el estudio principal). Entre conjuntos: victorias
de TabPFN-3.5 frente a XGBoost ajustado y Wilcoxon sobre las medias por conjunto.

Uso (desde esta carpeta):
  python analizar_robustez.py
Escribe resultados_finales/robustez/.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

import experimento_fraude as ef

COLAB = ef.BASE_DIR / "resultados_colab" / "variantes"
CONJUNTOS = {"base": "Base", "v1": "Variant I", "v2": "Variant II", "v3": "Variant III",
             "v4": "Variant IV", "v5": "Variant V"}
ABLACION = {"v5sx": "Variant V without x1, x2"}
ETIQUETA_CORTA = {"base": "Base", "v1": "I", "v2": "II", "v3": "III", "v4": "IV", "v5": "V",
                  "v5sx": "V w/o\nx1, x2"}
METODOS = [("TabPFN-3.5", "defecto"), ("TabPFN", "defecto"), ("XGBoost", "ajustado"),
           ("XGBoost", "defecto"), ("Random Forest", "defecto"), ("Logistic L1", "defecto")]
METRICAS = [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("recall_fpr5_test", "Recall at 5% FPR")]
ESCENARIOS = ["10000", "completo"]
REF_XGB = "XGBoost (ajustado)"
# Límite inferior del eje en la figura (Logistic L1 en la variante III cae mucho más abajo y
# aplastaría la escala); los valores por debajo se marcan con una flecha y su valor
PISO = {"pr_auc": -0.065}


def metodo(modelo, variante):
    return modelo + (" (ajustado)" if variante == "ajustado" else "")


NOMBRES_METODO = [metodo(m, v) for m, v in METODOS]


def cargar(salida_base):
    partes = []
    base = pd.read_csv(salida_base / "resultados_crudos.csv", dtype={"escenario": str})
    base["conjunto"] = "base"
    partes.append(base)
    for c in list(CONJUNTOS)[1:] + list(ABLACION):
        carpeta = COLAB / c
        if not carpeta.exists():
            if c in CONJUNTOS:
                raise SystemExit(f"Falta {carpeta}")
            continue
        for archivo, modelo in [("resultados_tiempos_colab.csv", None),
                                ("resultados_tabpfn35.csv", "TabPFN-3.5")]:
            ruta = carpeta / archivo
            if not ruta.exists():
                print(f"AVISO: falta {ruta}")
                continue
            d = pd.read_csv(ruta, dtype={"escenario": str})
            if modelo:
                d["modelo"] = modelo
            d["conjunto"] = c
            partes.append(d)
    res = pd.concat(partes, ignore_index=True)
    res["metodo"] = [metodo(m, v) for m, v in zip(res.modelo, res.variante)]
    res = res[res.escenario.isin(ESCENARIOS) & res.metodo.isin(NOMBRES_METODO)].copy()
    # Comprobaciones: 10 repeticiones por combinación, sin duplicados, mismo conjunto de prueba
    conteo = res.groupby(["conjunto", "escenario", "metodo"]).size()
    assert conteo.eq(10).all(), conteo[conteo != 10]
    for c, g in res.groupby("conjunto"):
        assert g.n_prueba.eq(50000).all() and (g.tp + g.fn).nunique() == 1, c
    return res


def holm(p):
    p = np.asarray(p, dtype=float)
    orden = np.argsort(p)
    k = len(p)
    ajust = np.minimum(1, np.maximum.accumulate((k - np.arange(k)) * p[orden]))
    salida = np.empty(k)
    salida[orden] = ajust
    return salida


def pruebas(res):
    """Friedman y Wilcoxon + Holm por (conjunto, escenario, métrica)."""
    friedman, pareadas = [], []
    for (c, esc), g in res.groupby(["conjunto", "escenario"], sort=False):
        for m, _ in METRICAS:
            tabla = g.pivot(index="rep", columns="metodo", values=m)
            tabla = tabla[[x for x in NOMBRES_METODO if x in tabla.columns]]
            est, p = friedmanchisquare(*[tabla[x] for x in tabla.columns])
            rangos = tabla.rank(axis=1, ascending=False).mean()
            friedman.append({"conjunto": c, "escenario": esc, "metrica": m, "chi2": est, "p": p,
                             "metodos": tabla.shape[1],
                             **{f"rango_{x}": rangos[x] for x in tabla.columns}})
            for ref in ["TabPFN-3.5", "TabPFN"]:
                if ref not in tabla.columns:
                    continue
                w = ef.wilcoxon_holm(tabla, ref)
                w.insert(0, "referencia", ref)
                w.insert(0, "metrica", m)
                w.insert(0, "escenario", esc)
                w.insert(0, "conjunto", c)
                pareadas.append(w)
    return pd.DataFrame(friedman), pd.concat(pareadas, ignore_index=True)


def resumen(res):
    filas = []
    for (c, esc, met), g in res.groupby(["conjunto", "escenario", "metodo"], sort=False):
        fila = {"conjunto": c, "escenario": esc, "metodo": met,
                "n_entrenamiento": int(g.n_entrenamiento.mean()),
                "fraudes_prueba": int((g.tp + g.fn).iloc[0])}
        for m in [x for x, _ in METRICAS] + ["t_ajuste_s", "t_prediccion_s", "t_busqueda_s"]:
            fila[m + "_media"] = g[m].mean()
            fila[m + "_ic95"] = ef.ic95(g[m].dropna()) if g[m].notna().any() else np.nan
        filas.append(fila)
    return pd.DataFrame(filas)


def diferencias(res):
    """Diferencia pareada (por repetición) de cada método frente a XGBoost ajustado."""
    filas = []
    for (c, esc), g in res.groupby(["conjunto", "escenario"], sort=False):
        for m, _ in METRICAS:
            tabla = g.pivot(index="rep", columns="metodo", values=m)
            for met in tabla.columns.drop(REF_XGB):
                d = tabla[met] - tabla[REF_XGB]
                filas.append({"conjunto": c, "escenario": esc, "metrica": m, "metodo": met,
                              "dif_media": d.mean(), "dif_ic95": ef.ic95(d),
                              "victorias": int((d > 0).sum()), "n": len(d)})
    return pd.DataFrame(filas)


def entre_conjuntos(res, texto):
    """TabPFN-3.5 (y v2) frente a XGBoost ajustado en los 6 conjuntos (sin la ablación)."""
    r = res[res.conjunto.isin(CONJUNTOS)]
    filas = []
    for esc in ESCENARIOS:
        for ref in ["TabPFN-3.5", "TabPFN"]:
            for m, nombre in METRICAS:
                g = r[(r.escenario == esc) & r.metodo.isin([ref, REF_XGB])]
                if ref not in set(g.metodo):
                    continue
                t = g.pivot_table(index=["conjunto", "rep"], columns="metodo", values=m)
                d = t[ref] - t[REF_XGB]
                medias = d.groupby(level=0).mean()
                p_medias = wilcoxon(medias).pvalue
                p_reps = wilcoxon(d).pvalue
                fila = {"escenario": esc, "referencia": ref, "metrica": m,
                        "victorias": int((d > 0).sum()), "n": len(d),
                        "conjuntos_con_media_mayor": int((medias > 0).sum()),
                        "conjuntos": len(medias), "dif_media": d.mean(),
                        "p_wilcoxon_medias_por_conjunto": p_medias,
                        "p_wilcoxon_todas_las_repeticiones": p_reps}
                filas.append(fila)
                texto.append(
                    f"[{esc}] {ref} vs XGBoost ajustado, {nombre}: gana {fila['victorias']}/{len(d)} "
                    f"repeticiones; media mayor en {fila['conjuntos_con_media_mayor']}/{len(medias)} "
                    f"conjuntos; Wilcoxon sobre medias por conjunto p = {p_medias:.4f}; "
                    f"sobre todas las repeticiones p = {p_reps:.3g} (descriptivo: no independientes)")
    return pd.DataFrame(filas)


def ablacion(res, texto):
    """Variante V completa frente a V sin x1 ni x2 (mismas filas y semillas): cambio por método."""
    if "v5sx" not in set(res.conjunto):
        texto.append("Ablación v5sx: sin resultados todavía.")
        return None
    filas = []
    for m, nombre in METRICAS:
        a = res[(res.conjunto == "v5") & (res.escenario == "10000")].pivot(
            index="rep", columns="metodo", values=m)
        b = res[(res.conjunto == "v5sx") & (res.escenario == "10000")].pivot(
            index="rep", columns="metodo", values=m)
        for met in [x for x in NOMBRES_METODO if x in b.columns]:
            d = b[met] - a[met]
            p = 1.0 if np.allclose(d, 0) else wilcoxon(b[met], a[met]).pvalue
            filas.append({"metrica": m, "metodo": met, "con_x": a[met].mean(),
                          "sin_x": b[met].mean(), "cambio_medio": d.mean(),
                          "cambio_ic95": ef.ic95(d), "p_wilcoxon": p})
            texto.append(f"Ablación {nombre} {met}: con x1,x2 {a[met].mean():.4f} -> sin "
                         f"{b[met].mean():.4f} (cambio {d.mean():+.4f}, p = {p:.4f})")
    return pd.DataFrame(filas)


def figura(dif, salida):
    """Diferencia pareada frente a XGBoost ajustado (media ± IC 95 %) por conjunto."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": ef.TINTA, "axes.labelcolor": ef.TINTA,
                         "xtick.color": ef.TINTA, "ytick.color": ef.TINTA})
    otros = [x for x in NOMBRES_METODO if x != REF_XGB]
    estilo = {"TabPFN-3.5": (ef.COLOR_MODELO["TabPFN-3.5"], "h", True),
              "TabPFN": (ef.COLOR_MODELO["TabPFN"], "o", True),
              "XGBoost": (ef.COLOR_MODELO["XGBoost"], "P", False),
              "Random Forest": (ef.COLOR_MODELO["Random Forest"], "D", False),
              "Logistic L1": (ef.COLOR_MODELO["Logistic L1"], "^", False)}
    conjuntos = [c for c in list(CONJUNTOS) + list(ABLACION) if c in set(dif.conjunto)]
    fig, ejes = plt.subplots(len(METRICAS), 2, figsize=(12, 10.5), sharey="row", squeeze=False)
    desp = np.linspace(-0.28, 0.28, len(otros))
    for j, esc in enumerate(ESCENARIOS):
        for i, (m, titulo) in enumerate(METRICAS):
            ax = ejes[i, j]
            ax.axhline(0, color=ef.TINTA, lw=1.2, zorder=1)
            cs = [c for c in conjuntos if not dif[(dif.conjunto == c) & (dif.escenario == esc)].empty]
            for k, met in enumerate(otros):
                color, marca, lleno = estilo[met.replace(" (ajustado)", "")]
                g = dif[(dif.escenario == esc) & (dif.metrica == m) & (dif.metodo == met)]
                if g.empty:
                    continue
                g = g.set_index("conjunto").reindex(cs).dropna(subset=["dif_media"])
                x = np.array([cs.index(c) for c in g.index]) + desp[k]
                fuera = g.dif_media.to_numpy() < PISO.get(m, -np.inf)
                ax.errorbar(x[~fuera], g.dif_media[~fuera], yerr=g.dif_ic95[~fuera], fmt=marca,
                            color=color, mfc=color if lleno else "white", mec=color, ms=7,
                            mew=1.6, elinewidth=1.6, capsize=0, zorder=3 if lleno else 2)
                # Valores fuera de escala: flecha en el borde inferior con el valor
                for xi, yi in zip(x[fuera], g.dif_media[fuera]):
                    ax.annotate(f"{yi:.3f}", xy=(xi, PISO[m]), xytext=(xi, PISO[m] + 0.012),
                                ha="center", fontsize=7.5, color=ef.TINTA,
                                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4))
            if m in PISO:
                ax.set_ylim(bottom=PISO[m])
            if "v5sx" in cs:
                ax.axvline(cs.index("v5sx") - 0.5, color=ef.TINTA, lw=0.8, ls=":", alpha=0.6)
            ax.set_xticks(range(len(cs)))
            ax.set_xticklabels([ETIQUETA_CORTA[c] for c in cs])
            ax.set_xlim(-0.6, len(cs) - 0.4)
            ax.grid(axis="y", alpha=0.3)
            if j == 0:
                ax.set_ylabel(f"{titulo}\ndifference vs tuned XGBoost")
            if i == 0:
                ax.set_title("10,000 training rows" if esc == "10000"
                             else "Full undersampled training set", fontsize=10)
            if i == len(METRICAS) - 1:
                ax.set_xlabel("Dataset (BAF Base and variants)")
    leyenda = [Line2D([], [], color=ef.TINTA, lw=1.2, label="Tuned XGBoost (reference, 0)")]
    for met in otros:
        color, marca, lleno = estilo[met.replace(" (ajustado)", "")]
        nombre = ef.NOMBRES_EN[met.replace(" (ajustado)", "")] + (
            " (default)" if met in ("XGBoost", "Random Forest", "Logistic L1") else "")
        leyenda.append(Line2D([], [], ls="none", marker=marca, color=color,
                              mfc=color if lleno else "white", mec=color, ms=7, mew=1.6,
                              label=nombre))
    fig.legend(handles=leyenda, loc="upper center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.005, "Points: mean paired difference over 10 repetitions (same partitions); "
             "bars: 95% CI. Above zero = better than tuned XGBoost. Arrows: values below the axis "
             "range.\nTabPFN v2 is limited to 10,000 training rows.", fontsize=8, color=ef.TINTA)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(salida / "fig_robustez.png", dpi=300)
    plt.close(fig)


def tabla_latex(resu, pareadas, salida):
    """Tabla de PR-AUC por conjunto con la comparación pareada frente a XGBoost ajustado."""
    def celda(c, esc, met):
        r = resu[(resu.conjunto == c) & (resu.escenario == esc) & (resu.metodo == met)]
        if r.empty:
            return "---"
        r = r.iloc[0]
        return f"${r.pr_auc_media:.3f} \\pm {r.pr_auc_ic95:.3f}$"

    def mejor_otro(c, esc):
        r = resu[(resu.conjunto == c) & (resu.escenario == esc) &
                 resu.metodo.isin(["XGBoost", "Random Forest", "Logistic L1"])]
        r = r.sort_values("pr_auc_media", ascending=False).iloc[0]
        nombre = {"XGBoost": "XGB", "Random Forest": "RF", "Logistic L1": "LR-$L_1$"}[r.metodo]
        return f"{nombre} ${r.pr_auc_media:.3f}$"

    def frente(c, esc, ref, m="pr_auc"):
        w = pareadas[(pareadas.conjunto == c) & (pareadas.escenario == esc) &
                     (pareadas.referencia == ref) & (pareadas.metrica == m) &
                     (pareadas.metodo == REF_XGB)]
        if w.empty:
            return "---"
        w = w.iloc[0]
        return f"{int(w.victorias)}/{int(w.n)} ({w.p_holm:.3f})"

    filas = []
    for c in [c for c in list(CONJUNTOS) + list(ABLACION) if c in set(resu.conjunto)]:
        nombre = CONJUNTOS.get(c) or ABLACION[c].replace("x1, x2", "$x_1$, $x_2$")
        fr = int(resu[resu.conjunto == c].fraudes_prueba.iloc[0])
        celdas = [nombre, f"{fr}",
                  celda(c, "10000", "TabPFN-3.5"), celda(c, "10000", "TabPFN"),
                  celda(c, "10000", REF_XGB), mejor_otro(c, "10000"),
                  frente(c, "10000", "TabPFN-3.5"), frente(c, "10000", "TabPFN"),
                  celda(c, "completo", "TabPFN-3.5"), celda(c, "completo", REF_XGB),
                  frente(c, "completo", "TabPFN-3.5")]
        if c in ABLACION:
            filas.append("\\midrule")
        filas.append(" & ".join(celdas) + " \\\\")
    tex = ("\\begin{table}[htbp]\n\\centering\n"
           "\\caption{Robustness across the BAF suite: test PR-AUC (mean $\\pm$ 95\\% CI over 10 "
           "repetitions; 50{,}000 test applications from months 6--7) with 10{,}000 training rows "
           "and with the full undersampled training set (67{,}400 rows in Base and Variants I--II; "
           "70{,}360 in Variants III--V). Frauds: frauds among the test applications. Best other: "
           "best default classical model (XGBoost, random forest, or $L_1$ logistic regression). "
           "Wins (Holm $p$): repetitions in which the TabPFN version obtained a higher PR-AUC than "
           "tuned XGBoost, and the Holm-adjusted $p$-value of the two-sided Wilcoxon signed-rank "
           "test; the correction covers the comparisons among the six methods of this analysis, so "
           "the adjusted $p$-values for Base differ from those of Table~\\ref{tab:learning}. Last "
           "row: ablation of Variant V without $x_1$ and $x_2$, run with 10{,}000 training rows "
           "only (Section~\\ref{sec:ablation}).}\n"
           "\\label{tab:robustness}\n\\footnotesize\n\\resizebox{\\textwidth}{!}{%\n"
           "\\begin{tabular}{l r c c c l c c c c c}\n\\toprule\n"
           " & & \\multicolumn{6}{c}{10{,}000 training rows} & \\multicolumn{3}{c}{Full training "
           "set} \\\\\n\\cmidrule(lr){3-8}\\cmidrule(lr){9-11}\n"
           " & & & & & & \\multicolumn{2}{c}{vs.\\ tuned XGBoost} & & & vs.\\ tuned XGB \\\\\n"
           "\\cmidrule(lr){7-8}\\cmidrule(lr){11-11}\n"
           "Dataset & Frauds & TabPFN-3.5 & TabPFN~v2 & Tuned XGB & Best other & TabPFN-3.5 & "
           "TabPFN~v2 & TabPFN-3.5 & Tuned XGB & TabPFN-3.5 \\\\\n\\midrule\n"
           + "\n".join(filas) + "\n\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n")
    (salida / "tabla_robustez.tex").write_text(
        "% === Generado por analizar_robustez.py: no editar a mano ===\n" + tex, encoding="utf-8")


def tabla_latex_paneles(resu, pareadas, salida):
    """La misma información que tabla_latex en dos paneles (10,000 filas y conjunto completo), con
    menos columnas, para plantillas de una columna estrecha (Springer sn-jnl)."""
    def media(c, esc, met):
        r = resu[(resu.conjunto == c) & (resu.escenario == esc) & (resu.metodo == met)]
        if r.empty:
            return "---"
        r = r.iloc[0]
        return f"${r.pr_auc_media:.3f} \\pm {r.pr_auc_ic95:.3f}$"

    def mejor_otro(c, esc):
        r = resu[(resu.conjunto == c) & (resu.escenario == esc) &
                 resu.metodo.isin(["XGBoost", "Random Forest", "Logistic L1"])]
        r = r.sort_values("pr_auc_media", ascending=False).iloc[0]
        nombre = {"XGBoost": "XGB", "Random Forest": "RF", "Logistic L1": "LR-$L_1$"}[r.metodo]
        return f"{nombre} ${r.pr_auc_media:.3f}$"

    def frente(c, esc, ref):
        w = pareadas[(pareadas.conjunto == c) & (pareadas.escenario == esc) &
                     (pareadas.referencia == ref) & (pareadas.metrica == "pr_auc") &
                     (pareadas.metodo == REF_XGB)]
        if w.empty:
            return "---"
        w = w.iloc[0]
        return f"{int(w.victorias)}/{int(w.n)} ({w.p_holm:.3f})"

    conjuntos = [c for c in list(CONJUNTOS) + list(ABLACION) if c in set(resu.conjunto)]
    filas = []
    for esc, titulo in [("10000", "(a) 10{,}000 training rows"),
                        ("completo", "(b) Full undersampled training set")]:
        filas.append(f"\\midrule\n\\multicolumn{{8}}{{l}}{{\\emph{{{titulo}}}}} \\\\")
        for c in conjuntos:
            if resu[(resu.conjunto == c) & (resu.escenario == esc)].empty:
                continue
            nombre = CONJUNTOS.get(c) or ABLACION[c].replace("x1, x2", "$x_1$, $x_2$")
            fr = int(resu[resu.conjunto == c].fraudes_prueba.iloc[0])
            filas.append(" & ".join([nombre, f"{fr}", media(c, esc, "TabPFN-3.5"),
                                     media(c, esc, "TabPFN"), media(c, esc, REF_XGB),
                                     mejor_otro(c, esc), frente(c, esc, "TabPFN-3.5"),
                                     frente(c, esc, "TabPFN")]) + " \\\\")
    tex = ("\\begin{table}[htbp]\n\\centering\n"
           "\\caption{Robustness across the BAF suite: test PR-AUC (mean $\\pm$ 95\\% CI over 10 "
           "repetitions; 50{,}000 test applications from months 6--7) with 10{,}000 training rows (a) "
           "and with the full undersampled training set (b) (67{,}400 rows in Base and Variants I--II; "
           "70{,}360 in Variants III--V). Frauds: frauds among the test applications. Best other: "
           "best default classical model (XGBoost, random forest, or $L_1$ logistic regression). Wins "
           "(Holm $p$): repetitions in which the TabPFN version obtained a higher PR-AUC than tuned "
           "XGBoost, and the Holm-adjusted $p$-value of the two-sided Wilcoxon signed-rank test; the "
           "correction covers the comparisons among the six methods of this analysis, so the adjusted "
           "$p$-values for Base differ from those of Table~\\ref{tab:learning}. TabPFN~v2 does not "
           "support the full training set. Variant V without $x_1$ and $x_2$: ablation run with "
           "10{,}000 training rows only (Section~\\ref{sec:ablation}).}\n"
           "\\label{tab:robustness}\n\\footnotesize\n"
           "\\begin{tabular}{l r c c c l c c}\n\\toprule\n"
           " & & & & & & \\multicolumn{2}{c}{vs.\\ tuned XGBoost: wins (Holm $p$)} \\\\\n"
           "\\cmidrule(lr){7-8}\n"
           "Dataset & Frauds & TabPFN-3.5 & TabPFN~v2 & Tuned XGB & Best other & TabPFN-3.5 & "
           "TabPFN~v2 \\\\\n"
           + "\n".join(filas) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (salida / "tabla_robustez_paneles.tex").write_text(
        "% === Generado por analizar_robustez.py: no editar a mano ===\n" + tex, encoding="utf-8")


NOMBRE_TEX = {"TabPFN-3.5": "TabPFN-3.5", "TabPFN": "TabPFN~v2",
              "XGBoost (ajustado)": "XGBoost (tuned)", "XGBoost": "XGBoost",
              "Random Forest": "Random Forest", "Logistic L1": "Logistic $L_1$"}


def tabla_ablacion_latex(abl, salida):
    """Variante V con y sin x1 y x2 (10,000 filas): media por método y cambio."""
    filas = []
    for met in NOMBRES_METODO:
        celdas = [NOMBRE_TEX[met]]
        for m, _ in METRICAS:
            r = abl[(abl.metrica == m) & (abl.metodo == met)].iloc[0]
            celdas += [f"{r.con_x:.3f}", f"{r.sin_x:.3f}",
                       f"${r.cambio_medio:+.3f} \\pm {r.cambio_ic95:.3f}$"]
        filas.append(" & ".join(celdas) + " \\\\")
    tex = ("\\begin{table}[htbp]\n\\centering\n"
           "\\caption{Ablation on Variant V with 10{,}000 training rows: test performance with and "
           "without $x_1$ and $x_2$ (means over the same 10 repetitions, with the same training "
           "rows and seeds) and paired change (mean $\\pm$ 95\\% CI). Every change is positive in "
           "all ten repetitions (two-sided Wilcoxon signed-rank test, $p = 0.002$).}\n"
           "\\label{tab:ablation}\n\\footnotesize\n\\resizebox{\\textwidth}{!}{%\n"
           "\\begin{tabular}{l c c c c c c c c c}\n\\toprule\n"
           " & \\multicolumn{3}{c}{PR-AUC} & \\multicolumn{3}{c}{ROC-AUC} & "
           "\\multicolumn{3}{c}{Recall@5\\%FPR} \\\\\n"
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n"
           "Method & With & Without & Change & With & Without & Change & With & Without & Change "
           "\\\\\n\\midrule\n" + "\n".join(filas) +
           "\n\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n")
    (salida / "tabla_ablacion.tex").write_text(
        "% === Generado por analizar_robustez.py: no editar a mano ===\n" + tex, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resultados", default=str(ef.BASE_DIR / "resultados_finales"))
    args = ap.parse_args()
    base = Path(args.resultados)
    salida = base / "robustez"
    salida.mkdir(exist_ok=True)

    res = cargar(base)
    res.to_csv(salida / "robustez_crudos.csv", index=False)
    resu = resumen(res)
    resu.to_csv(salida / "robustez_resumen.csv", index=False)
    fried, pareadas = pruebas(res)
    fried.to_csv(salida / "robustez_friedman.csv", index=False)
    pareadas.to_csv(salida / "robustez_wilcoxon_holm.csv", index=False)
    dif = diferencias(res)
    dif.to_csv(salida / "robustez_diferencias_vs_xgb.csv", index=False)

    texto = ["# Análisis de robustez (Base y variantes de BAF)", ""]
    for r in fried.itertuples():
        texto.append(f"[{r.conjunto} {r.escenario}] {r.metrica}: Friedman chi2 = {r.chi2:.2f}, "
                     f"p = {r.p:.3g} ({r.metodos} métodos)")
    texto.append("")
    w = pareadas[pareadas.metodo == REF_XGB]
    for r in w.itertuples():
        texto.append(f"[{r.conjunto} {r.escenario}] {r.metrica}: {r.referencia} vs XGBoost "
                     f"ajustado: gana {r.victorias}/{r.n}, dif. media {r.diferencia_media:+.4f}, "
                     f"p Holm = {r.p_holm:.4f}")
    texto.append("")
    entre = entre_conjuntos(res, texto)
    entre.to_csv(salida / "robustez_entre_conjuntos.csv", index=False)
    texto.append("")
    abl = ablacion(res, texto)
    if abl is not None:
        abl.to_csv(salida / "ablacion_v5_sin_x.csv", index=False)
        tabla_ablacion_latex(abl, salida)
    (salida / "pruebas_robustez.txt").write_text("\n".join(texto) + "\n", encoding="utf-8")

    figura(dif, salida)
    tabla_latex(resu, pareadas, salida)
    tabla_latex_paneles(resu, pareadas, salida)
    print("\n".join(texto))
    print("\nEscrito en", salida)


if __name__ == "__main__":
    main()
