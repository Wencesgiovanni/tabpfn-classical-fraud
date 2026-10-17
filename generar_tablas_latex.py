"""Genera las tablas del artículo (LaTeX, clase cas-sc de Elsevier) a partir de
resultados_finales/: Tablas 3-9 (desempeño, curva de aprendizaje, matrices de confusión y
costo).

Uso (con el entorno de requirements-clasicos-v2.txt):
  python generar_tablas_latex.py
Escribe resultados_finales/tablas.tex.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import experimento_fraude as ef

NOMBRE_TEX = {"TabPFN-3.5": "TabPFN-3.5", "TabPFN": "TabPFN~v2", "SVM lineal": "Linear SVM",
              "Logistic L1": "Logistic $L_1$", "Ridge": "Ridge", "Random Forest": "Random Forest",
              "XGBoost": "XGBoost", "MLP sklearn": "MLP", "Red Keras": "Keras NN"}


def nombre(modelo, variante):
    return NOMBRE_TEX[modelo] + (" (tuned)" if variante == "ajustado" else "")


def miles(x):
    return f"{x:,.0f}".replace(",", "{,}")


def rangos(res, esc, m="pr_auc"):
    g = res[res.escenario == esc].copy()
    g["metodo"] = list(zip(g.modelo, g.variante))
    t = g.pivot(index="rep", columns="metodo", values=m)
    return t.rank(axis=1, ascending=False).mean()


def fmt_ic(r, m, d=4):
    return f"${getattr(r, m + '_media'):.{d}f} \\pm {getattr(r, m + '_ic95'):.{d}f}$"


def tabla_principal(res, resu):
    """Desempeño con 10,000 filas y con el conjunto completo."""
    filas = []
    for esc, titulo in [("10000", "(a) 10{,}000 training rows"),
                        ("completo", "(b) Full undersampled training set, 67{,}400 rows")]:
        g = resu[resu.escenario == esc].sort_values("pr_auc_media", ascending=False)
        rk = rangos(res, esc)
        mejores = {m: g[m + "_media"].max() for m in
                   ["pr_auc", "roc_auc", "recall_fpr5_test", "precision_umbral_val", "f1_umbral_val"]}
        filas.append(f"\\midrule\n\\multicolumn{{7}}{{l}}{{\\emph{{{titulo}}}}} \\\\")
        for r in g.itertuples():
            celdas = [nombre(r.modelo, r.variante)]
            for m in ["pr_auc", "roc_auc", "recall_fpr5_test", "precision_umbral_val",
                      "f1_umbral_val"]:
                t = fmt_ic(r, m)
                if np.isclose(getattr(r, m + "_media"), mejores[m]):
                    t = t.replace("$", "$\\mathbf{", 1)[:-1] + "}$"
                celdas.append(t)
            rango = f"{rk[(r.modelo, r.variante)]:.1f}"
            if rango == f"{rk.min():.1f}":
                rango = f"\\textbf{{{rango}}}"
            celdas.append(rango)
            filas.append(" & ".join(celdas) + " \\\\")
    return ("\\begin{table}[htbp]\n"
            "\\caption{Test performance with 10{,}000 training rows (a) and with the full "
            "undersampled training set (b) (mean $\\pm$ 95\\% CI over 10 repetitions; 50{,}000 test "
            "applications from months 6--7, 702 frauds). Precision and F1 use the threshold fixed on "
            "the validation month at 5\\% FPR. Rank: average Friedman rank in PR-AUC within each "
            "panel (1 = best). Sorted by PR-AUC; best value per column in bold. TabPFN~v2 does not "
            "support the full training set.}\n"
            "\\label{tab:main10k}\n\\resizebox{\\textwidth}{!}{%\n"
            "\\begin{tabular*}{\\tblwidth}{@{}LLLLLLL@{}}\n\\toprule\n"
            "Method & PR-AUC & ROC-AUC & Recall@5\\%FPR & Precision & F1 & Rank \\\\\n"
            + "\n".join(filas) + "\n\\bottomrule\n\\end{tabular*}%\n}\n\\end{table}\n")


def tabla_curva(res, resu, dir_res):
    filas = []
    for esc in ["1000", "2500", "5000", "10000", "completo"]:
        g = resu[resu.escenario == esc]
        n = "67{,}400" if esc == "completo" else miles(int(esc))

        def val(mod, var="defecto"):
            r = g[(g.modelo == mod) & (g.variante == var)]
            return fmt_ic(r.iloc[0], "pr_auc") if len(r) else "--- (exceeds limit)"

        def mejor(var):
            c = g[(g.variante == var) & g.modelo.isin(ef.CLASICOS)].sort_values(
                "pr_auc_media", ascending=False).iloc[0]
            return f"{NOMBRE_TEX[c.modelo]}: {fmt_ic(c, 'pr_auc')}"

        def frente_xgb(sufijo):
            """Victorias y p de Holm (Wilcoxon pareado, PR-AUC) frente a XGBoost ajustado."""
            ruta = dir_res / f"wilcoxon_holm{sufijo}_{esc}_pr_auc.csv"
            if not ruta.exists():
                return "---"
            w = pd.read_csv(ruta).set_index("metodo").loc["XGBoost (ajustado)"]
            return f"{int(w.victorias)}/{int(w.n)} ({w.p_holm:.3f})"

        filas.append(" & ".join([n, val("TabPFN-3.5"), val("TabPFN"), mejor("ajustado"),
                                 mejor("defecto"), frente_xgb(""), frente_xgb("_v2")]) + " \\\\")
    return ("\\begin{table}[htbp]\n\\centering\n"
            "\\caption{Learning curve in PR-AUC (mean $\\pm$ 95\\% CI over 10 repetitions). For "
            "each training size: both TabPFN versions, the best classical model in each variant, and "
            "the paired comparison of each TabPFN version with tuned XGBoost: number of repetitions "
            "in which TabPFN obtained the higher PR-AUC and, in parentheses, the Holm-adjusted "
            "$p$-value of the two-sided Wilcoxon signed-rank test (Section~\\ref{sec:stats}).}\n"
            "\\label{tab:learning}\n\\footnotesize\n\\resizebox{\\textwidth}{!}{%\n"
            "\\begin{tabular}{r c c l l c c}\n\\toprule\n"
            " & & & & & \\multicolumn{2}{c}{vs.\\ tuned XGBoost: wins (Holm $p$)} \\\\\n"
            "\\cmidrule(lr){6-7}\n"
            "Rows & TabPFN-3.5 & TabPFN~v2 & Best tuned classical & Best default classical & "
            "TabPFN-3.5 & TabPFN~v2 \\\\\n\\midrule\n"
            + "\n".join(filas) + "\n\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n")


def filas_confusion(resu, esc):
    g = resu[resu.escenario == esc].sort_values("pr_auc_media", ascending=False)
    filas = []
    for r in g.itertuples():
        vals = [miles(r.tp_media), miles(r.fn_media), miles(r.fp_media), miles(r.tn_media),
                f"{100 * r.fpr_umbral_val_media:.1f}" if hasattr(r, "fpr_umbral_val_media") else "",
                f"{r.recall_umbral_val_media:.3f}", f"{r.precision_umbral_val_media:.3f}"]
        celdas = [nombre(r.modelo, r.variante)] + vals
        filas.append(" & ".join(celdas) + " \\\\")
    return "\n".join(filas)


ENCAB_CONF = ("\\begin{tabular*}{\\tblwidth}{@{}LLLLLLLL@{}}\n\\toprule\n"
              "Method & TP & FN & FP & TN & FPR (\\%) & Recall & Precision \\\\\n\\midrule\n")


def tabla_confusion(resu):
    return ("\\begin{table}[htbp]\n"
            "\\caption{Confusion matrices on the test set (50{,}000 applications from months 6--7: "
            "702 frauds and 49{,}298 legitimate applications) at the threshold fixed on the "
            "validation month at 5\\% FPR. Counts are means over 10 repetitions, rounded. TP: frauds "
            "detected; FN: frauds missed; FP: legitimate applications flagged (false alarms); TN: "
            "legitimate applications not flagged. FPR, recall, and precision are averaged per "
            "repetition, so they may differ slightly from ratios of the rounded mean counts. Methods "
            "are sorted by PR-AUC within each panel.}\n\\label{tab:confusion}\n" + ENCAB_CONF +
            "\\multicolumn{8}{l}{\\emph{(a) 10{,}000 training rows}} \\\\\n" +
            filas_confusion(resu, "10000") + "\n\\midrule\n"
            "\\multicolumn{8}{l}{\\emph{(b) Full undersampled training set, 67{,}400 rows "
            "(TabPFN~v2 not supported)}} \\\\\n" +
            filas_confusion(resu, "completo") +
            "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n")


def tabla_confusion_apendice(resu, esc):
    n = miles(int(esc))
    return ("\\begin{table}[htbp]\n"
            f"\\caption{{Confusion matrices on the test set with {n} training rows (means over 10 "
            "repetitions). TP: frauds detected; FN: frauds missed; FP: legitimate applications "
            "flagged; TN: legitimate applications not flagged.}\n"
            f"\\label{{tab:conf{esc}}}\n" + ENCAB_CONF + filas_confusion(resu, esc) +
            "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n")


def tabla_costo(resu):
    def t(x):
        if np.isnan(x):
            return "---"
        if x < 0.005:
            return "$<$0.01"
        return miles(x) + ".0" if x >= 1000 else (f"{x:.1f}" if x >= 10 else f"{x:.2f}")

    filas = []
    for mod in ef.MODELOS_ANALISIS:
        celdas = [NOMBRE_TEX[mod]]
        for esc in ["10000", "completo"]:
            d = resu[(resu.escenario == esc) & (resu.modelo == mod) & (resu.variante == "defecto")]
            a = resu[(resu.escenario == esc) & (resu.modelo == mod) & (resu.variante == "ajustado")]
            if d.empty:
                celdas.append("\\multicolumn{3}{c}{not supported}")
                continue
            celdas += [t(d.t_ajuste_s_media.iloc[0]),
                       t(a.t_busqueda_s_media.iloc[0]) if len(a) else "---",
                       t(d.t_prediccion_s_media.iloc[0])]
        filas.append(" & ".join(celdas) + " \\\\")
    return ("\\begin{table}[htbp]\n\\centering\n"
            "\\caption{Wall-clock cost in seconds (mean over 10 repetitions) on the same "
            "machine (NVIDIA A100 GPU; Section~\\ref{sec:hardware}): fit time of the default variant, "
            "total time of the 20-configuration random search (tuned variant), and inference time for "
            "50{,}000 test applications. TabPFN models on the GPU; all other models on the CPU.}\n"
            "\\label{tab:cost}\n\\footnotesize\n"
            "\\begin{tabular*}{\\tblwidth}{@{}LLLLLLL@{}}\n\\toprule\n"
            " & \\multicolumn{3}{c}{10{,}000 training rows} & \\multicolumn{3}{c}{67{,}400 training "
            "rows} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
            "Model & Fit & Search & Inference & Fit & Search & Inference \\\\\n\\midrule\n"
            + "\n".join(filas) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resultados", default=str(ef.BASE_DIR / "resultados_finales"))
    args = ap.parse_args()
    d = Path(args.resultados)
    res = pd.read_csv(d / "resultados_crudos.csv", dtype={"escenario": str})
    resu = pd.read_csv(d / "resumen.csv", dtype={"escenario": str})
    # FPR realizado (no está en resumen.csv): media por repetición
    fpr = res.groupby(["escenario", "modelo", "variante"]).fpr_umbral_val.mean().rename(
        "fpr_umbral_val_media").reset_index()
    resu = resu.merge(fpr, on=["escenario", "modelo", "variante"], how="left")

    partes = ["% === Generado por generar_tablas_latex.py: no editar a mano ===",
              "% --- Tabla principal", tabla_principal(res, resu),
              "% --- Curva de aprendizaje", tabla_curva(res, resu, d),
              "% --- Matrices de confusión", tabla_confusion(resu),
              "% --- Costo", tabla_costo(resu)]
    for esc in ["1000", "2500", "5000"]:
        partes += [f"% --- Apéndice {esc}", tabla_confusion_apendice(resu, esc)]
    (d / "tablas.tex").write_text("\n".join(partes), encoding="utf-8")
    print("escrito", d / "tablas.tex")


if __name__ == "__main__":
    main()
