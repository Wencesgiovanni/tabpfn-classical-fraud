# Tabular foundation models versus tuned classical classifiers for bank account fraud detection

Code and results for the article:

> Hernández-López, E., Cruz-Espinosa, D. J., Herrera-Zúñiga, L. D., and Wences, G.
> *Tabular Foundation Models versus Tuned Classical Classifiers for Bank Account Fraud
> Detection: A Learning-Curve Study of TabPFN v2 and TabPFN-3.5.* (under review)

The study compares two generations of TabPFN (v2 and 3.5) with seven classical classifiers
(linear SVM, L1 logistic regression, ridge, random forest, XGBoost, scikit-learn MLP and a
Keras network), each with default and tuned hyperparameters, on the Base variant of the Bank
Account Fraud (BAF) dataset. Models are trained on 1,000 to 67,400 rows under a temporal split
(months 0–4 train, 5 validation, 6–7 test), with ten repetitions per training size. A
robustness analysis repeats the comparison on the five biased variants of BAF (Variants I–V),
with an ablation of Variant V without the attributes `x1` and `x2`.

The source code is commented in Spanish; this README and all outputs used in the article are
in English.

## Repository structure

| Path | Content |
|---|---|
| `experimento_fraude.py` | Core of the protocol: data loading, temporal split, undersampling, preprocessing, models, random search, metrics, statistical tests (Friedman, paired Wilcoxon with Holm correction) and figures. |
| `exportar_para_colab.py` | Builds the exact partitions (validation, test and the 50 training sets) and the package for Google Colab; checks them against `particiones/`. |
| `tiempos_colab.py` | Runs the classical models and TabPFN v2 (and the permutation importance) on Colab, measuring fit, search and inference times. |
| `tabpfn35_colab.py` | Runs TabPFN-3.5 on Colab with the same partitions. |
| `preparar_resultados_finales.py` | Merges the Colab outputs and writes the summary, statistical tests and Figures 2–6 to `resultados_finales/`. |
| `figura_correlacion.py` | Figure 1 (correlation matrix of the BAF variables). |
| `generar_tablas_latex.py` | Tables 4–7 and A1–A3 of the article in LaTeX (`resultados_finales/tablas.tex`). |
| `analizar_robustez.py` | Robustness analysis on Base and Variants I–V and the ablation: statistical tests, Figure 7 and Tables 8–9 (`resultados_finales/robustez/`). |
| `diagnostico_x1_x2.py` | Univariate ROC-AUC per period and Kolmogorov–Smirnov statistic of `x1` and `x2` in Variants III and V. |
| `colab/ejecutar_en_colab.ipynb` | Notebook that runs the experiments on a Colab A100. |
| `colab/variantes_en_colab.ipynb` | Notebook that runs the variants and the ablation on a Colab A100. |
| `particiones/` | Reference partitions (`particiones.npz`) and SHA-256 fingerprints of every preprocessed test matrix (`huellas.json`) used in the article. |
| `resultados_colab/` | Raw outputs of the Colab runs used in the article (one row per model, training size and repetition) and the software environments; `resultados_colab/variantes/v1`–`v5` and `v5sx` hold those of the variants and of the ablation, with their partitions and fingerprints. |
| `resultados_finales/` | Everything reported in the article: summaries, statistical tests, figures and tables. |
| `requirements-clasicos-v2.txt`, `requirements-tabpfn35.txt` | The two software environments (see below). |

Internal model names are in Spanish: `SVM lineal` (linear SVM), `Logistic L1`, `Ridge`,
`Random Forest`, `XGBoost`, `MLP sklearn`, `Red Keras` (Keras NN), `TabPFN` (TabPFN v2) and
`TabPFN-3.5`; variants are `defecto` (default) and `ajustado` (tuned); the training size
`completo` is the full undersampled pool of 67,400 rows.

## Data

The BAF dataset is not redistributed here. Download `Base.csv` from the dataset authors
([Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022);
see also [github.com/feedzai/bank-account-fraud](https://github.com/feedzai/bank-account-fraud))
and place it at `data/Base.csv`; for the robustness analysis, also place `Variant I.csv` …
`Variant V.csv` in `data/`. Its use is subject to the license stated by its authors.

The partitions are defined by row position and random seeds. The copy of the dataset used in
the study contained the same rows as `Base.csv` but in a different order, so
`particiones/orden_filas.npz` (row indices only, no data) restores that order when the CSV is
read; with it, the seeds reproduce exactly the partitions of the article.
`exportar_para_colab.py` checks this and should print

```
Particiones idénticas a las del estudio: 52/52
Matrices de prueba preprocesadas idénticas a las de referencia (Base.csv de Kaggle): 50/50
```

That copy also stored the continuous variables rounded to at most nine decimal places or ten
significant digits (absolute differences of at most 5e-8, or 5e-10 relative for the three
velocity variables). The fingerprints in `particiones/huellas.json` were computed
from `Base.csv`; those of the copy used in the study are kept in
`particiones/huellas_copia_estudio.json` (they match the `huella_prueba` column of
`resultados_colab/`).

## Software environments

TabPFN v2 and TabPFN-3.5 require incompatible versions of the `tabpfn` package, so the
experiments used two software environments on the same type of machine (Section 4.9 of the
article):

* **Classical models and TabPFN v2** — Python 3.9, `requirements-clasicos-v2.txt`
  (PyTorch 2.7.1 with CUDA 11.8).
* **TabPFN-3.5** — Python 3.13 (the default Python of Colab at the time), `requirements-tabpfn35.txt`.
  Downloading the TabPFN-3.5 weights requires a free Prior Labs account and accepting the
  model license; the notebook reads the token from a Colab secret named `TABPFN_TOKEN`.

All models were fitted, scored and timed on Google Colab virtual machines of the same type,
with an NVIDIA A100-SXM4 GPU (80 GB) and a 12-core Intel Xeon CPU at 2.20 GHz: the TabPFN models
on the GPU and the classical models on the CPU. The exact package versions of each run are in
`resultados_colab/entorno_*.json`, and every per-repetition result, including all times, is in
`resultados_colab/`. The ablation ran on the same type of machine with the 40 GB version of the
A100.

The remaining scripts (`preparar_resultados_finales.py`, `figura_correlacion.py`,
`generar_tablas_latex.py`, `analizar_robustez.py` and `diagnostico_x1_x2.py`) fit no model and measure no time: they only aggregate these results
into summaries, statistical tests, figures and tables, and they give identical outputs on any
computer with the first environment (no GPU needed).

## Reproducing the article

### Tables and figures from the published runs (minutes, no GPU)

From the repository root, with the first environment:

```bash
python preparar_resultados_finales.py   # summary, statistical tests, Figures 2-6
python figura_correlacion.py            # Figure 1 (needs data/Base.csv)
python generar_tablas_latex.py          # Tables 4-7, A1-A3 -> resultados_finales/tablas.tex
python analizar_robustez.py             # robustness: Figure 7, Tables 8-9 -> resultados_finales/robustez/
python diagnostico_x1_x2.py             # x1, x2 diagnostic (needs data/Variant III.csv, data/Variant V.csv)
```

The first command reads `resultados_colab/` and regenerates every file in
`resultados_finales/`.

### Full rerun of the experiments (A100 GPU, several hours)

1. `python exportar_para_colab.py --datos data/Base.csv` creates `paquete_colab/` (data,
   partitions, fingerprints, scripts and the published results for comparison).
2. Copy `paquete_colab/` to `My Drive/paquete_colab` in Google Drive, open
   `colab/ejecutar_en_colab.ipynb` in Colab with an A100 runtime, and run its cells in order.
   Every script verifies the fingerprint of the preprocessed test matrix before running, saves
   one row per fitted model, and resumes where it stopped if the session is interrupted. Each
   line of the log shows the published PR-AUC next to the new one.
3. Copy the outputs listed at the end of the notebook to `resultados_colab/` and run the three
   commands of the previous section.

For the classical models, `Base.csv` and the copy used in the study give the same metrics (in a
check on two training sizes, PR-AUC agreed to six decimals and the confusion matrices were
identical). TabPFN is sensitive to tiny differences in its input and to the GPU on which it
runs, so its PR-AUC can differ slightly from the published values; in our checks the
differences were at most 0.001, below its standard deviation across repetitions
(0.002–0.010). Wall-clock times depend on the hardware and on the load of the virtual machine.

### Rerun of the robustness analysis (A100 GPU)

1. For each variant, `python exportar_para_colab.py --conjunto v1` (… `v5`) reads
   `data/Variant I.csv` (… `Variant V.csv`) and creates `paquete_colab/variantes/v1` (… `v5`);
   `--conjunto v5sx` creates the ablation (Variant V without `x1` and `x2`, 10,000 rows only).
   Each variant is used in the row order of its file.
2. Copy `paquete_colab/variantes/` to `My Drive/paquete_colab/variantes`, open
   `colab/variantes_en_colab.ipynb` with an A100 runtime and run its cells in order (Part A:
   TabPFN v2 and classical models; Part B: TabPFN-3.5; section 8: the ablation).
3. Copy the outputs to `resultados_colab/variantes/` (without `datos_baf.npz` and `puntajes/`)
   and run `python analizar_robustez.py`.

## Correspondence between the article and the outputs

| Article | File in `resultados_finales/` |
|---|---|
| Figure 1 | `fig_correlacion.png` |
| Figure 2 | `fig_desempeno.png` |
| Figure 3 | `fig_curva_aprendizaje.png` |
| Figures 4 and 5 | `fig_costo_10000.png`, `fig_costo_completo.png` |
| Figure 6 | `fig_importancia.png` (`importancia.csv`, `importancia_resumen.csv`, `importancia_concordancia_spearman.csv`) |
| Tables 4–7 and A1–A3 | `tablas.tex` |
| Figure 7, Tables 8–9 (Sections 5.8–5.9) | `robustez/fig_robustez.png`, `robustez/tabla_robustez_paneles.tex`, `robustez/tabla_ablacion.tex` |
| Robustness tests and summaries | `robustez/pruebas_robustez.txt`, `robustez/robustez_*.csv`, `robustez/ablacion_v5_sin_x.csv` |
| Diagnostic of `x1` and `x2` (Section 5.9) | `robustez/diagnostico_x1_x2.csv` |
| Section 5.3 (Friedman and paired Wilcoxon–Holm tests) | `pruebas_estadisticas.txt`, `wilcoxon_holm*_*.csv` |
| Means, standard deviations and 95% CIs | `resumen.csv`, `resumen.md` |
| Per-repetition results | `resultados_crudos.csv` |

## License

The code is released under the MIT License (`LICENSE`). The BAF dataset and the TabPFN model
weights are distributed by their respective authors under their own licenses.

## Citation

If you use this code, please cite the article above (full reference to be added upon
publication).
