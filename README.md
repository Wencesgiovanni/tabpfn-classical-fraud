# Tabular foundation models versus tuned classical classifiers for bank account fraud detection

Code and results for the article:

> Hernández-López, E., Cruz-Espinosa, D. J., Herrera-Zúñiga, L. D., and Wences, G.
> *Tabular Foundation Models versus Tuned Classical Classifiers for Bank Account Fraud
> Detection: A Learning-Curve Study of TabPFN v2 and TabPFN-3.5.* (under review)

The study compares two generations of TabPFN (v2 and 3.5) with seven classical classifiers
(linear SVM, L1 logistic regression, ridge, random forest, XGBoost, scikit-learn MLP and a
Keras network), each with default and tuned hyperparameters, on the Base variant of the Bank
Account Fraud (BAF) dataset. Models are trained on 1,000 to 67,400 rows under a temporal split
(months 0–4 train, 5 validation, 6–7 test), with ten repetitions per training size.

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
| `generar_tablas_latex.py` | Tables 3–9 of the article in LaTeX (`resultados_finales/tablas.tex`). |
| `colab/ejecutar_en_colab.ipynb` | Notebook that runs the experiments on a Colab A100. |
| `particiones/` | Reference partitions (`particiones.npz`) and SHA-256 fingerprints of every preprocessed test matrix (`huellas.json`) used in the article. |
| `resultados_colab/` | Raw outputs of the Colab runs used in the article (one row per model, training size and repetition) and the software environments. |
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
and place it at `data/Base.csv`. Its use is subject to the license stated by its authors.

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

TabPFN v2 and TabPFN-3.5 require incompatible versions of the `tabpfn` package, so two
environments were used (Section 4.8 of the article):

* **Classical models, TabPFN v2 and analysis** — Python 3.9, `requirements-clasicos-v2.txt`
  (PyTorch 2.7.1 with CUDA 11.8 on the GPU).
* **TabPFN-3.5** — Python 3.13 (the default Python of Colab at the time), `requirements-tabpfn35.txt`.
  Downloading the TabPFN-3.5 weights requires a free Prior Labs account and accepting the
  model license; the notebook reads the token from a Colab secret named `TABPFN_TOKEN`.

All reported results were obtained on Google Colab virtual machines with an NVIDIA
A100-SXM4 GPU (80 GB) and a 12-core Intel Xeon CPU at 2.20 GHz; the exact package versions of
each run are in `resultados_colab/entorno_*.json`. The TabPFN models ran on the GPU and the
classical models on the CPU.

## Reproducing the article

### Tables and figures from the published runs (minutes, no GPU)

With the classical environment, from the repository root:

```bash
python preparar_resultados_finales.py   # summary, statistical tests, Figures 2-6
python figura_correlacion.py            # Figure 1 (needs data/Base.csv)
python generar_tablas_latex.py          # Tables 3-9 -> resultados_finales/tablas.tex
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
identical). TabPFN is more sensitive to the rounding of the copy and to the GPU: on a GPU other
than the A100, its PR-AUC differed from the published value by up to 0.001, well below its
standard deviation across repetitions (0.002–0.010). Wall-clock times depend on the hardware
and on the load of the virtual machine.

`experimento_fraude.py` can also run the whole protocol on a single machine
(`python experimento_fraude.py --datos data/Base.csv`, with `--prueba-rapida` for a short
check); the results in the article come from the Colab runs described above.

## Correspondence between the article and the outputs

| Article | File in `resultados_finales/` |
|---|---|
| Figure 1 | `fig_correlacion.png` |
| Figure 2 | `fig_desempeno.png` |
| Figure 3 | `fig_curva_aprendizaje.png` |
| Figures 4 and 5 | `fig_costo_10000.png`, `fig_costo_completo.png` |
| Figure 6 | `fig_importancia.png` (`importancia.csv`, `importancia_resumen.csv`, `importancia_concordancia_spearman.csv`) |
| Tables 3–9 | `tablas.tex` |
| Section 5.3 (Friedman and paired Wilcoxon–Holm tests) | `pruebas_estadisticas.txt`, `wilcoxon_holm*_*.csv` |
| Means, standard deviations and 95% CIs | `resumen.csv`, `resumen.md` |
| Per-repetition results | `resultados_crudos.csv` |

## License

The code is released under the MIT License (`LICENSE`). The BAF dataset and the TabPFN model
weights are distributed by their respective authors under their own licenses.

## Citation

If you use this code, please cite the article above (full reference to be added upon
publication).
