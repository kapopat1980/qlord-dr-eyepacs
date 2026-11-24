# Q-LORD for diabetic retinopathy grading on EyePACS

Code, results and logs for **Q-LORD** (Quality-aware Lesion-graph Ordinal Retinopathy Detector): a five-grade diabetic retinopathy (DR) grading network that combines lesion-query attention, a patch graph with graph convolutions, a hybrid ordinal-evidential head, an auxiliary image-quality head and a learned abstention gate, with an evidence-weighted, temperature-calibrated three-fold ensemble.

> Status: research code accompanying a manuscript in preparation. Paper link and final citation: *to be added*.

## Important notes before you use this

- **Dataset:** EyePACS data from the Kaggle Diabetic Retinopathy Detection competition (2015), in the resized version [`tanlikesmath/diabetic-retinopathy-resized`](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized). The data are **not** included here; download them from Kaggle and follow their terms.
- **No clinical use.** This is research code. It has been evaluated on one dataset (a patient-grouped hold-out set) and has no external validation.
- **What the ablations show:** in the single-fold study, the ordinal head, the evidential head and class-balanced sampling clearly mattered, whereas removing the graph, lesion attention plus graph, or the quality head did **not** reduce QWK (all p > 0.4). The lesion-query attention and evidential ordinal head also follow earlier work (El Bellaj et al., arXiv:2602.10315).
- **Quality labels are a proxy** computed from image statistics (sharpness, contrast, brightness, retinal coverage), not clinical gradability labels.

## Repository layout

```
notebooks/
  qlord_eyepacs_main.ipynb    full Q-LORD: 3-fold training, ensemble, calibration, abstention (MODE = "main")
  qlord_eyepacs_study.ipynb   baselines + ablations, shared reduced protocol (MODE = "study")
src/
  qlord_eyepacs.py            script version of the notebook (same code; edit MODE in PART 0)
results/
  main/                       outputs of the main run (test predictions, split, metrics JSON, figures, log)
  study/                      outputs of the baseline/ablation study (12 variants, bootstrap tests, figures)
requirements.txt  LICENSE  CITATION.cff
```

## How to run (Kaggle, recommended)

1. Create a Kaggle notebook, upload `notebooks/qlord_eyepacs_main.ipynb` (or the study notebook).
2. Add the dataset `tanlikesmath/diabetic-retinopathy-resized` as input. Choose **GPU T4** and turn **Internet on** (needed to download the ImageNet-pretrained EfficientNet-B3 weights through timm).
3. Set `SMOKE_TEST = True` in PART 0 and run once to see the time estimate; then set it to `False`.
4. Use *Save Version, Save & Run All*. The run is resumable: if the session limit stops it, add the previous version's output as an input and run the same notebook again.

Approximate cost on a T4 (from the logged run): about 0.5 h preprocessing, about 2.1 to 2.2 h per fold at 512 px, about 7 h in total for the main run. The study run uses 384 px and a short schedule; allow roughly 5 h (estimate, not measured here).

Running locally is possible with a CUDA GPU and the dataset folder, but paths in PART 0/1 are written for Kaggle (`/kaggle/input`, `/kaggle/working`) and need adapting.

## Protocol in brief

| Item | Setting |
|---|---|
| Data | 35,108 labelled images, grades 0 to 4 (25,802 / 2,438 / 5,288 / 872 / 708) |
| Split | patient-grouped; 10% test (3,510 images), remaining 31,598 in 3 stratified folds; seed 42 |
| Preprocessing | retina crop, 512 x 512 (384 in study), Ben Graham enhancement |
| Backbone | EfficientNet-B3 (ImageNet), d = 256 tokens, K = 8 lesion queries, top-48 graph nodes, 2 GCN layers |
| Training | 3 frozen-backbone epochs + up to 25 fine-tuning epochs, AdamW, class-balanced sampling (power 0.75), 6,000 draws/epoch |
| Inference | flip TTA, per-fold temperature scaling, evidence-weighted ensemble of 3 fold models |

## Results

### Main run: Q-LORD three-fold ensemble (test n = 3,510)

| Accuracy | QWK (95% CI) | Macro-F1 | ECE | Brier | NLL |
|---|---|---|---|---|---|
| 0.841 | 0.818 (0.793 to 0.836) | 0.641 | 0.030 | 0.247 | 0.505 |

Mean single fold model: QWK 0.798, ECE 0.066. Per-grade recall: 0.961 / 0.308 / 0.573 / 0.691 / 0.590 (mild DR is the weakest grade). With the deployed abstention gate, 4.5% of cases are referred; retained-case accuracy 0.859 and QWK 0.832. Overall ECE hides weaker calibration for predicted minority grades (see the paper).

### Baselines and ablations (single fold, 384 px, same test images; not comparable to the main run)

| Variant | QWK | dQWK vs Q-LORD | p |
|---|---|---|---|
| Q-LORD (full) | 0.763 | n/a | n/a |
| Baseline: softmax | 0.733 | -0.030 | 0.004 |
| Baseline: MC-dropout | 0.741 | -0.022 | 0.016 |
| Baseline: ordinal only | 0.743 | -0.019 | 0.016 |
| Baseline: evidential only | 0.734 | -0.029 | 0.004 |
| w/o graph | 0.767 | +0.005 | 0.580 |
| w/o lesion attention and graph | 0.769 | +0.006 | 0.408 |
| w/o ordinal head | 0.731 | -0.031 | <0.004 |
| w/o evidential head | 0.735 | -0.028 | <0.004 |
| w/o quality head | 0.767 | +0.005 | 0.512 |
| w/o consistency losses | 0.755 | -0.008 | 0.380 |
| w/o balanced sampling | 0.720 | -0.043 | <0.004 |

p-values are two-sided, from a paired bootstrap (500 resamples) over the same test images; one fold and one seed, so differences of about 0.02 QWK or less are unresolved. Full tables: `results/study/comparison_table.csv` and `results/study/paper_table4.csv`.

## Output files

- `qlord_test_predictions.csv`: per-image reference grade, prediction, confidence, quality score, gate risk, vacuity, aleatoric and epistemic uncertainty, abstain flag.
- `split.csv`: the fixed patient-grouped split (role = test / trainval, fold), with the proxy quality statistics.
- `qlord.json`, `<variant>.json`: metrics, temperatures, selective-prediction and uncertainty summaries.
- `bootstrap.json`: paired bootstrap differences and p-values (study).
- `run_log.txt`: log of the main run. The study run's own log was not saved in this repository.

## Citation

If you use this code, please cite it (see `CITATION.cff`); update with the paper reference once published.

## Contact

Divyakant Meva, Faculty of Computer Applications, Marwadi University, Rajkot, India.
