# Overtone — Multi-label Emotion Classification

GDG AI/ML Wing recruitment submission. TF-IDF (word + char n-grams) features
feeding one logistic regression classifier per emotion label (27 labels,
multi-label), with per-label threshold tuning.

## Headline numbers

- **Honest macro-F1 (5-fold, tune/report split within val):** 0.4378 ± 0.0138
- **Shipping macro-F1 (thresholds tuned on full val, evaluated on val):** 0.4685

The honest number is the one that should be trusted as a generalization
estimate — it's computed by splitting `val` into 5 folds, tuning thresholds
on 4 folds and scoring on the held-out fold, so no split is ever tuned and
scored on itself. The shipping number tunes thresholds on all of `val` at
once and is reported for transparency, but it's optimistic by construction
(same data used to pick thresholds and to grade them) — it is **not** used
to touch `test.jsonl`, only to fix the final thresholds before scoring test.

## Setup

```bash
pip install -r requirements.txt
```

Download the dataset (`labels.txt`, `train.jsonl`, `val.jsonl`, `test.jsonl`,
and `score.py` if you have access to it) and place the files in `./data/`,
or point the `DATA_DIR` environment variable at wherever you put them:

```bash
export DATA_DIR=/path/to/overtone-data   # optional, defaults to ./data
```

## Run

```bash
jupyter nbconvert --to notebook --execute gdg-ai-ml-final.ipynb --output gdg-ai-ml-final.ipynb
```

or open the notebook and run all cells top to bottom.

## Outputs

- `val_predictions.jsonl` — predictions on the validation set, written for
  the format-check step below.
- `predictions.jsonl` — the actual submission: predictions on `test.jsonl`,
  written to the working directory the notebook is run from.

## Validating the output format (optional)

The last cell calls the organizer-provided `score.py` to sanity-check
`val_predictions.jsonl` against `val.jsonl`. This script is **not vendored
in this repo** — it's assumed to live alongside the dataset files in
`DATA_DIR`. If you don't have it, skip that cell; it doesn't affect
`predictions.jsonl` generation.

```bash
python $DATA_DIR/score.py --predictions val_predictions.jsonl --truth $DATA_DIR/val.jsonl
```

## Package versions

Pinned versions used for the numbers above (get exact versions from the
environment you ran in with `pip freeze | grep -E "numpy|scipy|scikit-learn"`
and paste them here — results can shift slightly across scikit-learn
versions because of changes to `TfidfVectorizer` tokenization defaults and
`LogisticRegression` solver behavior):

```
numpy==2.0.2
scikit-learn==1.6.1
scipy==1.16.3
```

## Method summary

- **Features:** TF-IDF word n-grams (1,2) + char n-grams (3,5), unioned,
  fit on train text only (no val/test leakage into the vocabulary).
- **Model:** one `LogisticRegression(C=4.0, class_weight="balanced")` per
  label, trained on all train rows (multi-label positives and zero-label
  negatives both included).
- **Thresholds:** per-label, swept in `[0.05, 0.95]` step `0.01`, chosen to
  maximize that label's F1.
- **Evaluation:** macro-F1 across all 27 labels, reported honestly via
  5-fold tune/report split on val (see Headline numbers above).
