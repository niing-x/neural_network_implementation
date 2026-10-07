## Requirements

- `torch`
- `scipy` (for sparse-matrix feature handling)
- `matplotlib` (confusion matrix plot)
- [`perf_eval.py`](perf_eval.py) — shared evaluation module, must be importable (same directory)
- A preprocessed data file (default `processed_data.pkl`) containing
  `train_features`, `train_labels`, `val_features`, `val_labels`,
  `test_features`, `test_labels` (dense or scipy-sparse arrays)

## Model

Three hidden layers of equal width (32 or 64 units, chosen via `--hidden-size`),
each `Linear → ReLU → Dropout(0.3)`, followed by a final `Linear` output layer
producing 2 raw class logits (no softmax — `CrossEntropyLoss` applies that
internally).

## Running it

```
python nn_model.py [options]
```

| Flag | Default | Meaning |
|---|---|---|
| `--hidden-size` | `64` | Hidden layer width — `32` or `64` only |
| `--processed-data` | `processed_data.pkl` | Path to the pickled, preprocessed dataset |
| `--batch-size` | `128` | Mini-batch size |
| `--epochs` | `50` | Maximum training epochs (early stopping usually cuts this short) |
| `--patience` | `5` | Stop if `val_loss` hasn't improved for this many consecutive epochs |
| `--learning-rate` | `3e-4` | Initial learning rate for `AdamW` |
| `--device` | `cuda` if available, else `cpu` | Torch device |
| `--output-dir` | `./output` | Where training artifacts are written |

Example — train the smaller model for longer with more patience:
```
python nn_model.py --hidden-size 32 --epochs 80 --patience 8
```

## Training behavior

- Optimizer: `AdamW` (decoupled weight decay, `weight_decay=1e-4`) — see
  "Why AdamW" note below.
- LR schedule: `ReduceLROnPlateau` halves the learning rate if `val_loss`
  hasn't improved for 3 epochs.
- Early stopping: training stops once `val_loss` hasn't improved for
  `--patience` consecutive epochs. Both the scheduler and early stopping
  watch the *same* metric (`val_loss`) so they act in a coordinated way.
- Checkpointing: independently of early stopping, the weights from whichever
  epoch had the best *validation accuracy* are restored before final
  evaluation and saving — so a run that trains past its best point still
  evaluates/saves that best snapshot, not the final epoch's weights.

**Why `AdamW` and not `Adam`:** plain `Adam`'s `weight_decay` gets folded into
the gradient before Adam's adaptive per-parameter scaling, which distorts how
much regularization each parameter actually receives. `AdamW` applies weight
decay as a separate, direct shrink of the weights, independent of that
scaling — the version generally recommended today.

## Evaluation

After training, `train_and_evaluate()` runs the best checkpoint through
[`perf_eval.PerfEvaluator`](perf_eval.py) — the same evaluation module used
for the other models in this project, so results are directly comparable.
This produces:

- Accuracy, precision, recall, F1-score
- ROC-AUC
- Confusion matrix (`[[TN, FP], [FN, TP]]`, positive class = `>50K`)
- Training time and inference time
- Parameter count and memory footprint

## Outputs (in `--output-dir`, default `./output`)

| File | Contents |
|---|---|
| `training_metrics.json` | Per-epoch train/val loss & accuracy |
| `model_checkpoint.pt` | Best model weights + run metadata (hidden size, epochs, LR, accuracies) |
| `nn_mlp_h{hidden_size}_confusion_matrix.png` | Confusion matrix plot |

Also written to `./results/nn_mlp_h{hidden_size}.json` (via `perf_eval`):
all metrics above, suitable for `perf_eval.comparison_table()` to merge
across models.
