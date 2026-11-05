# Training metrics and plot provenance

These files describe the completed GPT-2 Small one-epoch-equivalent run: **17,234 optimizer updates**, **9,035,579,392 training-token exposures**, and **124,439,808 unique parameters** across three ranks.

| Measurement | Result |
|---|---:|
| Final training loss (one update) | 3.1067519188 |
| Mean training loss, last 100 updates | 3.1128578424 |
| Final sampled validation loss | 3.1298661232 |
| Final sampled validation perplexity | 22.8709174522 |
| Best logged validation loss | 3.1280469894 at update 17,000 |
| Combined throughput across all recorded updates | 42,652.50 tokens/s |
| Combined throughput across the last 100 updates | 42,101.29 tokens/s |
| Final recorded learning rate | 0.0005991358875942747 |

The CSV retains every unique optimizer update, without downsampling. All updates 1–17,234 are present and finite. Update 5 was logged twice after replay; the last event is retained. Its loss, learning rate and token count match the earlier event, while timing and some runtime diagnostics differ. The 19 validation events include initialization, every 1,000 updates, and the final update. There are no missing update numbers. The source log has no wall-clock timestamps, so these plots cannot show interruption timing or duration.

Throughput is **global tokens divided by summed iteration durations**, not an arithmetic average of instantaneous throughput. It excludes evaluation, checkpoint saving, startup and downtime. The loss plot shows raw per-update training loss, a trailing 100-update arithmetic mean, and sampled validation measurements. Lines between validation points are visual guides. The separate full-file evaluation (perplexity 21.2342) uses different target coverage and context; it must not be presented as a later point on this training-validation curve.

The learning-rate schedule was configured for **600,000 updates**, with **2,000 warmup updates**, peak **0.0006** and minimum **0.00006**. Update n used schedule index n − 1. The fixed one-epoch-equivalent run stopped at 17,234 updates, so the LR remained near its peak. The dashed portion of the LR figure is the unexecuted configured continuation, not measured training or a forecast. This configuration alone does not establish the cause of generation-quality limitations.

## Reproduce the figures

From the repository root, install `requirements.txt`, then run:

```bash
python scripts/plot_training_metrics.py --data-dir results --output-dir docs/assets
```

The script reads only the two public CSV files and generates `training_loss.png/.svg` and `learning_rate.png/.svg`. It does not train, load model weights, access the network, or modify the CSV inputs. Source and CSV hashes, exact final values, coverage checks and definitions are recorded in `training_metrics_provenance.json`.
