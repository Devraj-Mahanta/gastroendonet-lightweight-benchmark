# Lightweight Deep Learning for Gastrointestinal Endoscopy Classification

A reproducible benchmark of eight lightweight CNN, vision-transformer and hybrid backbones for
four-class GI endoscopy classification (GERD, GERD-normal, polyp, polyp-normal) on the GastroEndoNet
dataset, with progressive fine-tuning, an eight-stage ablation, test-time augmentation, bootstrap
confidence intervals, statistical significance testing, calibration, near-duplicate leakage auditing,
CPU deployment profiling and zero-shot external validation on HyperKvasir.

## Requirements

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

Tested with Python 3.10 and TensorFlow 2.10 on an NVIDIA Quadro RTX 8000.

## Data

The GastroEndoNet dataset is publicly available at https://doi.org/10.17632/ffyn828yf4.3
(Mendeley Data, v3). Download it and arrange the four class folders under `data/`:

```
data/Gerd/  data/Gerd_Normal/  data/Polyp/  data/Polyp_Normal/
```

The exact train/validation/test split used in the paper is provided in `data/split.json`
(seed 42; 2804 / 600 / 602). Trained model weights are archived separately (see *Trained weights*).

## Reproducing the results

```bash
python generate_split.py                               # regenerate the fixed split (optional)
python run_all.py                                      # benchmark all eight backbones (3 runs each)
python finetune/ablation_runner.py                     # eight-stage fine-tuning ablation
python finetune/evaluate_finetune.py --model all --config all
```

Additional analyses (significance tests, calibration, leakage audit, CPU profiling, external
validation and Grad-CAM localization) live in `revision_analysis/`. These scripts read the trained
checkpoints and write their outputs to a local results directory; adjust the output path near the top
of each script to your environment before running.

## Repository structure

```
config.py, data_loader.py, train.py, evaluate.py, run_all.py   core benchmark pipeline
finetune/                                                      fine-tuning, ablation, TTA, Grad-CAM
revision_analysis/                                             significance / calibration / external / leakage
data/split.json                                               fixed split
results/, finetune/results/                                   per-seed evaluation logs and metrics
```

## Trained weights

Model checkpoints (benchmark and fine-tuning, ~1 GB) are archived on Zenodo:
https://doi.org/10.5281/zenodo.21350619

## Citation

```
Mahanta, D. R., & Mahanta, L. B. Lightweight Deep Learning for Gastrointestinal Endoscopy
Classification: A Reproducible Benchmark with Progressive Fine-Tuning and Ablation Analysis.
```

## License

Code released under the MIT License (see `LICENSE`). The GastroEndoNet and HyperKvasir datasets
retain their own licenses.
