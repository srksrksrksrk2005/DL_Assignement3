# DA6401 - Assignment 3: Transformer for Machine Translation

## Overview

This repository contains a PyTorch implementation of the Transformer architecture from "Attention Is All You Need" for German-to-English machine translation on the Multi30k dataset.

## Project Structure

```text
da6401_assignment_3/
├── dataset.py
├── lr_scheduler.py
├── model.py
├── train.py
├── requirements.txt
├── checkpoints/
└── run_logs/
```

## Training

Run training from the project directory:

```powershell
python train.py
```

Useful overrides are still available through CLI flags such as `--lr-strategy`, `--learning-rate`, `--positional-encoding`, `--no-attention-scaling`, and `--resume-checkpoint`.

The script saves checkpoints under `checkpoints/` and writes the best model to `checkpoint.pt` by default.

## Submission Notes

The code is kept submission-safe: For automated evaluation, the important outputs are the model implementation, the Noam scheduler, positional encoding, and the saved checkpoint.


Git link : https://github.com/ee23b067-cmd/da6401_assignement3/tree/main

wandb Report link : https://wandb.ai/sadamrk2005-indian-institute-of-technology-madras/DA6401_assignement3/reportlist