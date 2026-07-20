"""
Training, inference, BLEU evaluation, and checkpoint utilities for
DA6401 Assignment 3.
"""

import argparse
from collections import Counter
import math
import os
from pathlib import Path
import random
from typing import Optional

import gdown
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN, Multi30kDataset
from lr_scheduler import NoamScheduler
from model import (
    DEFAULT_CHECKPOINT_PATH,
    MultiHeadAttention,
    Transformer,
    _checkpoint_candidates,
    make_src_mask,
    make_tgt_mask,
)


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in the range [0, 1)")
        if not 0 <= pad_idx < vocab_size:
            raise ValueError("pad_idx must be a valid vocabulary index")

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.criterion = nn.KLDivLoss(reduction="sum")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: shape [batch * tgt_len, vocab_size]
            target: shape [batch * tgt_len]
        """
        logits = logits.reshape(-1, self.vocab_size)
        target = target.reshape(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            if self.vocab_size > 2:
                true_dist.fill_(self.smoothing / (self.vocab_size - 2))
            true_dist[:, self.pad_idx] = 0.0
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.pad_idx] = 0.0
            true_dist.masked_fill_((target == self.pad_idx).unsqueeze(1), 0.0)

        token_count = (target != self.pad_idx).sum().clamp_min(1)
        return self.criterion(log_probs, true_dist) / token_count


def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    step_state: Optional[dict] = None,
    max_train_steps: Optional[int] = None,
) -> tuple[float, float, float, float, bool]:
    """
    Run one epoch of teacher-forced training or validation.
    """
    model.train(is_train)
    total_loss = 0.0
    total_ce_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_confidence = 0.0
    stopped_early = False
    pad_idx = getattr(loss_fn, "pad_idx", 1)
    iterator = tqdm(
        data_iter,
        desc=f"{'train' if is_train else 'val'} {epoch_num}",
        leave=False,
        disable=_disable_tqdm(),
    )

    for src, tgt in iterator:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        src_mask = make_src_mask(src, pad_idx=pad_idx)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx=pad_idx)

        if is_train:
            if optimizer is None:
                raise ValueError("optimizer is required when is_train=True")
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        with torch.no_grad():
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_target = tgt_output.reshape(-1)
            correct, num_tokens = _token_accuracy(flat_logits, flat_target, pad_idx)
            confidence, confidence_tokens = _prediction_confidence(flat_logits, flat_target, pad_idx)
            ce_loss = F.cross_entropy(
                flat_logits,
                flat_target,
                ignore_index=pad_idx,
                reduction="sum",
            ).item()
            total_correct += correct
            total_confidence += confidence
            total_ce_loss += ce_loss

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if step_state is not None:
                step_state["global_step"] = step_state.get("global_step", 0) + 1
                if max_train_steps is not None and step_state["global_step"] >= max_train_steps:
                    stopped_early = True
                    break

        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        iterator.set_postfix(
            loss=total_loss / max(total_tokens, 1),
            acc=total_correct / max(total_tokens, 1),
        )

        if stopped_early:
            break

    avg_loss = total_loss / max(total_tokens, 1)
    avg_ce_loss = total_ce_loss / max(total_tokens, 1)
    avg_accuracy = total_correct / max(total_tokens, 1)
    avg_confidence = total_confidence / max(total_tokens, 1)
    return avg_loss, avg_accuracy, avg_confidence, avg_ce_loss, stopped_early


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int = 3,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.
    """
    was_training = model.training
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)

    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.full((1, 1), start_symbol, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=1).to(device)
            logits = model.decode(memory, src_mask, ys, tgt_mask)
            next_word = torch.argmax(logits[:, -1, :], dim=-1).item()
            ys = torch.cat(
                [ys, torch.tensor([[next_word]], dtype=torch.long, device=device)], dim=1
            )
            if next_word == end_symbol:
                break

    model.train(was_training)
    return ys


def _vocab_index(vocab, token: str, default: int) -> int:
    if hasattr(vocab, "stoi"):
        return int(vocab.stoi.get(token, default))
    try:
        return int(vocab[token])
    except Exception:
        return default


def _lookup_token(vocab, index: int) -> str:
    if hasattr(vocab, "lookup_token"):
        return vocab.lookup_token(index)
    if hasattr(vocab, "itos"):
        return vocab.itos[index]
    raise TypeError("tgt_vocab must expose lookup_token(index) or itos")


def _tokens_from_ids(vocab, ids: list[int], stop_at_eos: bool = True) -> list[str]:
    specials = {PAD_TOKEN, SOS_TOKEN, EOS_TOKEN}
    tokens = []
    for index in ids:
        token = _lookup_token(vocab, int(index))
        if stop_at_eos and token == EOS_TOKEN:
            break
        if token not in specials:
            tokens.append(token)
    return tokens


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _corpus_bleu(hypotheses: list[list[str]], references: list[list[str]]) -> float:
    if not hypotheses:
        return 0.0

    hyp_len = sum(len(hyp) for hyp in hypotheses)
    ref_len = sum(len(ref) for ref in references)
    if hyp_len == 0:
        return 0.0

    precisions = []
    for n in range(1, 5):
        matches = 0
        total = 0
        for hyp, ref in zip(hypotheses, references):
            hyp_ngrams = _ngrams(hyp, n)
            ref_ngrams = _ngrams(ref, n)
            total += sum(hyp_ngrams.values())
            for ngram, count in hyp_ngrams.items():
                matches += min(count, ref_ngrams.get(ngram, 0))
        if total == 0 or matches == 0:
            precisions.append(0.0)
        else:
            precisions.append(matches / total)

    if min(precisions) == 0.0:
        return 0.0

    brevity_penalty = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / hyp_len)
    bleu = brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / 4.0)
    return 100.0 * bleu


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.
    """
    if hasattr(model, "_ensure_checkpoint_loaded"):
        model._ensure_checkpoint_loaded()

    pad_idx = _vocab_index(tgt_vocab, PAD_TOKEN, 1)
    sos_idx = _vocab_index(tgt_vocab, SOS_TOKEN, 2)
    eos_idx = _vocab_index(tgt_vocab, EOS_TOKEN, 3)
    hypotheses = []
    references = []
    was_training = model.training
    model.eval()

    for src, tgt in tqdm(test_dataloader, desc="bleu", leave=False, disable=_disable_tqdm()):
        for row_idx in range(src.size(0)):
            src_row = src[row_idx : row_idx + 1].to(device)
            src_mask = make_src_mask(src_row, pad_idx=pad_idx)
            pred = greedy_decode(
                model,
                src_row,
                src_mask,
                max_len=max_len,
                start_symbol=sos_idx,
                end_symbol=eos_idx,
                device=device,
            )
            hypotheses.append(_tokens_from_ids(tgt_vocab, pred.squeeze(0).tolist()))
            references.append(_tokens_from_ids(tgt_vocab, tgt[row_idx].tolist()))

    model.train(was_training)
    return _corpus_bleu(hypotheses, references)


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.
    """
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model_config = model.get_config() if hasattr(model, "get_config") else {
        "src_vocab_size": model.src_vocab_size,
        "tgt_vocab_size": model.tgt_vocab_size,
        "d_model": model.d_model,
        "N": model.N,
        "num_heads": model.num_heads,
        "d_ff": model.d_ff,
        "dropout": model.dropout,
    }

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "model_config": model_config,
        },
        checkpoint_path,
    )


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model and optionally optimizer/scheduler state from disk.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        checkpoint_path = next(
            (candidate for candidate in _checkpoint_candidates() if candidate.exists()),
            DEFAULT_CHECKPOINT_PATH,
        )

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if hasattr(model, "_inference_checkpoint_loaded"):
        model._inference_checkpoint_loaded = True

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    return int(checkpoint["epoch"])


def download_checkpoint_from_gdrive(
    file_id: str = "1rXyhzh_9ozSHLu0uPwi_R7U0Tgqoi73n",
    output_path: str = "checkpoints/best_bleu_checkpoint.pt",
) -> None:
    """
    Download a checkpoint file from Google Drive using gdown.
    
    Args:
        file_id: The Google Drive file ID
        output_path: The local path to save the file
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists
    if Path(output_path).exists():
        print(f"Checkpoint already exists at {output_path}. Skipping download.")
        return
    
    print(f"Downloading checkpoint from Google Drive to {output_path}...")
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, output_path, quiet=False)
        print(f"Successfully downloaded checkpoint to {output_path}")
    except Exception as e:
        print(f"Error downloading checkpoint: {e}")
        print(f"Please manually download from: https://drive.google.com/file/d/{file_id}/view?usp=sharing")
        raise


def _env(name: str, default, cast):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return cast(value)


def _disable_tqdm() -> bool:
    return os.getenv("DISABLE_TQDM", "false").lower() in {"1", "true", "yes"}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_cli_args() -> dict:
    parser = argparse.ArgumentParser(description="Train the DA6401 assignment 3 Transformer")
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--num-epochs", default=None, type=int)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--best-checkpoint", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--lr-strategy", default=None)
    parser.add_argument("--learning-rate", default=None, type=float)
    parser.add_argument("--label-smoothing", dest="smoothing", default=None, type=float)
    parser.add_argument("--positional-encoding", default=None)
    parser.add_argument("--no-attention-scaling", action="store_true")
    parser.add_argument("--max-train-steps", default=None, type=int)
    return {key: value for key, value in vars(parser.parse_args()).items() if value is not None}


def _token_accuracy(logits: torch.Tensor, target: torch.Tensor, pad_idx: int) -> tuple[int, int]:
    predictions = logits.argmax(dim=-1)
    non_pad = target != pad_idx
    correct = ((predictions == target) & non_pad).sum().item()
    total = non_pad.sum().item()
    return correct, total


def _prediction_confidence(logits: torch.Tensor, target: torch.Tensor, pad_idx: int) -> tuple[float, int]:
    non_pad = target != pad_idx
    token_count = non_pad.sum().item()
    if token_count == 0:
        return 0.0, 0

    probs = F.softmax(logits, dim=-1)
    chosen = probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    confidence = chosen.masked_select(non_pad).sum().item()
    return confidence, token_count


def run_training_experiment(overrides: Optional[dict] = None) -> None:
    """
    Run the full Multi30k German-to-English training experiment.
    """
    overrides = overrides or {}
    config = {
        "seed": _env("SEED", 42, int),
        "batch_size": _env("BATCH_SIZE", 32, int),
        "num_epochs": overrides.get("num_epochs", _env("NUM_EPOCHS", 20, int)),
        "d_model": _env("D_MODEL", 256, int),
        "N": _env("NUM_LAYERS", 3, int),
        "num_heads": _env("NUM_HEADS", 8, int),
        "d_ff": _env("D_FF", 1024, int),
        "dropout": _env("DROPOUT", 0.1, float),
        "smoothing": overrides.get("smoothing", _env("LABEL_SMOOTHING", 0.1, float)),
        "warmup_steps": _env("WARMUP_STEPS", 4000, int),
        "min_freq": _env("MIN_FREQ", 2, int),
        "max_len": _env("MAX_LEN", 100, int),
        "num_workers": _env("NUM_WORKERS", 0, int),
        "checkpoint_dir": overrides.get("checkpoint_dir", os.getenv("CHECKPOINT_DIR", "checkpoints")),
        "best_checkpoint": overrides.get("best_checkpoint", os.getenv("BEST_CHECKPOINT", "checkpoint.pt")),
        "resume_checkpoint": overrides.get("resume_checkpoint", os.getenv("RESUME_CHECKPOINT", "")),
        "extra_epochs": _env("EXTRA_EPOCHS", 0, int),
        "task_name": overrides.get("task_name", _env("TASK_NAME", "baseline", str)),
        "lr_strategy": overrides.get("lr_strategy", os.getenv("LR_STRATEGY", "noam")),
        "learning_rate": overrides.get("learning_rate", _env("LEARNING_RATE", 1e-4, float)),
        "attention_scaling_label": "scaled" if not overrides.get("no_attention_scaling", False) else "no_scale",
        "positional_encoding": overrides.get(
            "positional_encoding", os.getenv("POSITIONAL_ENCODING", "sinusoidal")
        ),
        "use_attention_scaling": not overrides.get("no_attention_scaling", False),
        "max_train_steps": overrides.get("max_train_steps", _env("MAX_TRAIN_STEPS", 0, int)),
    }
    _set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_dataset = Multi30kDataset(
        "train", min_freq=config["min_freq"], max_len=config["max_len"]
    )
    val_dataset = Multi30kDataset(
        "validation",
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab,
        min_freq=config["min_freq"],
        max_len=config["max_len"],
    )
    test_dataset = Multi30kDataset(
        "test",
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab,
        min_freq=config["min_freq"],
        max_len=config["max_len"],
    )

    loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        **loader_kwargs,
    )

    model = Transformer(
        src_vocab_size=len(train_dataset.src_vocab),
        tgt_vocab_size=len(train_dataset.tgt_vocab),
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        use_learned_positional_encoding=config["positional_encoding"].lower() == "learned",
        use_attention_scaling=config["use_attention_scaling"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0 if config["lr_strategy"].lower() == "noam" else config["learning_rate"],
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = None
    if config["lr_strategy"].lower() == "noam":
        scheduler = NoamScheduler(
            optimizer,
            d_model=config["d_model"],
            warmup_steps=config["warmup_steps"],
        )
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_dataset.tgt_vocab),
        pad_idx=train_dataset.tgt_pad_idx,
        smoothing=config["smoothing"],
    )

    checkpoint_dir = Path(config["checkpoint_dir"])
    latest_path = checkpoint_dir / "latest_checkpoint.pt"
    best_path = Path(config["best_checkpoint"])
    best_val_loss = float("inf")
    start_epoch = 1
    step_state = {"global_step": 0}

    # Download checkpoint from Google Drive if it doesn't exist
    best_checkpoint_path = str(best_path)
    download_checkpoint_from_gdrive(
        file_id="1rXyhzh_9ozSHLu0uPwi_R7U0Tgqoi73n",
        output_path=best_checkpoint_path,
    )

    try:
        if config["resume_checkpoint"]:
            resume_path = Path(config["resume_checkpoint"])
            if resume_path.exists():
                start_epoch = load_checkpoint(str(resume_path), model, optimizer, scheduler) + 1
                print(f"resumed_from={resume_path} start_epoch={start_epoch}")
            else:
                raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")

        end_epoch = start_epoch + config["num_epochs"] + config["extra_epochs"] - 1

        for epoch in range(start_epoch, end_epoch + 1):
            (
                train_loss,
                train_accuracy,
                train_confidence,
                train_ce_loss,
                stopped_early,
            ) = run_epoch(
                train_loader,
                model,
                loss_fn,
                optimizer,
                scheduler,
                epoch_num=epoch,
                is_train=True,
                device=str(device),
                step_state=step_state,
                max_train_steps=config["max_train_steps"] or None,
            )
            if stopped_early:
                print(f"stopped_early=1 global_step={step_state['global_step']}")
                return
            val_loss, val_accuracy, val_confidence, val_ce_loss, _ = run_epoch(
                val_loader,
                model,
                loss_fn,
                optimizer=None,
                scheduler=None,
                epoch_num=epoch,
                is_train=False,
                device=str(device),
            )

            save_checkpoint(model, optimizer, scheduler, epoch, str(latest_path))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, epoch, str(best_path))

            lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]

            print(
                f"epoch={epoch} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} best_val_loss={best_val_loss:.4f}"
            )

        load_checkpoint(str(best_path), model, optimizer=None, scheduler=None)
        val_bleu = evaluate_bleu(
            model,
            val_loader,
            train_dataset.tgt_vocab,
            device=str(device),
            max_len=config["max_len"],
        )
        print(f"val_bleu={val_bleu:.2f}")
        test_bleu = evaluate_bleu(
            model,
            test_loader,
            train_dataset.tgt_vocab,
            device=str(device),
            max_len=config["max_len"],
        )
        print(f"test_bleu={test_bleu:.2f}")
    finally:
        pass


if __name__ == "__main__":
    run_training_experiment(_parse_cli_args())
