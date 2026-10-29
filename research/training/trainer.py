"""Robust GPU training engine optimized for NVIDIA GeForce RTX 3050 Laptop GPU."""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.device import get_device, get_gpu_info, get_memory_stats
from src.utils.audio import calculate_snr, calculate_snr_improvement, calculate_mse


class Trainer:
    """End-to-end model trainer with AMP mixed precision and RTX 3050 memory profiling."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        checkpoint_dir: Path = Path("outputs/checkpoints"),
        early_stopping_patience: int = 7,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or get_device()
        self.mixed_precision = mixed_precision and self.device.type == "cuda"
        self.grad_accum_steps = max(1, gradient_accumulation_steps)
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.patience = early_stopping_patience

        self.model.to(self.device)

        # PyTorch AMP scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision)

        self.history: Dict[str, list] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_snr_improvement_db": [],
            "learning_rate": [],
            "peak_gpu_mem_mb": [],
            "epoch_duration_sec": [],
        }

    def train_epoch(self, epoch: int) -> float:
        """Run single training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        self.optimizer.zero_grad()

        for batch_idx, (noisy, clean) in enumerate(self.train_loader):
            noisy = noisy.to(self.device, non_blocking=True)
            clean = clean.to(self.device, non_blocking=True)

            try:
                # Forward with optional AMP
                with torch.amp.autocast("cuda", enabled=self.mixed_precision):
                    pred = self.model(noisy)
                    loss = self.criterion(pred, clean)
                    loss_scaled = loss / self.grad_accum_steps

                # Backward pass
                self.scaler.scale(loss_scaled).backward()

                # Step optimizer on accumulation boundary
                if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
                    if self.max_grad_norm > 0.0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                total_loss += loss.item()

            except torch.cuda.OutOfMemoryError as oom:
                print("\n" + "!" * 60)
                print("CUDA OUT OF MEMORY ERROR DETECTED ON RTX 3050!")
                print(f"Batch index: {batch_idx}")
                print(f"Batch shape: {noisy.shape}")
                print(f"Allocated memory: {torch.cuda.memory_allocated() / (1024**2):.1f} MB")
                print("RECOMMENDATION:")
                print("  1. Reduce batch_size in configs")
                print("  2. Increase gradient_accumulation_steps")
                print("  3. Reduce model hidden_channels or window_size")
                print("!" * 60 + "\n")
                raise oom

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation loader."""
        self.model.eval()
        total_loss = 0.0
        snr_improvements = []
        num_batches = len(self.val_loader)

        for noisy, clean in self.val_loader:
            noisy = noisy.to(self.device, non_blocking=True)
            clean = clean.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.mixed_precision):
                pred = self.model(noisy)
                loss = self.criterion(pred, clean)

            total_loss += loss.item()

            # Batch SNR metrics
            for b in range(noisy.shape[0]):
                c_np = clean[b].cpu().numpy().squeeze()
                n_np = noisy[b].cpu().numpy().squeeze()
                p_np = pred[b].cpu().numpy().squeeze()
                delta_snr = calculate_snr_improvement(c_np, n_np, p_np)
                snr_improvements.append(delta_snr)

        avg_loss = total_loss / max(num_batches, 1)
        avg_delta_snr = float(np.mean(snr_improvements)) if snr_improvements else 0.0

        return {
            "val_loss": avg_loss,
            "val_snr_improvement_db": avg_delta_snr,
        }

    def fit(self, num_epochs: int = 10, experiment_name: str = "model") -> Dict[str, Any]:
        """Execute full training loop with checkpointing and early stopping."""
        print(f"\nBeginning Training for {num_epochs} Epochs on {self.device}...")
        best_val_loss = float("inf")
        patience_counter = 0
        total_start = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            train_loss = self.train_epoch(epoch)
            val_metrics = self.evaluate()
            val_loss = val_metrics["val_loss"]
            val_snr_gain = val_metrics["val_snr_improvement_db"]
            epoch_dur = time.time() - epoch_start

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Step learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            mem_stats = get_memory_stats()
            peak_vram = mem_stats["max_allocated_mb"]

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(round(train_loss, 6))
            self.history["val_loss"].append(round(val_loss, 6))
            self.history["val_snr_improvement_db"].append(round(val_snr_gain, 3))
            self.history["learning_rate"].append(current_lr)
            self.history["peak_gpu_mem_mb"].append(peak_vram)
            self.history["epoch_duration_sec"].append(round(epoch_dur, 2))

            print(
                f"Epoch [{epoch:2d}/{num_epochs:2d}] "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                f"Val Delta-SNR: {val_snr_gain:+5.2f} dB | LR: {current_lr:.1e} | "
                f"Peak VRAM: {peak_vram:4.0f} MB ({epoch_dur:.1f}s)"
            )

            # Checkpoint: Latest
            self.save_checkpoint(self.checkpoint_dir / f"{experiment_name}_latest.pt", epoch, val_loss)

            # Checkpoint: Best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(self.checkpoint_dir / f"{experiment_name}_best.pt", epoch, val_loss)
                print(f"  -> Saved new best checkpoint (Val Loss: {best_val_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"\n[Early Stopping] No improvement in validation loss for {self.patience} epochs.")
                    break

        total_dur = time.time() - total_start
        print(f"\nTraining completed in {total_dur:.1f} seconds. Best Val Loss: {best_val_loss:.6f}")

        # Save history log
        history_path = self.checkpoint_dir / f"{experiment_name}_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def save_checkpoint(self, path: Path, epoch: int, val_loss: float):
        """Save PyTorch model weights and state."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "device": str(self.device),
        }
        torch.save(state, path)
