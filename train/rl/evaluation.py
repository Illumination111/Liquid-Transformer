"""Fresh candidate training and validation-selected final checkpoints."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from deit_tiny_snn import SpikingDeiTTiny
from small_world import SmallWorldTopology
from torch import nn

from train.train import run_epoch, seed_everything


def load_snn_checkpoint(path: Path, device="cpu") -> SpikingDeiTTiny:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = SpikingDeiTTiny(
        topologies=[SmallWorldTopology.from_dict(item) for item in checkpoint["topologies"]],
        **checkpoint["model_config"],
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device)


class CandidateEvaluator:
    def __init__(self, model_config, train_config, loader_factory, device, run_dir):
        self.model_config, self.train_config = model_config, train_config
        self.loader_factory, self.device = loader_factory, torch.device(device)
        self.run_dir = Path(run_dir)

    def __call__(self, topologies, epochs: int, identifier: str, final=False):
        if epochs < 1:
            raise ValueError("candidate epochs must be positive")
        started = time.monotonic()
        cfg = self.train_config
        seed_everything(cfg["seed"])
        # Rebuild loaders to reset shuffle/augmentation worker seeds for each candidate.
        train_loader, validation_loader = self.loader_factory()
        if not len(train_loader) or not len(validation_loader):
            raise ValueError("candidate training and validation loaders must be nonempty")
        model = SpikingDeiTTiny(topologies=topologies, **self.model_config).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
        # FP32 is the default until mixed-precision SNN training is benchmarked.
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        best_accuracy, best_epoch, best_state = -1.0, 0, None
        best_loss = float("inf")
        history = []
        for epoch in range(epochs):
            train_loss, train_accuracy = run_epoch(
                model,
                train_loader,
                criterion,
                self.device,
                optimizer,
                scaler,
                amp=False,
                clip_grad=1.0,
            )
            loss, accuracy = run_epoch(model, validation_loader, criterion, self.device)
            if not all(
                torch.isfinite(torch.tensor(value))
                for value in (train_loss, train_accuracy, loss, accuracy)
            ):
                raise FloatingPointError(f"non-finite metrics for candidate {identifier}")
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "validation_loss": loss,
                    "validation_accuracy": accuracy,
                }
            )
            if accuracy > best_accuracy:
                best_accuracy, best_loss, best_epoch = accuracy, loss, epoch + 1
                if final:
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            scheduler.step()
        result = {
            "identifier": identifier,
            "accuracy": best_accuracy,
            "loss": best_loss,
            "best_epoch": best_epoch,
            "history": history,
            "runtime_seconds": time.monotonic() - started,
            "peak_memory_gb": (
                torch.cuda.max_memory_allocated(self.device) / 1024**3
                if self.device.type == "cuda"
                else 0.0
            ),
        }
        if final:
            path = self.run_dir / "checkpoints" / f"{identifier}-best.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": best_state,
                    "model_config": self.model_config,
                    "train_config": cfg,
                    "topologies": [item.to_dict() for item in topologies],
                    "result": result,
                },
                path,
            )
            result["checkpoint"] = str(path)
        return result

    def test(self, checkpoint: Path, test_loader):
        model = load_snn_checkpoint(checkpoint, self.device).eval()
        criterion = nn.CrossEntropyLoss(label_smoothing=self.train_config["label_smoothing"])
        loss, accuracy = run_epoch(model, test_loader, criterion, self.device)
        return {"test_loss": loss, "test_accuracy": accuracy}
