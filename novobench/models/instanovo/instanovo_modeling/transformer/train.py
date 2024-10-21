from __future__ import annotations

import argparse
import datetime
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytorch_lightning as ptl
import torch
import yaml
from pytorch_lightning.strategies import DDPStrategy
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from novobench.models.instanovo.instanovo_modeling.inference.beam_search import BeamSearchDecoder
from novobench.models.instanovo.instanovo_modeling.transformer.model import InstaNovo
from novobench.models.instanovo.instanovo_modeling.utils.metrics import Metrics
from novobench.models.instanovo.instanovo_dataloader import InstanovoDataModule
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class PTModule(ptl.LightningModule):
    """PTL wrapper for model."""

    def __init__(
        self,
        config: dict[str, Any],
        model: InstaNovo,
        decoder: BeamSearchDecoder,
        metrics: Metrics,
        # sw: SummaryWriter,
        optim: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        # device: str = 'cpu',
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.decoder = decoder
        self.metrics = metrics
        # self.sw = sw
        self.optim = optim
        self.scheduler = scheduler

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        self.running_loss = None
        self._reset_valid_metrics()
        self.steps = 0

        # Update rates based on bs=32
        self.step_scale = 32 / config["train_batch_size"]

    def forward(
        self,
        spectra: Tensor,
        precursors: Tensor,
        peptides: list[str] | Tensor,
        spectra_mask: Tensor,
        peptides_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Model forward pass."""
        return self.model(spectra, precursors, peptides, spectra_mask, peptides_mask)  # type: ignore

    def training_step(  # need to update this
        self,
        batch: tuple[Tensor, Tensor, Tensor, list[str] | Tensor, Tensor],
    ) -> torch.Tensor:
        """A single training step.

        Args:
            batch (tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]) :
                A batch of MS/MS spectra, precursor information, and peptide
                sequences as torch Tensors.

        Returns:
            torch.FloatTensor: training loss
        """
        spectra, precursors, spectra_mask, peptides, peptides_mask = batch
        spectra = spectra.to(self.device)
        precursors = precursors.to(self.device)
        spectra_mask = spectra_mask.to(self.device)
        # peptides = peptides.to(self.device)
        # peptides_mask = peptides_mask.to(self.device)

        preds, truth = self.forward(spectra, precursors, peptides, spectra_mask, peptides_mask)
        # cut off EOS's prediction, ignore_index should take care of masking
        # preds = preds[:, :-1].reshape(-1, preds.shape[-1])
        preds = preds[:, :-1, :].reshape(-1, self.model.decoder.vocab_size + 1)

        loss = self.loss_fn(preds, truth.flatten())

        if self.running_loss is None:
            self.running_loss = loss.item()
        else:
            self.running_loss = 0.99 * self.running_loss + (1 - 0.99) * loss.item()

        if ((self.steps + 1) % int(2000 * self.step_scale)) == 0:
            lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
            logging.info(
                f"[Step {self.steps-1:06d}]: train_loss_raw={loss.item():.4f}, running_loss={self.running_loss:.4f}, LR={lr}"
            )

        if (self.steps + 1) % int(500 * self.step_scale) == 0:
            lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
            # self.sw.add_scalar("train/loss_raw", loss.item(), self.steps - 1)
            # self.sw.add_scalar("train/loss_smooth", self.running_loss, self.steps - 1)
            # self.sw.add_scalar("optim/lr", lr, self.steps - 1)
            # self.sw.add_scalar("optim/epoch", self.trainer.current_epoch, self.steps - 1)

        self.steps += 1

        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor, Tensor, list[str] | Tensor, Tensor], *args: Any
    ) -> torch.Tensor:
        """Single validation step."""
        spectra, precursors, spectra_mask, peptides, peptides_mask = batch
        spectra = spectra.to(self.device)
        precursors = precursors.to(self.device)
        spectra_mask = spectra_mask.to(self.device)
        # peptides = peptides.to(self.device)
        # peptides_mask = peptides_mask.to(self.device)

        # Loss
        with torch.no_grad():
            preds, truth = self.forward(spectra, precursors, peptides, spectra_mask, peptides_mask)
        preds = preds[:, :-1, :].reshape(-1, self.model.decoder.vocab_size + 1)
        loss = self.loss_fn(preds, truth.flatten())

        # Greedy decoding
        with torch.no_grad():
            # y, _ = decoder(spectra, precursors, spectra_mask)
            p = self.decoder.decode(
                spectra=spectra,
                precursors=precursors,
                beam_size=self.config['instanovo']["n_beams"],
                max_length=self.config['instanovo']["max_length"],
            )

        # targets = self.model.batch_idx_to_aa(peptides)
        y = ["".join(x.sequence) if not isinstance(x, list) else "" for x in p]
        targets = peptides

        # aa_prec, aa_recall, pep_recall, _ = self.metrics.compute_precision_recall(targets, y)
        # aa_er = self.metrics.compute_aa_er(targets, y)

        self.valid_metrics["valid_loss"].append(loss.item())
        # self.valid_metrics["aa_er"].append(aa_er)
        # self.valid_metrics["aa_prec"].append(aa_prec)
        # self.valid_metrics["aa_recall"].append(aa_recall)
        # self.valid_metrics["pep_recall"].append(pep_recall)

        return loss.item()

    def on_train_epoch_end(self) -> None:
        """Log the training loss at the end of each epoch."""
        epoch = self.trainer.current_epoch
        # self.sw.add_scalar(f"eval/train_loss", self.running_loss, epoch)
        logging.info(f"[Epoch {epoch:02d}] train_loss={self.running_loss:.5f}")
        self.running_loss = None

    def on_validation_epoch_end(self) -> None:
        """Log the validation metrics at the end of each epoch."""
        epoch = self.trainer.current_epoch
        # for k, v in self.valid_metrics.items():
        #     self.sw.add_scalar(f"eval/{k}", np.mean(v), epoch)

        valid_loss = np.mean(self.valid_metrics["valid_loss"])
        logging.info(
            f"[Epoch {epoch:02d}] train_loss={self.running_loss:.5f}, valid_loss={valid_loss:.5f}"
        )
        logging.info(f"[Epoch {epoch:02d}] Metrics:")
        # for metric in ["aa_er", "aa_prec", "aa_recall", "pep_recall"]:
        #     val = np.mean(self.valid_metrics[metric])
        #     logging.info(f"[Epoch {epoch:02d}] - {metric:11s}{val:.3f}")

        self._reset_valid_metrics()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Save config with checkpoint."""
        checkpoint["config"] = self.config

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Attempt to load config with checkpoint."""
        self.config = checkpoint["config"]

    def configure_optimizers(
        self,
    ) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
        """Initialize the optimizer.

        This is used by pytorch-lightning when preparing the model for training.

        Returns
        -------
        Tuple[torch.optim.Optimizer, Dict[str, Any]]
            The initialized Adam optimizer and its learning rate scheduler.
        """
        return [self.optim], {"scheduler": self.scheduler, "interval": "step"}

    def _reset_valid_metrics(self) -> None:
        # valid_metrics = ["valid_loss", "aa_er", "aa_prec", "aa_recall", "pep_recall"]
        valid_metrics = ["valid_loss"]
        self.valid_metrics: dict[str, list[float]] = {x: [] for x in valid_metrics}


def train(
    train_df,
    valid_df,
    config,
    model_path, 
) -> None:
    # Transformer vocabulary, should we include an UNK token?
    if config['instanovo']["dec_type"] != "depthcharge":
        vocab = ["PAD", "<s>", "</s>"] + list(config['instanovo']["residues"].keys())
    else:
        vocab = list(config['instanovo']["residues"].keys())
    config["vocab"] = vocab
    s2i = {v: i for i, v in enumerate(vocab)}
    i2s = {i: v for i, v in enumerate(vocab)}
    logging.info(f"Vocab: {i2s}")

    train_dl = InstanovoDataModule(
        df = train_df,
        s2i = s2i,
        return_str = True,
        batch_size = config["train_batch_size"],
        n_workers = config["n_workers"]
    ).get_dataloader(shuffle=True)

    valid_dl = InstanovoDataModule(
        df = valid_df,
        s2i = s2i,
        return_str = True,
        batch_size = config["predict_batch_size"],
        n_workers = config["n_workers"]
    ).get_dataloader()

    # Update rates based on bs=32
    step_scale = 32 / config["train_batch_size"]
    logging.info(f"Updates per epoch: {len(train_dl):,}, step_scale={step_scale}")

    batch = next(iter(train_dl))
    spectra, precursors, spectra_mask, peptides, peptides_mask = batch

    logging.info("Sample batch:")
    logging.info(f" - spectra.shape={spectra.shape}")
    logging.info(f" - precursors.shape={precursors.shape}")
    logging.info(f" - spectra_mask.shape={spectra_mask.shape}")
    logging.info(f" - len(peptides)={len(peptides)}")
    logging.info(f" - peptides_mask={peptides_mask}")

    # init model
    model = InstaNovo(
        i2s=i2s,
        residues=config['instanovo']["residues"],
        dim_model=config['instanovo']["dim_model"],
        n_head=config['instanovo']["n_head"],
        dim_feedforward=config['instanovo']["dim_feedforward"],
        n_layers=config['instanovo']["n_layers"],
        dropout=config['instanovo']["dropout"],
        max_length=config['instanovo']["max_length"],
        max_charge=config["max_charge"],
        use_depthcharge=config['instanovo']["use_depthcharge"],
        enc_type=config['instanovo']["enc_type"],
        dec_type=config['instanovo']["dec_type"],
        dec_precursor_sos=config['instanovo']["dec_precursor_sos"],
    )

    if model_path is not None:
        logging.info(f"Loading model checkpoint from '{model_path}'")
        model_state = torch.load(model_path, map_location="cpu")
        # check if PTL checkpoint
        if "state_dict" in model_state:
            model_state = {k.replace("model.", ""): v for k, v in model_state["state_dict"].items()}

        k_missing = np.sum(
            [x not in list(model_state.keys()) for x in list(model.state_dict().keys())]
        )
        if k_missing > 0:
            logging.warning(f"Model checkpoint is missing {k_missing} keys!")
        k_missing = np.sum(
            [x not in list(model.state_dict().keys()) for x in list(model_state.keys())]
        )
        if k_missing > 0:
            logging.warning(f"Model state is missing {k_missing} keys!")
        model.load_state_dict(model_state, strict=False)

    logging.info(
        f"Model loaded with {np.sum([p.numel() for p in model.parameters()]):,d} parameters"
    )

    logging.info("Test forward pass:")
    with torch.no_grad():
        y, _ = model(spectra, precursors, peptides, spectra_mask, peptides_mask)
        logging.info(f" - y.shape={y.shape}")

    # Train on GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # decoder = GreedyDecoder(model, i2s, max_length=config["max_length"])
    decoder = BeamSearchDecoder(model=model)
    metrics = Metrics(config['instanovo']["residues"], config["isotope_error_range"])

    # init optim
    # assert s2i["PAD"] == 0  # require PAD token to be index 0, all padding should use zeros
    # loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optim = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = WarmupScheduler(optim, config["warmup_iters"])
    strategy = _get_strategy()

    ptmodel = PTModule(config, model, decoder, metrics, optim, scheduler)

    if config['instanovo']["save_model"]:
        callbacks = [
            ptl.callbacks.ModelCheckpoint(
                dirpath=config["model_save_folder_path"],
                save_top_k=config["save_top_k"],
                save_weights_only=config["save_weights_only"],
            )
        ]
    else:
        callbacks = None

    logging.info("Initializing PL trainer.")
    trainer = ptl.Trainer(
        accelerator="auto",
        auto_select_gpus=True,
        callbacks=callbacks,
        devices="auto",
        max_epochs=config["max_epochs"],
        num_sanity_val_steps=config["num_sanity_val_steps"],
        accumulate_grad_batches=config['instanovo']["grad_accumulation"],
        gradient_clip_val=config['instanovo']["gradient_clip_val"],
        val_check_interval=config["val_check_interval"],
        check_val_every_n_epoch=config["check_val_every_n_epoch"],
        strategy=strategy,
    )

    # Train the model.
    trainer.fit(ptmodel, train_dl, valid_dl)

    logging.info("Training complete.")


def _get_strategy() -> DDPStrategy | None:
    """Get the strategy for the Trainer.

    The DDP strategy works best when multiple GPUs are used. It can work for
    CPU-only, but definitely fails using MPS (the Apple Silicon chip) due to
    Gloo.

    Returns
    -------
    Optional[DDPStrategy]
        The strategy parameter for the Trainer.
    """
    if torch.cuda.device_count() > 1:
        return DDPStrategy(find_unused_parameters=False, static_graph=True)

    return None


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup scheduler."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup: int) -> None:
        self.warmup = warmup
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        """Get the learning rate at the current step."""
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch: int) -> float:
        """Get the LR factor at the current step."""
        lr_factor = 1.0
        if epoch <= self.warmup:
            lr_factor *= epoch / self.warmup
        return lr_factor


def train_instanovo(train_df,val_df,config,model_filename) -> None:
    """Train the model."""
    logging.info("Initializing training.")

    train(train_df, val_df, config, model_filename)



