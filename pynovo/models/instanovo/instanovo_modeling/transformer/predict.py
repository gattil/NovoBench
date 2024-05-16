from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
MASS_SCALE = 10000
from pynovo.models.instanovo.instanovo_modeling.inference.knapsack import Knapsack
from pynovo.models.instanovo.instanovo_modeling.inference.knapsack_beam_search import KnapsackBeamSearchDecoder
from pynovo.models.instanovo.instanovo_dataloader import InstanovoDataModule
from pynovo.models.instanovo.instanovo_dataloader import collate_batch
# from pynovo.models.instanovo.instanovo_modeling.transformer.dataset import SpectrumDataset
from pynovo.models.instanovo.instanovo_modeling.transformer.model import InstaNovo
from pynovo.models.instanovo.instanovo_modeling.utils.metrics import Metrics
import pandas as pd
from pynovo.metrics import evaluate
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_preds(
    df,
    model: InstaNovo,
    config: dict[str, Any],
    knapsack_path: str | None = None,
    device: str = "cuda",
) -> None:

    vocab = list(config["residues"].keys())
    config["vocab"] = vocab
    s2i = {v: i for i, v in enumerate(vocab)}
    i2s = {i: v for i, v in enumerate(vocab)}



    dl = InstanovoDataModule(
        df = df,
        s2i = s2i,
        return_str = True,
        batch_size = config["predict_batch_size"],
        n_workers = config["n_workers"]
    ).get_dataloader()

    model = model.to(device)
    model = model.eval()

    # setup decoder
    if knapsack_path is None or not os.path.exists(knapsack_path):
        logging.info("Knapsack path missing or not specified, generating...")
        knapsack = _setup_knapsack(model)
        decoder = KnapsackBeamSearchDecoder(model, knapsack)
        if knapsack_path is not None:
            logging.info(f"Saving knapsack to {knapsack_path}")
            knapsack.save(knapsack_path)
    else:
        logging.info("Knapsack path found. Loading...")
        decoder = KnapsackBeamSearchDecoder.from_file(model=model, path=knapsack_path)


    pred_df = {}
    preds = []
    targs = []
    probs = []

    start = time.time()
    for _, batch in tqdm(enumerate(dl), total=len(dl)):
        spectra, precursors, spectra_mask, peptides, _ = batch
        spectra = spectra.to(device)
        precursors = precursors.to(device)
        spectra_mask = spectra_mask.to(device)
        

        with torch.no_grad():
            p = decoder.decode(
                spectra=spectra,
                precursors=precursors,
                beam_size=config["n_beams"],
                max_length=config["max_length"],
            )

            preds += ["".join(x.sequence) if not isinstance(x, list) else "" for x in p]
            probs += [x.log_probability if not isinstance(x, list) else -1 for x in p]
            targs += list(peptides)
    
    delta = time.time() - start

    logging.info(f"Time taken  is {delta:.1f} seconds")
    logging.info(
        f"Average time per batch (bs={config['predict_batch_size']}): {delta/len(dl):.1f} seconds"
    )


    pred_df["targets"] = targs
    pred_df["preds"] = preds
    pred_df["probs"] = np.exp(probs)

    pred_df = pd.DataFrame(pred_df)


    metrics_dict = evaluate.aa_match_metrics(
        *evaluate.aa_match_batch(
            pred_df["targets"],
            pred_df["preds"],
            config["residues"]),
            pred_df["probs"]
        )
    
    print(metrics_dict)





def _setup_knapsack(model: InstaNovo) -> Knapsack:
    residue_masses = model.peptide_mass_calculator.masses
    residue_masses["$"] = 0
    residue_indices = model.decoder._aa2idx
    return Knapsack.construct_knapsack(
        residue_masses=residue_masses,
        residue_indices=residue_indices,
        max_mass=4000.00,
        mass_scale=MASS_SCALE,
    )


