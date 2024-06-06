import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import yaml
import pdb
import random
import datetime
import logging
import os
import sys

import click
import psutil
import pytorch_lightning as pl
import torch
import yaml
sys.path.append("/jingbo/PyNovo/")
import torch
from pynovo.models.contranovo.contranovo_modeling import utils
from pynovo.models.contranovo.contranovo_runner import ContranovoRunner
from pynovo.datasets import CustomDataset
from pynovo.models.helixnovo.helixnovo_runner import HelixnovoRunner

file_mapping = {
    "train" : "sample_train.parquet",
    "valid" : "sample_test.parquet",
}

def train():
    config_path = "/jingbo/PyNovo/pynovo/models/helixnovo/helixnovo_config.yaml"
    with open(config_path) as f_in:
        config = yaml.safe_load(f_in)
    config_types = dict(
        random_seed=int,
        n_peaks=int,
        min_mz=float,
        max_mz=float,
        min_intensity=float,
        remove_precursor_tol=float,
        max_charge=int,
        precursor_mass_tol=float,
        isotope_error_range=lambda min_max: (int(min_max[0]), int(min_max[1])),
        dim_model=int,
        n_head=int,
        dim_feedforward=int,
        n_layers=int,
        dropout=float,
        dim_intensity=int,
        max_length=int,
        n_log=int,
        warmup_iters=int,
        max_iters=int,
        learning_rate=float,
        weight_decay=float,
        train_batch_size=int,
        predict_batch_size=int,
        max_epochs=int,
        num_sanity_val_steps=int,
        train_from_scratch=bool,
        save_model=bool,
        model_save_folder_path=str,
        save_weights_only=bool,
        every_n_train_steps=int,
        decoding = str,
        n_beams=int,
        n_workers=int,
    )
    for k, t in config_types.items():
        try:
            if config[k] is not None:
                config[k] = t(config[k])
        except (TypeError, ValueError) as e:
            logger.error("Incorrect type for configuration value %s: %s", k, e)
            raise TypeError(f"Incorrect type for configuration value {k}: {e}")
    config["residues"] = {
        str(aa): float(mass) for aa, mass in config["residues"].items()
    }
   
    # Add extra configuration options and scale by the number of GPUs.
    #TODO
    gpu=[0]
    config["train_batch_size"] = config["train_batch_size"] // len(gpu)
    if gpu[0]==-1:
        gpu=None
    config["gpu"]=gpu
    pl.utilities.seed.seed_everything(seed=config["random_seed"], workers=True)

    dataset = CustomDataset("/jingbo/PyNovo/data", file_mapping)
    # dataset = CustomDataset("/usr/commondata/public/jingbo/seven_species/", file_mapping)

    # data = dataset.load_data()
    data = dataset.load_data(transform = ContranovoRunner.preprocessing_pipeline())
    model = HelixnovoRunner(config)
    model.train(data.get_train(), data.get_valid()) 

train()

# file_mapping = {
#     "train" : "sample_train.parquet",
#     "valid" : "sample_test.parquet",
# }


# def eval():
#     config_path = "/jingbo/PyNovo/pynovo/models/contranovo/contranovo_config.yaml"
#     with open(config_path) as f_in:
#         contranovo_config = yaml.safe_load(f_in)
#     contranovo_config["n_workers"] = utils.n_workers()
#     model = ContranovoRunner(contranovo_config)
#     model_path = '/jingbo/PyNovo/clipcasa/epoch=24-step=400-v2.ckpt'
#     dataset = CustomDataset("/jingbo/PyNovo/data", file_mapping)
#     data = dataset.load_data(transform = ContranovoRunner.preprocessing_pipeline())
#     model.eval(data.get_valid(), model_path)

# eval()





