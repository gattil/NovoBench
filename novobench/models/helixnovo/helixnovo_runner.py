import polars as pl
import os
import torch
import sys
import numpy as np
from torch import optim
import torch.nn.functional as F
import time
import math
import logging
from novobench.transforms import SetRangeMZ, FilterIntensity, RemovePrecursorPeak, ScaleIntensity
from novobench.transforms.misc import Compose
from novobench.models.helixnovo.helixnovo_modeling.denovo.model_runner import train, evaluate


def init_logger():
    output = "./test_helixnovo.log"
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter(
        "{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : "
        "{message}",
        style="{",
    )
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(log_formatter)
    root.addHandler(console_handler)
    file_handler = logging.FileHandler(output)
    file_handler.setFormatter(log_formatter)
    root.addHandler(file_handler)

class HelixnovoRunner(object):
    """A class to run Contranovo models.

    Parameters
    ----------
    config : Config object
        The contranovo configuration.
    """

    @staticmethod
    def preprocessing_pipeline(min_mz=50.5, max_mz=2500.0, n_peaks: int = 150,
                               min_intensity: float = 0.01, remove_precursor_tol: float = 2.0,):
        transforms = [
            SetRangeMZ(min_mz, max_mz), 
            RemovePrecursorPeak(remove_precursor_tol),
            FilterIntensity(min_intensity, n_peaks),
            ScaleIntensity()
        ]
        return Compose(*transforms)
    
    def __init__(
        self,
        config=None):
        init_logger()
        self.config = config

    def train(
        self,
        train_df,
        val_df):
        train(train_df, val_df, self.config)

    def eval(
        self,
        val_df,
        model_path):
        evaluate(val_df, model_path, self.config)