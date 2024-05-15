import polars as pl
import os
import torch
from torch import optim
import torch.nn.functional as F
import time
import math
import logging
from pynovo.transforms import SetRangeMZ, FilterIntensity, RemovePrecursorPeak, ScaleIntensity
from pynovo.transforms.misc import Compose
from pynovo.models.instanovo.instanovo_modeling.transformer.train import train
logger = logging.getLogger('instanovo')

def init_logger():
    output = "/jingbo/PyNovo/instanovo.log"
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

class InstanovoRunner(object):
    """A class to run Instanovo models.

    Parameters
    ----------
    config : Config object
        The instanovo configuration.
    """

    # TODO
    @staticmethod
    def preprocessing_pipeline(min_mz=50, max_mz=2500.0, n_peaks: int = 150,
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
        config):
        # init_logger()
        self.config = config

    def train(
        self,
        train_df,
        val_df):

        train(train_df, val_df, self.config)
    
    def evaluate(
        self,
        model,
        test_df):

        pass

