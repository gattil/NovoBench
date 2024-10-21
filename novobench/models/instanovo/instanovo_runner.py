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
from novobench.models.instanovo.instanovo_modeling.transformer.train import train_instanovo
from novobench.models.instanovo.instanovo_modeling.transformer.denovo import denovo_instanovo
from typing import Optional
logger = logging.getLogger('instanovo')

def init_logger(config):
    # Set up logging
    output = config.logger_save_path
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
    # Disable dependency non-critical log messages.
    logging.getLogger("depthcharge").setLevel(logging.INFO)
    logging.getLogger("github").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

class InstanovoRunner:
    """A class to run Instanovo model.
    Parameters
    ----------
    config : Config object
        The casanovo configuration.
    model_filename : str, optional
        The model filename is required for eval and de novo modes,
        but not for training a model from scratch.
    """

    @staticmethod
    def preprocessing_pipeline(config):
        transforms = [
            SetRangeMZ(config.min_mz, config.max_mz), 
            RemovePrecursorPeak(config.remove_precursor_tol),
            FilterIntensity(config.min_intensity, config.n_peaks),
            ScaleIntensity()
        ]
        return Compose(*transforms)
    
    def __init__(
        self,
        config,
        model_filename: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> None:
        
        
        init_logger(config)
        """Initialize a ModelRunner"""
        # adapt to the instanovo config type
        self.config = config.config
        self.model_filename = model_filename
        self.output_path = output_path


    def train(self, train_df, val_df):
        """Train the model"""
        """
        train_df: pd.DataFrame
            The training data.
        val_df: pd.DataFrame
            The validation data.
        config: Config object
        model_filename: checkpoint filename
        """
        train_instanovo(train_df, val_df, self.config,self.model_filename)

    def denovo(self, test_df):
        """De novo sequencing"""
        denovo_instanovo(test_df, self.config, self.model_filename, self.output_path)
        
            