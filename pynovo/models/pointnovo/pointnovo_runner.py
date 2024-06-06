import polars as pl
import os
import torch
from torch import optim
import torch.nn.functional as F
import time
import math
import logging
from .mode.train_func import train, build_model
from pynovo.utils.preprocessing import convert_mgf_ipc
from .pointnovo_dataloader import DeepnovoDataModule, DeepNovoDenovoDataset, denovo_collate_func
from .mode.denovo import  IonCNNDenovo
from .pointnovo_modeling import InferenceModelWrapper
from . import pointnovo_config as config
from pathlib import Path
from pynovo.data import SpectrumData
from pynovo.data import ms_io
from pynovo.metrics import evaluate
import numpy as np
from pynovo.data import ms_io
import sys
from pynovo.transforms import SetRangeMZ
from pynovo.transforms.misc import Compose
logger = logging.getLogger('pointnovo')

def init_logger():
    output = "/jingbo/PyNovo/pointnovo_seven.log"
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


class PointnovoRunner(object):
    """A class to run Pointnovo models.

    Parameters
    ----------
    config : Config object
        The pointnovo configuration.
    """
    @staticmethod
    def preprocessing_pipeline(min_mz=0.0, max_mz=config.MZ_MAX):
        transforms = [
            SetRangeMZ(min_mz, max_mz)
        ]
        return Compose(*transforms)

    def __init__(
        self):
        init_logger()

    def train(
        self,
        train_df,
        val_df):

        train_loader = DeepnovoDataModule(df = train_df).get_dataloader(shuffle=True)
        val_loader = DeepnovoDataModule(df = val_df).get_dataloader()
        train(train_loader, val_loader)
    
    def evaluate(
        self,
        valid_df,
        folder):
        dataset = DeepNovoDenovoDataset(data = valid_df)
        denovo_data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config.batch_size,
                                                         shuffle=False,
                                                         num_workers=config.num_workers,
                                                         collate_fn=denovo_collate_func)
        denovo_worker = IonCNNDenovo(config.MZ_MAX,
                                     config.knapsack_file,
                                     beam_size=config.FLAGS.beam_size)

        forward_deepnovo, backward_deepnovo, init_net = build_model(training=False,folder=folder)
        model_wrapper = InferenceModelWrapper(forward_deepnovo, backward_deepnovo, init_net)
        predict_pred, index = denovo_worker.search_denovo(model_wrapper, denovo_data_loader,'eval')
        predict_true = list(valid_df.modified_sequence)
        predict_true = list(np.array(predict_true)[np.array(index)])
        assert(len(predict_pred)==len(predict_true))
        metrics_dict = evaluate.aa_match_metrics(
            *evaluate.aa_match_batch(
                predict_true,
                predict_pred,
                dict(list(config.mass_AA.items())[3:])
            )
        )
        print(metrics_dict)
        logger.info(metrics_dict)
    
    def predict(
        self,
        peak_path,
        output_file,
        folder):

        writer = ms_io.MztabWriter(Path(output_file).with_suffix(".mztab"))
        peak_path = Path(peak_path)
        if peak_path.is_file():
            peak_path_list = [peak_path]
        else:
            peak_path_list = list(peak_path.iterdir())
        predict_df = convert_mgf_ipc(peak_path)
        predict_df = predict_df.sample(100)
        dataset = DeepNovoDenovoDataset(data = SpectrumData(predict_df))
        denovo_data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config.batch_size,
                                                         shuffle=False,
                                                         num_workers=config.num_workers,
                                                         collate_fn=denovo_collate_func)
        denovo_worker = IonCNNDenovo(config.MZ_MAX,
                                     config.knapsack_file,
                                     beam_size=config.FLAGS.beam_size)
        forward_deepnovo, backward_deepnovo, init_net = build_model(training=False,folder=folder)
        model_wrapper = InferenceModelWrapper(forward_deepnovo, backward_deepnovo, init_net)
        writer_psm = denovo_worker.search_denovo(model_wrapper, denovo_data_loader,'denovo')
        writer.psms = writer_psm
        writer.save()

