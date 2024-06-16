from __future__ import annotations
import re
import numpy as np
import pandas as pd
import polars as pl
import spectrum_utils.spectrum as sus
import torch
from novobench.data import SpectrumData
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader



class HelixnovoDataset(Dataset):
    def __init__(self, 
            data: SpectrumData,
            ) -> None:
        super().__init__()
        self.df = data.df

    def __len__(self) -> int:
        return self.df.height

    def __getitem__(self, idx: int):
        mz_array = torch.Tensor(self.df[idx, "mz_array"].to_list())
        int_array = torch.Tensor(self.df[idx, "intensity_array"].to_list())
        precursor_mz = self.df[idx, "precursor_mz"]
        precursor_charge = self.df[idx, "precursor_charge"]
        peptide = ''
        if 'modified_sequence' in self.df.columns:
            peptide = self.df[idx, 'modified_sequence'] 
            
        # peptide = re.split(r"(?<=.)(?=[A-Z])", peptide)

        spectrum = torch.stack([mz_array, int_array], dim=1)

        return spectrum, precursor_mz, precursor_charge, peptide

class HelixnovoDataModule:
    def __init__(
        self,
        df: pl.DataFrame,
        batch_size: int = 128,
        n_workers: int = 64,
    ):
        self.dataframe = df
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.dataset = HelixnovoDataset(df)
    
    def get_dataloader(self,shuffle=False) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=True,
            collate_fn=prepare_batch,
            shuffle = shuffle
        )

def prepare_batch(
    batch: List[Tuple[torch.Tensor, float, int, str]]
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Collate MS/MS spectra into a batch.

    The MS/MS spectra will be padded so that they fit nicely as a tensor.
    However, the padded elements are ignored during the subsequent steps.

    Parameters
    ----------
    batch : List[Tuple[torch.Tensor, float, int, str]]
        A batch of data from an AnnotatedSpectrumDataset, consisting of for each
        spectrum (i) a tensor with the m/z and intensity peak values, (ii), the
        precursor m/z, (iii) the precursor charge, (iv) the spectrum identifier.

    Returns
    -------
    spectra : torch.Tensor of shape (batch_size, n_peaks, 2)
        The padded mass spectra tensor with the m/z and intensity peak values
        for each spectrum.
    precursors : torch.Tensor of shape (batch_size, 3)
        A tensor with the precursor neutral mass, precursor charge, and
        precursor m/z.
    spectrum_ids : np.ndarray
        The spectrum identifiers (during de novo sequencing) or peptide
        sequences (during training).
    """
    spectra, precursor_mzs, precursor_charges, spectrum_ids = list(zip(*batch))
    spectra = torch.nn.utils.rnn.pad_sequence(spectra, batch_first=True)
    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - 1.007276) * precursor_charges
    precursors = torch.vstack(
        [precursor_masses, precursor_charges, precursor_mzs]
    ).T.float()
    return spectra, precursors, np.asarray(spectrum_ids)

