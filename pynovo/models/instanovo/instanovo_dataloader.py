import re
import numpy as np
import pandas as pd
import polars as pl
import spectrum_utils.spectrum as sus
import torch
from pynovo.data import SpectrumData
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


PROTON_MASS_AMU = 1.007276



class InstanovoDataset(Dataset):
    def __init__(self, 
            data: SpectrumData,
            s2i: dict[str, int],
            eos_symbol: str = "</s>",
            return_str: bool = False
            ) -> None:
        super().__init__()
        self.df = data.df
        self.s2i = s2i
        self.return_str = return_str

        if eos_symbol in self.s2i:
            self.EOS_ID = self.s2i[eos_symbol]
        else:
            self.EOS_ID = -1


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
        if not self.return_str:
            peptide = re.split(r"(?<=.)(?=[A-Z])", peptide)

        spectrum = torch.stack([mz_array, int_array], dim=1)

        return spectrum, precursor_mz, precursor_charge, peptide

class InstanovoDataModule:
    def __init__(
        self,
        df: pl.DataFrame,
        s2i: dict[str, int],
        eos_symbol: str = "</s>",
        return_str: bool = False,
        batch_size: int = 128,
        n_workers: int = 64,
    ):
        self.dataframe = df
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.dataset = InstanovoDataset(df,s2i,eos_symbol,return_str)
    
    def get_dataloader(self,shuffle=False) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=True,
            collate_fn=collate_batch,
            shuffle = shuffle
        )



def collate_batch(
    batch: list[tuple[Tensor, float, int, Tensor]]
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Collate batch of samples."""
    spectra, precursor_mzs, precursor_charges, peptides = zip(*batch)

    # Pad spectra
    ll = torch.tensor([x.shape[0] for x in spectra], dtype=torch.long)
    spectra = nn.utils.rnn.pad_sequence(spectra, batch_first=True)
    spectra_mask = torch.arange(spectra.shape[1], dtype=torch.long)[None, :] >= ll[:, None]

    # Pad peptide
    if isinstance(peptides[0], str):
        peptides_mask = None
    else:
        ll = torch.tensor([x.shape[0] for x in peptides], dtype=torch.long)
        peptides = nn.utils.rnn.pad_sequence(peptides, batch_first=True)
        peptides_mask = torch.arange(peptides.shape[1], dtype=torch.long)[None, :] >= ll[:, None]

    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - PROTON_MASS_AMU) * precursor_charges
    precursors = torch.vstack([precursor_masses, precursor_charges, precursor_mzs]).T.float()

    return spectra, precursors, spectra_mask, peptides, peptides_mask