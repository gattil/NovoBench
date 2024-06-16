import os
import torch
from torch.utils.data import Dataset
import time
import numpy as np
from typing import List
import pickle
import csv
import re
import logging
from dataclasses import dataclass
from . import pointnovo_config as config
import sys
import numpy as np
from pynovo.data import SpectrumData
from torch.utils.data import Dataset, DataLoader


logger = logging.getLogger('pointnovo')

mass_ID_np = config.mass_ID_np
GO_ID = config.GO_ID
EOS_ID = config.EOS_ID
mass_H2O = config.mass_H2O
mass_NH3 = config.mass_NH3
mass_H = config.mass_H
mass_CO = config.mass_CO
vocab_size = config.vocab_size
num_ion = config.num_ion


def get_sinusoid_encoding_table(n_position, embed_size, padding_idx=0):


    def cal_angle(position, hid_idx):
        return position / np.power(config.sinusoid_base, 2 * (hid_idx // 2) / embed_size)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(embed_size)]

    sinusoid_matrix = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position + 1)], dtype=np.float32)

    sinusoid_matrix[:, 0::2] = np.sin(sinusoid_matrix[:, 0::2])  # dim 2i
    sinusoid_matrix[:, 1::2] = np.cos(sinusoid_matrix[:, 1::2])  # dim 2i+1

    sinusoid_matrix[padding_idx] = 0.
    return sinusoid_matrix

sinusoid_matrix = get_sinusoid_encoding_table(config.n_position, config.embedding_size,
                                              padding_idx=config.PAD_ID)


def get_ion_index(peptide_mass, prefix_mass, direction):
    if direction == 0:
        candidate_b_mass = prefix_mass + mass_ID_np
        candidate_y_mass = peptide_mass - candidate_b_mass
    elif direction == 1:
        candidate_y_mass = prefix_mass + mass_ID_np
        candidate_b_mass = peptide_mass - candidate_y_mass
    candidate_a_mass = candidate_b_mass - mass_CO

    # b-ions
    candidate_b_H2O = candidate_b_mass - mass_H2O
    candidate_b_NH3 = candidate_b_mass - mass_NH3
    candidate_b_plus2_charge1 = ((candidate_b_mass + 2 * mass_H) / 2
                                - mass_H)

    # a-ions
    candidate_a_H2O = candidate_a_mass - mass_H2O
    candidate_a_NH3 = candidate_a_mass - mass_NH3
    candidate_a_plus2_charge1 = ((candidate_a_mass + 2 * mass_H) / 2
                                - mass_H)

    # y-ions
    candidate_y_H2O = candidate_y_mass - mass_H2O
    candidate_y_NH3 = candidate_y_mass - mass_NH3
    candidate_y_plus2_charge1 = ((candidate_y_mass + 2 * mass_H) / 2
                                - mass_H)

    # ion_2
    #~   b_ions = [candidate_b_mass]
    #~   y_ions = [candidate_y_mass]
    #~   ion_mass_list = b_ions + y_ions

    # ion_8
    b_ions = [candidate_b_mass,
            candidate_b_H2O,
            candidate_b_NH3,
            candidate_b_plus2_charge1]
    y_ions = [candidate_y_mass,
            candidate_y_H2O,
            candidate_y_NH3,
            candidate_y_plus2_charge1]
    a_ions = [candidate_a_mass,
            candidate_a_H2O,
            candidate_a_NH3,
            candidate_a_plus2_charge1]
    ion_mass_list = b_ions + y_ions + a_ions
    ion_mass = np.array(ion_mass_list, dtype=np.float32)  # 8 by 26

    # ion locations
    # ion_location = np.ceil(ion_mass * SPECTRUM_RESOLUTION).astype(np.int64) # 8 by 26

    in_bound_mask = np.logical_and(
        ion_mass > 0,
        ion_mass <= config.MZ_MAX).astype(np.float32)
    ion_location = ion_mass * in_bound_mask  # 8 by 26, out of bound index would have value 0
    return ion_location.transpose()  # 26 by 8


def pad_to_length(data: list, length, pad_token=0.):
    for i in range(length - len(data)):
        data.append(pad_token)
    return data


def process_peaks(spectrum_mz_list, spectrum_intensity_list, peptide_mass):
    charge = 1.0
    spectrum_intensity_max = np.max(spectrum_intensity_list)

    # charge 1 peptide location
    spectrum_mz_list.append(peptide_mass + charge * config.mass_H)
    spectrum_intensity_list.append(spectrum_intensity_max)

    # N-terminal, b-ion, peptide_mass_C
    # append N-terminal
    mass_N = config.mass_N_terminus - config.mass_H
    spectrum_mz_list.append(mass_N + charge * config.mass_H)
    spectrum_intensity_list.append(spectrum_intensity_max)
    # append peptide_mass_C
    mass_C = config.mass_C_terminus + config.mass_H
    peptide_mass_C = peptide_mass - mass_C
    spectrum_mz_list.append(peptide_mass_C + charge * config.mass_H)
    spectrum_intensity_list.append(spectrum_intensity_max)

    # C-terminal, y-ion, peptide_mass_N
    # append C-terminal
    mass_C = config.mass_C_terminus + config.mass_H
    spectrum_mz_list.append(mass_C + charge * config.mass_H)
    spectrum_intensity_list.append(spectrum_intensity_max)


    spectrum_mz_list = pad_to_length(spectrum_mz_list, config.MAX_NUM_PEAK)
    spectrum_intensity_list = pad_to_length(spectrum_intensity_list, config.MAX_NUM_PEAK)

    spectrum_mz = np.array(spectrum_mz_list, dtype=np.float32)
    spectrum_mz_location = np.ceil(spectrum_mz * config.spectrum_reso).astype(np.int32)

    neutral_mass = spectrum_mz - charge * config.mass_H
    in_bound_mask = np.logical_and(neutral_mass > 0., neutral_mass < config.MZ_MAX)
    neutral_mass[~in_bound_mask] = 0.
    # intensity
    spectrum_intensity = np.array(spectrum_intensity_list, dtype=np.float32)
    norm_intensity = spectrum_intensity / spectrum_intensity_max

    spectrum_representation = np.zeros(config.embedding_size, dtype=np.float32)
    for i, loc in enumerate(spectrum_mz_location):
        if loc < 0.5 or loc > config.n_position:
            continue
        else:
            spectrum_representation += sinusoid_matrix[loc] * norm_intensity[i]

    top_N_indices = np.argpartition(norm_intensity, -config.MAX_NUM_PEAK)[-config.MAX_NUM_PEAK:]
    intensity = norm_intensity[top_N_indices]
    mass_location = neutral_mass[top_N_indices]

    return mass_location, intensity, spectrum_representation


def parse_raw_sequence(raw_sequence: str):
    raw_sequence_len = len(raw_sequence)
    peptide = []
    index = 0
    while index < raw_sequence_len:
        if raw_sequence[index] == "(":
            if peptide[-1] == "C" and raw_sequence[index:index + 8] == "(+57.02)":
                peptide[-1] = "C(Carbamidomethylation)"
                index += 8
            elif peptide[-1] == 'M' and raw_sequence[index:index + 8] == "(+15.99)":
                peptide[-1] = 'M(Oxidation)'
                index += 8
            elif peptide[-1] == 'N' and raw_sequence[index:index + 6] == "(+.98)":
                peptide[-1] = 'N(Deamidation)'
                index += 6
            elif peptide[-1] == 'Q' and raw_sequence[index:index + 6] == "(+.98)":
                peptide[-1] = 'Q(Deamidation)'
                index += 6
            elif peptide[-1] == 'S' and raw_sequence[index:index + 8] == "(+79.97)":
                peptide[-1] = "S(Phosphorylation)"
                index += 8
            elif peptide[-1] == 'T' and raw_sequence[index:index + 8] == "(+79.97)":
                peptide[-1] = "T(Phosphorylation)"
                index += 8
            elif peptide[-1] == 'Y' and raw_sequence[index:index + 8] == "(+79.97)":
                peptide[-1] = "Y(Phosphorylation)"
                index += 8
            else:  # unknown modification
                logger.warning(f"unknown modification in seq {raw_sequence}")
                return False, peptide
        else:
            peptide.append(raw_sequence[index])
            index += 1

    for aa in peptide:
        if aa not in config.vocab:
            logger.warning(f"unknown modification in seq {raw_sequence}")
            return False, peptide
    return True, peptide


def to_tensor(data_dict: dict) -> dict:
    temp = [(k, torch.from_numpy(v)) for k, v in data_dict.items()]
    return dict(temp)



@dataclass
class DDAFeature:
    feature_id: int
    mz: float
    z: float
    peptide: list
    mass: float
    mz_array: list
    intensity_array: list


@dataclass
class _DenovoData:
    peak_location: np.ndarray
    peak_intensity: np.ndarray
    spectrum_representation: np.ndarray
    original_dda_feature: DDAFeature


@dataclass
class BatchDenovoData:
    peak_location: torch.Tensor
    peak_intensity: torch.Tensor
    spectrum_representation: torch.Tensor
    original_dda_feature_list: List[DDAFeature]


@dataclass
class TrainData:
    peak_location: np.ndarray
    peak_intensity: np.ndarray
    spectrum_representation: np.ndarray
    forward_id_target: list
    backward_id_target: list
    forward_ion_location_index_list: list
    backward_ion_location_index_list: list
    forward_id_input: list
    backward_id_input: list


class BaseDataset(Dataset):
    def __init__(self, data: SpectrumData):
        """
        read all feature information and store in memory,
        :param data: SpectrumData object
        """
        self.annotated = data.annotated
        self.df = data.df
        self.feature_list = []
        for i in range(len(self.df)):
            line = self.df[i]
            if self.annotated:
                peptide = line['modified_sequence'][0]
            else:
                peptide = line['sequence'][0]
            ptm, peptide = parse_raw_sequence(peptide)
            mass = (float(line['precursor_mz'][0]) - config.mass_H) * float(line['precursor_charge'][0])
            # skip the feature if it contains unknown modification
            if not ptm:
                continue
            if mass > config.MZ_MAX:
                continue
            if len(peptide) >= config.MAX_LEN - 2:
                continue
            new_feature = DDAFeature(feature_id=i,
                                     mz=float(line['precursor_mz'][0]),
                                     z=float(line['precursor_charge'][0]),
                                     peptide=peptide,
                                     mass = (float(line['precursor_mz'][0]) - config.mass_H) * float(line['precursor_charge'][0]),
                                     mz_array = list(line['mz_array'][0]),
                                     intensity_array = list(line['intensity_array'][0]))
            self.feature_list.append(new_feature)

    def __len__(self):
        return len(self.feature_list)


    def _get_feature(self, feature: DDAFeature):
        raise NotImplementedError("subclass should implement _get_feature method")

    def __getitem__(self, idx):
        feature = self.feature_list[idx]
        return self._get_feature(feature)


class DeepNovoTrainDataset(BaseDataset):
    def _get_feature(self, feature: DDAFeature) -> TrainData:
        mz_list, intensity_list = feature.mz_array, feature.intensity_array
        peak_location, peak_intensity, spectrum_representation = process_peaks(mz_list, intensity_list, feature.mass)

        assert np.max(peak_intensity) < 1.0 + 1e-5

        peptide_id_list = [config.vocab[x] for x in feature.peptide]
        forward_id_input = [config.GO_ID] + peptide_id_list
        forward_id_target = peptide_id_list + [config.EOS_ID]
        forward_ion_location_index_list = []
        prefix_mass = 0.
        for i, id in enumerate(forward_id_input):
            prefix_mass += config.mass_ID[id]
            ion_location = get_ion_index(feature.mass, prefix_mass, 0)
            forward_ion_location_index_list.append(ion_location)

        backward_id_input = [config.EOS_ID] + peptide_id_list[::-1]
        backward_id_target = peptide_id_list[::-1] + [config.GO_ID]
        backward_ion_location_index_list = []
        suffix_mass = 0
        for i, id in enumerate(backward_id_input):
            suffix_mass += config.mass_ID[id]
            ion_location = get_ion_index(feature.mass, suffix_mass, 1)
            backward_ion_location_index_list.append(ion_location)

        return TrainData(peak_location=peak_location,
                         peak_intensity=peak_intensity,
                         spectrum_representation=spectrum_representation,
                         forward_id_target=forward_id_target,
                         backward_id_target=backward_id_target,
                         forward_ion_location_index_list=forward_ion_location_index_list,
                         backward_ion_location_index_list=backward_ion_location_index_list,
                         forward_id_input=forward_id_input,
                         backward_id_input=backward_id_input)


def collate_func(train_data_list):
    """

    :param train_data_list: list of TrainData
    :return:
        peak_location: [batch, N]
        peak_intensity: [batch, N]
        forward_target_id: [batch, T]
        backward_target_id: [batch, T]
        forward_ion_index_list: [batch, T, 26, 8]
        backward_ion_index_list: [batch, T, 26, 8]
    """
    # sort data by seq length (decreasing order)
    train_data_list.sort(key=lambda x: len(x.forward_id_target), reverse=True)
    batch_max_seq_len = len(train_data_list[0].forward_id_target)
    ion_index_shape = train_data_list[0].forward_ion_location_index_list[0].shape
    assert ion_index_shape == (config.vocab_size, config.num_ion)

    peak_location = [x.peak_location for x in train_data_list]
    peak_location = np.stack(peak_location) # [batch_size, N]
    peak_location = torch.from_numpy(peak_location)

    peak_intensity = [x.peak_intensity for x in train_data_list]
    peak_intensity = np.stack(peak_intensity) # [batch_size, N]
    peak_intensity = torch.from_numpy(peak_intensity)

    spectrum_representation = [x.spectrum_representation for x in train_data_list]
    spectrum_representation = np.stack(spectrum_representation)  # [batch_size, embed_size]
    spectrum_representation = torch.from_numpy(spectrum_representation)

    batch_forward_ion_index = []
    batch_forward_id_target = []
    batch_forward_id_input = []
    for data in train_data_list:
        ion_index = np.zeros((batch_max_seq_len, ion_index_shape[0], ion_index_shape[1]),
                               np.float32)
        forward_ion_index = np.stack(data.forward_ion_location_index_list)
        ion_index[:forward_ion_index.shape[0], :, :] = forward_ion_index
        batch_forward_ion_index.append(ion_index)

        f_target = np.zeros((batch_max_seq_len,), np.int64)
        forward_target = np.array(data.forward_id_target, np.int64)
        f_target[:forward_target.shape[0]] = forward_target
        batch_forward_id_target.append(f_target)

        f_input = np.zeros((batch_max_seq_len,), np.int64)
        forward_input = np.array(data.forward_id_input, np.int64)
        f_input[:forward_input.shape[0]] = forward_input
        batch_forward_id_input.append(f_input)



    batch_forward_id_target = torch.from_numpy(np.stack(batch_forward_id_target))  # [batch_size, T]
    batch_forward_ion_index = torch.from_numpy(np.stack(batch_forward_ion_index))  # [batch, T, 26, 8]
    batch_forward_id_input = torch.from_numpy(np.stack(batch_forward_id_input))

    batch_backward_ion_index = []
    batch_backward_id_target = []
    batch_backward_id_input = []
    for data in train_data_list:
        ion_index = np.zeros((batch_max_seq_len, ion_index_shape[0], ion_index_shape[1]),
                             np.float32)
        backward_ion_index = np.stack(data.backward_ion_location_index_list)
        ion_index[:backward_ion_index.shape[0], :, :] = backward_ion_index
        batch_backward_ion_index.append(ion_index)

        b_target = np.zeros((batch_max_seq_len,), np.int64)
        backward_target = np.array(data.backward_id_target, np.int64)
        b_target[:backward_target.shape[0]] = backward_target
        batch_backward_id_target.append(b_target)

        b_input = np.zeros((batch_max_seq_len,), np.int64)
        backward_input = np.array(data.backward_id_input, np.int64)
        b_input[:backward_input.shape[0]] = backward_input
        batch_backward_id_input.append(b_input)

    batch_backward_id_target = torch.from_numpy(np.stack(batch_backward_id_target))  # [batch_size, T]
    batch_backward_ion_index = torch.from_numpy(np.stack(batch_backward_ion_index))  # [batch, T, 26, 8]
    batch_backward_id_input = torch.from_numpy(np.stack(batch_backward_id_input))

    return (peak_location,
            peak_intensity,
            spectrum_representation,
            batch_forward_id_target,
            batch_backward_id_target,
            batch_forward_ion_index,
            batch_backward_ion_index,
            batch_forward_id_input,
            batch_backward_id_input
            )


# helper functions
def chunks(l, n: int):
    for i in range(0, len(l), n):
        yield l[i:i + n]


class DeepNovoDenovoDataset(DeepNovoTrainDataset):
    # override _get_feature method
    def _get_feature(self, feature: DDAFeature) -> _DenovoData:
        mz_list, intensity_list = feature.mz_array, feature.intensity_array
        peak_location, peak_intensity, spectrum_representation = process_peaks(mz_list, intensity_list, feature.mass)

        return _DenovoData(peak_location=peak_location,
                           peak_intensity=peak_intensity,
                           spectrum_representation=spectrum_representation,
                           original_dda_feature=feature)


def denovo_collate_func(data_list: List[_DenovoData]):
    batch_peak_location = np.array([x.peak_location for x in data_list])
    batch_peak_intensity = np.array([x.peak_intensity for x in data_list])
    batch_spectrum_representation = np.array([x.spectrum_representation for x in data_list])

    batch_peak_location = torch.from_numpy(batch_peak_location)
    batch_peak_intensity = torch.from_numpy(batch_peak_intensity)
    batch_spectrum_representation = torch.from_numpy(batch_spectrum_representation)

    original_dda_feature_list = [x.original_dda_feature for x in data_list]

    return BatchDenovoData(batch_peak_location, batch_peak_intensity, batch_spectrum_representation,
                           original_dda_feature_list)

class DeepnovoDataModule:
    def __init__(self, df):
        self.dataframe = df
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.dataset = DeepNovoTrainDataset(df)

    def get_dataloader(self,shuffle=False):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=shuffle, pin_memory=True, collate_fn=collate_func, num_workers=self.num_workers)