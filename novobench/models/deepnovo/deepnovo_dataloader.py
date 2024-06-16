from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import csv
import re
import sys
import logging
from dataclasses import dataclass
from . import deepnovo_config
from novobench.data import SpectrumData
from torch.utils.data import Dataset, DataLoader


logger = logging.getLogger('deepnovo')


mass_ID_np = deepnovo_config.mass_ID_np
GO_ID = deepnovo_config.GO_ID
EOS_ID = deepnovo_config.EOS_ID
mass_H2O = deepnovo_config.mass_H2O
mass_NH3 = deepnovo_config.mass_NH3
mass_H = deepnovo_config.mass_H
SPECTRUM_RESOLUTION = deepnovo_config.SPECTRUM_RESOLUTION
WINDOW_SIZE = deepnovo_config.WINDOW_SIZE
vocab_size = deepnovo_config.vocab_size
num_ion = deepnovo_config.num_ion
MZ_SIZE = deepnovo_config.MZ_SIZE


def copy_values(candidate_intensity_view, spectrum_view,  location_sub, i1, i2):
  for j in range(WINDOW_SIZE):
    candidate_intensity_view[i2, i1, j] = spectrum_view[location_sub[i1, i2] + j]


def get_location(peptide_mass, prefix_mass, direction):
  if direction == 0:
    candidate_b_mass = prefix_mass + mass_ID_np
    candidate_y_mass = peptide_mass - candidate_b_mass
  elif direction == 1:
    candidate_y_mass = prefix_mass + mass_ID_np
    candidate_b_mass = peptide_mass - candidate_y_mass
  
  # b-ions
  candidate_b_H2O = candidate_b_mass - mass_H2O
  candidate_b_NH3 = candidate_b_mass - mass_NH3
  candidate_b_plus2_charge1 = ((candidate_b_mass + 2 * mass_H) / 2
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
  ion_mass_list = b_ions + y_ions
  ion_mass = np.array(ion_mass_list, dtype=np.float32)

  # ion locations
  location_sub50 = np.rint(ion_mass * SPECTRUM_RESOLUTION).astype(np.int32) # TODO(nh2tran): line-too-long
  # location_sub50 = np.int32(ion_mass * SPECTRUM_RESOLUTION)
  location_sub50 -= (WINDOW_SIZE // 2)
  location_plus50 = location_sub50 + WINDOW_SIZE
  ion_id_rows, aa_id_cols = np.nonzero(np.logical_and(
      location_sub50 >= 0,
      location_plus50 <= MZ_SIZE))
  return ion_id_rows, aa_id_cols, location_sub50, location_plus50


def get_candidate_intensity(spectrum_original, peptide_mass, prefix_mass, direction):
  """TODO(nh2tran): docstring."""
  ion_id_rows, aa_id_cols, location_sub50, location_plus50 = get_location(peptide_mass, prefix_mass, direction)
  # candidate_intensity
  candidate_intensity = np.zeros(shape=(vocab_size,
                                        num_ion,
                                        WINDOW_SIZE),
                                 dtype=np.float32)
  location_sub50_view = location_sub50
  location_plus50_view = location_plus50
  candidate_intensity_view = candidate_intensity
  row = ion_id_rows.astype(np.int32)
  col = aa_id_cols.astype(np.int32)
  for index in range(ion_id_rows.size):
    if col[index] < 3:
      continue
    copy_values(candidate_intensity_view, spectrum_original, location_sub50_view, row[index], col[index])

  # Nomalization to [0, 1]
  max_intensity = np.max(candidate_intensity)
  if max_intensity > 1.0:
    candidate_intensity /= max_intensity
  return candidate_intensity


def process_spectrum(spectrum_mz_list, spectrum_intensity_list, peptide_mass):
  """TODO(nh2tran): docstring."""

  # neutral mass, location, assuming ion charge z=1
  charge = 1.0
  spectrum_mz = np.array(spectrum_mz_list, dtype=np.float32)
  neutral_mass = spectrum_mz - charge*deepnovo_config.mass_H
  neutral_mass_location = np.rint(neutral_mass * deepnovo_config.SPECTRUM_RESOLUTION).astype(np.int32)
  neutral_mass_location_view = neutral_mass_location

  # intensity
  spectrum_intensity = np.array(spectrum_intensity_list, dtype=np.float32)
  # log-transform
#~   spectrum_intensity = np.log(spectrum_intensity)
  # find max intensity value for normalization and to assign to special locations
  spectrum_intensity_max = np.max(spectrum_intensity)
  # no normalization for each individual spectrum, we'll do it for multi-spectra
#~   norm_intensity = spectrum_intensity / spectrum_intensity_max
  norm_intensity = spectrum_intensity / (spectrum_intensity_max + 1e-6)
  norm_intensity_view = norm_intensity

  # fill spectrum holders
  spectrum_holder = np.zeros(shape=deepnovo_config.MZ_SIZE, dtype=np.float32)
  spectrum_holder_view = spectrum_holder
  # note that different peaks may fall into the same location, hence loop +=
  for index in range(neutral_mass_location.size):
#~     spectrum_holder_view[neutral_mass_location_view[index]] += norm_intensity_view[index]
    spectrum_holder_view[neutral_mass_location_view[index]] = max(spectrum_holder_view[neutral_mass_location_view[index]],
                                                                     norm_intensity_view[index])
  spectrum_original_forward = np.copy(spectrum_holder)
  spectrum_original_backward = np.copy(spectrum_holder)

  # add complement
  complement_mass = peptide_mass - neutral_mass
  complement_mass_location = np.rint(complement_mass * deepnovo_config.SPECTRUM_RESOLUTION).astype(np.int32) # TODO(nh2tran): line-too-long
  complement_mass_location_view = complement_mass_location
#~   cdef int index
  for index in np.nonzero(complement_mass_location > 0)[0]:
    spectrum_holder_view[complement_mass_location_view[index]] = max(norm_intensity_view[index],
                                                                     spectrum_holder_view[complement_mass_location_view[index]])

  # peptide_mass
  spectrum_original_forward[int(round(peptide_mass * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0
  spectrum_original_backward[int(round(peptide_mass * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0

  # N-terminal, b-ion, peptide_mass_C
  # append N-terminal
  mass_N = deepnovo_config.mass_N_terminus - deepnovo_config.mass_H
  spectrum_holder[int(round(mass_N * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0
  # append peptide_mass_C
  mass_C = deepnovo_config.mass_C_terminus + deepnovo_config.mass_H
  peptide_mass_C = peptide_mass - mass_C
  spectrum_holder[int(round(peptide_mass_C * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0
  spectrum_original_forward[int(round(peptide_mass_C * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0

  # C-terminal, y-ion, peptide_mass_N
  # append C-terminal
  mass_C = deepnovo_config.mass_C_terminus + deepnovo_config.mass_H
  spectrum_holder[int(round(mass_C * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0
  # append peptide_mass_N
  mass_N = deepnovo_config.mass_N_terminus - deepnovo_config.mass_H
  peptide_mass_N = peptide_mass - mass_N
  spectrum_holder[int(round(peptide_mass_N * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0
  spectrum_original_backward[int(round(peptide_mass_N * deepnovo_config.SPECTRUM_RESOLUTION))] = 1.0

  return spectrum_holder, spectrum_original_forward, spectrum_original_backward



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
                peptide[-1] = 'S(Phosphorylation)'
                index += 8
            elif peptide[-1] == 'T' and raw_sequence[index:index + 8] == "(+79.97)":
                peptide[-1] = 'T(Phosphorylation)'
                index += 8
            elif peptide[-1] == 'Y' and raw_sequence[index:index + 8] == "(+79.97)":
                peptide[-1] = 'Y(Phosphorylation)'
                index += 8
            else:  # unknown modification
                logger.warning(f"unknown modification in seq {raw_sequence}")
                return False, peptide
        else:
            peptide.append(raw_sequence[index])
            index += 1
    for aa in peptide:
        if aa not in deepnovo_config.vocab:
            logger.warning(f"unknown modification in seq {raw_sequence}")
            return False, peptide

    return True, peptide


def to_tensor(data_dict: dict) -> dict:
    temp = [(k, torch.from_numpy(v)) for k, v in data_dict.items()]
    return dict(temp)


def pad_to_length(input_data: list, pad_token, max_length: int) -> list:
    assert len(input_data) <= max_length
    result = input_data[:]
    for i in range(max_length - len(result)):
        result.append(pad_token)
    return result


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
class DenovoData:
    spectrum_holder: np.ndarray
    spectrum_original_forward: np.ndarray
    spectrum_original_backward: np.ndarray
    original_dda_feature: DDAFeature


@dataclass
class TrainData:
    spectrum_holder: np.ndarray
    forward_id_input: list
    forward_id_target: list
    backward_id_input: list
    backward_id_target: list
    forward_candidate_intensity: list
    backward_candidate_intensity: list


class DeepNovoTrainDataset(Dataset):
    def __init__(self,data: SpectrumData):
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
            mass = (float(line['precursor_mz'][0]) - deepnovo_config.mass_H) * float(line['precursor_charge'][0])
            # skip the feature if it contains unknown modification
            if not ptm:
                continue
            if mass > deepnovo_config.MZ_MAX:
                continue
            if len(peptide) >= deepnovo_config.MAX_LEN:
                continue
            new_feature = DDAFeature(feature_id=i,
                                     mz=float(line['precursor_mz'][0]),
                                     z=float(line['precursor_charge'][0]),
                                     peptide=peptide,
                                     mass = (float(line['precursor_mz'][0]) - deepnovo_config.mass_H) * float(line['precursor_charge'][0]),
                                     mz_array = list(line['mz_array'][0]),
                                     intensity_array = list(line['intensity_array'][0]))
            self.feature_list.append(new_feature)



    def __len__(self):
        return len(self.feature_list)

    def _get_feature(self, feature: DDAFeature) -> TrainData:
        mz_list, intensity_list = feature.mz_array, feature.intensity_array


        spectrum_holder, \
        spectrum_original_forward, \
        spectrum_original_backward = process_spectrum(mz_list, intensity_list, feature.mass)

        assert np.max(spectrum_holder) < 1.0 + 1e-5

        peptide_id_list = [deepnovo_config.vocab[x] for x in feature.peptide]
        forward_id_input = [deepnovo_config.GO_ID] + peptide_id_list
        forward_id_target = peptide_id_list + [deepnovo_config.EOS_ID]
        candidate_intensity_forward = []
        prefix_mass = 0.
        for i, id in enumerate(forward_id_input):
            prefix_mass += deepnovo_config.mass_ID[id]
            candidate_intensity = get_candidate_intensity(spectrum_original_forward, feature.mass, prefix_mass, 0)
            candidate_intensity_forward.append(candidate_intensity)
        backward_id_input = [deepnovo_config.EOS_ID] + peptide_id_list[::-1]
        backward_id_target = peptide_id_list[::-1] + [deepnovo_config.GO_ID]
        candidate_intensity_backward = []
        suffix_mass = 0
        for i, id in enumerate(backward_id_input):
            suffix_mass += deepnovo_config.mass_ID[id]
            candidate_intensity = get_candidate_intensity(spectrum_original_backward, feature.mass, suffix_mass, 1)
            candidate_intensity_backward.append(candidate_intensity)
        assert len(candidate_intensity_backward) == len(candidate_intensity_forward) == len(forward_id_target) == len(backward_id_target), \
            f"{len(candidate_intensity_backward)} {len(candidate_intensity_forward)} {len(forward_id_target)} {len(backward_id_target)}"
        return TrainData(spectrum_holder=spectrum_holder,
                         forward_id_input=forward_id_input,
                         forward_id_target=forward_id_target,
                         backward_id_input=backward_id_input,
                         backward_id_target=backward_id_target,
                         forward_candidate_intensity=candidate_intensity_forward,
                         backward_candidate_intensity=candidate_intensity_backward)

    def __getitem__(self, idx):
        feature = self.feature_list[idx]
        return self._get_feature(feature)
    



def collate_func(train_data_list):
    """

    :param train_data_list: list of TrainData
    :return:
    """
    # sort data by seq length (decreasing order)
    train_data_list.sort(key=lambda x: len(x.forward_id_input), reverse=True)
    batch_max_seq_len = len(train_data_list[0].forward_id_input)
    intensity_shape = train_data_list[0].forward_candidate_intensity[0].shape
    spectrum_holder = [x.spectrum_holder for x in train_data_list]
    spectrum_holder = np.stack(spectrum_holder) # [batch_size, mz_size]
    spectrum_holder = torch.from_numpy(spectrum_holder)

    batch_forward_intensity = []
    batch_forward_id_input = []
    batch_forward_id_target = []
    for data in train_data_list:
        f_intensity = np.zeros((batch_max_seq_len, intensity_shape[0], intensity_shape[1], intensity_shape[2]),
                               np.float32)
        forward_intensity = np.stack(data.forward_candidate_intensity)
        f_intensity[:forward_intensity.shape[0], :, :, :] = forward_intensity
        batch_forward_intensity.append(f_intensity)

        f_input = np.zeros((batch_max_seq_len,), np.int64)
        f_target = np.zeros((batch_max_seq_len,), np.int64)
        forward_input = np.array(data.forward_id_input, np.int64)
        f_input[:forward_input.shape[0]] = forward_input
        forward_target = np.array(data.forward_id_target, np.int64)
        f_target[:forward_target.shape[0]] = forward_target
        batch_forward_id_input.append(f_input)
        batch_forward_id_target.append(f_target)

    batch_forward_intensity = torch.from_numpy(np.stack(batch_forward_intensity))  # [batch_size, batch_max_seq_len, 26, 8, 10]
    batch_forward_id_input = torch.from_numpy(np.stack(batch_forward_id_input))  # [batch_size, batch_max_seq_len]
    batch_forward_id_target = torch.from_numpy(np.stack(batch_forward_id_target))  # [batch_size, batch_max_seq_len]

    batch_backward_intensity = []
    batch_backward_id_input = []
    batch_backward_id_target = []
    for data in train_data_list:
        b_intensity = np.zeros((batch_max_seq_len, intensity_shape[0], intensity_shape[1], intensity_shape[2]),
                               np.float32)
        backward_intensity = np.stack(data.backward_candidate_intensity)
        b_intensity[:backward_intensity.shape[0], :, :, :] = backward_intensity
        batch_backward_intensity.append(b_intensity)

        b_input = np.zeros((batch_max_seq_len,), np.int64)
        b_target = np.zeros((batch_max_seq_len,), np.int64)
        backward_input = np.array(data.backward_id_input, np.int64)
        b_input[:backward_input.shape[0]] = backward_input
        backward_target = np.array(data.backward_id_target, np.int64)
        b_target[:backward_target.shape[0]] = backward_target
        batch_backward_id_input.append(b_input)
        batch_backward_id_target.append(b_target)

    batch_backward_intensity = torch.from_numpy(
        np.stack(batch_backward_intensity))  # [batch_size, batch_max_seq_len, 26, 8, 10]
    batch_backward_id_input = torch.from_numpy(np.stack(batch_backward_id_input))  # [batch_size, batch_max_seq_len]
    batch_backward_id_target = torch.from_numpy(np.stack(batch_backward_id_target))  # [batch_size, batch_max_seq_len]

    return (spectrum_holder,
            batch_forward_intensity,
            batch_forward_id_input,
            batch_forward_id_target,
            batch_backward_intensity,
            batch_backward_id_input,
            batch_backward_id_target)


# helper functions
def chunks(l, n: int):
    for i in range(0, len(l), n):
        yield l[i:i + n]



class DeepnovoDataModule:
    def __init__(self, df):
        self.dataframe = df
        self.batch_size = deepnovo_config.batch_size
        self.num_workers = deepnovo_config.num_workers
        self.dataset = DeepNovoTrainDataset(df)

    def get_dataloader(self,shuffle=False):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=shuffle, pin_memory=True, collate_fn=collate_func, num_workers=self.num_workers)

class DeepNovoDenovoDataset(DeepNovoTrainDataset):
    # override _get_feature method
    def _get_feature(self, feature: DDAFeature) -> DenovoData:
        mz_list, intensity_list = feature.mz_array, feature.intensity_array
        spectrum_holder, \
        spectrum_original_forward, \
        spectrum_original_backward = process_spectrum(mz_list, intensity_list, feature.mass)

        return DenovoData(spectrum_holder=spectrum_holder,
                          spectrum_original_forward=spectrum_original_forward,
                          spectrum_original_backward=spectrum_original_backward,
                          original_dda_feature=feature)
