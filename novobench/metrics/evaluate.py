"""Methods to evaluate peptide-spectrum predictions."""
import re
from typing import Dict, Iterable, List, Tuple

import numpy as np
from spectrum_utils.utils import mass_diff
from pyteomics import proforma, mass, mgf
ALL_IONS_TYPES = ('b','b-NH3','b-H2O','y','y-NH3','y-H2O')
ERROR = 0.5
MAX_CHARGE = 1
STD_AA_MASS = {
    'G': 57.02146372057,
    'A': 71.03711378471,
    'S': 87.03202840427001,
    'P': 97.05276384885,
    'V': 99.06841391299,
    'T': 101.04767846841,
    'C': 103.00918478471,
    'L': 113.08406397713001,
    'I': 113.08406397713001,
    'J': 113.08406397713001,
    'N': 114.04292744114001,
    'D': 115.02694302383001,
    'Q': 128.05857750527997,
    'K': 128.09496301399997,
    'E': 129.04259308796998,
    'M': 131.04048491299,
    'H': 137.05891185845002,
    'F': 147.06841391298997,
    'U': 150.95363508471,
    'R': 156.10111102359997,
    'Y': 163.06332853254997,
    'W': 186.07931294985997,
    'O': 237.14772686284996}

def split_peptide(peptide, aa_dict):
    aa_list = aa_dict.keys()
    regex_pattern = '|'.join(map(re.escape, sorted(aa_list, key=len, reverse=True)))

    # 使用正则表达式分割字符串
    parts = re.findall(regex_pattern, peptide)
    # 移除空字符串
    parts = [part for part in parts if part]

    if ''.join(parts) != peptide:
        raise ValueError(f"Input string '{peptide}' contains unmatched characters in aa_dict: {aa_list}.")
    else:
        return parts

def aa_match_prefix(
    peptide1: List[str],
    peptide2: List[str],
    aa_dict: Dict[str, float],
    ptm_list: List[str],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
) -> Tuple[np.ndarray, bool]:
    """
    Find the matching prefix amino acids between two peptide sequences.

    This is a similar evaluation criterion as used by DeepNovo.

    Parameters
    ----------
    peptide1 : List[str]
        The first tokenized peptide sequence to be compared.
    peptide2 : List[str]
        The second tokenized peptide sequence to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.

    Returns
    -------
    aa_matches : np.ndarray of length max(len(peptide1), len(peptide2))
        Boolean flag indicating whether each paired-up amino acid matches across
        both peptide sequences.
    pep_match : bool
        Boolean flag to indicate whether the two peptide sequences fully match.
    """
    aa_matches = np.zeros(max(len(peptide1), len(peptide2)), np.bool_)
    ptm_matches_1 = np.zeros(max(len(peptide1), len(peptide2)), np.bool_)
    ptm_matches_2 = np.zeros(max(len(peptide1), len(peptide2)), np.bool_)
    # Find longest mass-matching prefix.
    i1, i2, cum_mass1, cum_mass2 = 0, 0, 0.0, 0.0
    while i1 < len(peptide1) and i2 < len(peptide2):
        aa_mass1 = aa_dict.get(peptide1[i1], 0)
        aa_mass2 = aa_dict.get(peptide2[i2], 0)
        if (
            abs(mass_diff(cum_mass1 + aa_mass1, cum_mass2 + aa_mass2, True))
            < cum_mass_threshold
        ):
            idx = max(i1, i2)
            aa_matches[idx] = (
                abs(mass_diff(aa_mass1, aa_mass2, True)) < ind_mass_threshold
            )
            if aa_matches[idx]:
                ptm_matches_1[idx] = (peptide1[i1] in ptm_list)
                ptm_matches_2[idx] = (peptide2[i2] in ptm_list)
            i1, i2 = i1 + 1, i2 + 1
            cum_mass1, cum_mass2 = cum_mass1 + aa_mass1, cum_mass2 + aa_mass2
        elif cum_mass2 + aa_mass2 > cum_mass1 + aa_mass1:
            i1, cum_mass1 = i1 + 1, cum_mass1 + aa_mass1
        else:
            i2, cum_mass2 = i2 + 1, cum_mass2 + aa_mass2
    return aa_matches, ptm_matches_1, ptm_matches_2, aa_matches.all()


def aa_match(
    peptide1: List[str],
    peptide2: List[str],
    aa_dict: Dict[str, float],
    ptm_list: List[str],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
    mode: str = "best",
) -> Tuple[np.ndarray, bool]:
    """
    Find the matching amino acids between two peptide sequences.

    Parameters
    ----------
    peptide1 : List[str]
        The first tokenized peptide sequence to be compared.
    peptide2 : List[str]
        The second tokenized peptide sequence to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.
    mode : {"best", "forward", "backward"}
        The direction in which to find matching amino acids.

    Returns
    -------
    aa_matches : np.ndarray of length max(len(peptide1), len(peptide2))
        Boolean flag indicating whether each paired-up amino acid matches across
        both peptide sequences.
    pep_match : bool
        Boolean flag to indicate whether the two peptide sequences fully match.
    """
    # Find longest mass-matching prefix.
    aa_matches, ptm_matches_1, ptm_matches_2, pep_match = aa_match_prefix(
        peptide1, peptide2, aa_dict, ptm_list, cum_mass_threshold, ind_mass_threshold
    )
    
    # No need to evaluate the suffixes if the sequences already fully match.
    if pep_match:
        return aa_matches, pep_match, ptm_matches_1, ptm_matches_2
    # Find longest mass-matching suffix.
    i1, i2 = len(peptide1) - 1, len(peptide2) - 1
    i_stop = np.argwhere(~aa_matches)[0]
    cum_mass1, cum_mass2 = 0.0, 0.0
    while i1 >= i_stop and i2 >= i_stop:
        aa_mass1 = aa_dict.get(peptide1[i1], 0)
        aa_mass2 = aa_dict.get(peptide2[i2], 0)
        if (
            abs(mass_diff(cum_mass1 + aa_mass1, cum_mass2 + aa_mass2, True))
            < cum_mass_threshold
        ):
            idx = max(i1, i2)
            aa_matches[idx] = (
                abs(mass_diff(aa_mass1, aa_mass2, True)) < ind_mass_threshold
            )
            if aa_matches[idx]:
                ptm_matches_1[idx] = (peptide1[i1] in ptm_list)
                ptm_matches_2[idx] = (peptide2[i2] in ptm_list)
            i1, i2 = i1 - 1, i2 - 1
            cum_mass1, cum_mass2 = cum_mass1 + aa_mass1, cum_mass2 + aa_mass2
        elif cum_mass2 + aa_mass2 > cum_mass1 + aa_mass1:
            i1, cum_mass1 = i1 - 1, cum_mass1 + aa_mass1
        else:
            i2, cum_mass2 = i2 - 1, cum_mass2 + aa_mass2

            
    return aa_matches, aa_matches.all(), ptm_matches_1, ptm_matches_2


def aa_match_batch(
    peptides1: Iterable,
    peptides2: Iterable,
    aa_dict: Dict[str, float],
    ptm_list = ['M(+15.99)','N(+.98)','Q(+.98)'],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
    mode: str = "best",
) -> Tuple[List[Tuple[np.ndarray, bool]], int, int]:
    """
    Find the matching amino acids between multiple pairs of peptide sequences.

    Parameters
    ----------
    peptides1 : Iterable
        The first list of peptide sequences to be compared.
    peptides2 : Iterable
        The second list of peptide sequences to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.
    mode : {"best", "forward", "backward"}
        The direction in which to find matching amino acids.

    Returns
    -------
    aa_matches_batch : List[Tuple[np.ndarray, bool]]
        For each pair of peptide sequences: (i) boolean flags indicating whether
        each paired-up amino acid matches across both peptide sequences, (ii)
        boolean flag to indicate whether the two peptide sequences fully match.
    n_aa1: int
        Total number of amino acids in the first list of peptide sequences.
    n_aa2: int
        Total number of amino acids in the second list of peptide sequences.
    """
    aa_matches_batch, n_aa1, n_aa2 = [], 0, 0
    n_ptm_1, n_ptm_2 = 0, 0
    length_list = [len(split_peptide(peptide, aa_dict)) for peptide in peptides1]
    
    for peptide1, peptide2 in zip(peptides1, peptides2):
        # Split peptides into individual AAs if necessary.
        if isinstance(peptide1, str):
            peptide1 = split_peptide(peptide1, aa_dict)
        if isinstance(peptide2, str):
            peptide2 = split_peptide(peptide2, aa_dict)
        n_aa1, n_aa2 = n_aa1 + len(peptide1), n_aa2 + len(peptide2)

        ptm_mask_1 = [1 if aa in ptm_list else 0 for aa in peptide1]
        ptm_mask_2 = [1 if aa in ptm_list else 0 for aa in peptide2]
        n_ptm_1 += sum(ptm_mask_1)
        n_ptm_2 += sum(ptm_mask_2)

        aa_matches_batch.append(    # List[aa_matches, pep_matches, ptm_matches_1, ptm_matches_2]
            aa_match(
                peptide1,
                peptide2,
                aa_dict,
                ptm_list,
                cum_mass_threshold,
                ind_mass_threshold,
                mode,
            )
        )
    return aa_matches_batch, n_aa1, n_aa2, n_ptm_1, n_ptm_2, length_list

def prec_by_length(pep_match_bool_list, length_list):
    
    correct_counts = {}
    total_counts = {}

    # 统计每种长度的正确匹配数和总数
    for is_correct, length in zip(pep_match_bool_list, length_list):
        if length in total_counts:
            total_counts[length] += 1
        else:
            total_counts[length] = 1

        if is_correct:
            if length in correct_counts:
                correct_counts[length] += 1
            else:
                correct_counts[length] = 1

    # 计算准确率
    accuracy_dict = {}
    for length in total_counts:
        # 如果某个长度没有正确匹配的记录，则准确率为0
        correct_count = correct_counts.get(length, 0)
        accuracy_dict[length] = correct_count / total_counts[length]

    return accuracy_dict

def aa_match_metrics(
    aa_matches_batch: List[Tuple[np.ndarray, bool]],
    n_aa_true: int,
    n_aa_pred: int,
    n_ptm_true: int,
    n_ptm_pred: int,
    length_list: List[int],
) -> Tuple[float, float, float]:
    """
    Calculate amino acid and peptide-level evaluation metrics.

    Parameters
    ----------
    aa_matches_batch : List[Tuple[np.ndarray, bool]]
        For each pair of peptide sequences: (i) boolean flags indicating whether
        each paired-up amino acid matches across both peptide sequences, (ii)
        boolean flag to indicate whether the two peptide sequences fully match.
    n_aa_true: int
        Total number of amino acids in the true peptide sequences.
    n_aa_pred: int
        Total number of amino acids in the predicted peptide sequences.

    Returns
    -------
    aa_precision: float
        The number of correct AA predictions divided by the number of predicted
        AAs.
    aa_recall: float
        The number of correct AA predictions divided by the number of true AAs.
    pep_precision: float
        The number of correct peptide predictions divided by the number of
        peptides.
    """
    # aa_matches_batch: List[aa_matches, pep_matches, ptm_matches_1, ptm_matches_2]
    n_aa_correct = sum(
        [aa_matches[0].sum() for aa_matches in aa_matches_batch]
    )
    aa_precision = n_aa_correct / (n_aa_pred + 1e-8)
    aa_recall = n_aa_correct / (n_aa_true + 1e-8)
    pep_precision = sum([aa_matches[1] for aa_matches in aa_matches_batch]) / (
        len(aa_matches_batch) + 1e-8
    )

    ptm_recall = sum([aa_matches[2].sum() for aa_matches in aa_matches_batch]) / (n_ptm_true + 1e-8)
    ptm_precision = sum([aa_matches[3].sum() for aa_matches in aa_matches_batch]) / (n_ptm_pred + 1e-8)

    pep_match_bool_list = [aa_matches[1] for aa_matches in aa_matches_batch]
    prec_by_len = prec_by_length(pep_match_bool_list, length_list)  # Dict[int, float]
    
    return aa_precision, aa_recall, pep_precision, ptm_recall, ptm_precision


def aa_precision_recall(
    aa_scores_correct: List[float],
    aa_scores_all: List[float],
    n_aa_total: int,
    threshold: float,
) -> Tuple[float, float]:
    """
    Calculate amino acid level precision and recall at a given score threshold.

    Parameters
    ----------
    aa_scores_correct : List[float]
        Amino acids scores for the correct amino acids predictions.
    aa_scores_all : List[float]
        Amino acid scores for all amino acids predictions.
    n_aa_total : int
        The total number of amino acids in the predicted peptide sequences.
    threshold : float
        The amino acid score threshold.

    Returns
    -------
    aa_precision: float
        The number of correct amino acid predictions divided by the number of
        predicted amino acids.
    aa_recall: float
        The number of correct amino acid predictions divided by the total number
        of amino acids.
    """
    n_aa_correct = sum([score > threshold for score in aa_scores_correct])
    n_aa_predicted = sum([score > threshold for score in aa_scores_all])
    return n_aa_correct / n_aa_predicted, n_aa_correct / n_aa_total

# (AUC-ROC) Calculate area under curve of precision-recall.
def calc_auc(pred_file, model_name, png_name, aa_dict):
    # `psm_sequences` is assumed to be a DataFrame with at least the following
    # three columns:
    #   - "sequence": The ground-truth peptide labels.
    #   - "sequence_pred": The predicted peptide labels.
    #   - "score": The prediction scores.
    psm_sequences = pd.read_csv(pred_file)  # TODO: Get the PSM information.

    # Sort the PSMs by descreasing prediction score.
    psm_sequences = psm_sequences.sort_values(
        "score", ascending=False
    )
    # Find matches between the true and predicted peptide sequences.
    aa_matches_batch = aa_match_batch(
        psm_sequences["sequence"],
        psm_sequences["sequence_pred"],
        aa_dict,
    ) 
    # aa_matches_batch[0]: List[aa_matches, aa_matches.all(), ptm_matches_1, ptm_matches_2]

    # Calculate the peptide precision and coverage.
    peptide_matches_bool = np.asarray([aa_match[1] for aa_match in aa_matches_batch[0]])
    precision = np.cumsum(peptide_matches_bool) / np.arange(1, len(peptide_matches_bool) + 1)
    recall = np.cumsum(peptide_matches_bool) / len(peptide_matches_bool)

    # Some results
    # print(f"Peptide precision @ coverage=1 = {precision[-1]:.6f}")
    # print(f"Peptide recall    @ coverage=1 = {recall[-1]:.6f}")

    # Plot the precision–coverage curve.
    # width = 4
    # height = width / 1.618
    # fig, ax = plt.subplots(figsize=(width, width))

    # ax.plot(
    #     recall, precision, label=f"{model_name} AUC = {auc(recall, precision):.3f}"
    # )

    # ax.set_xlim(0, 1)
    # ax.set_ylim(0.30, 1)

    # ax.set_xlabel("Recall")
    # ax.set_ylabel("Precision")
    # ax.legend(loc="lower left")

    # plt.savefig(f"/chenshaorong/pynovo/{png_name}.png", dpi=300, bbox_inches="tight")
    # plt.savefig(f"/chenshaorong/pynovo/{png_name}.pdf", dpi=300, bbox_inches="tight")
    # plt.close()
    
    return auc(recall, precision)


def fragments(peptide, types=ALL_IONS_TYPES, maxcharge=MAX_CHARGE, aa_mass_list=STD_AA_MASS):
    """
    The function generates all possible m/z for fragments of types
    `types` and of charges from 1 to `maxcharge`.
    """
    all_mz = []
    for i in range(1, len(peptide)):
        site_mz = []
        for ion_type in types:
            for charge in range(1, maxcharge+1):
                if ion_type[0] in 'abc':
                    site_mz.append( mass.fast_mass(peptide[:i], ion_type=ion_type, charge=charge, aa_mass = aa_mass_list) )
                else:
                    site_mz.append( mass.fast_mass(peptide[i:], ion_type=ion_type, charge=charge, aa_mass = aa_mass_list) )
        all_mz.append(site_mz)
                    
    return np.array(all_mz)


def calc_missing_ratio(peptide:List[str], spectra_mz_list:List[float], aa_mass_list:Dict[str,float]=STD_AA_MASS ):
    fragment_list = fragments(peptide, aa_mass_list=aa_mass_list)  # [cleave site num * (ion_type * charge_type)]
    missing_num = 0
    
    for theory_mz_list in fragment_list:    # theory_mz_list [(ion_type * charge_type)]
        match_bool = False
        for theory_mz in theory_mz_list:
            if any(abs(spectra_mz - theory_mz) <= ERROR for spectra_mz in spectra_mz_list):
                match_bool = True
                break;
            else:
                continue;
        if not match_bool:
            # print(f"Missing! theory_mz_list: {theory_mz_list}")
            missing_num += 1
    
    # print(f"Theory peak: {len(fragment_list)}, Missing peak: {missing_num}")
    return missing_num/len(fragment_list)

def calc_noise_signal_ratio(peptide:List[str], spectra_mz_list:List[float], aa_mass_list:Dict[str,float]=STD_AA_MASS ):
    theory_mz_list_1 = fragments(peptide, aa_mass_list=aa_mass_list)  # [cleave site num , (ion_type * charge_type)]
    theory_mz_list = [item for sublist in theory_mz_list_1 for item in sublist]
    print(f"theory_mz_list_1 shape {theory_mz_list_1.shape}, theory_mz_list shape {len(theory_mz_list)}")
    
    bool_matrix = []
    for theory_mz in theory_mz_list:    # theory_mz_list [(ion_type * charge_type)]
        bool_mask = np.isclose(np.array(spectra_mz_list), theory_mz, atol=ERROR)
        bool_matrix.append(bool_mask)
    
    bool_matrix = np.array(bool_matrix)        
    signal_peak_bool = np.any(bool_matrix, axis=0)
    signal_peak_num = np.sum(signal_peak_bool)
    print(f"signal_peak_num: {signal_peak_num}")
    
    return (signal_peak_bool.shape[0]-signal_peak_num) / signal_peak_num
