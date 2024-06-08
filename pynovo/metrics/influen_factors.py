from typing import Dict, Iterable, List, Tuple
from pyteomics import proforma, mass, mgf
from evaluate import split_peptide
import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    'O': 237.14772686284996,
    'M(+15.99)': 147.0354,
    'Q(+.98)': 129.0426,
    'N(+.98)': 115.02695,
    'C(+57.02)': 160.03065,
}

def get_png_file_name(file_name: str, factor:str):
    file_path_without_extension, _ = os.path.splitext(file_name)
    new_file_path = file_path_without_extension + "_" + factor + ".jpg"

    if os.path.exists(new_file_path):
        base, extension = os.path.splitext(new_file_path)
        counter = 1
        new_file_path = f"{base}_{counter}{extension}"
        while os.path.exists(new_file_path):
            counter += 1
            new_file_path = f"{base}_{counter}{extension}"
    
    return new_file_path

def fragments(
    peptides:List[List[str]], 
    types=ALL_IONS_TYPES, 
    maxcharge=MAX_CHARGE, 
    aa_mass_list=STD_AA_MASS
):
    """
    The function generates all possible m/z for fragments of types
    `types` and of charges from 1 to `maxcharge`.
    """
    all_theory_mz = []
    for peptide in peptides:
        theory_mz = []
        for i in range(1, len(peptide)):
            site_mz = []
            for ion_type in types:
                for charge in range(1, maxcharge+1):
                    if ion_type[0] in 'abc':
                        site_mz.append( mass.fast_mass(peptide[:i], ion_type=ion_type, charge=charge, aa_mass = aa_mass_list) )
                    else:
                        site_mz.append( mass.fast_mass(peptide[i:], ion_type=ion_type, charge=charge, aa_mass = aa_mass_list) )
            theory_mz.append(site_mz)  # [cleave site num , (ion_type * charge_type)]
        all_theory_mz.append(theory_mz) # [peptide_num, cleave site num, (ion_type * charge_type)]
                    
    return all_theory_mz


def calc_missing_ratio(
    peptides:List[List[str]], 
    spectra_mz_list:List[List[float]], 
    aa_mass_list:Dict[str,float]=STD_AA_MASS, 
):
    fragment_lists = fragments(peptides, aa_mass_list=aa_mass_list) # [peptide_num, cleave site num, (ion_type * charge_type)]
    ratio = []
    
    for pep_idx, pep_fragment_list in enumerate(fragment_lists):    # pep_fragment_list:[cleave site num, (ion_type * charge_type)]
        missing_num = 0
        for site_theory_mz_list in pep_fragment_list:               # site_theory_mz_list:[(ion_type * charge_type)]
            match_bool = False
            for theory_mz in site_theory_mz_list:
                if any(abs(spectra_mz - theory_mz) <= ERROR for spectra_mz in spectra_mz_list[pep_idx]):
                    match_bool = True
                    break;
                else:
                    continue;
            if not match_bool:
                # print(f"Missing! site_theory_mz_list: {site_theory_mz_list}")
                missing_num += 1

        ratio.append( missing_num / len(pep_fragment_list))
    
    return ratio    # [peptide_num, 1]


def calc_noise_signal_ratio(
    peptides:List[List[str]], 
    spectra_mz_list:List[List[float]], 
    aa_mass_list:Dict[str,float]=STD_AA_MASS 
):
    
    theory_mz_list = fragments(peptides, aa_mass_list=aa_mass_list) # [peptide_num, cleave site num, (ion_type * charge_type)]
    merge_theory_mz_list = []
    # 遍历第0个维度
    for sublist in theory_mz_list:
        # 合并内层列表
        merged_sublist = [item for inner_list in sublist for item in inner_list]
        # 添加到结果列表中
        merge_theory_mz_list.append(merged_sublist)
    # merge_theory_mz_list: [peptide_num, (cleave site num * ion_type * charge_type)]

    # print(f"theory_mz_list_1 shape {theory_mz_list_1.shape}, theory_mz_list shape {len(theory_mz_list)}")
    all_nsr = []
    for pep_idx, pep_theory_mz_list in enumerate(merge_theory_mz_list):   # [(cleave site num * ion_type * charge_type)]

        pep_bool_matrix = []
        for theory_mz in pep_theory_mz_list:                        # theory_mz_list [(ion_type * charge_type)]
            bool_mask = np.isclose(np.array(spectra_mz_list[pep_idx]), theory_mz, atol=ERROR)
            pep_bool_matrix.append(bool_mask)
        
        pep_bool_matrix = np.array(pep_bool_matrix)        
        signal_peak_bool = np.any(pep_bool_matrix, axis=0)
        signal_peak_num = np.sum(signal_peak_bool)
        # print(f"signal_peak_num: {signal_peak_num}")
        all_nsr.append( (signal_peak_bool.shape[0]-signal_peak_num) / (signal_peak_num + 1e-6) )
        
    return all_nsr


def prec_by_length(
    pep_match_bool: List[bool], 
    length_list: List[int], 
    save_file_path: str,
    aa_mass_list:Dict[str,float]=STD_AA_MASS, 
):
    assert len(pep_match_bool)==len(length_list)
    # length_list = [len(split_peptide(peptide, aa_dict)) for peptide in peptides1]
    correct_counts = {}
    total_counts = {}

    # 统计每种长度的正确匹配数和总数
    for is_correct, length in zip(pep_match_bool, length_list):
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
    len_threshold = 25
    above_total_count, above_correct_count = 0, 0
    for length in total_counts:
        if length <= len_threshold:
            # 如果某个长度没有正确匹配的记录，则准确率为0
            correct_count = correct_counts.get(length, 0)
            accuracy_dict[length] = correct_count / total_counts[length]
        else:
            above_total_count += total_counts[length]
            above_correct_count += correct_counts.get(length, 0)
    
    accuracy_dict['>25'] = above_correct_count/above_total_count
    total_counts['>25'] = above_total_count

    # 对字典按键排序
    sorted_keys = sorted(accuracy_dict.keys(), key=lambda x: (x if isinstance(x, int) else float('inf')))
    sorted_accuracy = [accuracy_dict[k] for k in sorted_keys]
    sorted_data_count = [total_counts[k] for k in sorted_keys]

    # # 创建图形
    # fig, ax1 = plt.subplots(figsize=(20, 12))
    # bar_width = 0.4
    # bar_positions = range(len(sorted_keys))

    # # 正确率柱状图
    # ax1.bar(bar_positions, sorted_accuracy, bar_width, color='b', label='Accuracy')
    # ax1.set_xlabel('Length')  # , fontproperties=font
    # ax1.set_ylabel('Accuracy', color='b')  # , fontproperties=font
    # ax1.tick_params(axis='y', labelcolor='b')

    # # 创建第二个y轴
    # ax2 = ax1.twinx()
    # # 数据量柱状图
    # ax2.bar([p + bar_width for p in bar_positions], sorted_data_count, bar_width, color='r', alpha=0.6, label='Data num')
    # ax2.set_ylabel('Peptides counts', color='r')  # , fontproperties=font
    # ax2.tick_params(axis='y', labelcolor='r')

    # # 设置x轴刻度
    # plt.xticks([p + bar_width / 2 for p in bar_positions], sorted_keys, ha='right', fontsize=10)
    # fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
    # fig.tight_layout()


    # # 保存图像
    # plt.savefig(save_file_path)
    # plt.show()
    # print(f"Picture saved in: {save_file_path}")
    # 对字典按键排序
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x_label = [str(num) for num in sorted_keys]

    # 创建第二个y轴，并绘制样本数量柱状图
    ax1.bar(x_label, sorted_data_count, alpha=0.6, color='#3682be', label='Total Count')
    ax1.set_xlabel('Peptide length')
    ax1.set_ylabel('Total Count')
    ax1.tick_params(axis='y')

    # 绘制精确度折线图
    ax2 = ax1.twinx()
    ax2.plot(x_label, sorted_accuracy, marker='o', linestyle='-', color='#df3881', label='Accuracy')
    ax2.set_ylabel('Accuracy Rates')
    ax2.tick_params(axis='y')
    ax2.set_ylim(0, 1)  # 设置y轴范围从0到1

    # 添加标题和图例
    plt.title('Model Accuracy and Sample Count at Different Peptide Length')
    fig.tight_layout()
    fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))

    # 显示图表
    plt.savefig(save_file_path, dpi=1000)
    plt.show()
    print(f"Picture saved in: {save_file_path}")

    return accuracy_dict

def prec_by_mfr(
    peptides:List[List[str]], 
    pep_match_bool: List[bool], 
    spectra_mz_list:List[List[float]], 
    save_file_path: str,
    aa_mass_list:Dict[str,float]=STD_AA_MASS, 
) -> List[float]:
    
    miss_ratio = calc_missing_ratio(peptides, spectra_mz_list, aa_mass_list)    # List[float]
    
    # 初始化统计计数器：每个区间的预测正确数和总数
    correct_count = [0] * 10
    total_count = [0] * 10
    
    # 遍历噪声列表和预测结果
    for pep_match, mfr in zip(pep_match_bool, miss_ratio):
        # 确定当前噪声值属于哪个区间
        if mfr == 1.0:
            index = 9  # 如果噪声值为1.0，归入最后一个区间
        else:
            index = int(mfr * 10)  # 其他情况根据噪声值计算区间索引
        
        # 更新统计数据
        total_count[index] += 1
        if pep_match:
            correct_count[index] += 1
    
    # 计算每个区间的预测正确比例
    accuracy_rates = []
    for correct, total in zip(correct_count, total_count):
        if total > 0:
            accuracy_rates.append(correct / total)
        else:
            accuracy_rates.append(0.0)  # 避免除以零错误

    ### plotting
    # 定义噪声水平区间
    mfr_levels = [f"[{i/10},{(i+1)/10}]" for i in range(10)]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 创建第二个y轴，并绘制样本数量柱状图
    ax1.bar(mfr_levels, total_count, alpha=0.6, color='#3682be', label='Total Count')
    ax1.set_xlabel('MFR Levels')
    ax1.set_ylabel('Total Count')
    ax1.tick_params(axis='y')
    
    # 绘制精确度折线图
    ax2 = ax1.twinx()
    ax2.plot(mfr_levels, accuracy_rates, marker='o', linestyle='-', color='#df3881', label='Accuracy')
    ax2.set_ylabel('Accuracy Rates')
    ax2.tick_params(axis='y')
    ax2.set_ylim(0, 1)  # 设置y轴范围从0到1

    # 添加标题和图例
    plt.title('Model Accuracy and Sample Count at Different MFR Levels')
    fig.tight_layout()
    fig.legend(loc='upper right', bbox_to_anchor=(0.9,0.9))

    # 显示图表
    plt.savefig(save_file_path, dpi=1000)
    plt.show()
    print(f"Picture saved in: {save_file_path}")
    
    return accuracy_rates

def prec_by_noise(
    peptides:List[List[str]], 
    pep_match_bool: List[bool], 
    spectra_mz_list:List[List[float]],  
    save_file_path: str,
    aa_mass_list:Dict[str,float]=STD_AA_MASS, 
) -> List[float]:
    
    assert len(peptides) == len(pep_match_bool)
    noise_ratio = calc_noise_signal_ratio(peptides, spectra_mz_list, aa_mass_list)    # List[float]
    
    # 初始化统计计数器：每个区间的预测正确数和总数
    noise_bound = 20
    correct_count = [0] * (noise_bound + 1)
    total_count = [0] * (noise_bound + 1)
    
    # 遍历噪声列表和预测结果
    for pep_match, noise in zip(pep_match_bool, noise_ratio):
        # 确定当前噪声值属于哪个区间
        if noise >= noise_bound:
            index = noise_bound  # 如果噪声值为1.0，归入最后一个区间
        else:
            index = int(noise)  # 其他情况根据噪声值计算区间索引
        
        # 更新统计数据
        total_count[index] += 1
        if pep_match:
            correct_count[index] += 1
    
    # 计算准确率
    accuracy_rates = []
    for correct, total in zip(correct_count, total_count):
        if total > 0:
            accuracy_rates.append(correct / total)
        else:
            accuracy_rates.append(0.0)  # 避免除以零错误

    # 对字典按键排序
    noise_levels = [f"[{i},{i+1})" for i in range(noise_bound)]
    noise_levels.append(f">={noise_bound}")

    fig, ax1 = plt.subplots(figsize=(15, 9))

    # 创建第二个y轴，并绘制样本数量柱状图
    ax1.bar(noise_levels, total_count, alpha=0.6, color='#3682be', label='Total Count')
    ax1.set_xlabel('Noise Levels')
    ax1.set_ylabel('Total Count')
    ax1.tick_params(axis='y')
    
    # 绘制精确度折线图
    ax2 = ax1.twinx()
    ax2.plot(noise_levels, accuracy_rates, marker='o', linestyle='-', color='#df3881', label='Accuracy')
    ax2.set_ylabel('Accuracy Rates')
    ax2.tick_params(axis='y')
    ax2.set_ylim(0, 1)  # 设置y轴范围从0到1

    # 添加标题和图例
    plt.title('Model Accuracy and Sample Count at Different Noise Levels')
    fig.tight_layout()
    fig.legend(loc='upper right', bbox_to_anchor=(0.9,0.9))

    # 显示图表
    plt.savefig(save_file_path, dpi=1000)
    plt.show()
    print(f"Picture saved in: {save_file_path}")
    
    return accuracy_rates

def main(factor:str, pred_file:str, spectra_file:str = ""):
    save_file_path = get_png_file_name(pred_file, factor)
    
    if factor == "length":    
        pred_result = pd.read_csv(pred_file)
        length_list = [ len(split_peptide(pep,STD_AA_MASS)) \
                                for pep in pred_result['peptides_true'] ]
        prec_by_length(pred_result['pep_matches'], length_list, save_file_path)
        
    elif factor == "mfr" or factor == "noise":
        if spectra_file == "" or (not os.path.exists(spectra_file)):
            raise FileNotFoundError(f"File {spectra_file} not found!")
        
        pred_result = pd.read_csv(pred_file)
        spectra_info = {}
        
        if os.path.splitext(spectra_file)[1] == '.mgf':
            with mgf.read(spectra_file) as spectra:
                # spectra: List[{params {title,charge,scans,rtinseconds,seq}, m/z array, intensity array'}]
                spectra_info['peptides_true'] = [elem['params']['seq'] for elem in  spectra]
                spectra_info['ms2_mz'] = [elem['m/z array'] for elem in  spectra]
        elif os.path.splitext(spectra_file)[1] == '.parquet':
            pddata = pd.read_parquet(spectra_file)
            spectra_info['peptides_true'] = pddata['modified_sequence']
            spectra_info['ms2_mz'] = pddata['mz_array']
        else:
            raise TypeError(f"Not supportive spectra file. Option: mgf, parquet.")
        
        if not np.array_equal(spectra_info['peptides_true'].values, pred_result['peptides_true'].values):
            raise ValueError(f"Discrepancy between spectra_file: {spectra_file} and pred_file: {pred_file}!")
        
        peptide_splited = [ split_peptide(pep,STD_AA_MASS) \
                                for pep in pred_result['peptides_true'] ]
        
        if factor == "mfr": 
            prec_by_mfr(peptide_splited,
                        pred_result['pep_matches'],
                        spectra_info['ms2_mz'],
                        save_file_path,)
        elif factor == "noise":
            prec_by_noise(  peptide_splited,
                            pred_result['pep_matches'],
                            spectra_info['ms2_mz'],
                            save_file_path,)
        else:
            pass
        
    else:
        raise ValueError(f"Wrong factor: {factor}! Options: length, mfr, noise.")
    
    
if __name__ == '__main__':
    main(*sys.argv[1:])
    
"""
Command case:

python /jingbo/PyNovo/pynovo/metrics/influen_factors.py length \
    /jingbo/PyNovo/predict/casanovo/test_9spec.csv

python /jingbo/PyNovo/pynovo/metrics/influen_factors.py mfr \
    /jingbo/PyNovo/predict/casanovo/test_9spec.csv \
        /usr/commondata/public/jingbo/nine_species/test.parquet
        
python /jingbo/PyNovo/pynovo/metrics/influen_factors.py noise \
    /jingbo/PyNovo/predict/casanovo/test_9spec.csv \
        /usr/commondata/public/jingbo/nine_species/test.parquet
"""