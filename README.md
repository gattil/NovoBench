# NovoBench: Benchmark $de$ $novo$ peptide sequencing algorithms
<p>
  <a href="https://github.com/pytorch/pytorch"> <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" height="22px"></a>
  <a href="https://github.com/Lightning-AI/pytorch-lightning"> <img src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white" height="22px"></a>
<p>

<p align="center" width="100%">
  <!-- <img src='./images/all.png' width="600%"> -->
  <img src='./images/novobench.png' width="100%">
</p>

## 📚 Introduction
**NovoBench provides a unified framework for *de novo* peptide sequencing, focusing on four key aspects:**  

- 💥 **Datasets** (diverse MS/MS spectrum data)
    - ✨ Multiple types of  spectrum data
    - ✨ **Standardized data splits** (ensures fair and reproducible evaluation)  

- 💥 **Models** (integrates state-of-the-art methods)  
    - ✨ **Included models** (DeepNovo, PointNovo, Casanovo, InstaNovo, AdaNovo, $\pi$-HelixNovo)  

- 💥 **Influencing factors** (evaluates model robustness)  
    - ✨ **Key factors** (peptide length, noise peaks, missing fragment ratio)  

- 💥 **Evaluation metrics** (comprehensive performance measures)  
    - ✨ **Comprehensive metrics** (amino acid-level and peptide-level precision/recall, PTM identification, efficiency, confidence)  

NovoBench abstracts *de novo* peptide sequencing into well-defined challenges, providing standardized datasets, integrated models, and evaluation metrics to facilitate large-scale comparative studies and drive future advancements in proteomics.

📑 Please see more details in our [NeurIPS 2024 paper](https://arxiv.org/abs/2406.11906).

## 📦 Installation 
This project has provided an environment setting file of conda, users can easily reproduce the environment by the following commands:
```shell
conda env create -f novobench.yaml
conda activate novobench
```

## 🍩 Dataset & Checkpoint download

- All the necessary data files can be downloaded from the [link](https://huggingface.co/datasets/jingbo02/NovoBench).

- The InstaNovo pretrain weight can be downloaded from the [Nine-species](https://github.com/instadeepai/InstaNovo/releases/download/0.1.4/instanovo.pt), [HC-PT](https://github.com/instadeepai/InstaNovo/releases/download/0.1.4/instanovo_yeast.pt).

## 🚀 Getting Started with NovoBench

> Train a New  Model

To train a model from scratch, run:
```shell
python tests/casanovo.py --mode train --data_path parquet_path --model_path ckpt_path  --config_path config_path
```

> Sequence Mass Spectra

To sequence the mass spectra with NovoBench, use the following command:
```shell
python tests/casanovo.py --mode seq --data_path parquet_path --model_path ckpt_path --denovo_output_path csv_path --config_path config_path
``` 

## ⚠️  Note

- DeepNovo and PointNovo <U>need more cpu</U> to process the dataset.
- The unified config file for DeepNovo and PointNovo is <U>in progress</U>. Currently, other models can run using a single config file.

## 🔗 Citation
```bibtex
@misc{zhou2024novobenchbenchmarkingdeeplearningbased,
      title={NovoBench: Benchmarking Deep Learning-based De Novo Peptide Sequencing Methods in Proteomics}, 
      author={Jingbo Zhou and Shaorong Chen and Jun Xia and Sizhe Liu and Tianze Ling and Wenjie Du and Yue Liu and Jianwei Yin and Stan Z. Li},
      year={2024},
      eprint={2406.11906},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM},
      url={https://arxiv.org/abs/2406.11906}, 
}
```
