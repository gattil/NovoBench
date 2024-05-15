import sys
import os
import yaml
import pdb

sys.path.append("/jingbo/PyNovo/")

from pynovo.models.instanovo import InstanovoRunner
from pynovo.datasets import CustomDataset

file_mapping = {
    "train" : "sample_train.parquet",
    "valid" : "sample_test.parquet",
}

def train():
    config_path = "/jingbo/PyNovo/pynovo/models/instanovo/instanovo_config.yaml"
    with open(config_path) as f_in:
        instanovo_config = yaml.safe_load(f_in)
    dataset = CustomDataset("/jingbo/PyNovo/data", file_mapping)
    data = dataset.load_data(transform = InstanovoRunner.preprocessing_pipeline())
    model = InstanovoRunner(instanovo_config)
    model.train(data.get_train(), data.get_valid()) 

train()
