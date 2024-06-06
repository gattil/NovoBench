import sys
import os
import yaml
import pdb
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
sys.path.append("/jingbo/PyNovo/")

from pynovo.models.instanovo.instanovo_runner import InstanovoRunner
from pynovo.datasets import CustomDataset

file_mapping = {
    "train" : "sample_train.parquet",
    "valid" : "sample_test.parquet",
}

def train():
    config_path = "/jingbo/PyNovo/pynovo/models/instanovo/instanovo_config.yaml"
    with open(config_path) as f_in:
        instanovo_config = yaml.safe_load(f_in)
    dataset = CustomDataset("/usr/commondata/public/jingbo/nine_species/", file_mapping)
    data = dataset.load_data(transform = InstanovoRunner.preprocessing_pipeline())
    model = InstanovoRunner(instanovo_config)
    model.train(data.get_train(), data.get_valid()) 


def eval():
    model_path = '/jingbo/PyNovo/pynovo/save_models/instanovo/instanovo_nine/epoch=28-step=450000.ckpt'
    model = InstanovoRunner()
    dataset = CustomDataset("/jingbo/PyNovo/data", file_mapping)
    data = dataset.load_data(transform = InstanovoRunner.preprocessing_pipeline())
    model.evaluate(data.get_valid(), model_path)

eval()
# train()
