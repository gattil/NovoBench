import sys
import os
sys.path.append("/jingbo/PyNovo/")

from pynovo.datasets import CustomDataset, NineSpeciesDataset
from pynovo.models.adanovo import AdanovoRunner, AdanovoConfig


# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

file_mapping = {
    "train" : "train.parquet",
    "valid" : "valid.parquet",
}

def train():
    dataset = CustomDataset("/usr/commondata/public/jingbo/seven_species/", file_mapping)
    data = dataset.load_data(transform = AdanovoRunner.preprocessing_pipeline())
    config = AdanovoConfig()
    model = AdanovoRunner(config)
    model.train(data.get_train(), data.get_valid())


def eval():
    dataset = CustomDataset("/jingbo/PyNovo/data/", file_mapping)
    data = dataset.load_data(transform = AdanovoRunner.preprocessing_pipeline())
    config = AdanovoConfig()
    model = AdanovoRunner(config,"/jingbo/PyNovo/pynovo/save_models/adanovo/epoch=99-step=1600.ckpt")
    model.evaluate(data.get_valid())


def predict():
    mgf_path = '/root/novo_7species/cross.7species_50k.exclude_yeast/cross.cat.mgf.test.repeat'
    output_file = '/jingbo/PyNovo/adsa'
    config = AdanovoConfig()
    model = AdanovoRunner(config,"/jingbo/PyNovo/pynovo/save_models/casanovo/epoch=99-step=1600.ckpt")
    model.predict(mgf_path,output_file)

train()
# eval()