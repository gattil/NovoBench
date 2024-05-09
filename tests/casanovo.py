import sys
import os
sys.path.append("/jingbo/PyNovo/")

from pynovo.datasets import CustomDataset, NineSpeciesDataset
from pynovo.models.casanovo import CasanovoRunner, CasanovoConfig


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

file_mapping = {
    # "train" : "train.parquet",
    "valid" : "test.parquet",
}

def train():
    dataset = CustomDataset("/usr/commondata/public/jingbo/seven_species/", file_mapping)
    data = dataset.load_data(transform = CasanovoRunner.preprocessing_pipeline())
    config = CasanovoConfig()
    model = CasanovoRunner(config)
    model.train(data.get_train(), data.get_valid())


def eval():
    dataset = CustomDataset("/usr/commondata/public/jingbo/nine_species/", file_mapping)
    data = dataset.load_data(transform = CasanovoRunner.preprocessing_pipeline())
    config = CasanovoConfig()
    model = CasanovoRunner(config,"/usr/commondata/local_public/jingbo/casanovo/nine_species/epoch=19-step=300000.ckpt")
    model.evaluate(data.get_valid())


def predict():
    mgf_path = '/root/novo_7species/cross.7species_50k.exclude_yeast/cross.cat.mgf.test.repeat'
    output_file = '/jingbo/PyNovo/casa'
    config = CasanovoConfig()
    model = CasanovoRunner(config,"/jingbo/PyNovo/pynovo/save_models/casanovo/epoch=99-step=1600.ckpt")
    model.predict(mgf_path,output_file)

# train()
eval()