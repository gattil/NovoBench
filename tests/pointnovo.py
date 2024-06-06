import sys
import os

sys.path.append("/jingbo/PyNovo")
from pynovo.datasets import CustomDataset, NineSpeciesDataset
from pynovo.models.pointnovo.pointnovo_runner import PointnovoRunner


# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

file_mapping = {
    "train" : "train.parquet",
    "valid" : "valid.parquet",
}

def train():
    dataset = CustomDataset("/usr/commondata/public/jingbo/seven_species/", file_mapping)
    data = dataset.load_data(transform=PointnovoRunner.preprocessing_pipeline())
    model = PointnovoRunner()
    model.train(data.get_train(), data.get_valid())


def eval():
    model_save_folder = '/jingbo/PyNovo/pynovo/save_models/pointnovo/'
    dataset = CustomDataset("/jingbo/PyNovo/data/", file_mapping)
    data = dataset.load_data()
    model = PointnovoRunner()
    model.evaluate(data.get_valid(),model_save_folder)

def predict():
    mgf_path = '/root/novo_7species/cross.7species_50k.exclude_yeast/cross.cat.mgf.test.repeat'
    output_file = '/jingbo/PyNovo/pointnovo'
    model_save_folder = '/jingbo/PyNovo/pynovo/save_models/pointnovo/'
    model = PointnovoRunner()
    model.predict(mgf_path, output_file,model_save_folder)

train()
# eval()
# predict()