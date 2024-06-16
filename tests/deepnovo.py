import sys
import os

sys.path.append("/path/to/PyNovo/")
from pynovo.models.deepnovo.deepnovo_dataloader  import  DeepNovoTrainDataset
from pynovo.datasets import CustomDataset, NineSpeciesDataset
from pynovo.models.deepnovo.deepnovo_runner import DeepnovoRunner


# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

file_mapping = {
    "train" : "train.parquet",
    "valid" : "valid.parquet",
}



def train():
    dataset = CustomDataset("/usr/commondata/public/jingbo/nine_species/", file_mapping)
    data = dataset.load_data(transform=DeepnovoRunner.preprocessing_pipeline())
    model = DeepnovoRunner()
    model.train(data.get_train(), data.get_valid())


# def eval():
#     # dataset = CustomDataset("/usr/commondata/public/jingbo/seven_species/", file_mapping)
#     dataset = CustomDataset("/jingbo/PyNovo/data/", file_mapping)
#     data = dataset.load_data(transform=DeepnovoRunner.preprocessing_pipeline())
#     model = DeepnovoRunner()
#     model.evaluate(data.get_valid(),'/jingbo/PyNovo/pynovo/save_models/deepnovo/seven_species/')

# def predict():
#     mgf_path = '/root/novo_7species/cross.7species_50k.exclude_yeast/cross.cat.mgf.test.repeat'
#     output_file = '/jingbo/PyNovo/deepnovo'
#     model = DeepnovoRunner()
#     model.predict(mgf_path, output_file)


# eval()
train()



