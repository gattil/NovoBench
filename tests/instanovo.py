import sys
import os
import csv
sys.path.append("/jingbo/PyNovo/")

from pynovo.datasets import CustomDataset, NineSpeciesDataset
from pynovo.models.instanovo import CasanovoRunner, CasanovoConfig


file_mapping = {
    "train" : "train.parquet",
    "valid" : "valid.parquet",
}

def train():
    dataset = CustomDataset("/usr/commondata/public/jingbo/seven_species/", file_mapping)
    data = dataset.load_data(transform = CasanovoRunner.preprocessing_pipeline())
    config = CasanovoConfig()
    model = CasanovoRunner(config)
    model.train(data.get_train(), data.get_valid())


train()


def eval(data_dir, model_file, saved_path):
    # dataset = CustomDataset("/usr/commondata/public/jingbo/nine_species/", file_mapping)
    # data = dataset.load_data(transform = CasanovoRunner.preprocessing_pipeline())
    # config = CasanovoConfig()
    # model = CasanovoRunner(config,"/usr/commondata/local_public/jingbo/casanovo/nine_species/epoch=19-step=300000.ckpt")
    # model.evaluate(data.get_valid())
    if os.path.exists(saved_path):
        base, extension = os.path.splitext(saved_path)
        counter = 1
        saved_path = f"{base}_{counter}{extension}"
        while os.path.exists(saved_path):
            counter += 1
            saved_path = f"{base}_{counter}{extension}"
        print(f"Saved file exist, use {saved_path} instead!")   
    with open(saved_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['peptides_true', 'peptides_pred', 'peptides_score'])
    
    file_mapping = {"valid" : "test.parquet",}
    dataset = CustomDataset(data_dir, file_mapping)
    data = dataset.load_data(transform = CasanovoRunner.preprocessing_pipeline())
    config = CasanovoConfig()
    config.calculate_precision = True
    
    model = CasanovoRunner(config, model_file, saved_path)
    model.evaluate(data.get_valid())



        