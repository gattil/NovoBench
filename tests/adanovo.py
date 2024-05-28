import sys
import os
import csv
sys.path.append("/jingbo/PyNovo/")

from pynovo.datasets import CustomDataset, NineSpeciesDataset
from pynovo.models.adanovo import AdanovoRunner, AdanovoConfig


# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

file_mapping = {
    "train" : "train.parquet",
    "valid" : "valid.parquet",
}

def train():
    dataset = CustomDataset("/usr/commondata/public/jingbo/nine_species/", file_mapping)
    data = dataset.load_data(transform = AdanovoRunner.preprocessing_pipeline())
    config = AdanovoConfig()
    model = AdanovoRunner(config)
    model.train(data.get_train(), data.get_valid())


# def eval():
#     dataset = CustomDataset("/jingbo/PyNovo/data/", file_mapping)
#     data = dataset.load_data(transform = AdanovoRunner.preprocessing_pipeline())
#     config = AdanovoConfig()
#     model = AdanovoRunner(config,"/jingbo/PyNovo/pynovo/save_models/adanovo/epoch=99-step=1600.ckpt")
#     model.evaluate(data.get_valid())

def eval(data_dir, model_file, saved_path):
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
    data = dataset.load_data(transform = AdanovoRunner.preprocessing_pipeline())
    config = AdanovoConfig()
    config.calculate_precision = True
    
    model = AdanovoRunner(config, model_file, saved_path)
    model.evaluate(data.get_valid())

def predict():
    mgf_path = '/root/novo_7species/cross.7species_50k.exclude_yeast/cross.cat.mgf.test.repeat'
    output_file = '/jingbo/PyNovo/adsa'
    config = AdanovoConfig()
    model = AdanovoRunner(config,"/jingbo/PyNovo/pynovo/save_models/casanovo/epoch=99-step=1600.ckpt")
    model.predict(mgf_path,output_file)

# train()
# eval()
if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Need mode: train, eval or predict!")
    elif sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "eval":
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[5]
        eval(sys.argv[2], sys.argv[3],sys.argv[4])
    elif sys.argv[1] == "predict":
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[5]
        predict(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("Error mode!")