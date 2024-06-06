import sys
import os
import csv
sys.path.append("/jingbo/PyNovo/")

from pynovo.datasets import CustomDataset, NineSpeciesDataset
from pynovo.models.casanovo import CasanovoRunner, CasanovoConfig


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

file_mapping = {
    "train" : "sample_train.parquet",
    "valid" : "sample_test.parquet",
}

def train():
    dataset = CustomDataset("/jingbo/PyNovo/data/", file_mapping)
    data = dataset.load_data(transform = CasanovoRunner.preprocessing_pipeline())
    config = CasanovoConfig()
    model = CasanovoRunner(config)
    model.train(data.get_train(), data.get_valid())


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


def predict(data_path, output_dir, model_file):
    mgf_path = '/root/novo_7species/cross.7species_50k.exclude_yeast/cross.cat.mgf.test.repeat'
    output_file = '/jingbo/PyNovo/casa'
    config = CasanovoConfig()
    model = CasanovoRunner(config,"/jingbo/PyNovo/pynovo/save_models/casanovo/epoch=99-step=1600.ckpt")
    model.predict(mgf_path,output_file)

train()
# eval()

# if __name__ == "__main__":
#     if len(sys.argv) == 1:
#         print("Need mode: train, eval or predict!")
#     elif sys.argv[1] == "train":
#         train()
#     elif sys.argv[1] == "eval":
#         os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[5]
#         eval(sys.argv[2], sys.argv[3],sys.argv[4])
#     elif sys.argv[1] == "predict":
#         os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[5]
#         predict(sys.argv[2], sys.argv[3], sys.argv[4])
#     else:
#         print("Error mode!")
        