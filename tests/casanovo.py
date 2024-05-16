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


def eval(data_dir, model_file):
    # dataset = CustomDataset("/usr/commondata/public/jingbo/nine_species/", file_mapping)
    # data = dataset.load_data(transform = CasanovoRunner.preprocessing_pipeline())
    # config = CasanovoConfig()
    # model = CasanovoRunner(config,"/usr/commondata/local_public/jingbo/casanovo/nine_species/epoch=19-step=300000.ckpt")
    # model.evaluate(data.get_valid())
    file_mapping = {"valid" : "test.parquet",}
    dataset = CustomDataset(data_dir, file_mapping)
    data = dataset.load_data(transform = CasanovoRunner.preprocessing_pipeline())
    config = CasanovoConfig()
    
    # Change config
    config.calculate_precision = True
    valid_data_len = data._data['valid']._df.shape[0]
    config.predict_batch_size = valid_data_len
    
    model = CasanovoRunner(config, model_file)
    model.evaluate(data.get_valid())


def predict(data_path, output_dir, model_file):
    mgf_path = '/root/novo_7species/cross.7species_50k.exclude_yeast/cross.cat.mgf.test.repeat'
    output_file = '/jingbo/PyNovo/casa'
    config = CasanovoConfig()
    model = CasanovoRunner(config,"/jingbo/PyNovo/pynovo/save_models/casanovo/epoch=99-step=1600.ckpt")
    model.predict(mgf_path,output_file)

# train()
# eval()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Need mode: train, eval or predict!")
    elif sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "eval":
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[4]
        eval(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "predict":
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[5]
        predict(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("Error mode!")
        