# import sys
# import os

# sys.path.append("/jingbo/PyNovo")
# from pynovo.models.deepnovo  import  DeepNovoTrainDataset
# from pynovo.datasets import CustomDataset, NineSpeciesDataset
# from pynovo.models.deepnovo import DeepnovoRunner
# from pynovo.models.deepnovo.deepnovo_dataloader import DeepnovoDataModule
# import torch

# file_mapping = {
#     "train" : "train.parquet",
#     "valid" : "test.parquet",
# }


# from tqdm import tqdm
# from multiprocessing import Pool



# l = []
# num_workers = 64
# dataset = CustomDataset("/usr/commondata/public/jingbo/seven_species/", file_mapping) 
# data = dataset.load_data()
# dataset = DeepnovoDataModule(df = data.get_train()).dataset

# def process_feature(feature):
#     global dataset
#     return dataset._get_feature(feature)

# with Pool(num_workers) as pool:
#     feature_list = dataset.feature_list
#     results = list(tqdm(pool.imap(process_feature, feature_list), total=len(feature_list)))

# l.extend(results)
# torch.save(l, "/usr/commondata/public/jingbo/seven_species/train.pt")

import re
peptide1 = 'AAGGK(.123)AK'
peptide1 = re.split(r"(?<=.)(?=[A-Z])", peptide1)
print(peptide1)