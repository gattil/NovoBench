import pandas as pd
import re
import evaluate
STD_AA_MASS = {
    'G': 57.02146372057,
    'A': 71.03711378471,
    'S': 87.03202840427001,
    'P': 97.05276384885,
    'V': 99.06841391299,
    'T': 101.04767846841,
    'C': 103.00918478471,
    'L': 113.08406397713001,
    'I': 113.08406397713001,
    'J': 113.08406397713001,
    'N': 114.04292744114001,
    'D': 115.02694302383001,
    'Q': 128.05857750527997,
    'K': 128.09496301399997,
    'E': 129.04259308796998,
    'M': 131.04048491299,
    'H': 137.05891185845002,
    'F': 147.06841391298997,
    'U': 150.95363508471,
    'R': 156.10111102359997,
    'Y': 163.06332853254997,
    'W': 186.07931294985997,
    'O': 237.14772686284996,
    'M(+15.99)': 147.0354,
    'Q(+.98)': 129.0426,
    'N(+.98)': 115.02695,
    'C(+57.02)': 160.03065,
}

file_path = "/jingbo/PyNovo/predict/casanovo/test_9spec.csv"
df = pd.read_csv(file_path, na_values=['', ' '])
df = df.fillna('')
match_ret = evaluate.aa_match_batch(
        df['peptides_true'],
        df['peptides_pred'],
        STD_AA_MASS,
)
metrics_dict = evaluate.aa_match_metrics(
    *match_ret, df['peptides_score']
)#  metrics_dict = {"aa_precision" : float,"aa_recall" : float,"pep_precision" : float,"ptm_recall" : float,"ptm_precision" : float,"curve_auc" : float}

df['pep_matches'] = [aa_matches[1] for aa_matches in match_ret[0]]
df.to_csv(file_path, index=False)

for key, value in metrics_dict.items():
    print(f"{key}:\t{value}")

print("END")

# data = pd.read_parquet("/usr/commondata/public/jingbo/seven_species/test.parquet")
# print("END")