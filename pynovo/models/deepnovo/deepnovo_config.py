# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import numpy as np
# # import tensorflow as tf
# import tensorflow.compat.v1 as tf


# # ==============================================================================
# # FLAGS (options) for this app
# # ==============================================================================


# tf.app.flags.DEFINE_string("train_dir", # flag_name
#                            "/jingbo/PyNovo/pynovo/save_models/deepnovo/nine_species", # default_value
#                            "Training directory.") # docstring

# tf.app.flags.DEFINE_integer("direction",
#                             2,
#                             "Set to 0/1/2 for Forward/Backward/Bi-directional.")

# tf.app.flags.DEFINE_boolean("use_intensity",
#                             True,
#                             "Set to True to use intensity-model.")

# tf.app.flags.DEFINE_boolean("shared",
#                             False,
#                             "Set to True to use shared weights.")

# tf.app.flags.DEFINE_boolean("lstm_kmer",
#                             False,
#                             "Set to True to use lstm model on k-mers instead of full sequence.")

# tf.app.flags.DEFINE_boolean("knapsack_build",
#                             False,
#                             "Set to True to build knapsack matrix.")

# tf.app.flags.DEFINE_boolean("train",
#                             True,
#                             "Set to True for training.")

# tf.app.flags.DEFINE_boolean("test_true_feeding",
#                             False,
#                             "Set to True for testing.")

# tf.app.flags.DEFINE_boolean("decode",
#                             False,
#                             "Set to True for decoding.")

# tf.app.flags.DEFINE_boolean("beam_search",
#                             False,
#                             "Set to True for beam search.")

# tf.app.flags.DEFINE_integer("beam_size",
#                             5,
#                             "Number of optimal paths to search during decoding.")

# tf.app.flags.DEFINE_boolean("search_db",
#                             False,
#                             "Set to True to do a database search.")

# tf.app.flags.DEFINE_boolean("search_denovo",
#                             False,
#                             "Set to True to do a denovo search.")

# tf.app.flags.DEFINE_boolean("search_hybrid",
#                             False,
#                             "Set to True to do a hybrid, db+denovo, search.")

# tf.app.flags.DEFINE_boolean("test",
#                             False,
#                             "Set to True to test the prediction accuracy.")

# tf.app.flags.DEFINE_boolean("header_seq",
#                             True,
#                             "Set to False if peptide sequence is not provided.")

# tf.app.flags.DEFINE_boolean("decoy",
#                             False,
#                             "Set to True to search decoy database.")

# tf.app.flags.DEFINE_integer("multiprocessor",
#                             1,
#                             "Use multi processors to read data during training.")

# FLAGS = tf.app.flags.FLAGS
# train_dir = FLAGS.train_dir
# use_lstm = True

# # ==============================================================================
# # GLOBAL VARIABLES for VOCABULARY
# # ==============================================================================


# # Special vocabulary symbols - we always put them at the start.
# _PAD = "_PAD"
# _GO = "_GO"
# _EOS = "_EOS"
# _START_VOCAB = [_PAD, _GO, _EOS]

# PAD_ID = 0
# GO_ID = 1
# EOS_ID = 2
# assert PAD_ID == 0
# vocab_reverse = ['A',
#                  'R',
#                  'N',
#                  'N(Deamidation)',
#                  'D',
#                  #~ 'C',
#                  'C(Carbamidomethylation)',
#                  'E',
#                  'Q',
#                  'Q(Deamidation)',
#                  'G',
#                  'H',
#                  'I',
#                  'L',
#                  'K',
#                  'M',
#                  'M(Oxidation)',
#                  'F',
#                  'P',
#                  'S',
#                  'S(Phosphorylation)',
#                  'T',
#                  'T(Phosphorylation)',
#                  'W',
#                  'Y',
#                  "Y(Phosphorylation)",
#                  'V',
#                 ]

# vocab_reverse = _START_VOCAB + vocab_reverse
# print("vocab_reverse ", vocab_reverse)

# vocab = dict([(x, y) for (y, x) in enumerate(vocab_reverse)])
# print("vocab ", vocab)

# vocab_size = len(vocab_reverse)
# print("vocab_size ", vocab_size)


# # ==============================================================================
# # GLOBAL VARIABLES for THEORETICAL MASS
# # ==============================================================================


# mass_H = 1.0078
# mass_H2O = 18.0106
# mass_NH3 = 17.0265
# mass_N_terminus = 1.0078
# mass_C_terminus = 17.0027
# mass_CO = 27.9949
# mass_Phosphorylation = 79.96633

# mass_AA = {'_PAD': 0.0,
#            '_GO': mass_N_terminus-mass_H,
#            '_EOS': mass_C_terminus+mass_H,
#            'A': 71.03711, # 0
#            'R': 156.10111, # 1
#            'N': 114.04293, # 2
#            'N(Deamidation)': 115.02695,
#            'D': 115.02694, # 3
#            #~ 'C(Carbamidomethylation)': 103.00919, # 4
#            'C(Carbamidomethylation)': 160.03065, # C(+57.02)
#            #~ 'C(Carbamidomethylation)': 161.01919, # C(+58.01) # orbi
#            'E': 129.04259, # 5
#            'Q': 128.05858, # 6
#            'Q(Deamidation)': 129.0426,
#            'G': 57.02146, # 7
#            'H': 137.05891, # 8
#            'I': 113.08406, # 9
#            'L': 113.08406, # 10
#            'K': 128.09496, # 11
#            'M': 131.04049, # 12
#            'M(Oxidation)': 147.0354,
#            'F': 147.06841, # 13
#            'P': 97.05276, # 14
#            'S': 87.03203, # 15
#            'S(Phosphorylation)': 87.03203 + mass_Phosphorylation,
#            'T': 101.04768, # 16
#            'T(Phosphorylation)': 101.04768 + mass_Phosphorylation,
#            'W': 186.07931, # 17
#            'Y': 163.06333, # 18
#            'Y(Phosphorylation)': 163.06333 + mass_Phosphorylation,
#            'V': 99.06841, # 19
#           }

# mass_ID = [mass_AA[vocab_reverse[x]] for x in range(vocab_size)]
# mass_ID_np = np.array(mass_ID, dtype=np.float32)

# mass_AA_min = mass_AA["G"] # 57.02146


# # ==============================================================================
# # GLOBAL VARIABLES for PRECISION, RESOLUTION, temp-Limits of MASS & LEN
# # ==============================================================================


# # if change, need to re-compile cython_speedup << NO NEED
# SPECTRUM_RESOLUTION = 10 # bins for 1.0 Da = precision 0.1 Da
# #~ SPECTRUM_RESOLUTION = 20 # bins for 1.0 Da = precision 0.05 Da
# #~ SPECTRUM_RESOLUTION = 40 # bins for 1.0 Da = precision 0.025 Da
# # SPECTRUM_RESOLUTION = 50 # bins for 1.0 Da = precision 0.02 Da
# # SPECTRUM_RESOLUTION = 100 # bins for 1.0 Da = precision 0.01 Da
# print("SPECTRUM_RESOLUTION ", SPECTRUM_RESOLUTION)

# # if change, need to re-compile cython_speedup << NO NEED
# WINDOW_SIZE = 10 # 10 bins
# print("WINDOW_SIZE ", WINDOW_SIZE)

# MZ_MAX = 3000.0
# MZ_SIZE = int(MZ_MAX * SPECTRUM_RESOLUTION) # 30k

# KNAPSACK_AA_RESOLUTION = 10000 # 0.0001 Da
# # KNAPSACK_AA_RESOLUTION = 50 # change
# mass_AA_min_round = int(round(mass_AA_min * KNAPSACK_AA_RESOLUTION)) # 57.02146
# KNAPSACK_MASS_PRECISION_TOLERANCE = 100 # 0.01 Da
# num_position = 0

# PRECURSOR_MASS_PRECISION_TOLERANCE = 0.01

# # ONLY for accuracy evaluation
# #~ PRECURSOR_MASS_PRECISION_INPUT_FILTER = 0.01
# #~ PRECURSOR_MASS_PRECISION_INPUT_FILTER = 1000
# AA_MATCH_PRECISION = 0.1

# # skip (x > MZ_MAX,MAX_LEN)
# MAX_LEN = 50 if FLAGS.search_denovo else 30
# print("MAX_LEN ", MAX_LEN)


# # ==============================================================================
# # HYPER-PARAMETERS of the NEURAL NETWORKS
# # ==============================================================================


# num_ion = 8 # 2
# print("num_ion ", num_ion)

# weight_decay = 0.0  # no weight decay lead to better result.
# print("weight_decay ", weight_decay)

# #~ encoding_cnn_size = 4 * (RESOLUTION//10) # 4 # proportion to RESOLUTION
# #~ encoding_cnn_filter = 4
# #~ print("encoding_cnn_size ", encoding_cnn_size)
# #~ print("encoding_cnn_filter ", encoding_cnn_filter)

# embedding_size = 512
# print("embedding_size ", embedding_size)

# num_lstm_layers = 1
# num_units = 512
# print("num_lstm_layers ", num_lstm_layers)
# print("num_units ", num_units)

# dropout_rate = 0.25

# batch_size = 32
# num_workers = 64
# print("batch_size ", batch_size)

# num_epoch = 30

# init_lr = 1e-3

# train_stack_size = 500 # 3000 # 5000
# valid_stack_size = 1500#1000 # 3000 # 5000
# test_stack_size = 5000
# decode_stack_size = 1000 # 3000
# print("train_stack_size ", train_stack_size)
# print("valid_stack_size ", valid_stack_size)
# print("test_stack_size ", test_stack_size)
# print("decode_stack_size ", decode_stack_size)

# steps_per_validation = 3000 # 100 # 2 # 4 # 200
# print("steps_per_validation ", steps_per_validation)

# max_gradient_norm = 5.0
# print("max_gradient_norm ", max_gradient_norm)


# # ==============================================================================
# # DATASETS
# # ==============================================================================


# data_format = "mgf"
# cleavage_rule = "trypsin"
# num_missed_cleavage = 2
# fixed_mod_list = ['C']
# var_mod_list = ['N', 'Q', 'M']
# num_mod = 3
# precursor_mass_tolerance = 0.01 # Da
# precursor_mass_ppm = 15.0/1000000 # ppm (20 better) # instead of absolute 0.01 Da
# knapsack_file = "knapsack.npy"
# topk_output = 1


# vocab_reverse_eval = ['A',
#                  'R',
#                  'N',
#                  'N(+.98)',
#                  'D',
#                  #~ 'C',
#                  'C(+57.02)',
#                  'E',
#                  'Q',
#                  'Q(+.98)',
#                  'G',
#                  'H',
#                  'I',
#                  'L',
#                  'K',
#                  'M',
#                  'M(+15.99)',
#                  'F',
#                  'P',
#                  'S',
#                  'S(Phosphorylation)',
#                  'T',
#                  'T(Phosphorylation)',
#                  'W',
#                  'Y',
#                  "Y(Phosphorylation)",
#                  'V',
#                 ]

# vocab_reverse_eval = _START_VOCAB + vocab_reverse_eval
# print("vocab_reverse_eval ", vocab_reverse_eval)



# mass_AA_eval = {'_PAD': 0.0,
#            '_GO': mass_N_terminus-mass_H,
#            '_EOS': mass_C_terminus+mass_H,
#            'A': 71.03711, # 0
#            'R': 156.10111, # 1
#            'N': 114.04293, # 2
#            'N(+.98)': 115.02695,
#            'D': 115.02694, # 3
#            #~ 'C(Carbamidomethylation)': 103.00919, # 4
#            'C(+57.02)': 160.03065, # C(+57.02)
#            #~ 'C(Carbamidomethylation)': 161.01919, # C(+58.01) # orbi
#            'E': 129.04259, # 5
#            'Q': 128.05858, # 6
#             'Q(+.98)': 129.0426,
#            'G': 57.02146, # 7
#            'H': 137.05891, # 8
#            'I': 113.08406, # 9
#            'L': 113.08406, # 10
#            'K': 128.09496, # 11
#            'M': 131.04049, # 12
#            'M(+15.99)': 147.0354,
#            'F': 147.06841, # 13
#            'P': 97.05276, # 14
#            'S': 87.03203, # 15
#            'S(Phosphorylation)': 87.03203 + mass_Phosphorylation,
#            'T': 101.04768, # 16
#            'T(Phosphorylation)': 101.04768 + mass_Phosphorylation,
#            'W': 186.07931, # 17
#            'Y': 163.06333, # 18
#            'Y(Phosphorylation)': 163.06333 + mass_Phosphorylation,
#            'V': 99.06841, # 19
#           }


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf


# ==============================================================================
# FLAGS (options) for this app
# ==============================================================================

# Check
dataset = 'nine_species'

# Check
tf.app.flags.DEFINE_string("train_dir", # flag_name
                           '/jingbo/PyNovo/pynovo/save_models/deepnovo/'+dataset, # default_value
                           "Training directory.") # docstring

tf.app.flags.DEFINE_integer("direction",
                            2,
                            "Set to 0/1/2 for Forward/Backward/Bi-directional.")

tf.app.flags.DEFINE_boolean("use_intensity",
                            True,
                            "Set to True to use intensity-model.")

tf.app.flags.DEFINE_boolean("shared",
                            False,
                            "Set to True to use shared weights.")

tf.app.flags.DEFINE_boolean("lstm_kmer",
                            False,
                            "Set to True to use lstm model on k-mers instead of full sequence.")

tf.app.flags.DEFINE_boolean("knapsack_build",
                            False,
                            "Set to True to build knapsack matrix.")

tf.app.flags.DEFINE_boolean("train",
                            True,
                            "Set to True for training.")

tf.app.flags.DEFINE_boolean("test_true_feeding",
                            False,
                            "Set to True for testing.")

tf.app.flags.DEFINE_boolean("decode",
                            False,
                            "Set to True for decoding.")

tf.app.flags.DEFINE_boolean("beam_search",
                            False,
                            "Set to True for beam search.")

tf.app.flags.DEFINE_integer("beam_size",
                            5,
                            "Number of optimal paths to search during decoding.")

tf.app.flags.DEFINE_boolean("search_db",
                            False,
                            "Set to True to do a database search.")

tf.app.flags.DEFINE_boolean("search_denovo",
                            False,
                            "Set to True to do a denovo search.")

tf.app.flags.DEFINE_boolean("search_hybrid",
                            False,
                            "Set to True to do a hybrid, db+denovo, search.")

tf.app.flags.DEFINE_boolean("test",
                            False,
                            "Set to True to test the prediction accuracy.")

tf.app.flags.DEFINE_boolean("header_seq",
                            True,
                            "Set to False if peptide sequence is not provided.")

tf.app.flags.DEFINE_boolean("decoy",
                            False,
                            "Set to True to search decoy database.")

tf.app.flags.DEFINE_integer("multiprocessor",
                            1,
                            "Use multi processors to read data during training.")

FLAGS = tf.app.flags.FLAGS
train_dir = FLAGS.train_dir
use_lstm = True

# ==============================================================================
# GLOBAL VARIABLES for VOCABULARY
# ==============================================================================


# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_START_VOCAB = [_PAD, _GO, _EOS]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
assert PAD_ID == 0
# vocab_reverse = ['A',
#                  'R',
#                  'N',
#                  'N(Deamidation)',
#                  'D',
#                  #~ 'C',
#                  'C(Carbamidomethylation)',
#                  'E',
#                  'Q',
#                  'Q(Deamidation)',
#                  'G',
#                  'H',
#                  'I',
#                  'L',
#                  'K',
#                  'M',
#                  'M(Oxidation)',
#                  'F',
#                  'P',
#                  'S',
#                  'S(Phosphorylation)',
#                  'T',
#                  'T(Phosphorylation)',
#                  'W',
#                  'Y',
#                  "Y(Phosphorylation)",
#                  'V',
#                 ]

# vocab_reverse = _START_VOCAB + vocab_reverse
# print("vocab_reverse ", vocab_reverse)

# vocab = dict([(x, y) for (y, x) in enumerate(vocab_reverse)])
# print("vocab ", vocab)

# vocab_size = len(vocab_reverse)
# print("vocab_size ", vocab_size)


# ==============================================================================
# GLOBAL VARIABLES for THEORETICAL MASS
# ==============================================================================


mass_H = 1.0078
mass_H2O = 18.0106
mass_NH3 = 17.0265
mass_N_terminus = 1.0078
mass_C_terminus = 17.0027
mass_CO = 27.9949
mass_Phosphorylation = 79.96633

if dataset == 'hc_pt':
    vocab_reverse = ['A',
                     'R',
                     'N',
                     'D',
                     'C',
                     'E',
                     'Q',
                     'G',
                     'H',
                     'I',
                     'L',
                     'K',
                     'M',
                     'M(Oxidation)',
                     'F',
                     'P',
                     'S',
                     'T',
                     'W',
                     'Y',
                     'V',
                     ]
    mass_AA = {'_PAD': 0.0,
           '_GO': mass_N_terminus-mass_H,
           '_EOS': mass_C_terminus+mass_H,
           'A': 71.03711, # 0
           'R': 156.10111, # 1
           'N': 114.04293, # 2
           'D': 115.02694, # 3
           'C': 160.03065, # C(+57.02)
           'E': 129.04259, # 5
           'Q': 128.05858, # 6
           'G': 57.02146, # 7
           'H': 137.05891, # 8
           'I': 113.08406, # 9
           'L': 113.08406, # 10
           'K': 128.09496, # 11
           'M': 131.04049, # 12
           'M(Oxidation)': 147.0354,
           'F': 147.06841, # 13
           'P': 97.05276, # 14
           'S': 87.03203, # 15
           'T': 101.04768, # 16
           'W': 186.07931, # 17
           'Y': 163.06333, # 18
           'V': 99.06841, # 19
          }
else:
    vocab_reverse = ['A',
                     'R',
                     'N',
                     'N(Deamidation)',
                     'D',
                     'C(Carbamidomethylation)',
                     'E',
                     'Q',
                     'Q(Deamidation)',
                     'G',
                     'H',
                     'I',
                     'L',
                     'K',
                     'M',
                     'M(Oxidation)',
                     'F',
                     'P',
                     'S',
                     'T',
                     'W',
                     'Y',
                     'V',
                     ]
    mass_AA = {'_PAD': 0.0,
           '_GO': mass_N_terminus-mass_H,
           '_EOS': mass_C_terminus+mass_H,
           'A': 71.03711, # 0
           'R': 156.10111, # 1
           'N': 114.04293, # 2
           'N(Deamidation)': 115.02695,
           'D': 115.02694, # 3
           #~ 'C(Carbamidomethylation)': 103.00919, # 4
           'C(Carbamidomethylation)': 160.03065, # C(+57.02)
           #~ 'C(Carbamidomethylation)': 161.01919, # C(+58.01) # orbi
           'E': 129.04259, # 5
           'Q': 128.05858, # 6
           'Q(Deamidation)': 129.0426,
           'G': 57.02146, # 7
           'H': 137.05891, # 8
           'I': 113.08406, # 9
           'L': 113.08406, # 10
           'K': 128.09496, # 11
           'M': 131.04049, # 12
           'M(Oxidation)': 147.0354,
           'F': 147.06841, # 13
           'P': 97.05276, # 14
           'S': 87.03203, # 15
           'T': 101.04768, # 16
           'W': 186.07931, # 17
           'Y': 163.06333, # 18
           'V': 99.06841, # 19
          }
vocab_reverse = _START_VOCAB + vocab_reverse
print("vocab_reverse ", vocab_reverse)

vocab = dict([(x, y) for (y, x) in enumerate(vocab_reverse)])
print("vocab ", vocab)

vocab_size = len(vocab_reverse)
print("vocab_size ", vocab_size)
# mass_AA = {'_PAD': 0.0,
#            '_GO': mass_N_terminus-mass_H,
#            '_EOS': mass_C_terminus+mass_H,
#            'A': 71.03711, # 0
#            'R': 156.10111, # 1
#            'N': 114.04293, # 2
#            'N(Deamidation)': 115.02695,
#            'D': 115.02694, # 3
#            #~ 'C(Carbamidomethylation)': 103.00919, # 4
#            'C(Carbamidomethylation)': 160.03065, # C(+57.02)
#            #~ 'C(Carbamidomethylation)': 161.01919, # C(+58.01) # orbi
#            'E': 129.04259, # 5
#            'Q': 128.05858, # 6
#            'Q(Deamidation)': 129.0426,
#            'G': 57.02146, # 7
#            'H': 137.05891, # 8
#            'I': 113.08406, # 9
#            'L': 113.08406, # 10
#            'K': 128.09496, # 11
#            'M': 131.04049, # 12
#            'M(Oxidation)': 147.0354,
#            'F': 147.06841, # 13
#            'P': 97.05276, # 14
#            'S': 87.03203, # 15
#            'S(Phosphorylation)': 87.03203 + mass_Phosphorylation,
#            'T': 101.04768, # 16
#            'T(Phosphorylation)': 101.04768 + mass_Phosphorylation,
#            'W': 186.07931, # 17
#            'Y': 163.06333, # 18
#            'Y(Phosphorylation)': 163.06333 + mass_Phosphorylation,
#            'V': 99.06841, # 19
#           }

mass_ID = [mass_AA[vocab_reverse[x]] for x in range(vocab_size)]
mass_ID_np = np.array(mass_ID, dtype=np.float32)

mass_AA_min = mass_AA["G"] # 57.02146


# ==============================================================================
# GLOBAL VARIABLES for PRECISION, RESOLUTION, temp-Limits of MASS & LEN
# ==============================================================================


# if change, need to re-compile cython_speedup << NO NEED
if dataset== 'seven_species':
    SPECTRUM_RESOLUTION = 10
elif dataset == 'nine_species':
    SPECTRUM_RESOLUTION = 100
elif dataset == 'hc_pt':
    SPECTRUM_RESOLUTION = 1000 
print("SPECTRUM_RESOLUTION ", SPECTRUM_RESOLUTION)

# if change, need to re-compile cython_speedup << NO NEED
WINDOW_SIZE = 10 # 10 bins
print("WINDOW_SIZE ", WINDOW_SIZE)

MZ_MAX = 3000.0
MZ_SIZE = int(MZ_MAX * SPECTRUM_RESOLUTION) # 30k

KNAPSACK_AA_RESOLUTION = 10000 # 0.0001 Da
# KNAPSACK_AA_RESOLUTION = 50 # change
mass_AA_min_round = int(round(mass_AA_min * KNAPSACK_AA_RESOLUTION)) # 57.02146
KNAPSACK_MASS_PRECISION_TOLERANCE = 100 # 0.01 Da
num_position = 0

PRECURSOR_MASS_PRECISION_TOLERANCE = 0.01

# ONLY for accuracy evaluation
#~ PRECURSOR_MASS_PRECISION_INPUT_FILTER = 0.01
#~ PRECURSOR_MASS_PRECISION_INPUT_FILTER = 1000
AA_MATCH_PRECISION = 0.1

# skip (x > MZ_MAX,MAX_LEN)
MAX_LEN = 50 if FLAGS.search_denovo else 30
print("MAX_LEN ", MAX_LEN)


# ==============================================================================
# HYPER-PARAMETERS of the NEURAL NETWORKS
# ==============================================================================


num_ion = 8 # 2
print("num_ion ", num_ion)

weight_decay = 0.0  # no weight decay lead to better result.
print("weight_decay ", weight_decay)

#~ encoding_cnn_size = 4 * (RESOLUTION//10) # 4 # proportion to RESOLUTION
#~ encoding_cnn_filter = 4
#~ print("encoding_cnn_size ", encoding_cnn_size)
#~ print("encoding_cnn_filter ", encoding_cnn_filter)

embedding_size = 512
print("embedding_size ", embedding_size)

num_lstm_layers = 1
num_units = 512
print("num_lstm_layers ", num_lstm_layers)
print("num_units ", num_units)

dropout_rate = 0.25

batch_size = 32
num_workers = 64
print("batch_size ", batch_size)

num_epoch = 30

init_lr = 1e-3

train_stack_size = 500 # 3000 # 5000
valid_stack_size = 1500#1000 # 3000 # 5000
test_stack_size = 5000
decode_stack_size = 1000 # 3000
print("train_stack_size ", train_stack_size)
print("valid_stack_size ", valid_stack_size)
print("test_stack_size ", test_stack_size)
print("decode_stack_size ", decode_stack_size)

steps_per_validation = 3000 # 100 # 2 # 4 # 200
print("steps_per_validation ", steps_per_validation)

max_gradient_norm = 5.0
print("max_gradient_norm ", max_gradient_norm)


# ==============================================================================
# DATASETS
# ==============================================================================


data_format = "mgf"
cleavage_rule = "trypsin"
num_missed_cleavage = 2
fixed_mod_list = ['C']
var_mod_list = ['N', 'Q', 'M']
num_mod = 3
precursor_mass_tolerance = 0.01 # Da
precursor_mass_ppm = 15.0/1000000 # ppm (20 better) # instead of absolute 0.01 Da
topk_output = 1


