# import keras
import utils
import data_utils
import network
# TODO:
# 1. Should I use CoNLL format?
#    Then, add label_col and word_col
# 2. add use_char, etc. options
# 3. add dropout, regularisaion
#    https://arxiv.org/pdf/1512.05287.pdf

# get the command line args
args = utils.parse_args()

train_path = args.train
dev_path = args.dev
test_path = args.test
batch_size = args.batch_size
emb_to_use = args.emb
emb_path = args.emb_path
lstm_size = args.num_units
num_filters = args.num_filters
lr = args.lr
decay_rate = args.decay_rate
grad_clip = args.grad_clip
gamma = args.gamma

logger = utils.get_logger("BiLSTM-CNN-CRF")

data = data_utils.load_data(args)

# data["train_word"] = train_data
# data["dev_word"] = dev_data
# data["test_word"] = test_data
# data["word_emb_table"] = word_emb_table
# data["word_collection"] = word_collection
# data["char"] = char_data
# data["label_collection"] = label_collection
# data["train_orth"] = train_data_orth
# data["dev_orth"] = dev_data_orth
# data["test_orth"] = test_data_orth
# data["orth_word_emb_table"] = orth_word_emb_table
# data["orth_char"] = orth_char_data

# numLabels = label_collection.size() - 1

logger.info("building network...")

network.build_network(args, data)
