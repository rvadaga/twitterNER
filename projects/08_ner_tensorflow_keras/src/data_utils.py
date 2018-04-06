from __future__ import print_function
# from __future__ import import_function

import sys
import gzip
import utils
import numpy as np
import constants as const
from collection import Collection
from gensim.models.word2vec import Word2Vec
from six.moves import range
from tqdm import tqdm


if sys.version_info[0] >= 3:
    unicode = str


def get_logger(LOG_FILE):
    global logger
    logger = utils.get_logger("PreprocessData", LOG_FILE)


# The dataset is divided into 3 partitions - train, dev and test - and
# all the parameters related to the dataset are stored as variables in
# the class below
class NERData(object):
    def __init__(self, args):
        # Path to data files
        self.train_path = args.wnut + "/" + const.TRAIN_FILE
        self.dev_path = args.wnut + "/" + const.DEV_FILE
        self.test_path = args.wnut + "/" + const.TEST_FILE
        self.train_path_orth = args.wnut + "/" + const.ORTH_TRAIN_FILE
        self.dev_path_orth = args.wnut + "/" + const.ORTH_DEV_FILE
        self.test_path_orth = args.wnut + "/" + const.ORTH_TEST_FILE
        self.emb_path = args.emb_path

        # Bells and whistles of the model
        self.fine_tune = args.fine_tune
        self.oov = args.oov
        self.emb_to_use = args.emb

        # Hyperparameters
        self.orth_word_emb_dim = args.orth_word_emb_dim
        self.word_emb_dim = 0

        # Objects to store collection of words, chars (both normal and
        # orthographic) and the labels
        self.word_collection = Collection("word")
        self.orth_word_collection = Collection("orth_word")
        self.char_collection = Collection("char")
        self.orth_char_collection = Collection("orth_char")
        self.label_collection = Collection("label")

        self.logger = logger

        # Variables to store the data from the files
        self.words = {}
        self.orth_words = {}
        self.labels = {}
        self.word_indices = {}
        self.orth_word_indices = {}
        self.label_indices = {}
        self.char_indices = {}
        self.orth_char_indices = {}

        # Store all the embeddings in this dict
        self.emb = {}
        self.emb_table = {}

        # Lengths of the sentences
        self.sequence_lengths = {}
        self.max_lengths = {}
        self.max_sentence_length = 0
        self.max_word_length = 0
        self.num_labels = 0

        # For final train, dev and test data
        self.train_data = {}
        self.dev_data = {}
        self.test_data = {}

    def read_data(self):
        logger = self.logger

        word_collection = self.word_collection
        orth_word_collection = self.orth_word_collection
        label_collection = self.label_collection

        words = self.words
        orth_words = self.orth_words
        labels = self.labels
        word_indices = self.word_indices
        orth_word_indices = self.orth_word_indices
        label_indices = self.label_indices

        logger.info("Reading words from train file")
        words["train"], labels["train"], word_indices["train"], \
            label_indices["train"] = readDataFromFile(
                self.train_path, word_collection, label_collection)

        logger.info("Reading orthographic words from train file")
        orth_words["train"], orth_word_indices["train"] = \
            readOrthDataFromFile(
                self.train_path_orth, orth_word_collection)

        # if oov is "random" and do not fine tune, close word_collection
        # and orth_word_collection
        if self.oov == "random" and not self.fine_tune:
            logger.warning("Closed word and orth_word collection")
            word_collection.close()
            orth_word_collection.close()

        logger.info("Reading words from dev file")
        words["dev"], labels["dev"], word_indices["dev"], \
            label_indices["dev"] = readDataFromFile(
                self.dev_path, word_collection, label_collection)

        logger.info("Reading orthographic words from dev file")
        orth_words["dev"], orth_word_indices["dev"] = \
            readOrthDataFromFile(
                self.dev_path_orth, orth_word_collection)

        logger.info("Reading words from test file")
        words["test"], labels["test"], word_indices["test"], \
            label_indices["test"] = readDataFromFile(
                self.test_path, word_collection, label_collection)

        logger.info("Reading orthographic words from test file")
        orth_words["test"], orth_word_indices["test"] = \
            readOrthDataFromFile(
                self.test_path_orth, orth_word_collection)

        logger.info("Closing collection sets")
        word_collection.close()
        orth_word_collection.close()
        label_collection.close()

        logger.info("Word collection size: %d" % (word_collection.size()-1))
        logger.info("Label collection size: %d" % (label_collection.size()-1))
        logger.info("Orthographic word collection size: %d" %
                    (orth_word_collection.size() - 1))

        self.num_labels = label_collection.size() - 1

        self.train_data["length"], self.max_lengths["train"] = \
            get_sequence_lengths(self.words["train"])
        self.dev_data["length"], self.max_lengths["dev"] = \
            get_sequence_lengths(self.words["dev"])
        self.test_data["length"], self.max_lengths["test"] = \
            get_sequence_lengths(self.words["test"])
        self.max_sentence_length = min(const.MAX_LENGTH,
                                       max(self.max_lengths["train"],
                                           self.max_lengths["dev"],
                                           self.max_lengths["test"]))

        logger.info("Max sentence length of train set: %d" %
                    self.max_lengths["train"])
        logger.info("Max sentence length of development set: %d" %
                    self.max_lengths["dev"])
        logger.info("Max sentence length of test set: %d" %
                    self.max_lengths["test"])
        logger.info("Max sentence length for training: %d" %
                    self.max_sentence_length)

    def preprocess_data_fine_tune(self):
        logger.info("Loading embeddings from file")
        self.emb["word"], self.word_emb_dim, self.caseless_emb = \
            loadEmbeddingsFromFile(self.emb_to_use, self.emb_path,
                                   self.word_collection)
        logger.info("Word embedding dimension: %d, case: %s" %
                    (self.word_emb_dim, not self.caseless_emb))

        logger.info("Initialising orthographic word embeddings")
        self.emb["orth_word"] = \
            randomlyInitialiseOrthEmbeddings(self.orth_word_collection,
                                             self.orth_word_emb_dim)
        logger.info("Orthographic word embedding dimension: %d" %
                    (self.orth_word_emb_dim))

        logger.info("Generating dataset for training")
        # train data
        self.train_data["word"], self.train_data["label"], \
            self.train_data["mask"] = construct_tensor_fine_tune(
                self.word_indices["train"], self.label_indices["train"],
                self.max_sentence_length)

        self.train_data["orth_word"] = construct_orth_tensor_fine_tune(
            self.orth_word_indices["train"], self.max_sentence_length)

        # dev data
        self.dev_data["word"], self.dev_data["label"], \
            self.dev_data["mask"] = construct_tensor_fine_tune(
                self.word_indices["dev"], self.label_indices["dev"],
                self.max_sentence_length)

        self.dev_data["orth_word"] = construct_orth_tensor_fine_tune(
            self.orth_word_indices["dev"], self.max_sentence_length)

        # test data
        self.test_data["word"], self.test_data["label"], \
            self.test_data["mask"] = construct_tensor_fine_tune(
                self.word_indices["test"], self.label_indices["test"],
                self.max_sentence_length)

        self.test_data["orth_word"] = construct_orth_tensor_fine_tune(
            self.orth_word_indices["test"], self.max_sentence_length)

        self.generate_character_data("char")
        self.generate_character_data("orth_char")

        self.emb_table["word"] = build_embedd_table(
            self.word_collection, self.emb["word"], self.word_emb_dim,
            self.caseless_emb)
        self.emb_table["orth_word"] = build_embedd_table(
            self.orth_word_collection, self.emb["orth_word"],
            self.orth_word_emb_dim, False)

    def generate_character_data(self, collection_type):
        # character data
        if collection_type == "char":
            collection = self.char_collection
            words = self.words
            chars = self.char_indices
            collection.get_index(const.WORD_END)
        elif collection_type == "orth_char":
            collection = self.orth_char_collection
            words = self.orth_words
            chars = self.orth_char_indices
            collection.get_index(const.ORTH_WORD_END)
        else:
            print("Invalid collection specified")

        chars["train"], self.max_lengths["train_char"] = \
            get_character_indexes(words["train"], collection)
        chars["dev"], self.max_lengths["dev_char"] = \
            get_character_indexes(words["dev"], collection)
        chars["test"], self.max_lengths["test_char"] = \
            get_character_indexes(words["test"], collection)

        # close collection
        collection.close()
        logger.info(collection_type +
                    " collection size: %d" %
                    (collection.size() - 1))

        self.max_word_length = min(
            const.MAX_CHAR_LENGTH,
            max(self.max_lengths["train_char"],
                self.max_lengths["dev_char"],
                self.max_lengths["test_char"]))
        logger.info(collection_type +
                    ": maximum char length in training set: %d" %
                    self.max_lengths["train_char"])
        logger.info(collection_type +
                    ": maximum char length in dev set: %d" %
                    self.max_lengths["dev_char"])
        logger.info(collection_type +
                    ": maximum char length in test set: %d" %
                    self.max_lengths["test_char"])
        logger.info(collection_type +
                    ": maximum char length used for training: %d" %
                    self.max_word_length)

        # fill character tensor
        self.train_data[collection_type] = np.empty([len(chars["train"]),
                                                     self.max_sentence_length,
                                                     self.max_word_length])
        construct_tensor_char(self.train_data[collection_type],
                              chars["train"],
                              collection_type,
                              collection)

        self.dev_data[collection_type] = np.empty([len(chars["dev"]),
                                                   self.max_sentence_length,
                                                   self.max_word_length])
        construct_tensor_char(self.dev_data[collection_type],
                              chars["dev"],
                              collection_type,
                              collection)

        self.test_data[collection_type] = np.empty([len(chars["test"]),
                                                    self.max_sentence_length,
                                                    self.max_word_length])
        construct_tensor_char(self.test_data[collection_type],
                              chars["test"],
                              collection_type,
                              collection)

        self.emb_table[collection_type] = build_char_embedd_table(collection)


def construct_tensor_char(char_array,
                          index_sentences,
                          collection_type,
                          collection):
    if collection_type == "char":
        word_end_id = collection.get_index(const.WORD_END)
    else:
        word_end_id = collection.get_index(const.ORTH_WORD_END)

    for i in range(len(index_sentences)):
        words = index_sentences[i]
        sent_length = len(words)
        for j in range(sent_length):
            chars = words[j]
            char_length = len(chars)
            for k in range(char_length):
                cid = chars[k]
                char_array[i, j, k] = cid
            # fill index of word end after the end of word
            char_array[i, j, char_length:] = word_end_id
        # Zero out char_array after the end of the sentence
        char_array[i, sent_length:, :] = 0
    char_array = np.reshape(char_array, (len(index_sentences), -1))
    return char_array


def construct_tensor_fine_tune(word_index_sentences, label_index_sentences,
                               max_sentence_length):
    X = np.empty([len(word_index_sentences), max_sentence_length],
                 dtype=np.int32)
    Y = np.empty([len(word_index_sentences), max_sentence_length],
                 dtype=np.int32)
    mask = np.zeros([len(word_index_sentences), max_sentence_length],
                    dtype=np.int32)

    for i in range(len(word_index_sentences)):
        word_ids = word_index_sentences[i]
        label_ids = label_index_sentences[i]
        length = len(word_ids)
        for j in range(length):
            wid = word_ids[j]
            label = label_ids[j]
            X[i, j] = wid
            Y[i, j] = label - 1

        # Zero out X after the end of the sequence
        X[i, length:] = 0
        # Copy the last label after the end of the sequence
        Y[i, length:] = Y[i, length - 1]
        # Make the mask for this sample 1 within the range of length
        mask[i, :length] = 1
    return X, Y, mask


def build_char_embedd_table(collection, char_embedd_dim=30):
    scale = np.sqrt(3.0 / char_embedd_dim)
    char_embedd_table = np.random.uniform(
        -scale,
        scale,
        [collection.size(), char_embedd_dim]).astype(
            np.float32)
    return char_embedd_table


def construct_orth_tensor_fine_tune(
        orth_word_index_sentences, max_sentence_length):
    X = np.empty([len(orth_word_index_sentences), max_sentence_length],
                 dtype=np.int32)

    for i in range(len(orth_word_index_sentences)):
        orth_word_ids = orth_word_index_sentences[i]
        length = len(orth_word_ids)
        for j in range(length):
            wid = orth_word_ids[j]
            X[i, j] = wid

        # Zero out X after the end of the sequence
        X[i, length:] = 0
    return X


def loadEmbeddingsFromFile(emb_to_use, emb_path, word_collection, emb_dim=100):
    """Load word embeddings from file
    Args:
        emb_to_use: should be one of "w2v", "w2v_twitter", "glove", "senna",
            "random"
        emb_path: embedding file location
        word_collection: word collection object
        emb_dim: embedding dimension to use for random embeddings

    Returns:
        embedding_dict: dictionary of word embeddings
        word_emb_dim: length of the embeddings
        caseless: True if words have case
    """
    if emb_to_use == 'word2vec':
        # loading word2vec
        logger.info("Loading word2vec embeddings")
        word2vec = Word2Vec.load_word2vec_format(emb_path, binary=True)
        emb_dim = word2vec.vector_size
        return word2vec, emb_dim, False
    elif emb_to_use == 'glove':
        # loading GloVe
        logger.info("Loading GloVe embeddings")
        emb_dim = -1
        emb_dict = dict()
        with gzip.open(emb_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split()
                if emb_dim < 0:
                    emb_dim = len(tokens) - 1
                else:
                    assert (emb_dim + 1 == len(tokens))
                embedd = np.empty([1, emb_dim], dtype=np.float32)
                embedd[:] = tokens[1:]
                emb_dict[tokens[0]] = embedd
        return emb_dict, emb_dim, True
    elif emb_to_use == 'senna':
        # loading Senna
        logger.info("Loading Senna embeddings")
        emb_dim = -1
        emb_dict = dict()
        with gzip.open(emb_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split()
                if emb_dim < 0:
                    emb_dim = len(tokens) - 1
                else:
                    assert (emb_dim + 1 == len(tokens))
                embedd = np.empty([1, emb_dim], dtype=np.float32)
                embedd[:] = tokens[1:]
                emb_dict[tokens[0]] = embedd
        return emb_dict, emb_dim, True
    elif emb_to_use == 'random':
        # loading random embedding table
        logger.info("Loading random embeddings")
        emb_dict = dict()
        words = word_collection.get_content()
        scale = np.sqrt(3.0 / emb_dim)
        for word in words:
            emb_dict[word] = np.random.uniform(-scale,
                                               scale,
                                               [1, emb_dim])
        return emb_dict, emb_dim, False
    elif emb_to_use == "w2v_twitter":
        logger.info("Loading w2v_twitter embeddings")
        emb_dict = dict()
        with open(emb_path, "rb") as f:
            header = f.readline()
            vocab_size, emb_size = map(int, header.split())
            binary_len = np.dtype(np.float32).itemsize * emb_size
            # FIXME: change this to vocab_size later
            for idx in tqdm(range(vocab_size)):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch != b'\n':  # ignore newlines in front of words
                        # (some binary files have newline, some don't)
                        word.append(ch)
                word = to_unicode(b''.join(word), encoding='latin-1')
                emb = np.empty([1, emb_size], dtype=np.float32)
                emb = np.fromstring(f.read(binary_len),
                                    dtype=np.float32)
                emb_dict[word] = emb
            return emb_dict, emb_size, False
    else:
        raise ValueError("Embedding should choose from [word2vec, senna]")


def randomlyInitialiseOrthEmbeddings(orth_word_collection, emb_dim=200):
    """Load word embeddings from file
    Args:
        orth_word_collection: collection object containing instance-idx and
            idx2instance mapping for orth words
        emb_dim: embedding dimension for random embeddings
    Returns:
        emb_dict
    """
    emb_dict = dict()
    words = orth_word_collection.get_content()
    scale = np.sqrt(3.0 / emb_dim)
    for word in words:
        emb_dict[word] = np.random.uniform(-scale, scale, [1, emb_dim])
    return emb_dict


def to_unicode(text, errors='strict', encoding='utf8'):
    """Convert a string (unicode or bytestring in `encoding`),
    to bytestring in utf8.
    """
    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors=errors).encode('utf8')


def readDataFromFile(path, word_collection, label_collection):
    """Reads data from the given path.

    Finds the words  and labels in each sentence and adds them to the
    Collection objects, which assigns the indices.

    Returns an a list of list of word and label indices

    Args:
        path: path to the file to be loaded
        word_collection: collection of words
        label_collection: collection of labels

    Returns:
        word_sentences: list of list of words in the given file
        label_sentences: list of list of labels in the given file
        word_index_sentences: list of list of word indices. Words for
            word indices can be obtained from word collection object
        label_index_sentences: list of list of label indices. Labels
            corresponding to the label indices can be obtained from
            label collection object

    """
    word_sentences = []
    label_sentences = []

    word_index_sentences = []
    label_index_sentences = []

    words = []
    labels = []

    word_ids = []
    label_ids = []

    num_tokens = 0
    with open(path) as file:
        for line in file:
            line.decode('utf-8')
            if line.strip() == "":
                if 0 < len(words) <= const.MAX_LENGTH:
                    word_sentences.append(words[:])
                    label_sentences.append(labels[:])

                    word_index_sentences.append(word_ids[:])
                    label_index_sentences.append(label_ids[:])

                    num_tokens += len(words)
                else:
                    if len(words) != 0:
                        logger.info("ignore sentence with length %d" %
                                    (len(words)))

                words = []
                labels = []

                word_ids = []
                label_ids = []
            else:
                tokens = line.strip().split()
                word = tokens[const.WORD_COLUMN]
                label = tokens[const.LABEL_COLUMN]

                words.append(word)
                labels.append(label)

                word_id = word_collection.get_index(word)
                label_id = label_collection.get_index(label)

                word_ids.append(word_id)
                label_ids.append(label_id)

    if 0 < len(words) <= const.MAX_LENGTH:
        word_sentences.append(words[:])
        label_sentences.append(labels[:])

        word_index_sentences.append(word_ids[:])
        label_index_sentences.append(label_ids[:])

        num_tokens += len(words)
    else:
        if len(words) != 0:
            logger.warning("Ignore sentence with length %d" % (len(words)))

    logger.info("#sentences: %d, #tokens: %d" % (len(word_sentences),
                                                 num_tokens))
    return word_sentences, label_sentences, word_index_sentences, \
        label_index_sentences


def readOrthDataFromFile(path, word_collection):
    """Read orthographic words from file and map them to IDs.
    Returns a list of list of word IDs for the file.

    Args:
        path: path of the file to be loaded
        orth_word_collection: collection of orthpgraphic words
    Returns:
        word_sentences: list of list of orthographic words in the given file
        word_index_sentences: list of list of orth word indices. Words for
            word indices can be obtained from word collection object

    """
    word_sentences = []
    word_index_sentences = []
    words = []
    word_ids = []
    num_tokens = 0
    with open(path) as file:
        for line in file:
            line.decode('utf-8')
            if line.strip() == "":
                if 0 < len(words) <= const.MAX_LENGTH:
                    word_sentences.append(words[:])
                    word_index_sentences.append(word_ids[:])
                    num_tokens += len(words)
                else:
                    if len(words) != 0:
                        logger.info("ignore sentence with length %d" %
                                    (len(words)))

                words = []

                word_ids = []
            else:
                tokens = line.strip().split()
                word = tokens[0]

                words.append(word)
                word_id = word_collection.get_index(word)
                word_ids.append(word_id)

    if 0 < len(words) <= const.MAX_LENGTH:
        word_sentences.append(words[:])
        word_index_sentences.append(word_ids[:])
        num_tokens += len(words)
    else:
        if len(words) != 0:
            logger.info("ignore sentence with length %d" % (len(words)))

    logger.info("#sentences: %d, #tokens: %d" % (len(word_sentences),
                                                 num_tokens))
    return word_sentences, word_index_sentences


def get_sequence_lengths(word_sentences):
    max_len = 0
    lengths = []
    for sentence in word_sentences:
        length = len(sentence)
        lengths.append(length)
        if length > max_len:
            max_len = length
    return np.array(lengths), max_len


def get_character_indexes(sentences, char_collection):
    index_sentences = []
    max_length = 0
    for words in sentences:
        index_words = []
        for word in words:
            index_chars = []
            if len(word) > max_length:
                max_length = len(word)

            for char in word[:const.MAX_CHAR_LENGTH]:
                char_id = char_collection.get_index(char)
                index_chars.append(char_id)

            index_words.append(index_chars)
        index_sentences.append(index_words)
    return index_sentences, max_length


def build_embedd_table(word_collection, word_emb_dict, word_emb_dim, caseless):
    scale = np.sqrt(3.0 / word_emb_dim)
    embedd_table = np.empty([word_collection.size(), word_emb_dim],
                            dtype=np.float32)
    embedd_table[word_collection.default_index, :] = \
        np.random.uniform(-scale, scale, [1, word_emb_dim])
    for word, index in word_collection.iteritems():
        ww = word.lower() if caseless else word
        embedd = word_emb_dict[ww] if ww in word_emb_dict else \
            np.random.uniform(-scale, scale, [1, word_emb_dim])
        embedd_table[index, :] = embedd
    return embedd_table


def shuffle_data(data):
    n_examples = data["word"].shape[0]
    idx = np.arange(n_examples)
    np.random.shuffle(idx)
    data_shuffled = {}
    for key in data.iterkeys():
        data_shuffled[key] = data[key][idx]
    return data_shuffled
