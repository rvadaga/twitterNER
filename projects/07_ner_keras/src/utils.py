import argparse
import sys
import logging
import numpy as np

if sys.version_info[0] >= 3:
    unicode = str


def parse_args():
    # initialise parser
    parser = argparse.ArgumentParser(
        description="Train a neural net for NER")

    # add arguments
    parser.add_argument("--train",
                        help="path to training data",
                        required=True)
    parser.add_argument("--dev",
                        help="path to dev data",
                        required=True)
    parser.add_argument("--test",
                        help="path to test data",
                        required=True)
    parser.add_argument("--batch_size",
                        type=int,
                        default=10,
                        help="batch size for training")
    parser.add_argument("--emb",
                        choices=['w2v', 'w2v_twitter', 'glove',
                                 'senna', 'random'],
                        help="embedding to use",
                        required=True)
    parser.add_argument("--emb_path",
                        help="path to embedding directory")
    # TODO is this cell state?
    # change the label appropriately
    parser.add_argument("--num_units",
                        type=int,
                        default=200,
                        help="cell size in LSTM")
    parser.add_argument("--num_filters",
                        type=int,
                        default=200,
                        help="num of filters in CNN")
    parser.add_argument("--fine_tune",
                        action="store_true",
                        help="fine tune the embeddings")
    parser.add_argument("--use_char",
                        action="store_true",
                        default=True,
                        help="use char embeddings")
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="learning rate")
    parser.add_argument("--decay_rate",
                        type=float,
                        default=0.1,
                        help="decay for learning rate")
    parser.add_argument("--grad_clip",
                        type=float,
                        default=0,
                        help="value for gradient clipping")
    parser.add_argument("--gamma",
                        type=float,
                        default=1e-6,
                        help="weight for regularisation")
    parser.add_argument('--oov',
                        choices=['random', 'embedding'],
                        help='embedding for OOV word',
                        required=True)
    parser.add_argument('--orth_word_emb_dim',
                        type=int,
                        default=200,
                        help='embedding dimension for orthographic words',
                        required=True)
    args = parser.parse_args()
    # FIXME:
    # 1. check if args are valid or not
    # 2. add extra args
    return args


def get_logger(name, handler=sys.stdout, level=logging.INFO,
               formatter="%(asctime)s - %(name)s - " +
                         "%(levelname)s - %(message)s"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def loadEmbeddingsFromFile(emb_to_use,
                           emb_path,
                           word_collection,
                           logger,
                           emb_dim=100):
    """
    load word embeddings from file
    :param embedding:
    :param embedding_path:
    :param logger:
    :return: embedding dict, embedding dimension, caseless
    """
    if emb_to_use == 'word2vec':
        # loading word2vec
        logger.info("loading word2vec ...")
        word2vec = Word2Vec.load_word2vec_format(emb_path, binary=True)
        emb_dim = word2vec.vector_size
        return word2vec, emb_dim, False
    elif emb_to_use == 'glove':
        # loading GloVe
        logger.info("loading GloVe ...")
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
        logger.info("loading Senna...")
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
        logger.info("loading random embeddings...")
        emb_dict = dict()
        words = word_collection.get_content()
        scale = np.sqrt(3.0 / emb_dim)
        for word in words:
            emb_dict[word] = np.random.uniform(-scale,
                                               scale,
                                               [1, emb_dim])
        return emb_dict, emb_dim, False
    elif emb_to_use == "w2v_twitter":
        logger.info("loading w2v_twitter embeddings...")
        emb_dict = dict()
        with open(emb_path, "rb") as f:
            header = f.readline()
            vocab_size, emb_size = map(int, header.split())
            binary_len = np.dtype(np.float32).itemsize * emb_size
            # FIXME: change this to vocab_size later
            for idx in xrange(200):
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
        raise ValueError("embedding should choose from [word2vec, senna]")


def randomlyInitialiseOrthEmbeddings(orth_word_collection,
                                     logger,
                                     emb_dim=200):
    """
    load word embeddings from file
    :param embedding:
    :param logger:
    :return: embedding dict, embedding dimension
    """
    logger.info("initialising orth word embeddings...")
    emb_dict = dict()
    words = orth_word_collection.get_content()
    scale = np.sqrt(3.0 / emb_dim)
    for word in words:
        emb_dict[word] = np.random.uniform(-scale, scale, [1, emb_dim])
    return emb_dict


def to_unicode(text, errors='strict', encoding='utf8'):
    """Convert a string (unicode or bytestring in `encoding`),
    to bytestring in utf8."""
    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors=errors).encode('utf8')
