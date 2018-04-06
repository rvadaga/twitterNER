# written by Xeuzhe Max
# link: https://github.com/XuezheMax/LasagneNLP

import numpy as np
from collection import Collection
import utils
from constants import *

logger = utils.get_logger("Load Data")


def readDataFromFile(path,
                     word_collection,
                     label_collection):
    """
    Read data from file in CoNLL format

    Input:
        path:
                path to the
                file to be loaded
        word_collection:
                collection of words
        label_collection:
                collection of labels

    Returns:
        word_sentences:
                list of list of words in the file,
                where inner list is list of words in
                a sentence
        label_sentences:
                list of list of labels in the file,
                with the inner list holding labels
                of each word in a sentence
        word_index_sentences:
                list of list of word indices, where each
                inner list holds indices of each word (from word_collection)
                in a sentence
        label_index_sentences:
                list of list of label indices, where
                each inner list holds indices of labels
                of each word (from word_collection)
                in a sentence
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
                if 0 < len(words) <= MAX_LENGTH:
                    word_sentences.append(words[:])  # TODO: Is this
                    # here necessary?
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
                word = tokens[WORD_COLUMN]
                label = tokens[LABEL_COLUMN]

                words.append(word)
                labels.append(label)

                word_id = word_collection.get_index(word)
                label_id = label_collection.get_index(label)

                word_ids.append(word_id)
                label_ids.append(label_id)

    if 0 < len(words) <= MAX_LENGTH:
        word_sentences.append(words[:])
        label_sentences.append(labels[:])

        word_index_sentences.append(word_ids[:])
        label_index_sentences.append(label_ids[:])

        num_tokens += len(words)
    else:
        if len(words) != 0:
            logger.info("ignore sentence with length %d" % (len(words)))

    logger.info("#sentences: %d, #tokens: %d" % (len(word_sentences),
                                                 num_tokens))
    return (word_sentences,
            label_sentences,
            word_index_sentences,
            label_index_sentences)


def readOrthDataFromFile(path,
                         word_collection):
    """
    Read orthographic words from file

    Input:
        path:
                path to the
                file to be loaded
        orth_word_collection:
                collection of words
    Returns:
        word_sentences:
                list of list of words in the file,
                where inner list is list of words in
                a sentence
        word_index_sentences:
                list of list of word indices, where each
                inner list holds indices of each word (from word_collection)
                in a sentence
    """
    word_sentences = []
    word_index_sentences = []
    words = []
    word_ids = []
    num_tokens = 0
    path = path + "_orth"
    with open(path) as file:
        for line in file:
            line.decode('utf-8')
            if line.strip() == "":
                if 0 < len(words) <= MAX_LENGTH:
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

    if 0 < len(words) <= MAX_LENGTH:
        word_sentences.append(words[:])
        word_index_sentences.append(word_ids[:])
        num_tokens += len(words)
    else:
        if len(words) != 0:
            logger.info("ignore sentence with length %d" % (len(words)))

    logger.info("#sentences: %d, #tokens: %d" % (len(word_sentences),
                                                 num_tokens))
    return word_sentences, word_index_sentences


def get_max_length(word_sentences):
    max_len = 0
    for sentence in word_sentences:
        length = len(sentence)
        if length > max_len:
            max_len = length
    return max_len


def generate_character_data(
        sentences_train,
        sentences_dev,
        sentences_test,
        max_sent_length,
        collectionType,
        char_embedd_dim=30):
    """
    generate data for charaters
    :param sentences_train:
    :param sentences_dev:
    :param sentences_test:
    :param max_sent_length:
    :return: C_train, C_dev, C_test, char_embedd_table
    """

    def get_character_indexes(sentences):
        index_sentences = []
        max_length = 0
        for words in sentences:
            index_words = []
            for word in words:
                index_chars = []
                if len(word) > max_length:
                    max_length = len(word)

                for char in word[:MAX_CHAR_LENGTH]:
                    char_id = char_collection.get_index(char)
                    index_chars.append(char_id)

                index_words.append(index_chars)
            index_sentences.append(index_words)
        return index_sentences, max_length

    def construct_tensor_char(index_sentences):
        C = np.empty([len(index_sentences), max_sent_length, max_char_length],
                     dtype=np.int32)
        if collectionType == "char":
            word_end_id = char_collection.get_index(WORD_END)
        else:
            word_end_id = char_collection.get_index(ORTH_WORD_END)

        for i in range(len(index_sentences)):
            words = index_sentences[i]
            sent_length = len(words)
            for j in range(sent_length):
                chars = words[j]
                char_length = len(chars)
                for k in range(char_length):
                    cid = chars[k]
                    C[i, j, k] = cid
                # fill index of word end after the end of word
                C[i, j, char_length:] = word_end_id
            # Zero out C after the end of the sentence
            C[i, sent_length:, :] = 0
        np.reshape(C, (len(index_sentences), max_sent_length * max_char_length))
        return C

    def build_char_embedd_table():
        scale = np.sqrt(3.0 / char_embedd_dim)
        char_embedd_table = np.random.uniform(
            -scale,
            scale,
            [char_collection.size(), char_embedd_dim]).astype(
                np.float32)
        return char_embedd_table

    char_collection = Collection(collectionType)
    if collectionType == "char":
        char_collection.get_index(WORD_END)
    else:
        char_collection.get_index(ORTH_WORD_END)

    index_sentences_train, max_char_length_train = \
        get_character_indexes(sentences_train)
    index_sentences_dev, max_char_length_dev = \
        get_character_indexes(sentences_dev)
    index_sentences_test, max_char_length_test = \
        get_character_indexes(sentences_test)

    # close character collection
    char_collection.close()
    logger.info(collectionType +
                ": char collection size: %d" %
                (char_collection.size() - 1))

    max_char_length = min(
        MAX_CHAR_LENGTH,
        max(max_char_length_train,
            max_char_length_dev,
            max_char_length_test))
    logger.info(collectionType +
                ": maximum char length in training set: %d" %
                max_char_length_train)
    logger.info(collectionType +
                ": maximum char length in dev set: %d" %
                max_char_length_dev)
    logger.info(collectionType +
                ": maximum char length in test set: %d" %
                max_char_length_test)
    logger.info(collectionType +
                ": maximum char length used for training: %d" %
                max_char_length)

    # fill character tensor
    C_train = construct_tensor_char(index_sentences_train)
    C_dev = construct_tensor_char(index_sentences_dev)
    C_test = construct_tensor_char(index_sentences_test)

    return (C_train, C_dev, C_test, build_char_embedd_table())


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


def load_data(args):
    # TODO: write doc string
    def construct_tensor_fine_tune(word_index_sentences,
                                   label_index_sentences):
        X = np.empty([len(word_index_sentences), max_length], dtype=np.int32)
        Y = np.empty([len(word_index_sentences), max_length], dtype=np.int32)
        mask = np.zeros([len(word_index_sentences), max_length],
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
        return (X, Y, mask)

    def construct_orth_tensor_fine_tune(orth_word_index_sentences):
        X = np.empty([len(orth_word_index_sentences), max_length],
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

    def generateDatasetFineTune():
        """
        generate data to be feed to the network
        Returns:
        # TODO
        """
        word_emb_dict, word_emb_dim, caseless = utils.loadEmbeddingsFromFile(
            emb_to_use,
            emb_path,
            word_collection,
            logger)
        logger.info("word embedding dimension: %d, case: %s" %
                    (word_emb_dim, not caseless))
        orth_word_emb_dict = \
            utils.randomlyInitialiseOrthEmbeddings(orth_word_collection,
                                                   logger,
                                                   orth_word_emb_dim)
        logger.info("orthographic word embedding dimension: %d" %
                    (orth_word_emb_dim))
        # fill data tensor (X.shape = [#data, max_length],
        #                   Y.shape = [#data, max_length])
        train_data = construct_tensor_fine_tune(
            word_index_sentences_train,
            label_index_sentences_train)
        # X_train_orth
        train_data_orth = construct_orth_tensor_fine_tune(
            orth_word_index_sentences_train)

        # X_dev, Y_dev, mask_dev
        dev_data = construct_tensor_fine_tune(
            word_index_sentences_dev,
            label_index_sentences_dev)
        X_dev_orth = construct_orth_tensor_fine_tune(
            orth_word_index_sentences_dev)

        # X_test, Y_test, mask_test = construct_tensor_fine_tune(
        test_data = construct_tensor_fine_tune(
            word_index_sentences_test,
            label_index_sentences_test)
        # X_test_orth = construct_orth_tensor_fine_tune(
        test_data_orth = construct_orth_tensor_fine_tune(
            orth_word_index_sentences_test)

        # C_train, C_dev, C_test, char_emb_table = generate_character_data(
        char_data = generate_character_data(
            word_sentences_train,
            word_sentences_dev,
            word_sentences_test,
            max_length,
            "char",
            30) if args.use_char else \
            (None, None, None, None)
        # orth_C_train, orth_C_dev, orth_C_test, orth_char_emb_table = \
        orth_char_data = generate_character_data(
            orth_word_sentences_train,
            orth_word_sentences_dev,
            orth_word_sentences_test,
            max_length,
            "orth_char",
            30) if args.use_char else \
            (None, None, None, None)
        word_emb_table = build_embedd_table(
            word_collection,
            word_emb_dict,
            word_emb_dim,
            caseless)
        orth_word_emb_table = build_embedd_table(
            orth_word_collection,
            orth_word_emb_dict,
            orth_word_emb_dim,
            False)
        data = {}
        data["train_word"] = train_data
        data["dev_word"] = dev_data
        data["test_word"] = test_data
        data["word_emb_table"] = word_emb_table
        data["word_collection"] = word_collection
        data["char"] = char_data
        data["label_collection"] = label_collection
        data["train_orth"] = train_data_orth
        data["dev_orth"] = dev_data_orth
        data["test_orth"] = test_data_orth
        data["orth_word_emb_table"] = orth_word_emb_table
        data["orth_char"] = orth_char_data
        return data

    # ####################
    # Start of load_data()
    # ####################

    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    batch_size = args.batch_size
    emb_to_use = args.emb
    emb_path = args.emb_path
    fine_tune = args.fine_tune
    oov = args.oov
    orth_word_emb_dim = args.orth_word_emb_dim

    word_collection = Collection("word")
    label_collection = Collection("label")
    orth_word_collection = Collection("orth_word")

    # read training data
    logger.info("reading data from train set...")
    train_data = readDataFromFile(
        train_path,
        word_collection,
        label_collection)
    word_sentences_train = train_data[0]
    word_index_sentences_train = train_data[2]
    label_index_sentences_train = train_data[3]

    logger.info("reading data from orth train set...")
    train_data_orth = readOrthDataFromFile(
        train_path,
        orth_word_collection)
    orth_word_sentences_train = train_data_orth[0]
    orth_word_index_sentences_train = train_data_orth[1]

    # if oov is "random" and do not fine tune, close word_collection
    if oov == "random" and not fine_tune:
        logger.info("closed word collection")
        word_collection.close()
        orth_word_collection.close()  # TODO: What's this for?

    # read dev data
    logger.info("reading data from dev set...")
    dev_data = readDataFromFile(
        dev_path,
        word_collection,
        label_collection)
    word_sentences_dev = dev_data[0]
    word_index_sentences_dev = dev_data[2]
    label_index_sentences_dev = dev_data[3]

    logger.info("reading data from orth dev set...")
    dev_data_orth = readOrthDataFromFile(
        dev_path,
        orth_word_collection)
    orth_word_sentences_dev = dev_data_orth[0]
    orth_word_index_sentences_dev = dev_data_orth[1]

    # read test data
    logger.info("reading data from test set...")
    test_data = readDataFromFile(
        test_path,
        word_collection,
        label_collection)
    word_sentences_test = test_data[0]
    word_index_sentences_test = test_data[2]
    label_index_sentences_test = test_data[3]

    logger.info("reading data from orth test set...")
    test_data_orth = readOrthDataFromFile(
        test_path,
        orth_word_collection)
    orth_word_sentences_test = test_data_orth[0]
    orth_word_index_sentences_test = test_data_orth[1]

    # close collection sets
    word_collection.close()
    label_collection.close()
    orth_word_collection.close()

    logger.info("word collection size: %d" % (word_collection.size() - 1))
    logger.info("label collection size: %d" % (label_collection.size() - 1))
    logger.info("orthographic word collection size: %d" %
                (orth_word_collection.size() - 1))

    # get maximum length
    max_length_train = get_max_length(word_sentences_train)
    max_length_dev = get_max_length(word_sentences_dev)
    max_length_test = get_max_length(word_sentences_test)
    max_length = min(MAX_LENGTH,
                     max(max_length_train,
                         max_length_dev,
                         max_length_test))
    logger.info("maximum sentence length of training set: %d" % max_length_train)
    logger.info("maximum sentence length of dev set: %d" % max_length_dev)
    logger.info("maximum sentence length of test set:  %d" % max_length_test)
    logger.info("maximum sentence length used for training: %d" % max_length)

    if fine_tune:
        logger.info("generating data with fine tuning...")
        return generateDatasetFineTune()
    else:
        logger.info("generating data without fine tuning")
        return generateDatasetWithoutFineTune()
