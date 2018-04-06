from keras.models import Model
from keras.layers import Input, Reshape, Permute, Concatenate
from keras.layers import Embedding, Conv2D, MaxPool2D, Dense
from keras.layers import LSTM, Bidirectional, TimeDistributed
from crf import ChainCRF
from constants import *
import sys


def get_word_embeddings(args, data, input_sequence, type):
    # shape of emb_table:
    # word_collection_size, emb_dim
    if type == "word":
        emb_table = data["word_emb_table"]
    elif type == "orth_word":
        emb_table = data["orth_word_emb_table"]
    else:
        print "Wrong embedding specified"
        sys.exit(1)

    max_sentence_len = data["train_word"][0].shape[1]

    # Input shape: batch_size, max_sentence_len
    # Output shape: batch_size, max_sentence_len, emb_dim
    emb_output = Embedding(
            input_dim=emb_table.shape[0],
            output_dim=emb_table.shape[1],
            weights=[emb_table],
            input_shape=(max_sentence_len,),
            trainable=args.fine_tune,
            name=type + "_emb") (input_sequence)

    return emb_output


def get_char_based_embeddings(args, data, input_sequence, type):
    # max_sentence_len and max_word_len are the same
    # for both char and orth_char embeddings.

    # data["char"][0] denotes char_train data
    # shape: batch_size, max_sentence_len, max_word_len
    max_sentence_len = data["char"][0].shape[1]
    max_word_len = data["char"][0].shape[-1]

    # FIXME Dropout

    # char_emb_table has the shape
    # char_collection_size, char_emb_dim
    if type == "char":
        char_emb_table = data["char"][3]
    elif type == "orth_char":
        char_emb_table = data["orth_char"][3]

    char_emb_dim = char_emb_table.shape[1]

    # Input shape: batch_size, max_sentence_len * max_word_len
    # Output shape: batch_size, max_sentence_len * max_word_len, char_emb_dim

    emb_output = Embedding(
        input_dim=char_emb_table.shape[0],
        output_dim=char_emb_dim,
        weights=[char_emb_table],
        input_length=(max_sentence_len * max_word_len),
        name=type + "_emb_1") (input_sequence)

    # Input shape: batch_size, max_sentence_len * max_word_len, char_emb_dim
    # Output shape: batch_size, max_sentence_len, max_word_len, char_emb_dim
    reshape_output_1 = Reshape(
        target_shape=(max_sentence_len, max_word_len, char_emb_dim),
        name=type + "_reshape_2") (emb_output)

    # Input shape: batch_size, max_sentence_len, max_word_len, char_emb_dim
    # Output shape: batch_size, char_emb_dim, max_sentence_len, max_word_len
    permute_output_1 = Permute(
        dims=(3, 1, 2),
        name=type + "_permute_3") (reshape_output_1)

    # Input Shape: batch_size, char_emb_dim, max_sentence_len, max_word_len
    # Output Shape: batch_size, num_filters, max_sentence_len, max_word_len
    num_filters = args.num_filters
    conv_output = Conv2D(
        filters=num_filters,
        kernel_size=(1, CONV_WINDOW),
        strides=1,
        padding="same",
        data_format="channels_first",
        activation="tanh",
        name=type + "_conv2d_4") (permute_output_1)

    # Input Shape: batch_size, num_filters, max_sentence_len, max_word_len
    # Output Shape: batch_size, num_filters, max_sentence_len, 1
    maxpool_output = MaxPool2D(
        pool_size=(1, max_word_len),
        data_format="channels_first",
        name=type + "_maxpool_5") (conv_output)

    # Input Shape: batch_size, num_filters, max_sentence_len, 1
    # Output shape: batch_size, num_filters, max_sentence_len
    reshape_output_2 = Reshape(
        target_shape=(num_filters, max_sentence_len),
        name=type + "_reshape_6") (maxpool_output)

    # Input shape: batch_size, num_filters, max_sentence_len
    # Output shape: batch_size, max_sentence_len, num_filters
    permute_output_2 = Permute(
        dims=(2, 1),
        name=type + "_word_emb") (reshape_output_2)

    return permute_output_2


def get_bi_lstm_output(args, data, inputs):
    # Inputs are
    #   1. char_based_word_emb
    #   2. word_emb
    #   3. orth_char_based_word_emb
    #   4. orth_word_emb
    # Output shape:
    # batch_size, max_sentence_len, <sum_of_all_emb_dim>
    concatenated_inputs = Concatenate(
        axis=-1) (inputs)

    # Output shape:
    # batch_size, max_sentence_len, 2*num_units
    output = Bidirectional(LSTM(
        units=args.num_units,
        activation="sigmoid",
        recurrent_activation="tanh",
        return_sequences=True)) (concatenated_inputs)

    return output


def build_network(args, data):
    max_sentence_len = data["char"][0].shape[1]
    max_word_len = data["char"][0].shape[-1]
    num_labels = data["label_collection"].size() - 1

    # Input tensor contains char indices
    # for all words in a given batch of sentences
    # Shape: batch_size, max_sentence_len * max_word_len
    char_input = Input(
        shape=(max_sentence_len * max_word_len,),
        name="char_input",
        dtype="int32")

    # 3D tensor containing char based word embeddings
    # for the given batch of sentences
    # Shape: batch_size, max_sentence_len, num_filters
    char_based_word_emb = get_char_based_embeddings(
        args, data, char_input, "char")

    # Input tensor containing word indices
    # for the given batch of sentences
    # Shape: batch_size, max_sentence_len
    word_input = Input(
        shape=(max_sentence_len,),
        name="word_input",
        dtype="int32")

    # 3D tensor containing word embeddings
    # for the given batch of sentences
    # Shape: batch_size, max_sentence_len, word_emb_dim
    word_emb = get_word_embeddings(args, data, word_input, "word")

    # Input tensor contains orth char indices
    # for all words in a given batch of sentences
    # Shape: batch_size, max_sentence_len * max_word_len
    orth_char_input = Input(
        shape=(max_sentence_len * max_word_len,),
        name="orth_char_input",
        dtype="int32")

    # 3D tensor containing orth char based word embeddings
    # for the given batch of sentences
    # Shape: batch_size, max_sentence_len, num_filters
    orth_char_based_word_emb = get_char_based_embeddings(
        args, data, orth_char_input, "orth_char")

    # Input tensor containing orth word indices
    # for the given batch of sentences
    # Shape: batch_size, max_sentence_len
    orth_word_input = Input(
        shape=(max_sentence_len,),
        name="orth_word_input",
        dtype="int32")

    # 3D tensor containing orth word embeddings
    # for the given batch of sentences
    # Shape: batch_size, max_sentence_len, orth_word_emb_dim
    orth_word_emb = get_word_embeddings(
        args, data, orth_word_input, "orth_word")

    inputs = [char_based_word_emb,
              word_emb,
              orth_char_based_word_emb,
              orth_word_emb]

    bi_lstm_output = get_bi_lstm_output(args, data, inputs)

    lstm_output_dim = bi_lstm_output.shape[2]
    
    hidden_layer_output = TimeDistributed(
        Dense(
            units=num_labels,
            input_shape=(max_sentence_len, lstm_output_dim))) (bi_lstm_output)
    
    crf_output = ChainCRF() (hidden_layer_output)

    model = Model(
        inputs=[char_input, word_input, orth_char_input, orth_word_input],
        outputs=crf_output)

    # model = Model(
    #     inputs=[char_input, word_input, orth_char_input, orth_word_input],
    #     outputs=bi_lstm_output)
    
    model.summary()
