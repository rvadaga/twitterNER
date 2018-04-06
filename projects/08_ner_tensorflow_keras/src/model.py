import tensorflow as tf
import data_utils
import constants as const


def unary_scores(word_input,
                 char_input,
                 orth_input,
                 orth_char_input,
                 seq_length_input,
                 args,
                 wnut):
    # Extract from the pretrained embeddings
    with tf.variable_scope("word"):
        table = wnut.emb_table["word"]
        word_emb_table = tf.get_variable(
            name="word_emb_table",
            shape=[table.shape[0], table.shape[1]],
            dtype=tf.float32,
            initializer=tf.constant_initializer(table))
        word_emb = tf.nn.embedding_lookup(
            word_emb_table, word_input, name="word_emb_output")

    # Get char level representations from char embeddings
    with tf.variable_scope("char"):
        # Get char embeddings
        # char_emb_table = tf.Variable(
        #     wnut.emb_table["char"], name="char_emb_table")
        table = wnut.emb_table["char"]
        char_vocab_size = wnut.emb_table["char"].shape[0]
        char_emb_dim = wnut.emb_table["char"].shape[1]

        char_emb_table = tf.get_variable(
            name="char_emb_table",
            shape=[char_vocab_size, char_emb_dim],
            dtype=tf.float32,
            initializer=tf.constant_initializer(table))

        char_emb = tf.nn.embedding_lookup(
            char_emb_table,
            char_input,
            name="char_emb_output")

        # Reshape the char_emb output
        char_emb_reshape = tf.reshape(
            char_emb,
            [-1, wnut.max_word_length, char_emb_dim])

        # permute the tensor
        char_emb_permute = tf.transpose(char_emb_reshape, perm=[0, 2, 1])

        char_conv_input = tf.layers.dropout(
            char_emb_permute,
            rate=0.5)

        # Perform a conv2d operation
        char_conv_output = tf.layers.conv1d(
            inputs=char_conv_input,
            filters=args.num_filters,
            kernel_size=const.CONV_WINDOW,
            strides=1,
            padding="same",
            data_format="channels_first",
            activation=tf.tanh,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="char_cnn")

        char_conv_output_permute = tf.transpose(char_conv_output,
                                                perm=[0, 2, 1])

        # do maxpooling on the conv layer output
        char_maxpool_output = tf.layers.max_pooling1d(
            inputs=char_conv_output_permute,
            pool_size=wnut.max_word_length,
            strides=wnut.max_word_length,
            padding="valid",
            name="char_max_pool")

        print(char_maxpool_output.get_shape().as_list())

        # squeeze the maxpool output
        char_maxpool_output_squeeze = tf.squeeze(
            char_maxpool_output, [1])

        char_level_emb = tf.reshape(
            char_maxpool_output_squeeze,
            [-1, wnut.max_sentence_length, args.num_filters])

        # fwd_cell = tf.contrib.rnn.LSTMCell(
        #     num_units=args.num_units, state_is_tuple=True)
        # bwd_cell = tf.contrib.rnn.LSTMCell(
        #     num_units=args.num_units, state_is_tuple=True)
        # fwd_cell = tf.contrib.rnn.MultiRNNCell(
        #     [fwd_cell], state_is_tuple=True)
        # bwd_cell = tf.contrib.rnn.MultiRNNCell(
        #     [bwd_cell], state_is_tuple=True)
        # lstm_input_dropout = tf.layers.dropout(
        #     char_emb_reshape,
        #     rate=0.5)

        # outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        #     fwd_cell, bwd_cell, lstm_input_dropout,
        #     sequence_length=seq_length_input, dtype=tf.float32)
        # fwd_emb = tf.slice(
        #     input_=outputs[0],
        #     begin=[0, wnut.max_word_length-1, 0],
        #     size=[-1, 1, args.num_units])
        # bwd_emb = tf.slice(
        #     input_=outputs[1],
        #     begin=[0, wnut.max_word_length-1, 0],
        #     size=[-1, 1, args.num_units])
        # concat_output = tf.squeeze(tf.concat([fwd_emb, bwd_emb], axis=2))
        # char_level_emb = tf.reshape(
        #     concat_output,
        #     [-1, wnut.max_sentence_length, 2*args.num_units])

    # Get embeddings of the orthographic words
    with tf.variable_scope("orth"):
        table = wnut.emb_table["orth_word"]
        orth_emb_table = tf.get_variable(
            name="orth_emb_table",
            shape=[table.shape[0], table.shape[1]],
            dtype=tf.float32,
            initializer=tf.constant_initializer(table))
        orth_emb = tf.nn.embedding_lookup(
            orth_emb_table, orth_input, name="orth_char_emb_output")

    # Get char level representations of the orthogrphic words
    # from orth char embeddings
    with tf.name_scope("orth_char"):
        # Get orth_char embeddings
        table = wnut.emb_table["orth_char"]
        emb_dim = wnut.emb_table["orth_char"].shape[1]
        orth_char_emb_table = tf.get_variable(
            name="orth_char_emb_table",
            shape=[table.shape[0], table.shape[1]],
            dtype=tf.float32,
            initializer=tf.constant_initializer(table))
        orth_char_emb = tf.nn.embedding_lookup(
            orth_char_emb_table, orth_char_input,
            name="orth_char_emb_output")

        orth_char_emb_reshape = tf.reshape(
            orth_char_emb,
            [-1, wnut.max_word_length, emb_dim])

        # permute the tensor
        orth_char_emb_permute = tf.transpose(
            orth_char_emb_reshape, perm=[0, 2, 1])

        orth_char_conv_input = tf.layers.dropout(
            orth_char_emb_permute,
            rate=0.5)

        # Perform a conv2d operation
        orth_char_conv_output = tf.layers.conv1d(
            inputs=orth_char_conv_input,
            filters=args.num_filters,
            kernel_size=const.CONV_WINDOW,
            strides=1,
            padding="same",
            data_format="channels_first",
            activation=tf.tanh,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="orth_char_cnn")

        orth_char_conv_output_permute = tf.transpose(orth_char_conv_output,
                                                     perm=[0, 2, 1])

        # do maxpooling on the conv layer output
        orth_char_maxpool_output = tf.layers.max_pooling1d(
            inputs=orth_char_conv_output_permute,
            pool_size=wnut.max_word_length,
            strides=wnut.max_word_length,
            padding="valid",
            name="orth_char_max_pool")

        # squeeze the maxpool output
        orth_char_maxpool_output_squeeze = tf.squeeze(
            orth_char_maxpool_output, [1])

        orth_char_level_emb = tf.reshape(
            orth_char_maxpool_output_squeeze,
            [-1, wnut.max_sentence_length, args.num_filters])

        # fwd_cell = tf.contrib.rnn.LSTMCell(
        #     num_units=args.num_units, state_is_tuple=True)
        # bwd_cell = tf.contrib.rnn.LSTMCell(
        #     num_units=args.num_units, state_is_tuple=True)
        # fwd_cell = tf.contrib.rnn.MultiRNNCell(
        #     [fwd_cell], state_is_tuple=True)
        # bwd_cell = tf.contrib.rnn.MultiRNNCell(
        #     [bwd_cell], state_is_tuple=True)
        # lstm_input_dropout = tf.layers.dropout(
        #     orth_char_emb_reshape,
        #     rate=0.5)

        # outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        #     fwd_cell, bwd_cell, lstm_input_dropout,
        #     sequence_length=seq_length_input, dtype=tf.float32)
        # fwd_emb = tf.slice(
        #     outputs[0],
        #     [0, wnut.max_word_length-1, 0],
        #     [-1, 1, -1])
        # bwd_emb = tf.slice(
        #     outputs[1],
        #     [0, wnut.max_word_length-1, 0],
        #     [-1, 1, -1])
        # concat_output = tf.squeeze(tf.concat([fwd_emb, bwd_emb], axis=2))
        # orth_char_level_emb = tf.reshape(
        #     concat_output,
        #     [-1, wnut.max_sentence_length, 2*args.num_units])

    with tf.variable_scope("lstm"):
        lstm_input = tf.concat([word_emb,
                                char_level_emb,
                                orth_emb,
                                orth_char_level_emb
                                ],
                               axis=2)
        fwd_cell = tf.contrib.rnn.GRUCell(
            num_units=args.num_units)
        bwd_cell = tf.contrib.rnn.GRUCell(
            num_units=args.num_units)
        # fwd_cell = tf.contrib.rnn.LSTMCell(
        #     num_units=args.num_units,
        #     use_peepholes=True,
        #     state_is_tuple=True)
        # bwd_cell = tf.contrib.rnn.LSTMCell(
        #     num_units=args.num_units,
        #     use_peepholes=True,
        #     state_is_tuple=True)
        fwd_cell = tf.contrib.rnn.MultiRNNCell(
            [fwd_cell], state_is_tuple=True)
        bwd_cell = tf.contrib.rnn.MultiRNNCell(
            [bwd_cell], state_is_tuple=True)
        lstm_input_dropout = tf.layers.dropout(
            lstm_input,
            rate=0.5)
        # lstm_input_dropout = tf.layers.dropout(
        #     word_emb,
        #     rate=0.5)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            fwd_cell, bwd_cell, lstm_input_dropout,
            sequence_length=seq_length_input, dtype=tf.float32)
        lstm_output = tf.concat([outputs[0], outputs[1]], axis=2)

    with tf.variable_scope("crf_hidden"):
        W = tf.get_variable("W", shape=[2*args.num_units, wnut.num_labels])
        lstm_output_dropout = tf.layers.dropout(lstm_output, rate=0.5)
        crf_input = tf.tensordot(lstm_output_dropout, W, [[2], [0]])

    return crf_input


def get_feed_dict(data,
                  batch_id,
                  batch_size,
                  word_input,
                  char_input,
                  orth_input,
                  orth_char_input,
                  seq_length_input,
                  labels):
    feed_dict = {
        word_input:
            data["word"][batch_id*batch_size:(batch_id+1)*batch_size],
        char_input:
            data["char"][batch_id*batch_size:(batch_id+1)*batch_size],
        orth_input:
            data["orth_word"][batch_id*batch_size:(batch_id+1)*batch_size],
        orth_char_input:
            data["orth_char"][batch_id*batch_size:(batch_id+1)*batch_size],
        seq_length_input:
            data["length"][batch_id*batch_size:(batch_id+1)*batch_size],
        labels:
            data["label"][batch_id*batch_size:(batch_id+1)*batch_size]
    }

    return feed_dict
