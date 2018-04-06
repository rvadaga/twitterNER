from __future__ import print_function

import utils
import time
import data_utils
import collection
import model
import numpy as np
import tensorflow as tf
import constants as const
from six.moves import range
from tqdm import tqdm
import os

# FIXME
# 1. change save_dir working
# 2. load_emb : increase vocab size
# 3. include out_predict
# 4. dropout
# 5. check tf logging system
# 6. may have to change constants.py file name
# 7. feed_dict function should cover all examples
#    in train data

args = utils.parse_args()

LOG_FILE = args.save_dir + "/log_file"

# Set up Logging.
# Logger will output to both sys.stdout and
# to log_file in save_dir.
if not args.bilstm_bilstm:
    logger = utils.get_logger("Bi-LSTM_CNN_CRF", LOG_FILE)
else:
    logger = utils.get_logger("Bi-LSTM_Bi-LSTM", LOG_FILE)

data_utils.get_logger(LOG_FILE)
collection.get_log_file_path(LOG_FILE)

# Generate data set for training
wnut = data_utils.NERData(args)
wnut.read_data()
wnut.preprocess_data_fine_tune()

# Some data related constants
num_labels = wnut.num_labels
max_sentence_len = wnut.max_sentence_length
max_word_len = wnut.max_word_length
n_train_examples = wnut.train_data["word"].shape[0]
n_dev_examples = wnut.dev_data["word"].shape[0]
n_test_examples = wnut.test_data["word"].shape[0]

# Some hyperparameters
batch_size = args.batch_size

# number of batches
n_train_batches = n_train_examples/batch_size
n_dev_batches = n_dev_examples/batch_size
n_test_batches = n_test_examples/batch_size

# Collection objects
word_collection = wnut.word_collection
label_collection = wnut.label_collection

# make directories to save predictions
os.makedirs(args.save_dir + "/predictions")
os.makedirs(args.save_dir + "/scores")

bufsize = 0
f = open(args.save_dir + "/results.txt", "a", bufsize)
f.write("# epoch, train_loss, train_acc, dev_loss, dev_acc, " +
        "dev_f1, test_loss, test_acc, test_f1\n")

# Model will be built into the default graph
with tf.Graph().as_default():
    with tf.Session() as session:
        # Build network
        logger.info("Building network")

        # Generate placeholders for inputs and labels
        word_input = tf.placeholder(
            tf.int32, shape=(None, max_sentence_len), name="word_input")
        char_input = tf.placeholder(
            tf.int32, shape=(None, max_sentence_len, max_word_len),
            name="char_input")
        orth_input = tf.placeholder(
            tf.int32, shape=(None, max_sentence_len),
            name="orth_input")
        orth_char_input = tf.placeholder(
            tf.int32, shape=(None, max_sentence_len, max_word_len),
            name="orth_char_input")
        seq_length_input = tf.placeholder(
            tf.int32, shape=(None),
            name="seq_length_input")
        char_seq_length_input = tf.placeholder(
            tf.int32, shape=(None),
            name="char_seq_length_input")
        labels = tf.placeholder(
            tf.int32, shape=(None, max_sentence_len),
            name="labels")

        dev_data = wnut.dev_data
        test_data = wnut.test_data

        # Build a network that finds the unary scores for the words in
        # the sentence. These scores are then fed to the crf layer
        # which finds the log likelihood

        crf_input = model.unary_scores(
            word_input,
            char_input,
            orth_input,
            orth_char_input,
            seq_length_input, 
            char_seq_length_input,
            args,
            wnut)
        # crf_input = model.unary_scores(
        #     word_input, char_input,
        #     seq_length_input, args, wnut)

        # add log likelihood op to the graph
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            crf_input, labels, seq_length_input)

        # add loss op to the graph, set the optimisers and
        # get the train op
        loss_nll = tf.reduce_mean(-log_likelihood)
        # vars = tf.trainable_variables()
        # loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * 0.001
        # loss = loss_nll + loss_l2
        loss = loss_nll
        optimizer = tf.train.MomentumOptimizer(learning_rate=args.lr, momentum=0.9)
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -args.grad_clip, args.grad_clip), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)

        session.run(tf.global_variables_initializer())

        # start training
        for epoch in range(const.MAX_EPOCHS):
            # shuffle the data for every epoch
            train_data_shuffled = data_utils.shuffle_data(wnut.train_data)

            print("Epoch %d" % (epoch))
            num_batches = 0
            start_time = time.time()

            for batch_id in tqdm(range(n_train_batches)):
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                # run_metadata = tf.RunMetadata()

                feed_dict = model.get_feed_dict(
                    train_data_shuffled, batch_id, batch_size, word_input,
                    char_input,
                    orth_input,
                    orth_char_input,
                    seq_length_input,
                    char_seq_length_input,
                    labels,
                    args)

                tf_crf_input, tf_transition_params, tf_loss, _ = \
                    session.run([crf_input,
                                 transition_params,
                                 loss,
                                 train_op],
                                feed_dict=feed_dict)

                # Find the accuracy of the model on the train data. This
                # requires that we compute Viterbi sequence of each example
                # in the current batch
                idx = np.arange(batch_id*batch_size, (batch_id+1)*batch_size)
                correct_labels = 0
                total_labels = 0
                for tf_crf_input_, y_, sequence_length_ in zip(
                        tf_crf_input,
                        train_data_shuffled["label"][idx],
                        train_data_shuffled["length"][idx]):
                    tf_crf_input_ = tf_crf_input_[:sequence_length_]
                    y_ = y_[:sequence_length_]

                    viterbi_seq, _ = tf.contrib.crf.viterbi_decode(
                        tf_crf_input_, tf_transition_params)

                    correct_labels += np.sum(np.equal(viterbi_seq, y_))
                    total_labels += sequence_length_

                num_batches += 1
                current_time = time.time()
                # if (batch_id % 5 == 0) or (batch_id == n_train_batches-1):
                #     accuracy = 100.0 * correct_labels / total_labels
                #     elapsed_time = current_time - start_time
                #     remaining_time = (n_train_batches - num_batches) * \
                #         elapsed_time / num_batches
                #     print(("Step %3d: Loss %.4f Accuracy %.4f Elapsed %.2fs " +
                #            "Remaining %.2fs") %
                #           (batch_id, tf_loss, accuracy, elapsed_time,
                #            remaining_time))

            # find loss on entire training data
            start_time = time.time()
            correct_labels = 0
            total_labels = 0
            train_loss = 0.0
            for batch_id in range(n_train_batches):
                feed_dict = model.get_feed_dict(
                    train_data_shuffled, batch_id, batch_size,
                    word_input,
                    char_input,
                    orth_input,
                    orth_char_input,
                    seq_length_input,
                    char_seq_length_input,
                    labels,
                    args)
                tf_crf_input, tf_transition_params, tf_loss = \
                    session.run([crf_input, transition_params, loss],
                                feed_dict=feed_dict)
                train_loss += tf_loss * batch_size
                idx = np.arange(batch_id*batch_size, (batch_id+1)*batch_size)
                for tf_crf_input_, y_, sequence_length_ in zip(
                        tf_crf_input,
                        train_data_shuffled["label"][idx],
                        train_data_shuffled["length"][idx]):
                    tf_crf_input_ = tf_crf_input_[:sequence_length_]
                    y_ = y_[:sequence_length_]

                    viterbi_seq, _ = tf.contrib.crf.viterbi_decode(
                        tf_crf_input_, tf_transition_params)

                    correct_labels += np.sum(np.equal(viterbi_seq, y_))
                    total_labels += sequence_length_
            current_time = time.time()

            train_loss /= n_train_examples
            train_acc = 100.0 * correct_labels/total_labels
            print("Loss: %.4f Accuracy %.4f Elapsed %.2fs" % (
                  train_loss,
                  train_acc,
                  current_time-start_time))

            # Run evals on dev data
            correct_labels = 0
            total_labels = 0
            dev_loss = 0.0
            count = np.zeros((num_labels, num_labels), dtype=int)
            start_time = time.time()
            for batch_id in range(n_dev_batches):
                # run_options = tf.RunOptions(
                    # trace_level=tf.RunOptions.FULL_TRACE)
                # run_metadata = tf.RunMetadata()

                feed_dict = model.get_feed_dict(
                    dev_data, batch_id, batch_size, word_input,
                    char_input,
                    orth_input,
                    orth_char_input,
                    seq_length_input,
                    char_seq_length_input,
                    labels,
                    args)

                tf_crf_input, tf_transition_params, tf_loss = \
                    session.run([crf_input,
                                 transition_params,
                                 loss],
                                feed_dict=feed_dict)

                dev_loss += tf_loss * batch_size

                # Find the accuracy of the model on the train data. This
                # requires that we compute Viterbi sequence of each example
                # in the current batch
                idx = np.arange(batch_id*batch_size, (batch_id+1)*batch_size)
                for tf_crf_input_, y_, sequence_length_, input_ in zip(
                        tf_crf_input,
                        dev_data["label"][idx],
                        dev_data["length"][idx],
                        dev_data["word"][idx]):
                    tf_crf_input_ = tf_crf_input_[:sequence_length_]
                    y_ = y_[:sequence_length_]

                    viterbi_seq, _ = tf.contrib.crf.viterbi_decode(
                        tf_crf_input_, tf_transition_params)

                    correct_labels += np.sum(np.equal(viterbi_seq, y_))
                    total_labels += sequence_length_
                    utils.save_predictions(
                        viterbi_seq,
                        y_,
                        input_,
                        sequence_length_,
                        word_collection,
                        label_collection,
                        num_labels,
                        args.save_dir+"/predictions/dev_predict_%d.txt"%(epoch),
                        count)

            dev_f1 = utils.evaluate_f1_score(
                args.save_dir + "/predictions/dev_predict_%d.txt" % (epoch),
                args.save_dir + "/scores/dev_scores_%d.txt" % (epoch),
                args.save_dir + "/scores/dev_conf_matrix_%d.txt" % (epoch),
                label_collection,
                num_labels,
                count,
                epoch)

            current_time = time.time()
            dev_loss /= n_dev_examples
            dev_acc = 100.0 * correct_labels / total_labels
            print("Dev Loss: %.4f, F1: %.2f, accuracy: %.2f, time: %.2fs" %
                  (dev_loss, dev_f1, dev_acc, current_time-start_time))

            # Run evals on test data
            correct_labels = 0
            total_labels = 0
            test_loss = 0.0
            count = np.zeros((num_labels, num_labels), dtype=int)
            start_time = time.time()
            for batch_id in range(n_test_batches):
                feed_dict = model.get_feed_dict(
                    test_data, batch_id, batch_size, word_input,
                    char_input,
                    orth_input,
                    orth_char_input,
                    seq_length_input,
                    char_seq_length_input,
                    labels,
                    args)

                tf_crf_input, tf_transition_params, tf_loss = \
                    session.run([crf_input,
                                 transition_params,
                                 loss],
                                feed_dict=feed_dict)

                test_loss += tf_loss * batch_size

                # Find the accuracy of the model on the train data. This
                # requires that we compute Viterbi sequence of each example
                # in the current batch
                idx = np.arange(batch_id*batch_size, (batch_id+1)*batch_size)
                for tf_crf_input_, y_, sequence_length_, input_ in zip(
                        tf_crf_input,
                        test_data["label"][idx],
                        test_data["length"][idx],
                        test_data["word"][idx]):
                    tf_crf_input_ = tf_crf_input_[:sequence_length_]
                    y_ = y_[:sequence_length_]

                    viterbi_seq, _ = tf.contrib.crf.viterbi_decode(
                        tf_crf_input_, tf_transition_params)

                    correct_labels += np.sum(np.equal(viterbi_seq, y_))
                    total_labels += sequence_length_
                    utils.save_predictions(
                        viterbi_seq,
                        y_,
                        input_,
                        sequence_length_,
                        word_collection,
                        label_collection,
                        num_labels,
                        args.save_dir + "/predictions/test_predict_%d.txt" % (epoch),
                        count)

            test_f1 = utils.evaluate_f1_score(
                args.save_dir + "/predictions/test_predict_%d.txt" % (epoch),
                args.save_dir + "/scores/test_scores_%d.txt" % (epoch),
                args.save_dir + "/scores/test_conf_matrix_%d.txt" % (epoch),
                label_collection,
                num_labels,
                count,
                epoch)

            current_time = time.time()
            test_loss /= n_test_examples
            test_acc = 100.0 * correct_labels / total_labels
            print("Test Loss: %.4f, F1: %.2f, accuracy: %.2f, time: %.2fs" %
                  (test_loss, test_f1, test_acc, current_time-start_time))
            f.write("%d %.4f %.2f %.4f %.2f %.2f %.4f %.2f %.2f\n" % (
                    epoch,
                    train_loss,
                    train_acc,
                    dev_loss,
                    dev_acc,
                    dev_f1,
                    test_loss,
                    test_acc,
                    test_f1))
f.close()
