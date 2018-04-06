from __future__ import print_function

import numpy as np
import utils
import data_utils
import collection
import network
import tensorflow as tf
import constants as const
import sys
import prog_bar as prog

# FIXME
# 1. change save_dir working
# 2. load_emb : increase vocab size
# 3. include out_predict 

args = utils.parse_args()

LOG_FILE = args.save_dir + "/log_file"

# set up logging
# logger will output to both sys.stdout and
# to log_file in save_dir
logger = utils.get_logger(
    "Bi-LSTM_CNN_CRF",
    LOG_FILE)
logger.info("#"*50)
data_utils.get_logger(LOG_FILE)
collection.get_log_file_path(LOG_FILE)

# generate data set for training
data = data_utils.load_data(args)
train_char_x = data["char"][0]
train_word_x = data["train_word"][0]
train_word_y = np.expand_dims(data["train_word"][1], -1)
train_orth_char_x = data["orth_char"][0]
train_orth_x = data["train_orth"]
n_train_examples = len(train_word_x)

dev_char_x = data["char"][1]
dev_word_x = data["dev_word"][0]
dev_word_y = data["dev_word"][1]
dev_mask = data["dev_word"][2]
dev_orth_char_x = data["orth_char"][1]
dev_orth_x = data["dev_orth"]
n_dev_examples = len(dev_word_x)

# Build network
logger.info("Building network")
model = network.build_network(args, data)
num_labels = data["label_collection"].size() - 1

# Compiling model
logger.info("Compiling model")
model.compile(optimizer="adam",
              loss=model.layers[-1].loss,
              metrics=["accuracy"])

# Callbacks
loss_train = []
acc_train = []
loss_history = keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: loss_train.append(logs['loss']))
acc_history = keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: acc_train.append(logs['acc']*100.0))
prog_bar = prog.ProgbarLogger(count_mode='samples')

f_train = open(args.save_dir + "/training.log", "a")
logger.info("Saving training set log data to " +
            args.save_dir + "/training.log")

f_dev = open(args.save_dir + "/dev.log", "a")
logger.info("Saving dev set log data to " +
            args.save_dir + "/dev.log")

# Training
logger.info("Training model")
idx = np.arange(n_train_examples)
batch_size = args.batch_size

for epoch in xrange(const.NUM_EPOCHS):
    print()
    print("Epoch", epoch)
    np.random.shuffle(idx)
    model.fit({"char_input": train_char_x[idx],
               "word_input": train_word_x[idx],
               "orth_char_input": train_orth_char_x[idx],
               "orth_word_input": train_orth_x[idx]},
              {"output": train_word_y[idx]},
              epochs=1,
              batch_size=args.batch_size,
              callbacks=[loss_history, acc_history, prog_bar],
              verbose=0)
    f_train.write("Epoch %d, Loss: %.4f, Accuracy: %.4f\n" % (
        epoch,
        loss_train[-1],
        acc_train[-1]))

    # evaluate performance on dev data
    tag_scores = model.predict(
        {"char_input": dev_char_x, "word_input": dev_word_x,
         "orth_char_input": dev_orth_char_x,
         "orth_word_input": dev_orth_x},
        batch_size=batch_size)
    predictions = np.max(tag_scores, axis=-1).astype(int)

    count = np.zeros((num_labels, num_labels), dtype=int)

    utils.save_predictions(
        predictions,
        dev_word_y,
        dev_word_x,
        dev_mask,
        data["word_collection"],
        data["label_collection"],
        num_labels,
        args.save_dir + "/dev_predictions/dev_predict_%d" % epoch,
        count, False)

    dev_f1 = utils.evaluate_f1_score(
        args.save_dir + "/dev_predictions/dev_predict_%d" % epoch,
        args.save_dir + "/dev_scores/dev_scores_%d" % epoch,
        args.save_dir + "/dev_scores/dev_matrix_%d" % epoch,
        data["label_collection"],
        num_labels,
        count,
        epoch)

    scores = model.evaluate(
        {"char_input": dev_char_x, "word_input": dev_word_x,
         "orth_char_input": dev_orth_char_x,
         "orth_word_input": dev_orth_x},
        {"output": np.expand_dims(dev_word_y, -1)},
        batch_size=batch_size)

    str_print = "dev_data: Epoch %d, %s: %.4f, %s: %.4f, F1: %.2f" % (
        epoch,
        model.metrics_names[0],
        scores[0],
        model.metrics_names[1],
        scores[1],
        dev_f1)

    f_dev.write(str_print + "\n")
    print("\n" + str_print)
