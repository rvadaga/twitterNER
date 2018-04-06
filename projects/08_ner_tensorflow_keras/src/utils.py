# script with general utilities
# for logging, cmd line args, etc.

from __future__ import print_function

import argparse
import os
import logging
import sys
import codecs


eval_script = "conlleval.pl"
if sys.version_info[0] >= 3:
    unicode = str


def to_unicode(text, errors='strict', encoding='utf8'):
    """Convert a string (unicode or bytestring in `encoding`),
    to bytestring in utf8."""
    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors=errors).encode('utf8')


# Read the command line arguments and parse them
def parse_args():
    # Initialize a parser
    parser = argparse.ArgumentParser(
        description="Train a Neural Net for NER")

    # Add arguments
    parser.add_argument("--wnut",
                        help="path to wnut data directory",
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
                        help="path to embedding directory",
                        required=True)
    parser.add_argument("--save_dir",
                        help="directory to save model",
                        required=True)
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
    parser.add_argument("--use_orth_char",
                        action="store_true",
                        default=True,
                        help="use orth char embeddings")
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
                        default=5.0,
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

    # validate args
    validate_args(args)

    return args


# Check if the arguments are valid
def validate_args(args):
    # Check if wnut data directory isn't empty
    assert os.path.isdir(args.wnut)

    # Check if the emb path provided is a file
    assert os.path.isfile(args.emb_path)

    # Results will be stored in args.save_dir
    # In order to prevent results from experiments being overwritten,
    # an error message is thrown if the directory isn't empy.
    if os.path.isdir(args.save_dir):
        assert len(os.listdir(args.save_dir)) == 0
        print(args.save_dir + " is empty!")
    else:
        os.makedirs(args.save_dir)
        print("Created " + args.save_dir)


# Create a logger
def get_logger(name, filename, level=logging.INFO,
               formatter="%(asctime)s - %(name)s - " +
                         "%(levelname)s - %(message)s"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)

    # Clear handlers from previous run
    if logger.handlers:
        logger.handlers = []

    # Log to stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Log to file
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def save_predictions(
        predictions,
        targets,
        inputs,
        sequence_length,
        word_collection,
        label_collection,
        num_labels,
        predict_file_name,
        count):

    with codecs.open(predict_file_name, 'a', 'utf-8') as f:
        for j in range(sequence_length):
            predictTag = predictions[j] + 1
            f.write(
                "%s %s %s\n" %
                (unicode(word_collection.get_instance(inputs[j]),
                         "utf8"),
                 label_collection.get_instance(targets[j] + 1),
                 label_collection.get_instance(predictTag)))
            count[targets[j], predictTag-1] += 1
        f.write("\n")
        f.close()


def evaluate_f1_score(
        predict_file_name,
        score_file_name,
        conf_mat_file_name,
        label_collection,
        num_labels,
        count,
        epoch):
    os.system("perl %s < %s > %s" % (eval_script,
                                     predict_file_name,
                                     score_file_name))
    # parse the CoNLL evaluation results
    eval_lines = [l.rstrip() for l in codecs.open(score_file_name, 'r', 'utf8')]
    # # print the lines from the result
    # for line in eval_lines:
    #     print(line)

    f = open(conf_mat_file_name, "w")
    # print dataset + " confusion matrix for " + str(epoch) + ":"
    # # Confusion matrix with accuracy for each tag
    # # uncomment the print statements if you want to display the confusion
    # # matrix for each epoch
    # print ("{: >2} {: >5} {: >7} %s{: >9}" % ("{: >5} " * num_labels)).format(
    #     "ID", "NE", "Total",
    #     *([label_alphabet.get_instance(i + 1)[:5]
    #         for i in xrange(num_labels)] + ["Percent"])
    # )
    f.write(
        ("{: >2} {: >5} {: >7} %s{: >9}\n" % ("{: >5} " * num_labels)).format(
            "ID", "NE", "Total",
            *([label_collection.get_instance(i + 1)[:5]
                for i in xrange(num_labels)] + ["Percent"])
            )
        )
    for i in xrange(num_labels):
        # # uncomment the print statement if you want to display the confusion
        # # matrix for each epoch
        # print ("{: >2} {: >5} {: >7} %s{: >9}" % ("{: >5} " * num_labels)).format(
        #     str(i), label_alphabet.get_instance(i+1)[:5], str(count[i].sum()),
        #     *([count[i][j] for j in xrange(num_labels)] +
        #       ["%.3f" % (count[i][i] * 100. / max(1, count[i].sum()))])
        # )
        f.write(
            ("{: >2} {: >5} {: >7} %s{: >9}\n" % (
                "{: >5} " * num_labels)).format(
                str(i), label_collection.get_instance(i+1)[:5],
                str(count[i].sum()),
                *([count[i][j] for j in xrange(num_labels)] +
                  ["%.3f" % (count[i][i] * 100. / max(1, count[i].sum()))])
                )
            )

    # # Global accuracy
    # print "%i/%i (%.5f%%)" % (
    #     count.trace(), count.sum(), 100. * count.trace() / max(1,
    #     count.sum())
    # )
    f.write(
        "%i/%i (%.5f%%)\n" % (
         count.trace(), count.sum(), 100.*count.trace() / max(1, count.sum())
        ))
    f.close()
    # F1 on all entities
    return float(eval_lines[1].strip().split()[-1])
