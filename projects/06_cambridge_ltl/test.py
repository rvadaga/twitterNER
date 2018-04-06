import time
import sys
import argparse
from lasagne_nlp.utils import utils
from lasagne_nlp.utils.objectives import crf_loss, crf_accuracy
import lasagne_nlp.utils.data_processor as data_processor
import lasagne
import theano
import theano.tensor as T
from lasagne_nlp.networks.networks import build_BiLSTM_CNN_CRF
import numpy as np
from six.moves import cPickle
import os
import shutil
import csv


# Initialize parser
parser = argparse.ArgumentParser(
    description='Testing the noisy text architecture')

# Add arguments
parser.add_argument('--load_model', default=None,
                    help='path to the model to be loaded')
parser.add_argument('--test_data', default=None,
                    help='path to the test data')
parser.add_argument('--save_path', default=None,
                    help='path to save predictions')

# Obtain the arguments
args = parser.parse_args()
model_path = args.load_model
test_path = args.test_data
save_path = args.save_path

assert(os.path.isfile(model_path))
