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


def main():
    def construct_input_layer():
        if fineTune:
            layer_input = lasagne.layers.InputLayer(
                shape=(None, max_sentence_len),
                input_var=input_var,
                name='input')
            layer_embedding = lasagne.layers.EmbeddingLayer(
                layer_input,
                input_size=wordAlphabetSize,
                output_size=wordEmbeddingDim,
                W=word_emb_table, name='embedding')
            return layer_embedding
        else:
            layer_input = lasagne.layers.InputLayer(
                shape=(None, max_sentence_len, wordEmbeddingDim),
                input_var=input_var,
                name='input')
            return layer_input

    def construct_char_input_layer():
        layer_char_input = lasagne.layers.InputLayer(
            shape=(None, max_sentence_len_char, max_word_len),
            input_var=char_input_var,
            name='char-input')
        layer_char_input = lasagne.layers.reshape(
            layer_char_input,
            (-1, [2]))
        layer_char_embedding = lasagne.layers.EmbeddingLayer(
            layer_char_input,
            input_size=char_alphabet_size,
            output_size=char_emb_dim,
            W=char_emb_table,
            name='char_embedding')
        # TODO: why this?
        layer_char_input = lasagne.layers.DimshuffleLayer(
            layer_char_embedding,
            pattern=(0, 2, 1))
        return layer_char_input

    def construct_orth_input_layer():
        if fineTune:
            orth_layer_input = lasagne.layers.InputLayer(
                shape=(None, max_sentence_len),
                input_var=orth_input_var,
                name='orth-input')
            orth_layer_embedding = lasagne.layers.EmbeddingLayer(
                orth_layer_input,
                input_size=orth_word_alphabet_size,
                output_size=orth_word_emb_dim,
                W=orth_word_emb_table, name='orth_embedding')
            return orth_layer_embedding
        else:
            orth_layer_input = lasagne.layers.InputLayer(
                shape=(None, max_sentence_len, orth_word_emb_dim),
                input_var=orth_input_var,
                name='orth-input')
            return orth_layer_input

    def construct_orth_char_input_layer():
        layer_orth_char_input = lasagne.layers.InputLayer(
            shape=(None, max_sentence_len_char, max_word_len),
            input_var=orth_char_input_var,
            name='orth-char-input')
        layer_orth_char_input = lasagne.layers.reshape(
            layer_orth_char_input,
            (-1, [2]))
        layer_orth_char_embedding = lasagne.layers.EmbeddingLayer(
            layer_orth_char_input,
            input_size=orth_char_alphabet_size,
            output_size=orth_char_emb_dim,
            W=orth_char_emb_table,
            name='orth_char_embedding')
        # TODO: why this?
        layer_orth_char_input = lasagne.layers.DimshuffleLayer(
            layer_orth_char_embedding,
            pattern=(0, 2, 1))
        return layer_orth_char_input

    # initialise a parser
    parser = argparse.ArgumentParser(
        description='Tuning with bi-directional LSTM-CNN-CRF')

    # add arguments
    parser.add_argument('--fine_tune', action='store_true',
                        help='Fine tune the word embeddings')
    parser.add_argument('--embedding', choices=['word2vec', 'glove', 'senna',
                        'random', 'w2v_twitter'], help='Embedding for words',
                        required=True)
    parser.add_argument('--embedding_dict', default=None,
                        help='path for embedding dict')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Number of sentences in each batch')
    parser.add_argument('--num_units', type=int, default=100,
                        help='Number of hidden units in LSTM')
    parser.add_argument('--num_filters', type=int, default=20,
                        help='Number of filters in CNN')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1,
                        help='Decay rate of learning rate')
    parser.add_argument('--grad_clipping', type=float, default=0,
                        help='Gradient clipping')
    parser.add_argument('--gamma', type=float, default=1e-6,
                        help='weight for regularization')
    parser.add_argument('--peepholes', action='store_true',
                        help='Peepholes for LSTM')
    parser.add_argument('--oov', choices=['random', 'embedding'],
                        help='Embedding for oov word', required=True)
    parser.add_argument('--update', choices=['sgd', 'momentum', 'nesterov',
                        'adadelta'], help='update algorithm', default='sgd')
    parser.add_argument('--regular', choices=['none', 'l2'],
                        help='regularization for training', required=True)
    parser.add_argument('--dropout', action='store_true',
                        help='Apply dropout layers')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping')
    parser.add_argument('--char_emb_dim', type=int, default=30,
                        help='Character based embedding size')
    parser.add_argument('--save_model', default=None,
                        help='path for saving models')
    # parser.add_argument('--orth-emb-size', type=int, default=30,
    #                     help='Character based embedding size')
    parser.add_argument('--train')
    parser.add_argument('--dev')
    parser.add_argument('--test')

    args = parser.parse_args()

    fineTune = args.fine_tune
    oov = args.oov
    regular = args.regular
    embeddingToUse = args.embedding
    embeddingPath = args.embedding_dict
    trainPath = args.train
    devPath = args.dev
    testPath = args.test
    updateAlgo = args.update
    grad_clipping = args.grad_clipping
    peepholes = args.peepholes
    numFilters = args.num_filters
    gamma = args.gamma
    dropout = args.dropout
    char_emb_dim = args.char_emb_dim
    saveModel = args.save_model

    assert(len(os.listdir(saveModel)) == 0 or len(os.listdir(saveModel)) == 1)

    os.makedirs(saveModel + "/model")
    os.makedirs(saveModel + "/predictions")

    logger = utils.get_logger("BiLSTM-CNN-CRF")
    csvWriter = csv.writer(open(saveModel + "/training.csv", "w"), delimiter=',')

    # TODO: without finetuning
    train_X, train_Y, train_mask, train_X_orth, \
        dev_X, dev_Y, dev_mask, dev_X_orth, \
        test_X, test_Y, test_mask, test_X_orth, \
        word_emb_table, word_alphabet, orth_word_emb_table, \
        label_alphabet, char_train, char_dev, char_test, char_emb_table, \
        orth_char_train, orth_char_dev, orth_char_test, orth_char_emb_table \
        = data_processor.loadDataForSequenceLabeling(
            trainPath,
            devPath,
            testPath,
            char_emb_dim,
            word_column=0,
            label_column=1,
            oov=oov,
            fine_tune=fineTune,
            embeddingToUse=embeddingToUse,
            embedding_path=embeddingPath,
            use_character=True)

    # Because index 0 in label alphabet
    # has no meaning
    numLabels = label_alphabet.size() - 1

    logger.info("constructing network...")

    # create variables
    target_var = T.imatrix(name='targets')
    mask_var = T.matrix(name='masks', dtype=theano.config.floatX)
    if fineTune:
        input_var = T.imatrix(name='inputs')
        numTrainExamples, max_sentence_len = train_X.shape
        wordAlphabetSize, wordEmbeddingDim = word_emb_table.shape
        orth_input_var = T.imatrix(name='orth-inputs')
        numTrainExamplesOrth, max_sentence_len_orth = train_X_orth.shape
        orth_word_alphabet_size, orth_word_emb_dim = orth_word_emb_table.shape
    else:
        input_var = T.tensor3(name='inputs', dtype=theano.config.floatX)
        numTrainExamples, max_sentence_len, wordEmbeddingDim = train_X.shape
        orth_input_var = T.tensor3(name='orth-inputs',
                                   dtype=theano.config.floatX)
        numTrainExamplesOrth, max_sentence_len_orth, orth_word_emb_dim = \
            train_X_orth.shape
    char_input_var = T.itensor3(name='char-inputs')
    orth_char_input_var = T.itensor3(name='orth-char-inputs')
    numTrainExamplesChar, max_sentence_len_char, \
        max_word_len = char_train.shape
    numTrainExamplesOrthChar, max_sentence_len_orth_char, \
        max_word_len_orth = orth_char_train.shape
    char_alphabet_size, char_emb_dim = char_emb_table.shape
    orth_char_alphabet_size, orth_char_emb_dim = orth_char_emb_table.shape
    assert (max_sentence_len == max_sentence_len_char)
    assert (max_sentence_len_orth == max_sentence_len_orth_char)
    assert (max_sentence_len_orth == max_sentence_len)
    assert (numTrainExamples == numTrainExamplesChar)
    assert (numTrainExamplesOrthChar == numTrainExamplesOrth)
    assert (numTrainExamples == numTrainExamplesOrth)

    # construct input and mask layers
    layer_incoming1 = construct_char_input_layer()
    # output shape of layer_incoming1 is
    # [batch_size * max_sentence_len, char_emb_dim, max_word_len]
    layer_incoming2 = construct_input_layer()
    # output_shape of layer_incoming2 is
    # [batch_size, max_sentence_len, word_emb_dim]
    layer_incoming3 = construct_orth_char_input_layer()
    # output shape of layer_incoming3 is
    # [batch_size * max_sentence_len, orth_char_emb_dim, max_word_len]
    layer_incoming4 = construct_orth_input_layer()
    # output_shape of layer_incoming4 is
    # [batch_size, max_sentence_len, orth_word_emb_dim]

    layer_mask = lasagne.layers.InputLayer(
        shape=(None, max_sentence_len),
        input_var=mask_var,
        name='mask')

    # construct bilstm-cnn-crf
    # FIXME Dropout
    num_units = args.num_units

    bi_lstm_cnn_crf = build_BiLSTM_CNN_CRF(
        layer_incoming1,
        layer_incoming2,
        layer_incoming3,
        layer_incoming4,
        num_units,
        numLabels,
        mask=layer_mask,
        grad_clipping=grad_clipping,
        peepholes=peepholes,
        num_filters=numFilters,
        dropout=dropout)

    logger.info("Network structure: hidden=%d, filter=%d" %
                (num_units, numFilters))

    # compute loss
    num_tokens = mask_var.sum(dtype=theano.config.floatX)

    # get output of bi-lstm-cnn-crf shape [batch, length, numLabels, numLabels]
    energies_train = lasagne.layers.get_output(bi_lstm_cnn_crf)
    energies_eval = lasagne.layers.get_output(bi_lstm_cnn_crf,
                                              deterministic=True)

    loss_train = crf_loss(energies_train, target_var, mask_var).mean()
    loss_eval = crf_loss(energies_eval, target_var, mask_var).mean()

    # l2 regularization?
    if regular == 'l2':
        l2_penalty = lasagne.regularization.regularize_network_params(
            bi_lstm_cnn_crf,
            lasagne.regularization.l2)
        loss_train = loss_train + gamma * l2_penalty

    _, corr_train = crf_accuracy(energies_train, target_var)
    corr_train = (corr_train * mask_var).sum(dtype=theano.config.floatX)
    prediction_eval, corr_eval = crf_accuracy(energies_eval,
                                              target_var)
    corr_eval = (corr_eval * mask_var).sum(dtype=theano.config.floatX)

    # Create update expressions for training.
    # hyper parameters to tune: learning rate, momentum, regularization.
    batch_size = args.batch_size
    learning_rate = 1.0 if updateAlgo == 'adadelta' else args.learning_rate
    decay_rate = args.decay_rate
    momentum = 0.9
    params = lasagne.layers.get_all_params(bi_lstm_cnn_crf, trainable=True)
    updates = utils.create_updates(loss_train,
                                   params,
                                   updateAlgo,
                                   learning_rate,
                                   momentum=momentum)

    # Compile a function performing a training step on a mini-batch
    train_fn = theano.function([input_var, orth_input_var, target_var,
                                mask_var, char_input_var, orth_char_input_var],
                               [loss_train, corr_train, num_tokens],
                               updates=updates)
    # Compile a second function evaluating the loss and accuracy of network
    eval_fn = theano.function([input_var, orth_input_var, target_var,
                               mask_var, char_input_var, orth_char_input_var],
                              [loss_eval, corr_eval, num_tokens,
                               prediction_eval])

    # Finally, launch the training loop.
    logger.info(
        "Start training: %s with regularization: %s(%f), dropout: %s, fine tune: %s \
        (#training data: %d, batch size: %d, clip: %.1f, peepholes: %s)..." %
        (updateAlgo,
         regular,
         (0.0 if regular == 'none' else gamma),
         dropout,
         fineTune,
         numTrainExamples,
         batch_size,
         grad_clipping,
         peepholes))
    csvWriter.writerow(["epoch", "train_loss", "train_acc", "dev_loss", "dev_acc", "dev_f1"])
    num_batches = numTrainExamples / batch_size
    num_epochs = 50
    best_loss = 1e+12
    best_f1 = 0.0
    best_epoch_loss = 0
    best_epoch_f1 = 0
    best_loss_test_loss = 0.
    best_loss_test_f1 = 0.
    best_f1_test_loss = 0.
    best_f1_test_f1 = 0.
    stop_count = 0
    lr = learning_rate
    patience = args.patience
    for epoch in range(1, num_epochs + 1):
        print 'Epoch %d (learning rate=%.4f, decay rate=%.4f): ' % (epoch,
                                                                    lr,
                                                                    decay_rate)
        train_loss = 0.0
        train_corr = 0.0
        train_total = 0
        train_inst = 0
        start_time = time.time()
        train_batches = 0
        for batch in utils.iterate_minibatches(
                        train_X,
                        train_Y,
                        train_X_orth,
                        masks=train_mask,
                        char_inputs=char_train,
                        orth_char_inputs=orth_char_train,
                        batch_size=batch_size,
                        shuffle=True):
            inputs, orth_inputs, targets, masks, char_inputs, \
                    orth_char_inputs = batch
            loss, corr, num = train_fn(inputs,
                                       orth_inputs,
                                       targets,
                                       masks,
                                       char_inputs,
                                       orth_char_inputs)
            train_loss += loss * inputs.shape[0]
            train_corr += corr
            train_total += num
            train_inst += inputs.shape[0]
            train_batches += 1
            time_ave = (time.time() - start_time) / train_batches
            time_left = (num_batches - train_batches) * time_ave

            # update log
            # sys.stdout.write("\b" * num_back)
            log_info = ("train: %d/%d loss: %.4f, acc: %.2f "
                        "time left (estimated): %.2fs\n") % (
                    min(train_batches * batch_size, numTrainExamples),
                    numTrainExamples,
                    train_loss / train_inst,
                    train_corr * 100 / train_total,
                    time_left)
            sys.stdout.write(log_info)
            # num_back = len(log_info)

        # update training log after each epoch
        assert train_inst == numTrainExamples
        # sys.stdout.write("\b" * num_back)
        print 'train: %d/%d loss: %.4f, acc: %.2f, time: %.2fs' % (
            min(train_batches * batch_size, numTrainExamples),
            numTrainExamples,
            train_loss / numTrainExamples,
            train_corr * 100 / train_total,
            time.time() - start_time)

        # for each epoch also
        # evaluate performance on dev data
        dev_loss = 0.0
        dev_corr = 0.0
        dev_total = 0
        dev_inst = 0
        dev_f1 = 0.0
        count = np.zeros((numLabels, numLabels), dtype=np.int32)
        for batch in utils.iterate_minibatches(
                dev_X,
                dev_Y,
                dev_X_orth,
                masks=dev_mask,
                char_inputs=char_dev,
                orth_char_inputs=orth_char_dev,
                batch_size=batch_size):
            inputs, orth_inputs, targets, masks, char_inputs, \
                orth_char_inputs = batch
            loss, corr, num, predictions = eval_fn(inputs,
                                                   orth_inputs,
                                                   targets,
                                                   masks,
                                                   char_inputs,
                                                   orth_char_inputs)
            dev_loss += loss * inputs.shape[0]
            dev_corr += corr
            dev_total += num
            dev_inst += inputs.shape[0]
            utils.evaluate(predictions,
                           targets,
                           inputs,
                           masks,
                           word_alphabet,
                           label_alphabet,
                           numLabels,
                           saveModel + '/predictions/%d_dev_predict.txt' % epoch,
                           count,
                           isFlattened=False)

        dev_f1 = utils.evaluateF1Score(
            saveModel + '/predictions/%d_dev_predict.txt' % epoch,
            saveModel + '/predictions/%d_dev_scores.txt' % epoch,
            saveModel + '/predictions/%d_dev_confusion.txt' % epoch,
            label_alphabet,
            numLabels,
            count,
            "dev",
            epoch)
        print 'dev loss: %.4f, f1: %.2f' % (
            dev_loss / dev_inst, dev_f1)
        csvWriter.writerow([
            "%d" % epoch,
            "%.4f" % (train_loss/numTrainExamples),
            "%.2f" % (train_corr * 100.0/train_total),
            "%.4f" % (dev_loss/dev_inst),
            "%.2f" % (dev_corr * 100.0/dev_total),
            "%.2f" % dev_f1])
        # increase stop_count (for patience) if
        # 1. f1 score on dev is worse (less) than
        #    the best dev f1 score so far
        # 2. loss on dev is worse (more) than
        #    the best loss on dev so far
        if best_loss < dev_loss and best_f1 > dev_f1:
            stop_count += 1
            print "Bad dev f1 and dev loss performance..."
            print "Increasing stop_count and starting next epoch..."
            print "\n"
        else:
            update_loss = False
            update_f1 = False
            stop_count = 0
            # if better loss on dev in this epoch
            # then save this model
            if best_loss > dev_loss:
                update_loss = True
                print "Improvement in dev loss..."
                best_loss = dev_loss
                best_epoch_loss = epoch
                params = lasagne.layers.get_all_param_values(bi_lstm_cnn_crf)
                modelName = saveModel + "/model/model_" + \
                    str(epoch) + ".pkl"
                print "Saving model to :" + modelName
                cPickle.dump(params, open(modelName, "wb"))
            # if better f1 on dev in this epoch
            # then save this model
            if best_f1 < dev_f1:
                print "Improvement in dev f1..."
                update_f1 = True
                best_f1 = dev_f1
                best_epoch_f1 = epoch
                modelName = saveModel + "/model/model_" + \
                    str(epoch) + ".pkl"
                print "Saving model to :" + modelName
                if not os.path.isfile(modelName):
                    cPickle.dump(params, open(saveModel + "/model/model_" +
                                 str(epoch) + ".pkl", "wb"))
            print "Evaluating on test data... "
            # evaluate on test data when better performance detected
            # i.e. if either dev loss decreases on
            # dev f1 increases, then evaluate on test data
            test_loss = 0.0
            test_corr = 0.0
            test_total = 0
            test_inst = 0
            test_f1 = 0.0
            count = np.zeros((numLabels, numLabels), dtype=np.int32)
            for batch in utils.iterate_minibatches(
                                            test_X,
                                            test_Y,
                                            test_X_orth,
                                            masks=test_mask,
                                            char_inputs=char_test,
                                            orth_char_inputs=orth_char_test,
                                            batch_size=batch_size):
                inputs, orth_inputs, targets, masks, char_inputs, \
                    orth_char_inputs = batch
                loss, corr, num, predictions = eval_fn(inputs,
                                                       orth_inputs,
                                                       targets,
                                                       masks,
                                                       char_inputs,
                                                       orth_char_inputs)
                test_loss += loss * inputs.shape[0]
                test_corr += corr
                test_total += num
                test_inst += inputs.shape[0]
                utils.evaluate(
                    predictions,
                    targets,
                    inputs,
                    masks,
                    word_alphabet,
                    label_alphabet,
                    numLabels,
                    saveModel + '/predictions/%d_test_predict.txt' % epoch,
                    count,
                    isFlattened=False)

            test_f1 = utils.evaluateF1Score(
                saveModel + '/predictions/%d_test_predict.txt' % epoch,
                saveModel + '/predictions/%d_test_scores.txt' % epoch,
                saveModel + '/predictions/%d_test_confusion.txt' % epoch,
                label_alphabet,
                numLabels,
                count,
                "test",
                epoch)
            print 'test loss: %.4f, f1: %.2f\n' % (
                test_loss / test_inst, test_f1)

            # update_loss is True when dev_loss
            # is the best so far
            # so find the best test loss so far
            if update_loss:
                best_loss_test_loss = test_loss
                best_loss_test_f1 = test_f1
            # likewise update_f1 is True when dev_f1
            # is the best so far
            # so find best test_f1 and save it
            if update_f1:
                best_f1_test_loss = test_loss
                best_f1_test_f1 = test_f1

        # stop if dev f1 decrease 3 time straightly.
        # if stop_count == patience:
        #     break

        # re-compile a function with new learning rate for training
        if updateAlgo != 'adadelta':
            lr = learning_rate / (1.0 + epoch * decay_rate)
            updates = utils.create_updates(loss_train,
                                           params,
                                           updateAlgo,
                                           lr,
                                           momentum=momentum)
            train_fn = theano.function([input_var,
                                        orth_input_var,
                                        target_var,
                                        mask_var,
                                        char_input_var,
                                        orth_char_input_var],
                                       [loss_train,
                                        corr_train,
                                        num_tokens],
                                       updates=updates)
    # print best performance on test data.
    logger.info("final best loss test performance (at epoch %d)" %
                best_epoch_loss)
    print 'test loss: %.4f, f1: %.2f' % (
        best_loss_test_loss / test_inst,
        best_loss_test_f1)
    logger.info("final best f1 test performance (at epoch %d)" %
                best_epoch_f1)
    print 'test loss: %.4f, f1: %.2f' % (
        best_f1_test_loss / test_inst,
        best_f1_test_f1)


def test():
    energies_var = T.tensor4('energies', dtype=theano.config.floatX)
    targets_var = T.imatrix('targets')
    masks_var = T.matrix('masks', dtype=theano.config.floatX)
    layer_input = lasagne.layers.InputLayer([2, 2, 3, 3],
                                            input_var=energies_var)
    out = lasagne.layers.get_output(layer_input)
    loss = crf_loss(out, targets_var, masks_var)
    prediction, acc = crf_accuracy(energies_var, targets_var)

    fn = theano.function([energies_var, targets_var, masks_var], [loss, prediction, acc])

    energies = np.array([[[[10, 15, 20], [5, 10, 15], [3, 2, 0]], [[5, 10, 1], [5, 10, 1], [5, 10, 1]]],
                         [[[5, 6, 7], [2, 3, 4], [2, 1, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]], dtype=np.float32)

    targets = np.array([[0, 1], [0, 2]], dtype=np.int32)

    masks = np.array([[1, 1], [1, 0]], dtype=np.float32)

    l, p, a = fn(energies, targets, masks)
    print l
    print p
    print a


if __name__ == '__main__':
    main()
