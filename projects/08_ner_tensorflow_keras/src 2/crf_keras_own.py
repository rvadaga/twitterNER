# -*- coding: utf-8 -*-
from __future__ import absolute_import

# from .. import backend as K
# from .. import initializers
# from .. import regularizers
# from .. import constraints
# from ..engine import Layer, InputSpec

# import tensorflow as tf
from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras.engine.topology import Layer, \
    InputSpec
from tensorflow.contrib.keras.python.keras import initializers
from tensorflow.contrib.keras.python.keras import regularizers
from tensorflow.contrib.keras.python.keras import constraints
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

import numpy as np

def crf_sequence_score():

class ChainCRF_tensorflow(Layer):
    '''ChainCRF implementation based on
    tensorflow.contrib.crf
    Unary scores must be computed and fed to this layer.
    '''
    def __init__(
            self,
            init="glorot_uniform",
            U_regularizer=None,
            U_constraint=None,
            **kwargs):
        super(ChainCRF_tensorflow, self).__init__(**kwargs)

        # TODO
        # What is weights, b_start and b_end used for?
        self.init = initializers.get(init)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.U_constraint = constraints.get(U_constraint)

        self.supports_masking = True
        self.uses_learning_phase = True
        self.input_spec = [InputSpec(ndim=3)]

    def build(self, input_shape):
        assert len(input_shape) == 3
        n_classes = input_shape[2]
        n_steps = input_shape[1]
        assert n_steps is None or n_steps >= 2
        self.input_spec = [InputSpec(dtype=K.floatx(),
                           shape=(None, n_steps, n_classes))]

        # Transition params
        self.U = self.add_weight(
            (n_classes, n_classes),
            initializer=self.init,
            name='U',
            regularizer=self.U_regularizer,
            constraint=self.U_constraint)

        # if self.initial_weights is not None:
        #     self.set_weights(self.initial_weights)
        #     del self.initial_weights

        super(ChainCRF_tensorflow, self).build(input_shape)
        # Can alternatively use
        # self.built = True

    def call(self, x, mask=None):
        num_tags = self.input_spec[0].shape[2]

        sequence_scores = crf_sequence_score(
            x,
            tag_indices,
            sequence_lengths,
            transition_params)
        y_pred = viterbi_decode()


def viterbi_decode(x, U):
    """Decode the highest scoring sequence of tags.
    Done inside TensorFlow

    Args:
        score: A [seq_len, num_tags] matrix of unary potentials.
        transition_params: A [num_tags, num_tags] matrix of binary potentials.

    Returns:
        viterbi: A [seq_len] list of integers containing the highest scoring
            tag indicies.
        viterbi_score: A float containing the score for the Viterbi sequence.
    """
    U_expanded = array_ops.expand_dims(U, 0)

    def _forward_step(inputs, state):
        state = K.expand_dims(state, 2)
        transition_scores = state + U_expanded
        new_alphas = inputs + math_ops.reduce_logsumexp(transition_scores, [1])
        return new_alphas, new_alphas

    first_input = array_ops.slice(x, [0, 0, 0], [-1, 1, -1])
    first_input = array_ops.squeeze(x, [1])
    rest_of_input = array_ops.slice(x, [0, 1, 0], [-1, -1, -1])

    U_shared = array_ops.e

    # Compute the alpha values in the forward algorithm in order to get the
    # partition function.
    forward_cell = CrfForwardRnnCell(U)
    _, alphas = rnn.dynamic_rnn(
        cell=forward_cell,
        inputs=rest_of_input,
        sequence_length=sequence_lengths - 1,
        initial_state=first_input,
        dtype=dtypes.float32)
    log_norm = math_ops.reduce_logsumexp(alphas, [1])
    return log_norm


class CrfForwardRnnCell(core_rnn_cell.RNNCell):
    """Computes the alpha values in a linear-chain CRF.

    See http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.
    """

    def __init__(self, transition_params):
        """Initialize the CrfForwardRnnCell.

        Args:
        transition_params: A [num_tags, num_tags] matrix of binary potentials.
        This matrix is expanded into a [1, num_tags, num_tags] in preparation
        for the broadcast summation occurring within the cell.
        """
        self._transition_params = array_ops.expand_dims(transition_params, 0)
        self._num_tags = transition_params.get_shape()[0].value

    @property
    def state_size(self):
        return self._num_tags

    @property
    def output_size(self):
        return self._num_tags

    def __call__(self, inputs, state, scope=None):
        """Build the CrfForwardRnnCell.

        Args:
        inputs: A [batch_size, num_tags] matrix of unary potentials.
        state: A [batch_size, num_tags] matrix containing the previous alpha
        values.
        scope: Unused variable scope of this cell.

        Returns:
        new_alphas, new_alphas: A pair of [batch_size, num_tags] matrices
        values containing the new alpha values.
        """
        state = array_ops.expand_dims(state, 2)

        # This addition op broadcasts self._transitions_params along the zeroth
        # dimension and state along the second dimension. This performs the
        # multiplication of previous alpha values and the current binary potentials
        # in log space.
        transition_scores = state + self._transition_params
        new_alphas = inputs + math_ops.reduce_logsumexp(transition_scores, [1])

        # Both the state and the output of this RNN cell contain the alphas values.
        # The output value is currently unused and simply satisfies the RNN API.
        # This could be useful in the future if we need to compute marginal
        # probabilities, which would require the accumulated alpha values at every
        # time step.
        return new_alphas, new_alphas
