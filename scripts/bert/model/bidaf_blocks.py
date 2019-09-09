from mxnet import initializer
from mxnet.gluon import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.rnn import LSTM

class BiDAFOutputLayer(HybridBlock):
    """
    ``BiDAFOutputLayer`` produces the final prediction of an answer. The output is a tuple of
    start and end index of token in the paragraph per each batch.
    It accepts 2 inputs:
        `x` : the output of Attention layer of shape:
        seq_max_length x batch_size x 8 * span_start_input_dim
        `m` : the output of Modeling layer of shape:
         seq_max_length x batch_size x 2 * span_start_input_dim
    Parameters
    ----------
    batch_size : `int`
        Size of a batch
    span_start_input_dim : `int`, default 100
        The number of features in the hidden state h of LSTM
    nlayers : `int`, default 1
        Number of recurrent layers.
    biflag: `bool`, default True
        If `True`, becomes a bidirectional RNN.
    dropout: `float`, default 0
        If non-zero, introduces a dropout layer on the outputs of each
        RNN layer except the last layer.
    prefix : `str` or None
        Prefix of this `Block`.
    params : `ParameterDict` or `None`
        Shared Parameters for this `Block`.
    """
    def __init__(self, span_start_input_dim=100, nlayers=1, biflag=True,
                 dropout=0.2, prefix=None, params=None):
        super(BiDAFOutputLayer, self).__init__(prefix=prefix, params=params)

        with self.name_scope():
            self._dropout = nn.Dropout(rate=dropout)
            self._start_index_combined = nn.Dense(units=1, in_units=8 * span_start_input_dim,
                                                  flatten=False)
            self._start_index_model = nn.Dense(units=1, in_units=2 * span_start_input_dim,
                                               flatten=False)
            self._end_index_lstm = LSTM(hidden_size=span_start_input_dim,
                                        num_layers=nlayers, dropout=dropout, bidirectional=biflag,
                                        input_size=2 * span_start_input_dim)
            self._end_index_combined = nn.Dense(units=1, in_units=8 * span_start_input_dim,
                                                flatten=False)
            self._end_index_model = nn.Dense(units=1, in_units=2 * span_start_input_dim,
                                             flatten=False)

    def hybrid_forward(self, F, x, m, mask):
        # pylint: disable=arguments-differ,missing-docstring
        # setting batch size as the first dimension
        x = F.transpose(x, axes=(1, 0, 2))

        start_index_dense_output = self._start_index_combined(self._dropout(x)) + \
                                   self._start_index_model(self._dropout(
                                       F.transpose(m, axes=(1, 0, 2))))

        m2 = self._end_index_lstm(m)
        end_index_dense_output = self._end_index_combined(self._dropout(x)) + \
                                 self._end_index_model(self._dropout(F.transpose(m2,
                                                                                 axes=(1, 0, 2))))

        start_index_dense_output = F.squeeze(start_index_dense_output)
        start_index_dense_output_masked = start_index_dense_output + ((1 - mask) *
                                                                      get_very_negative_number())

        end_index_dense_output = F.squeeze(end_index_dense_output)
        end_index_dense_output_masked = end_index_dense_output + ((1 - mask) *
                                                                  get_very_negative_number())

        return start_index_dense_output_masked, \
               end_index_dense_output_masked

