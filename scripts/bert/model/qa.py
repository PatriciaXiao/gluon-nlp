# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""BertForQA models."""

__all__ = ['BertForQA', 'BertForQALoss']

from mxnet.gluon import Block, loss, nn, rnn
from mxnet.gluon.loss import Loss
from mxnet import gluon, nd
import mxnet as mx
from mxnet.initializer import MSRAPrelu, Normal, Uniform, Xavier
from mxnet.ndarray import GridGenerator, BilinearSampler

import gluonnlp as nlp
from gluonnlp.model.attention_cell import DotProductAttentionCell, MultiHeadAttentionCell
from gluonnlp.model.transformer import TransformerEncoder

def mask_logits(x, mask):
    r"""Implement mask logits computation.

        Parameters
        -----------
        x : NDArray
            input tensor with shape `(batch_size, sequence_length)`
        mask : NDArray
            input tensor with shape `(batch_size, sequence_length)`

        Returns
        --------
        return : NDArray
            output tensor with shape `(batch_size, sequence_length)`
        """
    return x + -1e30 * (1 - mask)

class CoAttention(Block):
    r"""
    An implementation of co-attention block.
    """

    def __init__(self, bert_out_dim, params=None):
        super(CoAttention, self).__init__("co_attention")
        with self.name_scope():
            self.w4c = gluon.nn.Dense(
                units=1,
                flatten=False,
                weight_initializer=Xavier(),
                use_bias=False
            )
            self.w4q = gluon.nn.Dense(
                units=1,
                flatten=False,
                weight_initializer=Xavier(),
                use_bias=False
            )
            self.w4mlu = self.params.get(
                'linear_kernel', shape=(1, 1, bert_out_dim), init=mx.init.Xavier())
            self.bias = self.params.get(
                'coattention_bias', shape=(1,), init=mx.init.Zero())

    def forward(self, context, query, context_mask, query_mask,
                       context_max_len, query_max_len):
        """Implement forward computation.

        Parameters
        -----------
        context : NDArray
            input tensor with shape `(batch_size, context_sequence_length, hidden_size)`
        query : NDArray
            input tensor with shape `(batch_size, query_sequence_length, hidden_size)`
        context_mask : NDArray
            input tensor with shape `(batch_size, context_sequence_length)`
        query_mask : NDArray
            input tensor with shape `(batch_size, query_sequence_length)`
        context_max_len : int
        query_max_len : int

        Returns
        --------
        return : NDArray
            output tensor with shape `(batch_size, context_sequence_length, 4*hidden_size)`
        """
        F = nd
        ctx = context.context
        w4mlu = self.w4mlu.data(ctx)
        bias = self.bias.data(ctx)
        context_mask = F.expand_dims(context_mask, axis=-1)
        query_mask = F.expand_dims(query_mask, axis=1)

        similarity = self._calculate_trilinear_similarity(
            context, query, context_max_len, query_max_len, w4mlu, bias)

        similarity_dash = F.softmax(mask_logits(similarity, query_mask))
        similarity_dash_trans = F.transpose(F.softmax(
            mask_logits(similarity, context_mask), axis=1), axes=(0, 2, 1))
        c2q = F.batch_dot(similarity_dash, query)
        q2c = F.batch_dot(F.batch_dot(
            similarity_dash, similarity_dash_trans), context)
        return F.concat(context, c2q, context * c2q, context * q2c, dim=-1)

    def _calculate_trilinear_similarity(self, context, query, context_max_len, query_max_len,
                                        w4mlu, bias):
        """Implement the computation of trilinear similarity function.

            refer https://github.com/NLPLearn/QANet/blob/master/layers.py#L505

            The similarity function is:
                    f(w, q) = W[w, q, w * q]
            where w and q represent the word in context and query respectively,
            and * operator means hadamard product.

        Parameters
        -----------
        context : NDArray
            input tensor with shape `(batch_size, context_sequence_length, hidden_size)`
        query : NDArray
            input tensor with shape `(batch_size, query_sequence_length, hidden_size)`
        context_max_len : int
        context_max_len : int

        Returns
        --------
        similarity_mat : NDArray
            output tensor with shape `(batch_size, context_sequence_length, query_sequence_length)`
        """

        subres0 = nd.tile(self.w4c(context), [1, 1, query_max_len])
        subres1 = nd.tile(nd.transpose(
            self.w4q(query), axes=(0, 2, 1)), [1, context_max_len, 1])
        subres2 = nd.batch_dot(w4mlu * context,
                               nd.transpose(query, axes=(0, 2, 1)))
        similarity_mat = subres0 + subres1 + subres2 + bias
        return similarity_mat


class BertForQA(Block):
    """Model for SQuAD task with BERT.

    The model feeds token ids and token type ids into BERT to get the
    pooled BERT sequence representation, then apply a Dense layer for QA task.

    Parameters
    ----------
    bert: BERTModel
        Bidirectional encoder with transformer.
    prefix : str or None
        See document of `mx.gluon.Block`.
    params : ParameterDict or None
        See document of `mx.gluon.Block`.
    n_rnn_layers: number of rnn layers added to the output of bert, 
        before the dense layers (if any) and the span_classifier layer
    rnn_hidden_size: the hidden size of each rnn layer
    n_dense_layers : the number of dense layers,
        before the final output (span_classifier) layer
    units_dense    : the units of each dense layer
    n_features     : # input features, needed to enable initialization
    """

    def __init__(self, bert, prefix=None, params=None,
                    n_rnn_layers=0, rnn_hidden_size=200,
                    n_dense_layers=0, units_dense=200, 
                    add_query=False,
                    apply_coattention=False, bert_out_dim=768):
        super(BertForQA, self).__init__(prefix=prefix, params=params)
        self.add_query=add_query
        self.apply_coattention = apply_coattention
        self.bert = bert
        self.span_classifier = nn.HybridSequential()
        if self.apply_coattention:
            with self.name_scope():
                self.co_attention = CoAttention(bert_out_dim)
        with self.span_classifier.name_scope():
            for i in range(n_rnn_layers):
                self.span_classifier.add(rnn.LSTM(hidden_size=rnn_hidden_size, bidirectional=True))
            for i in range(n_dense_layers):
                self.span_classifier.add(nn.Dense(units=units_dense, flatten=False, activation='relu'))
            self.span_classifier.add(nn.Dense(units=2, flatten=False))

    def forward(self, inputs, token_types, valid_length=None):  # pylint: disable=arguments-differ
        """Generate the unnormalized score for the given the input sequences.

        Parameters
        ----------
        inputs : NDArray, shape (batch_size, seq_length)
            Input words for the sequences.
        token_types : NDArray, shape (batch_size, seq_length)
            Token types for the sequences, used to indicate whether the word belongs to the
            first sentence or the second one.
        valid_length : NDArray or None, shape (batch_size,)
            Valid length of the sequence. This is used to mask the padded tokens.

        Returns
        -------
        outputs : NDArray
            Shape (batch_size, seq_length, 2)
        """
        bert_output = self.bert(inputs, token_types, valid_length)
        if self.add_query:
            o = mx.ndarray.transpose(bert_output, axes=(2,0,1))
            mask = 1 - token_types
            avg_q = mx.nd.sum(mx.nd.multiply(mask, o), axis=2) / mx.nd.sum(mask, axis=1)
            o = mx.nd.add(o, mx.nd.multiply(avg_q.expand_dims(axis=2), token_types))
            attended_output = mx.ndarray.transpose(o, axes=(1,2,0))
        if self.apply_coattention:
            o = mx.ndarray.transpose(bert_output, axes=(2,0,1))
            context_mask = token_types
            query_mask = 1 - context_mask
            context_max_len = bert_output.shape[1] # int(context_mask.sum(axis=1).max().asscalar())
            query_max_len = bert_output.shape[1] # int(query_mask.sum(axis=1).max().asscalar())
            context_emb_encoded = mx.ndarray.transpose(mx.nd.multiply(context_mask, o), axes=(1,2,0))
            query_emb_encoded = mx.ndarray.transpose(mx.nd.multiply(query_mask, o), axes=(1,2,0))
            context_mask = (context_emb_encoded != 0).max(axis=2)
            query_mask = (query_emb_encoded != 0).max(axis=2)
            attended_output = self.co_attention(context_emb_encoded, query_emb_encoded, 
                                                context_mask, query_mask, 
                                                context_max_len, query_max_len)
        if self.add_query or self.apply_coattention:
            output = self.span_classifier(attended_output)
        else:
            output = self.span_classifier(bert_output)
        return output


class BertForQALoss(Loss):
    """Loss for SQuAD task with BERT.
    """
    def __init__(self, weight=None, batch_axis=0, customize_loss=False, **kwargs):  # pylint: disable=unused-argument
        super(BertForQALoss, self).__init__(
            weight=None, batch_axis=0, **kwargs)
        self.customize_loss = customize_loss
        if self.customize_loss:
            self.loss = loss.SoftmaxCELoss(sparse_label=False)
        else:
            self.loss = loss.SoftmaxCELoss()
    # def hybrid_forward(self, F, pred, label):  # pylint: disable=arguments-differ
    def forward(self, pred, label):  # False / True
        """
        Parameters
        ----------
        pred : NDArray, shape (batch_size, seq_length, 2)
            BERTSquad forward output.
        label : list, length is 2, each shape is (batch_size,1)
            label[0] is the starting position of the answer,
            label[1] is the ending position of the answer.

        Returns
        -------
        outputs : NDArray
            Shape (batch_size,)
        """
        F = nd
        pred = F.split(pred, axis=2, num_outputs=2)
        start_pred = pred[0].reshape((0, -3))
        start_label = label[0]
        end_pred = pred[1].reshape((0, -3))
        end_label = label[1]   
        if self.customize_loss:
            # changed to a soft, non-sparse labeling system
            assert len(start_label) == len(end_label), "number of start labels doesn't match number of end labels."
            assert start_pred.shape[1] == end_pred.shape[1], "start encoding dimension doesn't match end encoding dimension."
            batch_size = len(start_label)
            seq_length = start_pred.shape[1]
            start_label_idx = start_label.astype(int).asnumpy().tolist() # start_label_idx[i].asscalar()
            end_label_idx = end_label.astype(int).asnumpy().tolist()
            # one_hot by itself will be the same with the default version
            start_label = mx.ndarray.one_hot(start_label, seq_length)
            end_label = mx.ndarray.one_hot(end_label, seq_length)
            a = 0.8
            b = 0.1
            assert a + 2 * b == 1
            for i in range(batch_size):
                # 0 should be treated separately: it is the digit for no-answer; leave it there be 0 if there is an answer
                # if no answer
                if start_label_idx[i] == 0: # then end index must be also 0
                    continue
                # if there is an answer
                for j in range(1, seq_length):
                    start_label[i, j] = b / (2 ** abs(j - start_label_idx[i]) )
                    end_label[i, j] = b / (2 ** abs(j - end_label_idx[i]) )
                start_label[i, start_label_idx[i]] = a
                end_label[i, end_label_idx[i]] = a
            # start_label = start_label.softmax(axis=1) # too-----slow
            # end_label = end_label.softmax(axis=1)
        return (self.loss(start_pred, start_label) + self.loss(
            end_pred, end_label)) / 2
