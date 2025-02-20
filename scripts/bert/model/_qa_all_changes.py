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

from .bidaf_blocks import BiDAFOutputLayer

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

    def __init__(self, name, bert_out_dim, params=None, concat_out=True):
        super(CoAttention, self).__init__(name)
        self.in_dim = bert_out_dim
        self.concat_out = concat_out
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
            if not self.concat_out:
                self.out_weight = self.params.get(
                    'weight_of_output', shape=(1,4), init=mx.init.Xavier())

    def forward(self, context, query, context_mask, query_mask,
                       context_max_len, query_max_len, cls_emb_encoded=None):
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
        cls_emb_encoded : used for null-entry prediction

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
        if self.concat_out:
            return F.concat(context, c2q, context * c2q, context * q2c, dim=-1), \
                   F.concat(query,   q2c, query * q2c, query * c2q, dim=-1)
        else:
            out_weight = self.out_weight.data(ctx)
            return out_weight[0,0] * context + out_weight[0,1] * c2q + out_weight[0,2] * context * c2q + out_weight[0,3] * context * q2c, \
                    out_weight[0,0] * query  + out_weight[0,1] * q2c + out_weight[0,2] * query * q2c   + out_weight[0,3] * query * c2q

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

    def __init__(self, bert, prefix=None, params=None, \
                    n_rnn_layers=0, rnn_hidden_size=600, num_rnn_layers=1, n_dense_layers=0, units_dense=600, \
                    add_query=False, \
                    apply_coattention=False, bert_out_dim=768,\
                    apply_self_attention=False, self_attention_dimension=None, n_attention_heads=4,
                    apply_transformer=False,
                    qanet_style_out=False,
                    bidaf_style_out=False,
                    remove_special_token=False):
        super(BertForQA, self).__init__(prefix=prefix, params=params)
        self.add_query=add_query
        self.apply_coattention = apply_coattention
        self.apply_self_attention = apply_self_attention
        self.apply_transformer = apply_transformer
        self.qanet_style_out = qanet_style_out
        self.bidaf_style_out = bidaf_style_out
        self.remove_special_token = remove_special_token
        self.bert = bert
        if self.apply_coattention:
            with self.name_scope():
                #self.co_attention_ = CoAttention("co-attention_", bert_out_dim) # try multiple layers
                self.co_attention = CoAttention("co-attention", bert_out_dim)
                if self.qanet_style_out:
                    self.project = gluon.nn.Dense(
                        units=bert_out_dim,
                        flatten=False,
                        use_bias=False,
                        weight_initializer=Xavier(),
                        prefix='projection_'
                    )
                    self.dropout = gluon.nn.Dropout(0.1)
                    self.model_encoder = TransformerEncoder(units=bert_out_dim)
                    self.predict_begin = gluon.nn.Dense(
                        units=1,
                        use_bias=True,
                        flatten=False,
                        weight_initializer=Xavier(
                            rnd_type='uniform', factor_type='in', magnitude=1),
                        bias_initializer=Uniform(1.0/bert_out_dim),
                        prefix='predict_start_'
                    )
                    self.predict_end = gluon.nn.Dense(
                        units=1,
                        use_bias=True,
                        flatten=False,
                        weight_initializer=Xavier(
                            rnd_type='uniform', factor_type='in', magnitude=1),
                        bias_initializer=Uniform(1.0/bert_out_dim),
                        prefix='predict_end_'
                    )
                    self.flatten = gluon.nn.Flatten()
                elif self.bidaf_style_out:
                    # BiDAF mode
                    self.modeling_layer = rnn.LSTM( hidden_size=int(bert_out_dim / 2), 
                                                    num_layers=2, 
                                                    dropout=0.0, 
                                                    bidirectional=True,
                                                    input_size=int(bert_out_dim * 4))
                    self.output_layer = BiDAFOutputLayer(span_start_input_dim=int(bert_out_dim / 2),
                                                        nlayers=1,
                                                        dropout=0.2)
                # '''
                # for the cls's encoding
                # used in version 2.0
                self.cls_mapping = nn.Dense(
                    units=2,
                    flatten=False,
                    weight_initializer=Xavier(),
                    prefix='cls_mapping_'
                )
                # '''
        if self.apply_self_attention:
            if self_attention_dimension is None:
                self_attention_dimension = bert_out_dim
            with self.name_scope():
                self.multi_head_attention = MultiHeadAttentionCell(DotProductAttentionCell(), \
                        self_attention_dimension, self_attention_dimension, self_attention_dimension, n_attention_heads)
        if self.apply_transformer:
            with self.name_scope():
                self.transformer = TransformerEncoder(units=bert_out_dim)
        if self.apply_coattention and (self.qanet_style_out or self.bidaf_style_out):
            self.span_classifier = None
        else:
            self.span_classifier = nn.HybridSequential()
            with self.span_classifier.name_scope():
                for i in range(n_rnn_layers):
                    self.span_classifier.add(rnn.LSTM( hidden_size=rnn_hidden_size, 
                                                        num_layers=num_rnn_layers, 
                                                        dropout=0.0, 
                                                        bidirectional=True))
                for i in range(n_dense_layers):
                    self.span_classifier.add(nn.Dense(units=units_dense, flatten=False, activation='relu'))
                self.span_classifier.add(nn.Dense(units=2, flatten=False))

    def shift_ndarray(self, data, mask, raw_offset):
        '''
        Parameters
        ----------
        data: NDArray, shape(dim, batch_size, seq_length)
                mostly we use o = mx.ndarray.transpose(bert_output, axes=(2,0,1))
        mask: NDArray, shape(batch_size, seq_length)
                the mask; in most cases,
                        context_mask = token_types
                        query_mask = 1 - context_mask
        raw_offset: NDArray, shape(batch_size, seq_length)
                    For example, if it is:
                    [[1, 1, 1, 1, 1, .... 1],
                     [-2, -2, -2, -2, -2, .... -2],
                     [0, 0, 0, 0, 0, .... 0]]
                    we'll shift every entry of the targeting matrix, data (o) 
                                           first line left-ward 1 position
                                           second line right-ward 1 position
                                           third line remains unchanged
                                           and after shifting, the blank positions will be fillled in with zeros
                    one way of computing this offset matrix:
                    raw_offset_contx = query_mask.sum(axis=1).reshape(len(query_mask),1).tile(bert_output.shape[1])
                    raw_offset_query = mx.nd.ones(inputs.shape).as_in_context(inputs.context)
                                    or mx.nd.zeros(inputs.shape).as_in_context(inputs.context)
            in case that ndarray is shifted, we need new valid length
                valid_query_length = query_mask.sum(axis=1)
                valid_contx_length = valid_length - valid_query_length
            and if remove special token, 
                valid_query_length = valid_query_length - 2
                valid_contx_length = valid_contx_length - 1

        Returns
        -------
        result: the final result we want is mx.ndarray.transpose(result, axes=(1,2,0))
        mask_result: the mask of the result (shifted)
        '''
        data_raw = mx.ndarray.expand_dims(mx.nd.multiply(mask, data), 0)
        raw_offset = raw_offset.astype(float)
        warp_matrix = mx.ndarray.expand_dims(mx.ndarray.stack(raw_offset, 
                                                mx.nd.zeros(raw_offset.shape).astype(float).as_in_context(raw_offset.context)), 
                                            0)
        grid = GridGenerator(data=warp_matrix, transform_type='warp')
        warpped_out = BilinearSampler(data_raw.astype(float), grid.astype(float))
        result = mx.ndarray.squeeze(warpped_out, axis=0).astype('float32')
        # correction needed for the first digit
        col_offsets = raw_offset[:,0].as_in_context(data.context)
        row_offsets = mx.nd.arange(len(col_offsets)).as_in_context(data.context)
        # mask shifted
        mask_result = (result != 0).max(axis=0)
        return result, mask_result

    def forward(self, inputs, token_types, valid_length=None, additional_masks=None):  # pylint: disable=arguments-differ
        """Generate the unnormalized score for the given the input sequences.

        Parameters
        ----------
        inputs : NDArray, shape (batch_size, seq_length, dim)
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
            # get the two encodings separated
            o = mx.ndarray.transpose(bert_output, axes=(2,0,1))
            context_mask = token_types
            query_mask = 1 - context_mask
            if self.remove_special_token:
                cls_mask, sep_mask_1, sep_mask_2 = additional_masks
                context_mask = context_mask - sep_mask_2
                query_mask = query_mask - (sep_mask_1 + cls_mask)
            context_max_len = bert_output.shape[1]
            query_max_len = bert_output.shape[1]
            context_emb_encoded = mx.ndarray.transpose(mx.nd.multiply(context_mask, o), axes=(1,2,0))
            query_emb_encoded = mx.ndarray.transpose(mx.nd.multiply(query_mask, o), axes=(1,2,0))
            context_mask = (context_emb_encoded != 0).max(axis=2)
            query_mask = (query_emb_encoded != 0).max(axis=2)
            attended_output, attended_query = self.co_attention(context_emb_encoded, query_emb_encoded, 
                                                context_mask, query_mask, 
                                                context_max_len, query_max_len)
            if self.qanet_style_out:
                M = self.project(attended_output)
                M = self.dropout(M)
                M_0, _ = self.model_encoder(M, valid_length=valid_length)
                M_1, _ = self.model_encoder(M_0, valid_length=valid_length)
                M_2, _ = self.model_encoder(M_1, valid_length=valid_length)
                begin_hat = self.flatten(
                    self.predict_begin(nd.concat(M_0, M_1, dim=-1)))
                end_hat = self.flatten(self.predict_end(nd.concat(M_0, M_2, dim=-1)))
                predicted_begin = mask_logits(begin_hat, context_mask)
                predicted_end = mask_logits(end_hat, context_mask)
                prediction = nd.stack(predicted_begin, predicted_end, axis=2)
                # deal with the null-score score
                cls_emb_encoded = mx.ndarray.expand_dims(bert_output[:, 0, :], 1)
                cls_reshaped = self.cls_mapping(cls_emb_encoded)
                output = mx.ndarray.concat(cls_reshaped, prediction[:,1:,:], dim=1)
                return (output, bert_output)
            elif self.bidaf_style_out:
                modeled_output = self.modeling_layer(attended_output)
                predicted_begin, predicted_end = self.output_layer(attended_output, modeled_output, context_mask)
                prediction = nd.stack(predicted_begin, predicted_end, axis=2)
                cls_emb_encoded = mx.ndarray.expand_dims(bert_output[:, 0, :], 1)
                cls_reshaped = self.cls_mapping(cls_emb_encoded)
                output = mx.ndarray.concat(cls_reshaped, prediction[:,1:,:], dim=1)
                return (output, bert_output)
        if self.apply_self_attention:
            attended_output, att_weights = self.multi_head_attention(bert_output, bert_output)   
        if self.apply_transformer:
            attended_output, additional_outputs = self.transformer(bert_output)
        if self.add_query or self.apply_self_attention or self.apply_transformer:
            output = self.span_classifier(attended_output)
        elif self.apply_coattention and not (self.qanet_style_out or self.bidaf_style_out):
            context_output_raw = self.span_classifier(attended_output)
            # mask the output - have to do this, because the rest digits are previously masked as 0, which is incorrect
            #     many valid digits have values smaller than 0, it is improper to use 0 as "impossible value"
            context_output_mask_raw = context_mask.expand_dims(-1)
            context_output_mask = nd.concat(context_output_mask_raw, context_output_mask_raw, dim=-1)
            context_output = mask_logits(context_output_raw, context_output_mask)
            # deal with the null-score score
            cls_emb_encoded = mx.ndarray.expand_dims(bert_output[:, 0, :], 1)
            cls_reshaped = self.cls_mapping(cls_emb_encoded)
            output = mx.ndarray.concat(cls_reshaped, context_output[:,1:,:], dim=1)
            # output = context_output
        else:
            output = self.span_classifier(bert_output)
        return (output, bert_output)

    def loss(self, weight=None, batch_axis=0, customize_loss=False, **kwargs):
        return BertForQALoss(weight=weight, batch_axis=batch_axis, customize_loss=customize_loss, **kwargs)

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

