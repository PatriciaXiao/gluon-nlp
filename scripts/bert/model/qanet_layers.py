import mxnet as mx
from mxnet import gluon
from mxnet.gluon import Block

EMB_ENCODER_CONV_CHANNELS = 768
p_L = 0.9

import math
from mxnet import nd
from mxnet.initializer import MSRAPrelu, Normal, Uniform, Xavier
from gluonnlp.model import (DotProductAttentionCell, Highway,
                            MultiHeadAttentionCell)

F = nd

class Encoder(Block):
    r"""
    Stacked block of Embedding encoder or Model encoder.
    """

    def __init__(self, kernel_size, num_filters, conv_layers=2, num_heads=8,
                 num_blocks=1, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.dropout = gluon.nn.Dropout(0.1)
        total_layers = float((conv_layers + 2) * num_blocks)
        sub_layer_idx = 1
        self.num_blocks = num_blocks
        self.stack_encoders = gluon.nn.Sequential()
        with self.stack_encoders.name_scope():
            for _ in range(num_blocks):
                self.stack_encoders.add(
                    OneEncoderBlock(
                        kernel_size=kernel_size,
                        num_filters=num_filters,
                        conv_layers=conv_layers,
                        num_heads=num_heads,
                        total_layers=total_layers,
                        sub_layer_idx=sub_layer_idx
                    )
                )
                sub_layer_idx += (conv_layers + 2)

    def forward(self, x, mask):
        r"""Implement forward computation.

        Parameters
        -----------
        x : NDArray
            input tensor with shape `(batch_size, sequence_length, features)`
        mask : NDArray
            input tensor with shape `(batch_size, sequence_length)`

        Returns, NDArray
        --------
            output tensor with shape `(batch_size, sequence_length, features)`
        """
        for encoder in self.stack_encoders:
            x = encoder(x, mask)
            x = F.Dropout(x, p=0.1)
        return x

class OneEncoderBlock(Block):
    r"""The basic encoder block.

    Parameters
    ----------
    kernel_size : int
        The kernel size for all depthwise convolution layers.
    num_filters : int
        The number of filters for all convolution layers.
    conv_layers : int
        The number of convolution layers in one encoder block.
    num_heads : int
        The number of heads in multi-head attention layer.
    total_layers : int
    sub_layer_idx : int
        The sub_layer_idx / total_layers is the dropout probability for layer.
    """

    def __init__(self, kernel_size, num_filters, conv_layers, num_heads, total_layers,
                 sub_layer_idx, **kwargs):
        super(OneEncoderBlock, self).__init__(**kwargs)
        self.position_encoder = PositionEncoder()
        self.convs = gluon.nn.Sequential()
        with self.convs.name_scope():
            for _ in range(conv_layers):
                one_conv_module = gluon.nn.Sequential()
                with one_conv_module.name_scope():
                    one_conv_module.add(
                        gluon.nn.LayerNorm(epsilon=1e-06)
                    )
                    one_conv_module.add(
                        gluon.nn.Dropout(0.1)
                    )
                    one_conv_module.add(
                        DepthwiseConv(
                            kernel_size=kernel_size,
                            num_filters=num_filters,
                            input_channels=num_filters
                        )
                    )
                    one_conv_module.add(
                        StochasticDropoutLayer(
                            dropout=(sub_layer_idx / total_layers) * (1 - p_L)
                        )
                    )
                sub_layer_idx += 1
                self.convs.add(one_conv_module)

        with self.name_scope():
            self.dropout = gluon.nn.Dropout(0.1)
            self.attention = SelfAttention(num_heads=num_heads)
            self.attention_dropout = StochasticDropoutLayer(
                (sub_layer_idx / total_layers) * (1 - p_L))
            sub_layer_idx += 1
            self.attention_layer_norm = gluon.nn.LayerNorm(epsilon=1e-06)

        self.positionwise_ffn = gluon.nn.Sequential()
        with self.positionwise_ffn.name_scope():
            self.positionwise_ffn.add(
                gluon.nn.LayerNorm(epsilon=1e-06)
            )
            self.positionwise_ffn.add(
                gluon.nn.Dropout(rate=0.1)
            )
            self.positionwise_ffn.add(
                gluon.nn.Dense(
                    units=EMB_ENCODER_CONV_CHANNELS,
                    activation='relu',
                    use_bias=True,
                    weight_initializer=MSRAPrelu(),
                    flatten=False
                )
            )
            self.positionwise_ffn.add(
                gluon.nn.Dense(
                    units=EMB_ENCODER_CONV_CHANNELS,
                    use_bias=True,
                    weight_initializer=Xavier(),
                    flatten=False
                )
            )
            self.positionwise_ffn.add(
                StochasticDropoutLayer(
                    dropout=(sub_layer_idx / total_layers) * (1 - p_L)
                )
            )

    def forward(self, x, mask):
        r"""Implement forward computation.

        Parameters
        -----------
        x : NDArray
            input tensor with shape `(batch_size, sequence_length, hidden_size)`
        mask : NDArray
            input tensor with shape `(batch_size, sequence_length)`

        Returns
        --------
        x : NDArray
            output tensor with shape `(batch_size, sequence_length, hidden_size)`
        mask : NDArray
            output tensor with shape `(batch_size, sequence_length)`
        """
        x = self.position_encoder(x)
        for conv in self.convs:
            residual = x
            x = conv(x) + residual
        residual = x
        x = self.attention_layer_norm(x)
        x = F.Dropout(x, p=0.1)
        x = self.attention(x, mask)
        x = self.attention_dropout(x) + residual
        return x + self.positionwise_ffn(x)


class StochasticDropoutLayer(Block):
    r"""
    Stochastic dropout a layer.
    """

    def __init__(self, dropout, **kwargs):
        super(StochasticDropoutLayer, self).__init__(**kwargs)
        self.dropout = dropout
        with self.name_scope():
            self.dropout_fn = gluon.nn.Dropout(dropout)

    def forward(self, inputs):
        ctx = inputs.ctx
        if F.random.uniform().asscalar() < self.dropout:
            return F.zeros(shape=(1,)).as_in_context(ctx)
        else:
            return self.dropout_fn(inputs)


class SelfAttention(Block):
    r"""
    Implementation of self-attention with gluonnlp.model.MultiHeadAttentionCell
    """

    def __init__(self, num_heads, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        with self.name_scope():
            self.attention = MultiHeadAttentionCell(
                num_heads=num_heads,
                base_cell=DotProductAttentionCell(
                    scaled=True,
                    dropout=0.1,
                    use_bias=False
                ),
                query_units=EMB_ENCODER_CONV_CHANNELS,
                key_units=EMB_ENCODER_CONV_CHANNELS,
                value_units=EMB_ENCODER_CONV_CHANNELS,
                use_bias=False,
                weight_initializer=Xavier()
            )

    def forward(self, x, mask):
        r"""Implement forward computation.

        Parameters
        -----------
        x : NDArray
            input tensor with shape `(batch_size, sequence_length, hidden_size)`
        mask : NDArray
            input tensor with shape `(batch_size, sequence_length)`

        Returns
        --------
        x : NDArray
            output tensor with shape `(batch_size, sequence_length, hidden_size)`
        """
        mask = F.batch_dot(mask.expand_dims(axis=2), mask.expand_dims(axis=1))
        return self.attention(x, x, mask=mask)[0]


class PositionEncoder(Block):
    r"""
    An implementation of position encoder.
    """

    def __init__(self, **kwargs):
        super(PositionEncoder, self).__init__(**kwargs)
        with self.name_scope():
            pass

    def forward(self, x, min_timescale=1.0, max_timescale=1e4):
        r"""Implement forward computation.

        Parameters
        -----------
        x : NDArray
            input tensor with shape `(batch_size, sequence_length, hidden_size)`

        Returns
        --------
         : NDArray
            output tensor with shape `(batch_size, sequence_length, hidden_size)`
        """
        length = x.shape[1]
        channels = x.shape[2]
        position = nd.array(range(length))
        num_timescales = channels // 2
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) / (num_timescales - 1))
        inv_timescales = min_timescale * \
            nd.exp(nd.array(range(num_timescales)) * -log_timescale_increment)
        scaled_time = F.expand_dims(
            position, 1) * F.expand_dims(inv_timescales, 0)
        signal = F.concat(F.sin(scaled_time), F.cos(scaled_time), dim=1)
        signal = F.reshape(signal, (1, length, channels))
        return x + signal.as_in_context(x.context)


class DepthwiseConv(Block):
    r"""
    An implementation of depthwise-convolution net.
    """

    def __init__(self, kernel_size, num_filters, input_channels, **kwargs):
        super(DepthwiseConv, self).__init__(**kwargs)
        with self.name_scope():
            self.depthwise_conv = gluon.nn.Conv1D(
                channels=input_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=input_channels,
                use_bias=False,
                dilation = 1,
            	strides = 1,
                weight_initializer=MSRAPrelu()
            )
            
            # out_width = floor((width+2*padding-dilation*(kernel_size-1)-1)/strides)+1
            
            self.pointwise_conv = gluon.nn.Conv1D(
                channels=num_filters,
                kernel_size=1,
                activation='relu',
                use_bias=True,
                weight_initializer=MSRAPrelu(),
                bias_initializer='zeros'
            )

    def forward(self, inputs):
        r"""Implement forward computation.

        Parameters
        -----------
        inputs : NDArray
            input tensor with shape `(batch_size, sequence_length, hidden_size)`

        Returns
        --------
        x : NDArray
            output tensor with shape `(batch_size, sequence_length, new_hidden_size)`
        """
        # print("inputs.shape",inputs.shape)
        tmp = F.transpose(inputs, axes=(0, 2, 1))
        depthwise_conv = self.depthwise_conv(tmp)
        outputs = self.pointwise_conv(depthwise_conv)
        return F.transpose(outputs, axes=(0, 2, 1))

