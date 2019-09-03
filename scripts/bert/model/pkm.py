# product-key memory
# translated from https://github.com/facebookresearch/XLM/blob/master/PKM-layer.ipynb
# reference https://github.com/zackchase/mxnet-the-straight-dope/blob/master/cheatsheets/pytorch_gluon.md
#           https://gist.github.com/zhanghang1989/3d646f71d60c17048cf8ad582393ac6c

import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import Block, loss, nn

import math
import numpy as np

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_uniform_keys(n_keys, dim, seed):
    """
    Generate random uniform keys (same initialization as nn.Linear).
    """
    rng = np.random.RandomState(seed)
    bound = 1 / math.sqrt(dim)
    keys = rng.uniform(-bound, bound, (n_keys, dim))
    return keys.astype(np.float32)

class HashingMemory(Block):
    r'''
    A simple implementation of the product-key memory block
    '''
    def __init__(self, input_dim, output_dim, params):
        super(HashingMemory, self).__init__()
        # global parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k_dim = params.k_dim
        self.v_dim = output_dim
        self.n_keys = params.n_keys
        self.size = self.n_keys ** 2
        self.heads = params.heads
        self.knn = params.knn
        assert self.k_dim >= 2 and self.k_dim % 2 == 0
        # dropout
        self.input_dropout = params.input_dropout
        self.query_dropout = params.query_dropout
        self.value_dropout = params.value_dropout
        # initialize keys / values
        self.initialize_keys()
        # no embedding-bag available, thus need to sum it up latter
        self.values = nn.Embedding(self.size, self.v_dim)
        # query network
        self.query_proj = nn.HybridSequential()
        with self.name_scope():
            self.query_proj.add(nn.Dense(units=self.heads * self.k_dim,
                                        in_units=self.input_dim,
                                        use_bias=True))
            if params.query_batchnorm:
                self.query_proj.add(nn.BatchNorm(center=True))

        if params.query_batchnorm:
            print("WARNING: Applying batch normalization to queries improves the performance "
                  "and memory usage. But if you use it, be sure that you use batches of "
                  "sentences with the same size at training time (i.e. without padding). "
                  "Otherwise, the padding token will result in incorrect mean/variance "
                  "estimations in the BatchNorm layer.\n")


    def initialize_keys(self):
        """
        Create two subkey sets per head.
        `self.keys` is of shape (heads, 2, n_keys, k_dim // 2)
        """
        half = self.k_dim // 2
        keys = gluon.Parameter('keys', 
            init=nd.array(np.array([
                get_uniform_keys(self.n_keys, half, seed=(2 * i + j))
                for i in range(self.heads)
                for j in range(2)
            ])).reshape(self.heads, 2, self.n_keys, half))
        self.keys = gluon.Parameter(keys)

    def forward(self, input):
        """
        Read from the memory.
        """
        # input dimensions
        assert input.shape[-1] == self.input_dim
        prefix_shape = input.shape[:-1]
        bs = np.prod(prefix_shape)

        print(bs)
        exit(0)

params = AttrDict({
    "sparse": False,
    "k_dim": 128,
    "heads": 4,
    "knn": 32,
    "n_keys": 512,  # the memory will have (n_keys ** 2) values
    "query_batchnorm": True,
    "input_dropout": 0,
    "query_dropout": 0,
    "value_dropout": 0,
})



if __name__ == "__main__":
    ctx = mx.gpu(0) if mx.context.num_gpus() else mx.cpu()
    input_dim = 50
    output_dim = 100
    memory = HashingMemory(input_dim, output_dim, params)
    # net.collect_params()
    memory.initialize(init=mx.init.Normal(output_dim ** -0.5), ctx=ctx)
    memory.hybridize()
    print(memory)
    # the input
    x = mx.ndarray.random.randn(2, 3, 4, input_dim).as_in_context(ctx)
    output = memory(x)




