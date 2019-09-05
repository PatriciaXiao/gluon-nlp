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
    def __init__(self, input_dim, output_dim, ctx, params):
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
        self.ctx = ctx
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
        keys = nd.array(np.array([
                get_uniform_keys(self.n_keys, half, seed=(2 * i + j))
                for i in range(self.heads)
                for j in range(2)
            ])).reshape(self.heads, 2, self.n_keys, half)
        self.keys = gluon.Parameter("keys", shape=keys.shape, init=mx.init.Zero())
        self.keys.initialize(ctx=self.ctx)
        self.keys.set_data(keys)

    def _get_indices(self, query, subkeys):
        """
        Generate scores and indices for a specific head.
        """
        assert len(query.shape) == 2 and query.shape[1] == self.k_dim
        bs = query.shape[0]
        knn = self.knn
        half = self.k_dim // 2
        n_keys = len(subkeys[0])
        # split query for product quantization
        q1 = query[:, :half]                                            # (bs,half)
        q2 = query[:, half:]                                            # (bs,half)

        #q1 = nd.ones(q1.shape)
        #q2 = nd.ones(q2.shape)
        # compute indices with associated scores
        scores1 = nd.dot(q1, subkeys[0].T)                              # (bs,n_keys)
        scores2 = nd.dot(q2, subkeys[1].T)                              # (bs,n_keys)
        indices1 = scores1.topk(k=knn, axis=1)                          # (bs,knn)
        indices2 = scores2.topk(k=knn, axis=1)                          # (bs,knn)
        scores1 = nd.array([scores1[i][indices1[i]].asnumpy() for i in range(bs)])
        scores2 = nd.array([scores2[i][indices2[i]].asnumpy() for i in range(bs)])

        # cartesian product on best candidate keys
        all_scores = (
            scores1.reshape(bs, knn, 1).broadcast_to([bs, knn, knn]) +
            scores2.reshape(bs, 1, knn).broadcast_to([bs, knn, knn])
        ).reshape(bs, -1)                                                # (bs,knn**2)
        all_indices = (
            indices1.reshape(bs, knn, 1).broadcast_to([bs, knn, knn]) * n_keys +
            indices2.reshape(bs, 1, knn).broadcast_to([bs, knn, knn])
        ).reshape(bs, -1)                                                # (bs,knn**2)

        # select best scores with associated indices
        best_indices = all_scores.topk(k=knn, axis=1)                    # (bs,knn) 
        scores = all_scores[best_indices]             
        scores = nd.array([all_scores[i][best_indices[i]].asnumpy() for i in range(bs)])
        indices = nd.array([all_indices[i][best_indices[i]].asnumpy() for i in range(bs)])    # (bs,knn) 

        assert scores.shape == indices.shape == (bs, knn)
        return scores, indices

    def get_indices(self, query):
        """
        Generate scores and indices.
        """
        assert len(query.shape) == 2 and query.shape[1] == self.k_dim
        query = query.reshape(-1, self.heads, self.k_dim)
        bs = len(query)
        outputs = [self._get_indices(query[:, i], self.keys.data(self.ctx)[i]) for i in range(self.heads)] # 4
        s = mx.ndarray.concat(*[s.reshape(bs, 1, self.knn) for s, _ in outputs], dim=1)  # (bs,heads,knn)
        i = mx.ndarray.concat(*[i.reshape(bs, 1, self.knn) for _, i in outputs], dim=1)  # (bs,heads,knn)
        return s.reshape(-1, self.knn), i.reshape(-1, self.knn)

    def forward(self, _input):
        """
        Read from the memory.
        """
        # input dimensions
        assert _input.shape[-1] == self.input_dim
        prefix_shape = _input.shape[:-1]
        bs = np.prod(prefix_shape)
        # compute query
        _input = mx.ndarray.Dropout(_input, p=self.input_dropout)       # (...,i_dim)
        query = self.query_proj(_input.reshape(-1, self.input_dim))     # (bs,heads*k_dim)
        query = query.reshape(bs * self.heads, self.k_dim)              # (bs*heads,k_dim)
        query = mx.ndarray.Dropout(query, p=self.query_dropout)         # (bs*heads,k_dim)
        assert query.shape == (bs * self.heads, self.k_dim)
        # retrieve indices and scores
        scores, indices = self.get_indices(query)                       # (bs*heads,knn)
        scores = mx.ndarray.softmax(scores, axis=-1)                    # (bs*heads,knn)
        # merge heads / knn (since we sum heads)
        indices = indices.reshape(bs, self.heads * self.knn)                       # (bs,heads*knn)
        scores = scores.reshape(bs, self.heads * self.knn)                         # (bs,heads*knn)
        # weighted sum of values
        output_raw = self.values(indices)                                          # (bs,knn,v_dim)
        # print(self.values.weight.data())
        output = mx.ndarray.squeeze(mx.ndarray.batch_dot(mx.ndarray.expand_dims(scores, 1), output_raw), axis=1)
        output = mx.ndarray.Dropout(output, p=self.value_dropout)                   # (bs,v_dim)
        # reshape output
        if len(prefix_shape) >= 2:
            output = output.reshape(prefix_shape + (self.v_dim,))                  # (...,v_dim)
        return output

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
    memory = HashingMemory(input_dim, output_dim, ctx, params)
    # net.collect_params()
    memory.initialize(init=mx.init.Normal(output_dim ** -0.5), ctx=ctx)
    memory.hybridize()
    print(memory)
    # the input
    x = mx.ndarray.random.randn(2, 3, 4, input_dim).as_in_context(ctx)
    output = memory(x)
    print(output.sum().asscalar())
    print(output.shape)




