# product-key memory
# translated from https://github.com/facebookresearch/XLM/blob/master/PKM-layer.ipynb

import mxnet as mx

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


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