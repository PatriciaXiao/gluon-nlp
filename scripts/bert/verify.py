import mxnet as mx

from bert_qa_evaluate import PredResult

class AnswerVerify(object):
    def __init__(self, name='simple'):
        self.name = name

    def train(self, train_features, example_ids, out):
        output = mx.nd.split(out, axis=2, num_outputs=2)
        example_ids = example_ids.asnumpy().tolist()
        pred_start = output[0].reshape((0, -3)).asnumpy()
        pred_end = output[1].reshape((0, -3)).asnumpy()
        for example_id, start, end in zip(example_ids, pred_start, pred_end):
            print(PredResult(start=start, end=end))
        exit(0)