import mxnet as mx

class AnswerVerify(object):
	def __init__(self, name='simple'):
		self.name = name

	def train(self, train_features, example_ids, out):
		output = mx.nd.split(out, axis=2, num_outputs=2)
        example_ids = example_ids.asnumpy().tolist()
        pred_start = output[0].reshape((0, -3)).asnumpy()
        pred_end = output[1].reshape((0, -3)).asnumpy()
        print(pred_start)
        print("*****************")
        print(pred_end)
        exit(0)