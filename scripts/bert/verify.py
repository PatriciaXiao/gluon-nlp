import mxnet as mx
import gluonnlp as nlp

from bert_qa_evaluate import PredResult, predict

class AnswerVerify(object):
    def __init__(self, tokenizer=nlp.data.BERTBasicTokenizer(lower=True),
                max_answer_length=30,
                null_score_diff_threshold=-2.0,
                n_best_size=20,
                version_2=True):
        self.tokenizer=tokenizer
        self.max_answer_length=max_answer_length
        self.null_score_diff_threshold=null_score_diff_threshold
        self.n_best_size=n_best_size
        self.version_2=version_2

    def train(self, train_features, example_ids, out):
        output = mx.nd.split(out, axis=2, num_outputs=2)
        example_ids = example_ids.asnumpy().tolist()
        pred_start = output[0].reshape((0, -3)).asnumpy()
        pred_end = output[1].reshape((0, -3)).asnumpy()
        for example_id, start, end in zip(example_ids, pred_start, pred_end):
            results = [PredResult(start=start, end=end)]
            features = train_features[example_id]
            prediction, _ = predict(
                features=features,
                results=results,
                tokenizer=self.tokenizer,
                max_answer_length=self.max_answer_length,
                null_score_diff_threshold=self.null_score_diff_threshold,
                n_best_size=self.n_best_size,
                version_2=self.version_2)
            print(prediction)
            print(features[0].is_impossible)
            exit(0)