import mxnet as mx
import gluonnlp as nlp

from bert_qa_evaluate import PredResult, predict

from mxnet.gluon.data.dataset import Dataset

import model

class VerifierDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)

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
        dataset = self.parse_sentences(train_features, example_ids, out)
        i = 0
        for data in dataset:
            i+=1
            print(i)
        exit(0)

    def parse_sentences(self, train_features, example_ids, out):
        output = mx.nd.split(out, axis=2, num_outputs=2)
        example_ids = example_ids.asnumpy().tolist()
        pred_start = output[0].reshape((0, -3)).asnumpy()
        pred_end = output[1].reshape((0, -3)).asnumpy()
        raw_data = []
        for example_id, start, end in zip(example_ids, pred_start, pred_end):
            results = [PredResult(start=start, end=end)]
            features = train_features[example_id]
            label = 0 if features[0].is_impossible else 1
            # if features[0].is_impossible:
            #     prediction = ""
            prediction, _ = predict(
                features=features,
                results=results,
                tokenizer=self.tokenizer,
                max_answer_length=self.max_answer_length,
                null_score_diff_threshold=self.null_score_diff_threshold,
                n_best_size=self.n_best_size,
                version_2=self.version_2)
            # print("context:", ' '.join(features[0].doc_tokens)) # the original context
            # print("question:", features[0].question_text)
            # print("prediction:", prediction)
            # print("unanswerable:", features[0].is_impossible)
            question_text = features[0].question_text
            raw_data.append([question_text, prediction, label])
        dataset = VerifierDataset(raw_data)
        return dataset


