import mxnet as mx
import gluonnlp as nlp

from bert_qa_evaluate import PredResult, predict

from mxnet.gluon.data.dataset import Dataset

# https://gluon-nlp.mxnet.io/examples/sentence_embedding/bert.html

import model, data

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
                version_2=True,
                ctx=ctx):
        self.tokenizer=tokenizer
        self.max_answer_length=max_answer_length
        self.null_score_diff_threshold=null_score_diff_threshold
        self.n_best_size=n_best_size
        self.version_2=version_2

        # The labels for the two classes [(0 = not proper) or  (1 = proper)]
        self.all_labels = [0, 1]
        # whether to transform the data as sentence pairs.
        # for single sentence classification, set pair=False
        # for regression task, set class_labels=None
        # for inference without label available, set has_label=False
        self.pair = True
        # The maximum length of an input sequence
        self.max_len = 128

        self.get_model(ctx)
        self.get_loss()

        self.metric = mx.metric.Accuracy()

        self.get_data_transform()


    def get_model(self, ctx):
        self.bert_base, self.vocabulary = nlp.model.get_model('bert_12_768_12',
                                             dataset_name='book_corpus_wiki_en_uncased',
                                             pretrained=True, ctx=ctx, use_pooler=True,
                                             use_decoder=False, use_classifier=False)
        self.bert_classifier = model.classification.BERTClassifier(bert_base, num_classes=2, dropout=0.1)
        self.bert_classifier.classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
        self.bert_classifier.hybridize(static_alloc=True)

    def get_loss(self):
        self.loss_function = mx.gluon.loss.SoftmaxCELoss()
        self.loss_function.hybridize(static_alloc=True)

    def get_data_transform(self):
        self.bert_tokenizer = nlp.data.BERTTokenizer(self.vocabulary, lower=True)
        self.transform = data.transform.BERTDatasetTransform(self.bert_tokenizer, self.max_len,
                                                        class_labels=self.all_labels,
                                                        has_label=True,
                                                        pad=True,
                                                        pair=self.pair)

    def train(self, train_features, example_ids, out):
        dataset_raw = self.parse_sentences(train_features, example_ids, out)
        # print(len(dataset))
        dataset = dataset_raw.transform(self.transform)
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


