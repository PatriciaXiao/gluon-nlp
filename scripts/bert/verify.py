import mxnet as mx
import gluonnlp as nlp

from bert_qa_evaluate import PredResult, predict, predict_span

from mxnet.gluon.data.dataset import Dataset

import random

# https://gluon-nlp.mxnet.io/examples/sentence_embedding/bert.html

import model, data

from mxnet.gluon import Block
from mxnet.gluon import nn

# https://blog.csdn.net/HappyRocking/article/details/80900890
import re
pattern = r'\?|\.|\!|;'

class VerifierDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)

class verifier_layers(Block):
    def __init__(self, dropout=0.0, num_classes=2, in_units=768, prefix=None, params=None):
        super(verifier_layers, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.classifier = nn.HybridSequential(prefix=prefix)
            self.classifier.add(nn.Dense(units=in_units, flatten=False, activation='tanh'))
            if dropout:
                self.classifier.add(nn.Dropout(rate=dropout))
            self.classifier.add(nn.Dense(units=num_classes))
    def forward(self, inputs):
        '''
        inputs are bert outputs
        '''
        return self.classifier(inputs)

class AnswerVerifyThreshold(object):
    def __init__(self, tokenizer=nlp.data.BERTBasicTokenizer(lower=True),
                max_answer_length=30,
                n_best_size=20,
                max_len=384,
                version_2=True,
                ctx=mx.cpu()):
        self.tokenizer=tokenizer
        self.max_answer_length=max_answer_length
        self.n_best_size=n_best_size
        self.version_2=version_2

        self.data = list()
        self.null_score_diff_threshold = 0.0 # normally between -5 and -1

    def train(self, train_features, example_ids, out, token_types=None, bert_out=None, num_epochs=1, verbose=False):
        if not self.version_2:
            return        
        raw_data = self.get_training_data(train_features, example_ids, out, token_types=token_types)
        self.data.extend(raw_data)
        # exit(0)

    def evaluate(self, dev_feature, prediction):
        # asserted that prediction is not null
        # reset the data
        self.data = list()

    def update(self):
        pass

    def get_training_data(self, train_features, example_ids, out, token_types=None):
        output = mx.nd.split(out, axis=2, num_outputs=2)
        example_ids = example_ids.asnumpy().tolist()
        pred_start = output[0].reshape((0, -3)).asnumpy()
        pred_end = output[1].reshape((0, -3)).asnumpy()
        raw_data = []
        for example_id, start, end in zip(example_ids, pred_start, pred_end):
            results = [PredResult(start=start, end=end)]
            features = train_features[example_id]
            label = 0 if features[0].is_impossible else 1
            prediction, score_diff, top_predict = predict(
                    features=features,
                    results=results,
                    tokenizer=self.tokenizer,
                    max_answer_length=self.max_answer_length,
                    n_best_size=self.n_best_size,
                    version_2=self.version_2)
            non_empty_top = 1. if top_predict else 0.
            print(prediction, "," , top_predict, ",", features[0].orig_answer_text)
            if top_predict:
                exit(0)
            raw_data.append([score_diff, non_empty_top, label])
        return raw_data



class AnswerVerifyDense(object):
    def __init__(self,
                max_answer_length=30,
                null_score_diff_threshold=-2.0,
                n_best_size=20,
                max_len=384,
                dropout=0.0,
                in_units=768,
                version_2=True,
                mode='classification',
                extract_sentence=True,
                ctx=mx.cpu(),
                prefix=None,
                params=None):
        self.max_answer_length=max_answer_length
        self.null_score_diff_threshold=null_score_diff_threshold
        self.n_best_size=n_best_size
        self.version_2=version_2
        self.ctx = ctx
        self.mode = mode
        assert mode in ['classification', 'regression']
        self.num_classes = 2 if mode == 'classification' else 1

        # the model's definition
        self.dense_layer = verifier_layers(dropout=dropout, 
                                        num_classes=self.num_classes, 
                                        in_units=in_units, 
                                        prefix=prefix, 
                                        params=params)
        self.dense_layer.collect_params().initialize(init=mx.init.Normal(0.02), ctx=self.ctx)

        # the trainer's definition
        self.step_cnt = 0
        self.schedule = mx.lr_scheduler.FactorScheduler(step=1000, factor=0.9)
        self.schedule.base_lr = 3e-5
        self.eps = 5e-9
        self.extract_sentence = extract_sentence
        self.trainer = mx.gluon.Trainer(self.dense_layer.collect_params(), 'adam',
                           {'learning_rate': self.schedule.base_lr, 'epsilon': self.eps}, update_on_kvstore=False)
        self.params = [p for p in self.dense_layer.collect_params().values() if p.grad_req != 'null']

        # loss function
        self.loss_function = self.get_loss()
        self.loss_function.hybridize(static_alloc=True)

    def get_loss(self):
        if self.num_classes == 1:
            return mx.gluon.loss.L2Loss()
        elif self.num_classes == 2:
            return mx.gluon.loss.SoftmaxCELoss()

    def parse_sentences(self, all_features, example_ids, out, token_types, bert_out):
        output = mx.nd.split(out, axis=2, num_outputs=2)
        example_ids = example_ids.asnumpy().tolist()
        pred_start = output[0].reshape((0, -3)).asnumpy()
        pred_end = output[1].reshape((0, -3)).asnumpy()
        verifier_input_shape = (bert_out.shape[0], bert_out.shape[1] + self.max_answer_length, bert_out.shape[2])
        verifier_input = mx.nd.zeros(verifier_input_shape, ctx=self.ctx)
        labels = mx.nd.array([[0 if all_features[eid][0].is_impossible else 1] \
                                        for eid in example_ids]).as_in_context(self.ctx)
        labels_pred = mx.nd.zeros(labels.shape, ctx=self.ctx)
        for idx, data in enumerate(zip(example_ids, pred_start, pred_end, token_types)):
            example_id, start, end, token = data
            results = [PredResult(start=start, end=end)]
            features = all_features[example_id]
            prediction = predict_span(
                features=features,
                results=results,
                max_answer_length=self.max_answer_length,
                n_best_size=self.n_best_size,
                version_2=self.version_2)
            num_total_tokens = len(features[0].tokens)
            num_query_tokens = int((1 - token).sum().max().asscalar()) - 2
            num_contx_tokens = num_total_tokens - num_query_tokens - 3
            num_answr_tokens = prediction[1] - prediction[0] + 1

            if self.extract_sentence:
                # the sentence
                if num_answr_tokens == 0:
                    sentence_idx = (num_query_tokens + 2, num_contx_tokens + num_query_tokens + 2)
                    num_sentc_tokens = num_contx_tokens
                else:
                    sentence_begin = num_query_tokens + 2
                    sentence_end = num_contx_tokens + num_query_tokens + 2
                    sequence_tokens = features[0].tokens
                    sentence_ends_included = { i \
                                                for i in range(len(sequence_tokens)) \
                                                if sequence_tokens[i].find('.') != -1 or sequence_tokens[i].find('?') != -1 or sequence_tokens[i].find('!') != -1}
                    sentence_ends_included.add(num_total_tokens - 2) # the ending
                    sentence_begins_included = {i + 1 for i in sentence_ends_included}
                    if num_total_tokens - 1 in sentence_begins_included:
                        sentence_begins_included.remove(num_total_tokens - 1)
                    if num_query_tokens + 1 in sentence_begins_included:
                        sentence_begins_included.remove(num_query_tokens + 1)
                    sentence_begins_included.add(1)
                    sentence_begins_included.add(num_query_tokens + 2)
                    begin_idxs = sorted(list(sentence_begins_included))
                    end_idxs = sorted(list(sentence_ends_included))
                    for i in range(len(begin_idxs) - 1):
                        if begin_idxs[i] <= prediction[0] and begin_idxs[i+1] > prediction[0]:
                            sentence_begin = begin_idxs[i]
                            break 
                    for i in range(len(end_idxs) - 1):
                        if end_idxs[i] < prediction[1] and end_idxs[i+1] >= prediction[1]:
                            sentence_end = end_idxs[i+1]
                            break
                    sentence_idx = (sentence_begin, sentence_end)
                    num_sentc_tokens = sentence_end - sentence_begin + 1
                # the beginning
                verifier_input[idx, 0, :] = bert_out[idx, 0, :]
                # the sentence embedding
                verifier_input[idx, 1:num_sentc_tokens+1, :] = bert_out[idx, sentence_idx[0]:sentence_idx[1]+1, :]
                # the query embedding
                verifier_input[idx, num_sentc_tokens+1: num_query_tokens+num_sentc_tokens+1, :] \
                                    = bert_out[idx, 1:num_query_tokens+1, :]
                # the separater
                verifier_input[idx, num_query_tokens+num_sentc_tokens+1, :] = bert_out[idx, num_query_tokens+1, :]
                # the answer
                if num_answr_tokens > 0:
                    verifier_input[idx, num_query_tokens+num_sentc_tokens+2:num_answr_tokens+num_query_tokens+num_sentc_tokens+2, :] \
                                    = bert_out[idx, prediction[0]:prediction[1]+1,:]
                # the ending
                verifier_input[idx, num_answr_tokens+num_query_tokens+num_sentc_tokens+2, :] \
                                    = bert_out[idx, num_query_tokens + num_contx_tokens+2, :]
            else:
                # the beginning
                verifier_input[idx, 0, :] = bert_out[idx, 0, :]
                # the context embedding
                verifier_input[idx, 1:num_contx_tokens+1, :] = bert_out[idx, num_query_tokens + 2: num_contx_tokens + num_query_tokens + 2, :]
                # the query embedding
                verifier_input[idx, num_contx_tokens+1: num_query_tokens+num_contx_tokens+1, :] \
                                    = bert_out[idx, 1:num_query_tokens+1, :]
                # the separater
                verifier_input[idx, num_query_tokens+num_contx_tokens+1, :] = bert_out[idx, num_query_tokens+1, :]
                # the answer
                if num_answr_tokens > 0:
                    verifier_input[idx, num_query_tokens+num_contx_tokens+2:num_answr_tokens+num_query_tokens+num_contx_tokens+2, :] \
                                    = bert_out[idx, prediction[0]:prediction[1]+1,:]
                # the ending
                verifier_input[idx, num_answr_tokens+num_query_tokens+num_contx_tokens+2, :] \
                                    = bert_out[idx, num_query_tokens + num_contx_tokens+2, :]
                # the predicted answerability
        return verifier_input, labels

    def train(self, train_features, example_ids, out, token_types=None, bert_out=None, num_epochs=1, verbose=False):
        if not self.version_2:
            return
        data = self.parse_sentences(train_features, example_ids, out, token_types, bert_out)
        verifier_input, labels = data
        for epoch_id in range(num_epochs):
            with mx.autograd.record():
                verify_out = self.dense_layer(verifier_input)
                ls = self.loss_function(verify_out, labels).mean()
            ls.backward()
            # Gradient clipping
            self.trainer.allreduce_grads()
            nlp.utils.clip_grad_global_norm(self.params, 1)
            self.trainer.update(1)
            self.trainer.set_learning_rate(self.schedule(self.step_cnt))
            self.step_cnt += 1
            if verbose:
                print("epoch {0} in dense-layer verifier ({2}), loss {1}".format(epoch_id, ls.asscalar(), self.mode))
        
    def evaluate(self, dev_features, example_ids, out, token_types, bert_out):
        if not self.version_2:
            return mx.nd.ones(example_ids.shape)
        data = self.parse_sentences(dev_features, example_ids, out, token_types, bert_out)
        verifier_input, _ = data
        verifier_output = self.dense_layer(verifier_input)
        if self.num_classes == 1:
            pred = verifier_output.reshape(-1)
        elif self.num_classes == 2:
            pred = mx.ndarray.argmax(verifier_output, axis=1)
        return pred


class AnswerVerify(object):
    def __init__(self, tokenizer=nlp.data.BERTBasicTokenizer(lower=True),
                max_answer_length=30,
                null_score_diff_threshold=-2.0,
                n_best_size=20,
                max_len=384,
                version_2=True,
                extract_sentence=True,
                ctx=mx.cpu()):
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
        self.max_len = min(max_len, 256) # TODO: try to increase this size

        self.lr = 5e-6
        self.eps = 1e-9
        self.batch_size = 3

        self.extract_sentence = extract_sentence

        self.get_model(ctx)
        self.get_loss()

        self.metric = mx.metric.Accuracy()

        self.get_data_transform()

    def get_model(self, ctx):
        bert_base, self.vocabulary = nlp.model.get_model('bert_12_768_12',
                                             dataset_name='book_corpus_wiki_en_uncased',
                                             pretrained=True, ctx=ctx, use_pooler=True,
                                             use_decoder=False, use_classifier=False)
        self.ctx = ctx
        self.bert_classifier = model.classification.BERTClassifier(bert_base, num_classes=2, dropout=0.0)
        self.bert_classifier.classifier.initialize(init=mx.init.Normal(0.02), ctx=self.ctx)
        self.bert_classifier.hybridize(static_alloc=True)
        self.trainer = mx.gluon.Trainer(self.bert_classifier.collect_params(), 'adam',
                           {'learning_rate': self.lr, 'epsilon': self.eps}, update_on_kvstore=False)
        # The gradients for these params are clipped later
        self.params = [p for p in self.bert_classifier.collect_params().values() if p.grad_req != 'null']

    def get_loss(self):
        self.loss_function = mx.gluon.loss.SoftmaxCELoss()
        self.loss_function.hybridize(static_alloc=True)

    def get_data_transform(self):
        bert_tokenizer = nlp.data.BERTTokenizer(self.vocabulary, lower=True)
        self.transform = data.transform.BERTDatasetTransform(bert_tokenizer, self.max_len,
                                                        class_labels=self.all_labels,
                                                        has_label=True,
                                                        pad=True,
                                                        pair=self.pair)

    def train(self, train_features, example_ids, out, token_types=None, bert_out=None, num_epochs=1, verbose=False):
        if not self.version_2:
            return
        dataset_raw = self.parse_sentences(train_features, example_ids, out)
        dataset = dataset_raw.transform(self.transform)

        # The FixedBucketSampler and the DataLoader for making the mini-batches
        train_sampler = nlp.data.FixedBucketSampler(lengths=[int(item[1]) for item in dataset],
                                                    batch_size=self.batch_size,
                                                    num_buckets=1, # number of buckets (mini-batches), by default 10;
                                                    shuffle=True)
        dataloader = mx.gluon.data.DataLoader(dataset, batch_sampler=train_sampler)

        for epoch_id in range(num_epochs):
            if verbose:
                self.metric.reset()
                step_loss = 0
            for batch_id, data in enumerate(dataloader):
                token_ids, valid_length, segment_ids, label = data
                with mx.autograd.record():

                    # Load the data to the GPU (or CPU if GPU disabled)
                    token_ids = token_ids.as_in_context(self.ctx)
                    valid_length = valid_length.as_in_context(self.ctx)
                    segment_ids = segment_ids.as_in_context(self.ctx)
                    label = label.as_in_context(self.ctx)

                    # Forward computation
                    out = self.bert_classifier(token_ids, segment_ids, valid_length.astype('float32'))
                    ls = self.loss_function(out, label).mean()

                # And backwards computation
                ls.backward()
                # Gradient clipping
                self.trainer.allreduce_grads()
                nlp.utils.clip_grad_global_norm(self.params, 1)
                self.trainer.update(1)

                if verbose:
                    # update the loss and metric
                    step_loss += ls.asscalar()
                    self.metric.update([label], [out])
            if verbose:
                print('[Epoch {}] loss={:.4f}, lr={:.7f}, acc={:.3f}'
                         .format(epoch_id,
                                 step_loss / len(dataloader),
                                 self.trainer.learning_rate,  # TODO: add learning rate scheduler latter
                                 self.metric.get()[1]))
            step_loss = 0           
        # exit(0)

    def evaluate(self, dev_feature, prediction):
        # asserted that prediction is not null
        if not self.version_2:
            # return True
            return 1.0
        raw_data = []
        for feature in dev_feature:
            context_text = ' '.join(feature.doc_tokens)
            question_text = feature.question_text
            label = 0 if feature.is_impossible else 1
            sentences = re.split(pattern, context_text)
            sentence_text = self.find_sentence(sentences, prediction)
            first_part = sentence_text + '. ' + question_text
            second_part = prediction
            raw_data.append([first_part, second_part, label])
        dataset_raw = VerifierDataset(raw_data)
        dataset = dataset_raw.transform(self.transform)
        train_sampler = nlp.data.FixedBucketSampler(lengths=[int(item[1]) for item in dataset],
                                                    batch_size=1,
                                                    num_buckets=1,
                                                    shuffle=True)
        dataloader = mx.gluon.data.DataLoader(dataset, batch_sampler=train_sampler)
        for data in dataloader:
            token_ids, valid_length, segment_ids, label = data
            out = self.bert_classifier(token_ids.as_in_context(self.ctx), segment_ids.as_in_context(self.ctx),
                        valid_length.astype('float32').as_in_context(self.ctx))
            # result = out.asnumpy().reshape(-1).tolist()
            # pred = mx.ndarray.argmax(out, axis=1).astype(int)[0]
            pred = mx.ndarray.argmax(out, axis=1)[0]
            # print(out, pred, label)
        # exit(0)

        # eval_result = pred == 1 # True
        eval_result = pred
        return eval_result

    def parse_sentences(self, train_features, example_ids, out, token_types=None):
        output = mx.nd.split(out, axis=2, num_outputs=2)
        example_ids = example_ids.asnumpy().tolist()
        pred_start = output[0].reshape((0, -3)).asnumpy()
        pred_end = output[1].reshape((0, -3)).asnumpy()
        raw_data = []
        for example_id, start, end in zip(example_ids, pred_start, pred_end):
            results = [PredResult(start=start, end=end)]
            features = train_features[example_id]
            label = 0 if features[0].is_impossible else 1
            context_text = ' '.join(features[0].doc_tokens)
            question_text = features[0].question_text
            answer_text = features[0].orig_answer_text
            prediction, _, _ = predict( # TODO: use this more wisely, for example, GAN
                    features=features,
                    results=results,
                    tokenizer=self.tokenizer,
                    max_answer_length=self.max_answer_length,
                    # null_score_diff_threshold=self.null_score_diff_threshold,
                    n_best_size=self.n_best_size,
                    version_2=self.version_2)
            if len(prediction) == 0:
                continue # not validating for n/a output
            if self.extract_sentence:
                sentences =  list(filter(lambda x: len(x.strip())>0, re.split(pattern, context_text) ))
                sentence_text = self.find_sentence(sentences, prediction)
                raw_data.append([sentence_text + '. ' + question_text, prediction, label])
                '''
                if label == 1:
                    answer_sentence = self.find_sentence(sentences, answer_text)
                    raw_data.append([answer_sentence + ' ' + question_text, answer_text, label])
                '''
            else:
                first_part = context_text + '. ' + question_text
                raw_data.append([first_part, prediction, label])
                '''
                if label == 1:
                    raw_data.append([first_part, answer_text, label])
                '''
        dataset = VerifierDataset(raw_data)
        return dataset

    def find_sentence(self, sentences, target):
        if len(target) == 0:
            return random.choice(sentences)
        sentence_text = ''
        for s in sentences:
            if s.find(target) != -1:
                sentence_text = s
                break
        if len(sentence_text) == 0:
            sentence_text = random.choice(sentences)
        return sentence_text

