import mxnet as mx
import gluonnlp as nlp

from bert_qa_evaluate import PredResult, predict

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

class AnswerVerify3(object):
    '''
    add additional pooling layer on top of traditional bert output
    regression version
    '''
    def __init__(self, version_2=True, ctx=mx.cpu(), dropout=0.0, in_units=768, prefix=None, params=None):
        self.regression = verifier_layers(dropout=dropout, 
                                        num_classes=1, 
                                        in_units=in_units, 
                                        prefix=prefix, 
                                        params=params)
        self.version_2 = version_2
        self.ctx = ctx

        self.lr = 3e-5
        self.eps = 5e-9

        self.regression.collect_params().initialize(init=mx.init.Normal(0.02), ctx=self.ctx)

        self.trainer = mx.gluon.Trainer(self.regression.collect_params(), 'adam',
                           {'learning_rate': self.lr, 'epsilon': self.eps}, update_on_kvstore=False)
        self.params = [p for p in self.regression.collect_params().values() if p.grad_req != 'null']
        self.loss_function = mx.gluon.loss.L2Loss()
        self.loss_function.hybridize(static_alloc=True)

    def train(self, train_features, example_ids, out, num_epochs=1, verbose=False):
        if not self.version_2:
            return
        example_ids = example_ids.asnumpy().tolist()
        labels = mx.nd.array([[0 if train_features[eid][0].is_impossible else 1] for eid in example_ids]).as_in_context(self.ctx)
        out = out.as_in_context(self.ctx)
        for epoch_id in range(num_epochs):
            with mx.autograd.record():
                reg_out = self.regression(out)
                ls = self.loss_function(reg_out, labels).mean()
            ls.backward()
            # Gradient clipping
            self.trainer.allreduce_grads()
            nlp.utils.clip_grad_global_norm(self.params, 1)
            self.trainer.update(1)

            if verbose:
                print("epoch {0} in verifier3, loss {1}".format(epoch_id, ls.asscalar()))

    def evaluate(self, dev_features, example_ids, out):
        if not self.version_2:
            return mx.nd.ones(example_ids.shape)
        example_ids = example_ids.asnumpy().tolist()
        labels = mx.nd.array([[0 if dev_features[eid][0].is_impossible else 1] for eid in example_ids]).as_in_context(self.ctx)
        reg_out = self.regression(out).reshape(-1)
        return reg_out


class AnswerVerify2(object):
    '''
    add additional pooling layer on top of traditional bert output
    '''
    def __init__(self, version_2=True, ctx=mx.cpu(), dropout=0.0, num_classes=2, in_units=768, prefix=None, params=None):
        self.classifier = verifier_layers(dropout=dropout, 
                                        num_classes=num_classes, 
                                        in_units=in_units, 
                                        prefix=prefix, 
                                        params=params)
        self.version_2 = version_2
        self.ctx = ctx

        self.lr = 3e-5
        self.eps = 5e-9

        self.classifier.collect_params().initialize(init=mx.init.Normal(0.02), ctx=self.ctx)

        self.trainer = mx.gluon.Trainer(self.classifier.collect_params(), 'adam',
                           {'learning_rate': self.lr, 'epsilon': self.eps}, update_on_kvstore=False)
        self.params = [p for p in self.classifier.collect_params().values() if p.grad_req != 'null']
        self.loss_function = mx.gluon.loss.SoftmaxCELoss()
        self.loss_function.hybridize(static_alloc=True)

    def train(self, train_features, example_ids, out, num_epochs=1, verbose=False):
        if not self.version_2:
            return
        example_ids = example_ids.asnumpy().tolist()
        labels = mx.nd.array([[0 if train_features[eid][0].is_impossible else 1] for eid in example_ids]).as_in_context(self.ctx)
        out = out.as_in_context(self.ctx)
        for epoch_id in range(num_epochs):
            with mx.autograd.record():
                class_out = self.classifier(out)
                ls = self.loss_function(class_out, labels).mean()
            ls.backward()
            # Gradient clipping
            self.trainer.allreduce_grads()
            nlp.utils.clip_grad_global_norm(self.params, 1)
            self.trainer.update(1)

            if verbose:
                print("epoch {0} in verifier2, loss {1}".format(epoch_id, ls.asscalar()))

    def evaluate(self, dev_features, example_ids, out):
        if not self.version_2:
            return mx.nd.ones(example_ids.shape)
        example_ids = example_ids.asnumpy().tolist()
        labels = mx.nd.array([[0 if dev_features[eid][0].is_impossible else 1] for eid in example_ids]).as_in_context(self.ctx)
        class_out = self.classifier(out)
        pred = mx.ndarray.argmax(class_out, axis=1)
        return pred


class AnswerVerify(object):
    def __init__(self, tokenizer=nlp.data.BERTBasicTokenizer(lower=True),
                max_answer_length=30,
                null_score_diff_threshold=-2.0,
                n_best_size=20,
                max_len=384,
                version_2=True,
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
        self.batch_size = 2

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

    def train(self, train_features, example_ids, out, num_epochs=1, verbose=False):
        if not self.version_2:
            return
        dataset_raw = self.parse_sentences(train_features, example_ids, out)
        # print(len(dataset_raw))
        dataset = dataset_raw.transform(self.transform)

        # The FixedBucketSampler and the DataLoader for making the mini-batches
        train_sampler = nlp.data.FixedBucketSampler(lengths=[int(item[1]) for item in dataset],
                                                    batch_size=self.batch_size,
                                                    num_buckets=1, # number of buckets (mini-batches), by default 10; 2 will cause oom
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
            return True
        raw_data = []
        for feature in dev_feature:
            context_text = ' '.join(feature.doc_tokens)
            question_text = feature.question_text
            label = 0 if feature.is_impossible else 1
            sentences = re.split(pattern, context_text)
            sentence_text = ''
            if label == 1:
                for s in sentences:
                    if s.find(prediction) != -1:
                        sentence_text = s
                        break
            # raw_data.append([question_text, prediction, label])
            first_part = sentence_text + ' ' + question_text
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
            pred = mx.ndarray.argmax(out, axis=1).astype(int)[0]
            # print(out, pred, label)
        # exit(0)

        eval_result = pred == 1 # True
        return eval_result

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
            '''
            prediction, _ = predict(
                features=features,
                results=results,
                tokenizer=self.tokenizer,
                max_answer_length=self.max_answer_length,
                null_score_diff_threshold=self.null_score_diff_threshold,
                n_best_size=self.n_best_size,
                version_2=self.version_2)
            '''
            context_text = ' '.join(features[0].doc_tokens)
            sentences = context_text.strip
            question_text = features[0].question_text
            answer_text = features[0].orig_answer_text # TODO: use this more wisely, for example, GAN
            sentences =  list(filter(lambda x: len(x.strip())>0, re.split(pattern, context_text) ))
            if label == 1:
                sentence_text = ''
                for s in sentences:
                    if s.find(answer_text) != -1:
                        sentence_text = s
                        break
            else:
                sentence_text = random.choice(sentences)
                answer_text = random.choice(sentence_text.split())
            # raw_data.append([question_text, prediction, label]) # TODO: might should use whole context if answer not available
            # raw_data.append([question_text, answer_text, label])
            first_part = sentence_text + ' ' + question_text
            second_part = answer_text
            raw_data.append([first_part, second_part, label])
        dataset = VerifierDataset(raw_data)
        return dataset


