"""changed evaluation script
"""
import argparse
import collections
import json
import numpy as np
import os
import re
import string
import sys

OPTS = None

# python myeval.py dev-v2.0.json XXX/predictions.json

def parse_args():
    parser = argparse.ArgumentParser('Official evaluation script for SQuAD version 2.0.')
    parser.add_argument('data_file', metavar='data.json', help='Input data JSON file.')
    parser.add_argument('pred_file', metavar='pred.json', help='Model predictions.')
    parser.add_argument('--na-prob-thresh', '-t', type=float, default=1.0,
                        help='Predict "" if no-answer probability exceeds this (default = 1.0).')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def make_qid_to_has_ans(dataset):
    qid_to_has_ans = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid_to_has_ans[qa['id']] = bool(qa['answers'])
    return qid_to_has_ans

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_raw_scores(dataset, preds):
    exact_scores = {}
    f1_scores = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid = qa['id']
                gold_answers = [a['text'] for a in qa['answers'] if normalize_answer(a['text'])]
                if not gold_answers:
                    # For unanswerable questions, only correct answer is empty string
                    gold_answers = ['']
                if qid not in preds:
                    print('Missing prediction for %s' % qid)
                    continue
                a_pred = preds[qid]
                # Take max over all gold answers
                exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
                f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
    return exact_scores, f1_scores

def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores

def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = max(len(exact_scores), 1)
        return collections.OrderedDict([
                ('exact', 100.0 * sum(exact_scores.values()) / total),
                ('f1', 100.0 * sum(f1_scores.values()) / total),
                ('total', total),
        ])
    else:
        total = max(len(qid_list), 1)
        return collections.OrderedDict([
                ('exact', 100.0 * sum(exact_scores[k] for k in qid_list if k in exact_scores.keys()) / total),
                ('f1', 100.0 * sum(f1_scores[k] for k in qid_list if k in f1_scores.keys()) / total),
                ('total', total),
        ])

def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval['%s_%s' % (prefix, k)] = new_eval[k]

def main():
    with open(OPTS.data_file) as f:
        dataset_json = json.load(f)
        dataset = dataset_json['data']
    with open(OPTS.pred_file) as f:
        preds = json.load(f)
    na_probs = {k: 0.0 for k in preds}
    qid_to_has_ans = make_qid_to_has_ans(dataset)    # maps qid to True/False
    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
    exact_raw, f1_raw = get_raw_scores(dataset, preds)
    exact_thresh = apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans, OPTS.na_prob_thresh)
    f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans, OPTS.na_prob_thresh)
    out_eval = make_eval_dict(exact_thresh, f1_thresh)
    if has_ans_qids:
        has_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=has_ans_qids)
        merge_eval(out_eval, has_ans_eval, 'HasAns')
    if no_ans_qids:
        no_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=no_ans_qids)
        merge_eval(out_eval, no_ans_eval, 'NoAns')
    print(json.dumps(out_eval, indent=2))

if __name__ == '__main__':
    OPTS = parse_args()
    main()

