from rank_bm25 import BM25Okapi
import pickle
import os
from tqdm import tqdm
import json
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from transformers import AutoModel, AutoTokenizer, BartForConditionalGeneration, GPT2LMHeadModel, GPT2Config
from datetime import datetime
import argparse
import torch
from itertools import chain
import random
import re

HOME = os.path.dirname(os.path.realpath(__file__))
VERSION = 2
DATA_DIR = os.path.join(HOME, 'data', str(VERSION))
BERT_NAME = 'bert-base-uncased'
CACHE_DIR = os.path.join(HOME, "model_cache", BERT_NAME)
stop_words = set(stopwords.words('english'))
word_piece_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir=CACHE_DIR)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def custom_tokenizer(text):
    text = " ".join([word for word in text.split() if word not in stop_words])
    tokens = word_piece_tokenizer.encode(text)[1:-1]
    return tokens


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def readDic(filename, out=None):
    output_idx_str = dict()
    output_idx_int = dict()
    with open(filename, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                k, idx = line.strip().split('\t')
            except:
                print(line)
                k, idx = line.strip().split()
            output_idx_str[k] = int(idx)
            output_idx_int[int(idx)] = k
    if out == 'str':
        return output_idx_str
    elif out == 'idx':
        return output_idx_int
    else:
        return {'str': output_idx_str, 'int': output_idx_int}


def clean_know_texts(know):
    output = []
    output.append(clean_know_text(know[0]))
    output.append(clean_know_text(know[1]))
    output.append(clean_know_text(know[2]))
    return output


def clean_know_text(text):
    output = text.replace('℃', ' degrees Celsius')
    return output


def clean_join_triple(know):
    if isinstance(know, list) and len(know) > 0:
        know = clean_know_texts(know)
        if know[1] == 'Sings':
            return ' '.join([know[0], 'singer', know[2]])
        elif know[1] == 'Stars':
            return ' '.join([know[0], 'star', know[2]])
        elif know[1] == 'Comments':
            return ' '.join([know[0], 'is known', know[2]])
        elif know[1] == 'Intro':
            return ' '.join([know[0], 'is', know[2]])
        elif know[1] == 'Birthday':
            return ' '.join([know[0], know[1], datetime.strptime(know[2].replace(' ', ''), '%Y-%m-%d').strftime('%Y %B %dth')])
        else:
            return ' '.join(know)
    else:
        return ""


def make_prev(args, mode, dialogs):
    cnt = 0
    filtered_corpus = args.train_know_tokens if mode == 'train' else args.all_know_tokens
    bm25 = BM25Okapi(filtered_corpus)

    # print(mode)
    corpus = list(args.all_knowledges)
    dataset_psd = []
    # with open(f'{HOME}/data/2/en_{mode}.txt', 'r', encoding='UTF-8') as f:
    for index in tqdm(range(len(dialogs)), desc=f"{mode.upper()}_Dataset Read", bar_format='{l_bar} | {bar:23} {r_bar}'):
        cnt += 1
        dialog = dialogs[index]
        # if cnt==20: break
        # dialog = json.loads(line)
        dialog['know_candidates'] = []
        conversation, knowledge_seq = dialog['conversation'], dialog['knowledge']
        topic_seq, goal_seq = dialog['goal_topic_list'], dialog['goal_type_list']

        role_seq = ["User", "System"] if dialog['goal_type_list'][0] != 'Greetings' else ["System", "User"]
        for i in range(2, len(conversation)):
            role_seq.append(role_seq[i % 2])

        for i in range(len(conversation)):
            conversation[i] = conversation[i] if conversation[i][0] != '[' else conversation[i][4:]

        uidx = -1
        prev_topic = ''
        for (goal, role, utt, know, topic) in zip(goal_seq, role_seq, conversation, knowledge_seq, topic_seq):
            uidx += 1

            if uidx > 0:
                response = conversation[uidx - 1] + utt
            else:
                response = utt

            if goal == 'Food recommendation': response = ' '.join(conversation[:uidx]) + utt
            response = response.replace('℃', ' degrees Celsius')

            response = word_piece_tokenizer.decode(word_piece_tokenizer.encode(response)[1:-1])
            if prev_topic != topic:
                response = prev_topic + "|" + topic + "|" + response
            else:
                response = topic + "|" + response

            if know:
                # know = clean_know_texts(know)
                know = clean_join_triple(know)
                # know_idx = corpus.index(know)
                # response_knowledge.append((response.lower(), know.lower(), know_idx, goal, topic, prev_topic))

                tokenized_query = custom_tokenizer(response.lower())
                doc_scores = bm25.get_scores(tokenized_query)

                # doc_scores_tensor = torch.Tensor(doc_scores).to('cuda')
                doc_scores = np.array(doc_scores)

                sorted_rank = doc_scores.argsort()[::-1]
                # sorted_rank_tensor = doc_scores_tensor.argsort(descending=False)
                top1000_retrieved = [corpus[idx] for idx in sorted_rank[:1000]]
                # top1000_retrieved = [corpus[idx] for idx in sorted_rank_tensor[:1000]]
                for rank in range(len(top1000_retrieved)):
                    if topic not in top1000_retrieved[rank]:
                        doc_scores[sorted_rank[rank]] = -1
                        # doc_scores[sorted_rank_tensor[rank]] = -1
                re_sorted_rank = doc_scores.argsort()[::-1]
                # re_sorted_rank_tensor = doc_scores_tensor.argsort(descending=True)
                # prob = softmax(doc_scores)

                candidates_positive_triple = [args.all_knowledges[corpus[idx]] for idx in re_sorted_rank[:20]]
                canditates_postivie_probs = [doc_scores[idx] for idx in re_sorted_rank[:20]]

                # candidates_positive_triple = [args.all_knowledges[corpus[idx]] for idx in re_sorted_rank_tensor[:20]]
                # canditates_postivie_probs = [doc_scores[idx] for idx in re_sorted_rank_tensor[:20]]

                know_candidates = []
                for idx, (tokens, prob) in enumerate(zip(candidates_positive_triple, canditates_postivie_probs)):
                    know_candidates.append((tokens, prob))
                dialog["know_candidates"].append(know_candidates)
            else:
                dialog["know_candidates"].append([])
            prev_topic = topic
        dataset_psd.append(dialog)
        # if m: m.append([index, dialog])
    # eval(dataset_psd)
    # if args.save: save(mode, dataset_psd)
    return dataset_psd


def make(args, mode, dialogs, augmendted_dialogs, topic_mode):
    cnt = 0
    filtered_corpus = args.train_know_tokens if mode == 'train' else args.all_know_tokens
    bm25 = BM25Okapi(filtered_corpus)

    # print(mode)
    corpus_full = list(args.all_knowledges)
    dataset_psd = []
    # with open(f'{HOME}/data/2/en_{mode}.txt', 'r', encoding='UTF-8') as f:
    for data in tqdm(augmendted_dialogs):
        dialog = data['dialog']
        goal = data['goal']
        # corpus = [p for p in corpus_full if p not in data['candidate_knowledges_refined_gpt']]
        corpus = corpus_full

        if topic_mode == 'top1':
            topic = data['predicted_topic'][0]
        elif topic_mode == 'top2':
            topic = data['predicted_topic'][1]
        elif topic_mode == 'top3':
            topic = data['predicted_topic'][2]
        else:
            topic = data['topic']

        if 'beverly hills cop (1984)' == topic.lower():
            print()

        pattern = r'\(\d+\)'
        match = re.search(pattern, topic)
        name = topic[:match.start()].strip()

        if 'last_topic' in data:
            last_topic = data['last_topic']
        else:
            last_topic = topic

        if len(dialog.split('\n')) > 1:
            utt = dialog.split('\n')[-1]
        else:
            utt = ''

        # 임시로 꺼놓은거
        response = data['response']

        if goal == 'Food recommendation':
            response = dialog + response
        else:
            response = utt + response

        response = response.replace('℃', ' degrees Celsius')
        response = response.replace('[SEP]', ' ')

        response = word_piece_tokenizer.decode(word_piece_tokenizer.encode(response)[1:-1])

        # if last_topic != topic:
        #     response = last_topic + "|" + topic + "|" + response
        # else:
        response = topic + "|" + response

        tokenized_query = custom_tokenizer(response.lower())
        doc_scores = bm25.get_scores(tokenized_query)

        # doc_scores_tensor = torch.Tensor(doc_scores).to('cuda')
        doc_scores = np.array(doc_scores)

        sorted_rank = doc_scores.argsort()[::-1]
        # sorted_rank_tensor = doc_scores_tensor.argsort(descending=False)
        top1000_retrieved = [corpus[idx] for idx in sorted_rank[:1000]]
        # top1000_retrieved = [corpus[idx] for idx in sorted_rank_tensor[:1000]]
        for rank in range(len(top1000_retrieved)):
            if topic not in top1000_retrieved[rank]:
                doc_scores[sorted_rank[rank]] = doc_scores[sorted_rank[rank]] - 10
            if name not in top1000_retrieved[rank]:
                doc_scores[sorted_rank[rank]] = doc_scores[sorted_rank[rank]] - 20

                # doc_scores[sorted_rank_tensor[rank]] = -1
        re_sorted_rank = doc_scores.argsort()[::-1]
        # re_sorted_rank_tensor = doc_scores_tensor.argsort(descending=True)
        # prob = softmax(doc_scores)

        # candidates_positive_triple = [args.all_knowledges[corpus[idx]] for idx in re_sorted_rank[:20]]
        candidates_positive_triple = [corpus[idx] for idx in re_sorted_rank[:20]]
        # candidates_positive_triple = [p for p in candidates_positive_triple if p not in data['candidate_knowledges_gpt']][:10]
        canditates_postivie_probs = [doc_scores[idx] for idx in re_sorted_rank[:20]]

        # candidates_positive_triple = [args.all_knowledges[corpus[idx]] for idx in re_sorted_rank_tensor[:20]]
        # canditates_postivie_probs = [doc_scores[idx] for idx in re_sorted_rank_tensor[:20]]

        # know_candidates = []
        # for idx, (tokens, prob) in enumerate(zip(candidates_positive_triple, canditates_postivie_probs)):
        #     know_candidates.append((tokens, prob))
        # data["candidate_knowledges2"] = know_candidates

        data[f'refined_candidate_knowledges'] = candidates_positive_triple  # [clean_join_triple(x) for x in candidates_positive_triple]
        # data[f'candidate_knowledges_{topic_mode}'] = [clean_join_triple(x) for x in candidates_positive_triple]
    return augmendted_dialogs


def save(dataset_psd, mode):
    with open(f'{HOME}/data/2/en_{mode}_know_cand_score20.txt', 'a', encoding='utf8') as fw:
        for dialog in dataset_psd:
            fw.write(json.dumps(dialog) + "\n")


def eval(augmendted_dialogs):
    hit1 = len([x for x in augmendted_dialogs if x['target_knowledge'] == x['candidate_knowledges_top1'][0]]) / len(augmendted_dialogs)
    hit2 = len([x for x in augmendted_dialogs if x['target_knowledge'] == x['candidate_knowledges_top1'][1]]) / len(augmendted_dialogs)

    print("HIT:%.3f\t%.3f" % (hit1, hit2))


def default_parser(parser):
    # Default For All
    parser.add_argument("--version", default='2', type=str, help="Choose the task")
    parser.add_argument('--bert_name', default='bert-base-uncased', type=str, help="BERT Model Name")
    parser.add_argument('--mode', default='train_test', type=str, help="Train/dev/test")
    parser.add_argument('--topic', default='top-1', type=str, help="Train/dev/test")

    parser.add_argument('--home', default=os.path.dirname(os.path.realpath(__file__)), type=str, help="Home path")
    parser.add_argument("--save", action='store_true', help="Whether to SAVE")
    return parser


if __name__ == "__main__":

    # test_pred_aug_dataset_inspired_new7 = pickle.load(open("test_pred_aug_dataset_inspired_new7.pkl", 'rb'))
    # test_pred_aug_dataset_inspired_new8 = pickle.load(open("test_pred_aug_dataset_inspired_new8.pkl", 'rb'))
    # for i,j in zip(test_pred_aug_dataset_inspired_new7, test_pred_aug_dataset_inspired_new8):
    #     i['candidate_knowledges'] = j['candidate_knowledges']
    #
    # train_pred_aug_dataset_inspired_new7 = pickle.load(open("train_pred_aug_dataset_inspired_new7.pkl", 'rb'))
    # train_pred_aug_dataset_inspired_new8 = pickle.load(open("train_pred_aug_dataset_inspired_new8.pkl", 'rb'))
    # for i,j in zip(train_pred_aug_dataset_inspired_new7, train_pred_aug_dataset_inspired_new8):
    #     i['candidate_knowledges'] = j['candidate_knowledges']

    # print()
    # # for i,j in zip(test_pred_aug_dataset_inspired_new5, test_pred_aug_dataset_inspired_new7):
    # #     # print()
    # #     i['candidate_knowledges'] = j['candidate_knowledges']
    #
    # train_pred_aug_dataset_inspired_new6 = pickle.load(open("train_pred_aug_dataset_inspired_new6.pkl", 'rb'))
    # train_pred_aug_dataset_inspired_new7 = pickle.load(open("train_pred_aug_dataset_inspired_new7.pkl", 'rb'))
    #
    # for i,j in zip(train_pred_aug_dataset_inspired_new6, train_pred_aug_dataset_inspired_new7):
    #     # print()
    #     i['candidate_knowledges'] = j['candidate_knowledges']
    # print()
    # for data in test_pred_aug_dataset_inspired_new:
    #     data['candidate_confidences'] = [1] * len(data['candidate_knowledges'])
    #     data['target_knowledge'] = data['candidate_knowledges'][0]
    #     data['gpt_selection'] = data['gpt_selection'][0]
    #     data['candidate_knowledges_gpt'] = [data['candidate_knowledges'][0]]

    args = default_parser(argparse.ArgumentParser(description="ours_main.py")).parse_args()
    args.home = os.path.dirname(os.path.realpath(__file__))

    all_knowledges, train_knowledges, valid_knowledges, test_knowledges = dict(), dict(), dict(), dict()
    train_dialogs, dev_dialogs, test_dialogs = list(), list(), list()
    # with open(f'{HOME}/data/en_train.txt', 'r', encoding='utf8') as f:
    #     for line in tqdm(f, desc="Dataset Read", bar_format='{l_bar} | {bar:23} {r_bar}'):
    #         dialog = json.loads(line)
    #         train_dialogs.append(dialog)
    #         for know in dialog['knowledge']:
    #             if know:
    #                 train_knowledges[clean_join_triple(know)] = know
    #
    # with open(f'{HOME}/data/en_dev.txt', 'r', encoding='UTF-8') as f:
    #     for line in tqdm(f, desc="Dataset Read", bar_format='{l_bar} | {bar:23} {r_bar}'):
    #         dialog = json.loads(line)
    #         dev_dialogs.append(dialog)
    #         for know in dialog['knowledge']:
    #             if know: valid_knowledges[clean_join_triple(know)] = know
    #
    # with open(f'{HOME}/data/en_test.txt', 'r', encoding='UTF-8') as f:
    #     for line in tqdm(f, desc="Dataset Read", bar_format='{l_bar} | {bar:23} {r_bar}'):
    #         dialog = json.loads(line)
    #         test_dialogs.append(dialog)
    #         for know in dialog['knowledge']:
    #             if know: test_knowledges[clean_join_triple(know)] = know
    # knowledgeDB_inspired
    ##### CH 추가 내용
    # original_DB = json.load(open('training/crs_data/inspired2_new/inspired2_original_mentioned_items_db.jsonl', 'r', encoding='utf8'))
    refined_DB = json.load(open('training/crs_data/inspired2_refine/inspired2_refined_mentioned_items_db.jsonl', 'r', encoding='utf8'))
    knowledgeDB_inspired = []
    for features in refined_DB.values():
        for feature in features:
            if 'N/A' not in feature:
                knowledgeDB_inspired.append(feature)
    ####### knowledge refined로 바꿔주기
    # knowledgeDB_inspired = refined_passages ## CH: refined 가 knowledge가 되도록 수정
    augmendted_train_dialogs = pickle.load(open('training/crs_data/inspired2_refine/inspired2_train_refined_aug_dataset_gpt_labeling.pkl', "rb"))
    augmendted_test_dialogs = pickle.load(open("training/crs_data/inspired2/test_dataset_inspired2.pkl", "rb"))

    args.train_knowledges = list(set(knowledgeDB_inspired))
    args.all_knowledges = args.train_knowledges

    # filtered_corpus_train = []
    filtered_corpus_all = []

    # for sent in tqdm(args.train_knowledges, desc="Train_know_tokenize", bar_format='{l_bar} | {bar:23} {r_bar}'):
    #     filtered_corpus_train.append(custom_tokenizer(sent))
    for sent in tqdm(args.all_knowledges, desc="Test_know_tokenize", bar_format='{l_bar} | {bar:23} {r_bar}'):
        filtered_corpus_all.append(custom_tokenizer(sent))
    args.train_know_tokens, args.all_know_tokens = filtered_corpus_all[:], filtered_corpus_all[:]

    if 'train' in args.mode:
        augmendted_train_dialogs = make(args, 'train', train_dialogs, augmendted_train_dialogs, 'target')
        # augmendted_train_dialogs = make(args, 'train', train_dialogs, augmendted_train_dialogs, 'top2')
        # augmendted_train_dialogs = make(args, 'train', train_dialogs, augmendted_train_dialogs, 'top3')
        pickle.dump(augmendted_train_dialogs, open(
            'training/crs_data/inspired2_refine/train_refined_dataset_inspired2_mentioned_items.pkl', 'wb'))

        # eval(augmendted_train_dialogs)

    ### CH: test는 필요 없어서 꺼둠
    # if 'test' in args.mode:
    #     augmendted_test_dialogs = make(args, 'test', test_dialogs, augmendted_test_dialogs, 'target')
    #     pickle.dump(augmendted_test_dialogs, open(f'inspired2/test_pred_aug_dataset_inspired2_know_new.pkl', 'wb'))
    #     # eval(augmendted_test_dialogs)
