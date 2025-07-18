import nltk
from rank_bm25 import BM25Okapi
import json
import pickle
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pytz import timezone
from datetime import datetime
import os


def custom_tokenizer(text):
    # text = " ".join([word for word in text.split() if word.lower() not in stop_words])
    # tokens = [t.strip() for t in text.split()]
    tokens = tokenizer.encode(text)[1:-1]
    return tokens


def make_documents_items(db_path: str):
    db = json.load(open(db_path, 'r', encoding='utf-8'))
    documents = []
    items = list(db.keys())
    for item, features in db.items():
        # features = json.dumps(features, indent=2, ensure_ascii=False)
        documents.append(f"{item} {features}")
    return documents, items

def hit_score(prediction, label):
    ks = [1, 3, 5, 10, 20]
    hits = []
    for k in ks:
        hits.append(int(label in prediction[:k]))
    return hits


def BM25(db_path, data_path, save_file):
    # tokenizer = AutoTokenizer.from_pretrained("GritLM/GritLM-7B")
    # stop_words = set(stopwords.words('english'))

    # Make Documents from DB
    documents, items = make_documents_items(db_path)
    documents = [custom_tokenizer(doc) for doc in documents]
    bm25 = BM25Okapi(documents)

    # 저장 관련 처리
    saved_time = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d-%H%M%S'))
    save_path = os.path.join('bm25_results', f"{saved_time}_{save_file}")

    # hit score
    hits = [0, 0, 0, 0, 0]
    cnt = 0

    dataset = pickle.load(open(data_path, 'rb'))
    for sample in tqdm(dataset):
        cnt +=1
        dialog = sample['dialog']
        tokenized_dialog = custom_tokenizer(dialog)
        doc_scores = bm25.get_scores(tokenized_dialog)
        doc_scores = np.array(doc_scores)
        sorted_rank = doc_scores.argsort()[::-1]
        topk_items = [items[idx] for idx in sorted_rank[:20]]

        # hit score 계산
        hit = hit_score(topk_items, sample['topic'])
        hits = [x+y for x, y in zip(hit, hits)]

        # 저장
        processing_data = {"Dialog": dialog, "label": sample['topic'], "topk_items": topk_items, "hit": hit, "hits": hits}
        with open(save_path, "a", encoding="utf-8") as fw:
            fw.write(json.dumps(processing_data, ensure_ascii=False) + '\n')

    avg_hit_score = [round(x / len(dataset), 5) for x in hits]
    print(avg_hit_score)
    print()


if __name__ == "__main__":
    nltk.download('stopwords')
    tokenizer = AutoTokenizer.from_pretrained('GritLM/GritLM-7B')
    # "GritLM/GritLM-7B"
    # stop_words = set(stopwords.words('english'))

    db_path = 'refine_results/0712-105956_E1_padding.jsonl'
    data_path = 'training/crs_data/inspired2/test_dataset_inspired2.pkl'
    save_file = '0712-105956_E1_padding_bm25_BPE.jsonl'

    BM25(db_path, data_path, save_file)
