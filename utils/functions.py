import re
import os
import json
from collections import defaultdict


# def extract_title_with_year(text):
#     # 괄호 안에 4자리 숫자(연도) + ) + 공백 패턴을 가장 마지막에서 찾기
#     matches = list(re.finditer(r'\(\d{4}\)\s', text))
#     if matches:
#         end = matches[-1].end()  # 마지막 연도 ') ' 이후 인덱스
#         return text[:end-1]  # 공백 제외, 닫는 괄호는 유지
#     return text  # 연도가 없으면 원문 그대로 반환

def extract_title_with_year(text):
    # 괄호 안에 4자리 숫자(연도) + ) + 공백 패턴을 가장 마지막에서 찾기
    matches = re.search(r'\(\d{4}\)', text)
    if matches:
        end = matches.end()  # 마지막 연도 ') ' 이후 인덱스
        return text[:end].strip()  # 공백 제외, 닫는 괄호는 유지
    return text  # 연도가 없으면 원문 그대로 반환


def dict_passage_to_string(db_path):
    original_db = json.load(open('../training/crs_data/inspired2/inspired2_item-passage_dict_db.jsonl', 'r', encoding='utf-8'))
    db = json.load(open(db_path, 'r', encoding='utf-8'))

    # padding_items = set([original_db.keys()]) - set([db.keys()])

    string_db = defaultdict(list)
    for key in original_db.keys():
        if key not in db:
            item_features = original_db[key]
        else:
            item_features = db[key]

        for f in item_features:
            category = f
            value = item_features[f]
            string_db[key].append(f"{key} | {category} | {value}")

    save_path = f"{db_path.split('.jsonl')[0]}_padding_string.jsonl"
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(string_db, f, ensure_ascii=False, indent=2)

    print("저장 완료: ", save_path)


def make_train_dataset_with_label(labeling_result_path):
    save_path = f"../training/crs_data/inspired2/{labeling_result_path.split('.jsonl', 1)[0].split('/', -1)[-1]}.jsonl"

    labeling_result = []
    with open(labeling_result_path, 'r', encoding='utf-8') as f:
        for line in f:
            labeling_result.append(json.loads(line))

    for sample in labeling_result:
        query_inst = 'Retrieve relevant passages based on user conversation history: '
        item_inst = 'Represent the item feature for retrieval: '

        query = [query_inst, sample['dialog']]
        pos = [item_inst, sample['selected_feature']]

        data = {
            "query": query,
            "pos": [pos],
            "task_id": 1
        }

        with open(save_path, "a", encoding="utf-8") as fw:
            fw.write(json.dumps(data, ensure_ascii=False) + '\n')

    print("저장 완료: ", save_path)


if __name__ == "__main__":

    refined = json.load(open('../refine_results/0722-233911_E1_padding_string.jsonl', 'r', encoding='utf-8'))
    original = json.load(open('../training/crs_data/inspired2/inspired2_item-passage_dict_db.jsonl', 'r', encoding='utf-8'))

    dict_passage_to_string('../refine_results/0722-233911_E1.jsonl')
    print()
