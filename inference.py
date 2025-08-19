from gritlm import GritLM
from peft import PeftModel
from collections import defaultdict
from parser import parse_args
from tqdm import tqdm

import torch.nn.functional as F
import re
import os
import json
import torch
import numpy as np
from gritlm_prompter import Prompter


# git 250723

def extract_title_with_year(text):
    # 괄호 안에 4자리 숫자(연도) + ) + 공백 패턴을 가장 마지막에서 찾기
    matches = list(re.finditer(r'\(\d{4}\)\s', text))
    if matches:
        end = matches[-1].end()  # 마지막 연도 ') ' 이후 인덱스
        return text[:end - 1]  # 공백 제외, 닫는 괄호는 유지
    return text  # 연도가 없으면 원문 그대로 반환


def gritlm_instruction(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"


def recall_score(gt_list, pred_list, ks, verbose=True):
    hits = defaultdict(list)
    for gt, preds in zip(gt_list, pred_list):
        for k in ks:
            hits[k].append(len(list(set(gt).intersection(set(preds[:k])))) / len(gt))
    if verbose:
        for k in ks:
            print("Recall@{}: {:.4f}".format(k, np.mean(hits[k])))
    return hits


def inference(args):
    model_path = os.path.join(args.home, 'model_weights', args.target_model_path)
    data_path = os.path.join(args.home, 'training/crs_data', args.data_json)
    db_path = os.path.join(args.home, 'training/crs_data', args.db_json)
    embeddings_path = os.path.join(args.home, 'training/crs_data', args.embeddings_path)
    saved_time = model_path.split('/')[-2]
    to_json = os.path.join(args.home, 'results', f"{saved_time}_{args.to_json}.jsonl")

    # merged_model = lora_model.merge_and_unload()
    # model.model = lora_model
    # [(i,k) for i,k in lora_model.named_parameters()][0]
    prompter = Prompter(args)
    query_instr, doc_instr = prompter.get_instruction()

    with open(data_path) as fd:
        lines = fd.readlines()
        test_data = [json.loads(line) for line in lines]
        print(len(test_data))

    queries = [prompter.generate_prompt(i) for i in test_data]
    labels = [i['rec'][0] for i in test_data]

    db = json.load(open(db_path, 'r', encoding='utf-8'))
    if isinstance(next(iter(db.values())), list):  # passage인 경우에 list 형태로 되어있음
        documents = []
        for passage_list in db.values():
            documents.extend(passage_list)
        all_names = list(db.keys())
        # documents = [passage for passages in db.values() for passage in passages]
    else:
        documents = list(db.values())
        all_names = [extract_title_with_year(v) for v in db.values()]
    documents = [doc[:prompter.max_char_len * 10] for doc in documents]

    # print( list(db.values())[:10])
    # print(documents[:10])
    # all_names_prev = [extract_title_with_year(v) for v in documents]
    # if len(all_names) != len(all_names_prev):
    #     print("Length of all_names_prev does not match length of all_names")
    # else:
    #     for i in range(len(all_names)):
    #         if all_names_prev[i] != all_names[i]:
    #             print(all_names_prev[i], all_names[i])

    if args.debug_mode:
        documents = documents[:12]
        all_names = all_names[:2]
        args.top_k = 2
    print(len(documents))

    # if isinstance(next(iter(db.values())), list):
    #     full_passages = []
    #     for key in db.keys():
    #         full_passages = [", ".join(db[key]) for key in db.keys()]

    # all_names = [extract_title_with_year(v) for v in db.values()]
    name2id = {all_names[index]: index for index in range(len(all_names))}
    print("name2id:", len(name2id))
    id2name = {v: k for k, v in name2id.items()}

    rec_lists = [[name2id[i]] for i in labels if i in name2id]  # target item id로 바꿔서 저장
    print(f"rec_lists: {len(rec_lists)}")
    rec_lists_prev = [[name2id[i]] for i in labels]  # target item id로 바꿔서 저장
    # print(f"(temp) rec_lists_prev: {len(rec_lists_prev)}")

    # Loads the model for both capabilities; If you only need embedding pass `mode="embedding"` to save memory (no lm head)
    model = GritLM("GritLM/GritLM-7B", mode='embedding', torch_dtype="auto")
    # model = GritLM("GritLM/GritLM-7B", torch_dtype="auto")

    if args.target_model_path != '':
        model.model = PeftModel.from_pretrained(model.model, model_path)
    else:
        print("BASE MODEL")

    # if args.linear:
    #     non_lora_path = os.path.join(model_path, "non_lora_trainables.bin")
    #     non_lora_state_dict = torch.load(non_lora_path)
    #     model.load_state_dict(non_lora_state_dict, strict=False)    
    # else:
    #     print("linear parameter X")

    # "item": [passage list] 형태로 바꿔줌 
    # name2passages = defaultdict(list)
    # for doc in documents:
    #     name = extract_title_with_year(doc)
    #     if 'N/A' not in doc:
    #         name2passages[name].append(doc)

    # passages_for_instruction = []
    # if args.instruction_aug:
    #     for target_p in tqdm(db.values()):
    #         name = extract_title_with_year(target_p)
    #         related_passages = name2passages[name]
    #         temp = [passage for passage in related_passages if passage != target_p]
    #         instruction_passage = ''
    #         for p in temp:
    #             p = p+' '
    #             instruction_passage += p
    #         passages_for_instruction.append(instruction_passage)

    if args.embeddings_path != '':
        print("Embedding save path: ", embeddings_path)

    d_rep = []
    for i, sample in enumerate(tqdm(documents, desc="Encoding documents")):
        batch_documents = documents[i]
        instruction = doc_instr
        d_rep.append(model.encode(batch_documents, instruction=gritlm_instruction(
            instruction)))  # self-attention 적용하려면 encode batch size 아이템에 대한 passage 개수로 설정해야함
        # d_rep.append([i] * 100)  # self-attention 적용하려면 encode batch size 아이템에 대한 passage 개수로 설정해야함
    d_rep = np.stack(d_rep, axis=0)
    print('document shape:', torch.from_numpy(d_rep).shape)

    if args.embeddings_path != '':
        torch.save(documents, embeddings_path)
        print("Embedding 저장 완료")

    # 마스크 생성 (N/A가 포함된 passage에 마스킹)
    masks = torch.tensor([0 if "N/A" in doc else 1 for doc in documents])  # [N]
    print("mask shape: ", masks.shape)
    print("mask sum: ", torch.sum(masks))

    max_rank, mean_rank, category_mean_reank = [], [], []
    num_categories = len(list(db.values())[0])
    print("num_categories: ", num_categories)
    mean_k_rank = [[] for _ in range(num_categories)]

    # top1_mean_rank, top2_mean_rank, top3_mean_rank, top4_mean_rank, top5_mean_rank = [], [], [], [], []
    passages = []
    hard_passages = []

    cosine_sim_value = []

    answer_passages = []
    for i in tqdm(range(0, len(queries), args.batch_size)):
        batch_queries = queries[i: i + args.batch_size]
        q_rep = model.encode(batch_queries, instruction=gritlm_instruction(query_instr))  # [B, d]

        q_rep_tensor = torch.from_numpy(q_rep)
        d_rep_tensor = torch.from_numpy(d_rep)
        mask_tensor = masks.view(len(name2id), -1).unsqueeze(0)  # [1, num_items, # categories]
        mask_tensor = mask_tensor.repeat(len(q_rep), 1, 1)  # [B, I, C]
        # print('queries shape:', torch.from_numpy(q_rep).shape) 

        cos_sim = F.cosine_similarity(q_rep_tensor.unsqueeze(1), d_rep_tensor.unsqueeze(0),
                                      dim=-1)  # [B, 1, d] x [1, N, d] = [B, N]
        cos_sim = torch.where(torch.isnan(cos_sim), torch.full_like(cos_sim, 0), cos_sim)  # [B, N]
        # cos_sim = cos_sim.masked_fill(masks.unsqueeze(0) == 0, float('-inf'))
        top20_passages = torch.topk(cos_sim, k=20, dim=-1).indices  # [B, 20]
        hard_passages += top20_passages.tolist()
        cos_sim = torch.softmax(cos_sim/0.02, dim=-1)

        for b_idx in range(len(batch_queries)):
            cosine_sim_value.append(cos_sim[b_idx].tolist())

        # cos_sim = cos_sim.view(len(q_rep), len(name2id), len(documents) // len(name2id))  # [B, I, P] where N = I x P

        # max pooling
        cos_sim_max = cos_sim.masked_fill(masks.unsqueeze(0) == 0, float('-inf'))

        cos_sim_max = cos_sim_max.view(len(q_rep), len(name2id), len(documents) // len(name2id))
        max_pooled_sim = cos_sim_max.max(dim=-1).values  # [B, I]
        max_pooled_indices = cos_sim_max.max(dim=-1).indices  # [B, I] idx of selected passages by batch
        max_topk_sim_values, max_topk_sim_indices = torch.topk(max_pooled_sim, k=args.top_k, dim=-1)  # [B, K], [B, K]
        gatherd_selected_passage_idx = torch.gather(max_pooled_indices, 1,
                                                    max_topk_sim_indices)  # [B, K] passage idx correspond to top-k item by batch

        max_rank += max_topk_sim_indices.tolist()  # extend
        passages += gatherd_selected_passage_idx.tolist()
        # print('rank length:', len(max_rank))
        # print('passages length:', len(passages))

        # mean pooling
        cos_sim_mean = cos_sim.view(len(q_rep), len(name2id), len(documents) // len(name2id))
        cos_sim_mean = cos_sim_mean * mask_tensor
        sum_sim = cos_sim_mean.sum(dim=-1)  # [B, num_items]
        passage_count = mask_tensor.sum(dim=-1)
        mean_pooled_sim = sum_sim / (passage_count + 1e-10)
        mean_topk_sim_values, mean_topk_sim_indices = torch.topk(mean_pooled_sim, k=args.top_k, dim=-1)

        mean_rank += mean_topk_sim_indices.tolist()

        # top-k mean pooling
        for mean_k in range(1, len(list(db.values())[0]) + 1):
            top1_cos_sim_mean_indices = torch.topk(cos_sim_mean, k=mean_k, dim=2).indices  # [B, I, k]
            top1_cos_sim_mean_value = torch.topk(cos_sim_mean, k=mean_k, dim=-1).values  # [B, I, k]
            top1_cos_sum_mean_mask = torch.gather(mask_tensor, dim=2, index=top1_cos_sim_mean_indices)
            top1_sum_sim = top1_cos_sim_mean_value.sum(dim=-1)  # [B, num_items]
            top1_passage_count = top1_cos_sum_mean_mask.sum(dim=-1)
            top1_mean_pooled_sim = top1_sum_sim / (top1_passage_count + 1e-10)
            top1_mean_topk_sim_values, top1_mean_topk_sim_indices = torch.topk(top1_mean_pooled_sim, k=args.top_k,
                                                                               dim=-1)

            mean_k_rank[mean_k - 1].extend(top1_mean_topk_sim_indices.tolist())


        # category-aware pooling
        if args.category_aware_pooling:
            category_mask = torch.tensor([query['category_mask'] for query in test_data[i: i + args.batch_size]]) # [B, C]
            category_mask = category_mask.unsqueeze(1).expand(-1, len(name2id), -1) # [B, I, C]
            mask_tensor = mask_tensor.mul(category_mask)

            cos_sim_category = cos_sim.view(len(q_rep), len(name2id), len(documents) // len(name2id))
            cos_sim_category = cos_sim_category * mask_tensor
            sum_sim = cos_sim_category.sum(dim=-1)  # [B, num_items]
            passage_count = mask_tensor.sum(dim=-1)
            category_mean_pooled_sim = sum_sim / (passage_count + 1e-10)
            mean_topk_sim_values, mean_topk_sim_indices = torch.topk(category_mean_pooled_sim, k=args.top_k, dim=-1)

            category_mean_reank += mean_topk_sim_indices.tolist()


        print('rank length:', len(mean_rank))

    # Hit@k 성능 확인
    print('model path:', model_path)
    # print('Max pooling')
    # recall_score(rec_lists, max_rank, ks=[1, 3, 5, 10, 20])
    # print()
    #
    # print('Mean pooling')
    # recall_score(rec_lists, mean_rank, ks=[1, 3, 5, 10, 20])
    # print()

    for mean_k in range(1, len(list(db.values())[0]) + 1):
        print(f'Top-{mean_k} Mean pooling')
        recall_score(rec_lists, mean_k_rank[mean_k - 1], ks=[1, 3, 5, 10, 20])
        print()

    
    if args.category_aware_pooling:
        print("Category-aware pooling")
        recall_score(rec_lists, category_mean_reank, ks=[1,3,5,10])
    
    if args.store_results:
        for i in tqdm(range(len(max_rank))):
            # ranked_list = {j: id2name[j] for j in rank[i]}
            max_item_list = [id2name[j] for j in max_rank[i]][:args.top_k]
            mean_item_list = [id2name[j] for j in mean_rank[i]][:args.top_k]

            passage_list = [db[item][j] for item, j in zip(max_item_list, passages[i])]  # K passages
            hard_passage_list = [documents[i] for i in hard_passages[i]]
            cosine_value = cosine_sim_value[i]

            # test_data[i]["cand_list"] = ranked_list
            test_data[i]["max_cand_list"] = max_item_list
            test_data[i]["mean_cand_list"] = mean_item_list
            # test_data[i]["cosine_value"] = cosine_value

            for mean_k in range(1, len(list(db.values())[0]) + 1):
                top3_mean_item_list = [id2name[j] for j in mean_k_rank[mean_k - 1][i]][:args.top_k]
                test_data[i][f"top_{mean_k}mean_cand_list"] = top3_mean_item_list

            if passages:
                test_data[i]['max_passages'] = passage_list
                test_data[i]['hard_passages'] = hard_passage_list

            with open(to_json, "w", encoding="utf-8") as fwr:
                for example in test_data:
                    fwr.write(json.dumps(example))
                    fwr.write("\n")


if __name__ == '__main__':
    args = parse_args()
    inference(args)
