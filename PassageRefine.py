import json
import os
import random
import time
import pickle
from tqdm import tqdm
import re
import math
from collections import defaultdict

from pytz import timezone
from datetime import datetime

from openai import OpenAI
import textgrad as tg

from utils.args import parse_args



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



def passages_post_processing(error_items:list, text: str):
    processed = []
    for p in text.strip().split('\n'):
        match = re.match(r"Passage \d+\.\s*(.*)", p.strip())
        if match:
            processed.append(match.group(1))
        else:  # target item 다시 돌릴 수 있도록 함
            print(f"\n[Format Error] target_item: {target_item}")
            print("→ Full text:")
            print(text)
            error_items.append(target_item)
            return None
    return processed


prompt = """[Problem Statement]
Your task is to refine a list of passages that clearly describe the key features of a given target item.
Your passages will be used to help a conversational recommender system provide accurate recommendations.

I will give you a few dialog samples in which the target item will be recommended by the assistant.  
Use these dialogs to infer which key features should be included in the passage list.
If none of the given passages adequately describe the key features, you may add a new passage to the list.

[Requirements for passages]
- Start each passage with "Passage N." (e.g., "Passage 1.", "Passage 2.", etc.).
- Immediately after the passage number, begin with the item title (e.g., "Passage 1. Inception (2010) director Christopher Nolan.").
- Each passage should concisely describe a single, distinguishing feature of the item, such as its genre, director, plot, themes, style, tone, or critical reception.
- Avoid overly specific or trivial details that are not helpful for recommendation (e.g., "The protagonist wears a red scarf in scene 4").
- List each passage on a separate line, without bullet points or additional formatting.

[Reasoning process]
The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively, i.e., <think> reasoning process here</think> <answer>answer here</answer>


### Input:
Multiple user-system dialogs, the target item, the passages.

[Dialogs]  
{Dialogs}

[Target item]  
{target_item}

[Passages]
{passages}

### Output:"""


print()

if __name__ == "__main__":

    args = parse_args()

    # OPENAI api key setting
    OPENAI_API_KEY = args.gpt_api
    client = OpenAI(api_key=OPENAI_API_KEY)
    MODEL = args.gpt_model

    # Train dataset load & represent as a dictionary
    train_dataset = pickle.load(open('dataset/train_dataset_inspired2.pkl', 'rb'))
    dict_dialogs = defaultdict(list)
    for sample in train_dataset:
        dict_dialogs[sample['topic']].append('\n'.join(sample['dialog'].split('\n')[-5:]))

    # PassageDB load
    db = json.load(open(args.db_path, 'r', encoding='utf-8'))
    dict_db = defaultdict(list)
    for p in db.values():
        if isinstance(p, list):
            dict_db = db
            break
        else:
            dict_db[extract_title_with_year(p)].append(p)

    # Save refined corpus
    saved_time = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d-%H%M%S'))
    save_path = os.path.join('output_refine', f"{saved_time}_{args.output_file}.jsonl")
    processing_dict = defaultdict(dict)
    refined_dict = defaultdict(list)

    error_items = []
    items = dict_dialogs.keys()

    # Refine
    for epoch in range(args.epochs):
        print(f"\n===== EPOCH {epoch + 1} =====")
        pbar = tqdm(total=len(dict_dialogs.keys()), desc=f"Refine corpus (epoch {epoch + 1})", unit="dialog")
        for target_item in items:
            dialogs = dict_dialogs[target_item]
            random.shuffle(dialogs)
            dialogs = dialogs[:args.batch_size * 2]

            # dict_db[target_item] = '\n'.join([f"Passage {idx+1}. {passage}" for idx, passage in enumerate(dict_db[target_item])])
            num_batch = math.ceil(len(dialogs)/args.batch_size)
            for i in range(num_batch):
                task_prompt = prompt

                # dialog 처리
                batch_dialogs = dialogs[i* args.batch_size:(i+1) * args.batch_size]
                dialog_str = '\n\n'.join([f"Dialog {d_idx+1}\n{dialog}" for d_idx, dialog in enumerate(batch_dialogs)])

                # passages 처리
                if target_item in dict_db:
                    passage_list = dict_db[target_item]
                else:
                    # 없을 때 처리
                    print(f"[Warning] {target_item} not found in dict_db")
                    break
                passages_str = '\n'.join([f"Passage {idx+1}. {passage}" for idx, passage in enumerate(passage_list)])

                # Prompt
                task_prompt = task_prompt.format(Dialogs=dialog_str, target_item=target_item, passages=passages_str)


                # Output
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": task_prompt}
                    ],
                )
                response = response.choices[0].message.content

                # 파싱
                # match_answer = re.search(r'<answer>\s*([\s\S]+?)\s*</answer>', response)
                match_think = response.split('<answer>')[0].strip()
                match_answer = response.split('<answer>')[1].strip()
                if '</answer>' in match_answer:
                    passages = match_answer.split('</answer>')[0].strip()
                else:
                    passages = match_answer

                # match_think = re.search(r'<think>\s*([\s\S]+?)\s*</think>', response)
                if match_answer:
                    passage_list = [line.strip() for line in passages.split('\n') if line.strip()]
                    passage_list = [re.sub(r'^Passage \d+\.\s*', '', passage) for passage in passage_list]
                    dict_db[target_item] = passage_list

                    refined_dict[target_item] = passage_list
                    processing_data = {"Dialogs": dialog_str, "initial_passages": passages_str, "Think": match_think, "Answer": passage_list}
                else:
                    print(f"[Format error] {target_item} - (No <answer> tag found.)")
                    print(response)
                    error_items.append(target_item)
                    break

                # log 저장하기
                save_path_processing = os.path.join('log_w_reasoning', f"{saved_time}_E{epoch + 1}.jsonl")
                with open(save_path_processing, "a", encoding="utf-8") as fw:
                    fw.write(json.dumps(processing_data, ensure_ascii=False)+ '\n')

            pbar.update(1)

            save_path_result = os.path.join('output_result', f"{saved_time}_E{epoch + 1}.jsonl")
            with open(save_path_result, "w", encoding="utf-8") as fw:
                json.dump(refined_dict, fw, ensure_ascii=False, indent=2)

        pbar.close()

            # new_passage_list = response.split()
            # new_passage_list = passages_post_processing(error_items, passages.value)
            #         if new_passage_list is None:
            #             print(f"Skipping target_item due to formatting error: {target_item}")
            #             break
            #         else:
            #             dict_db[target_item] = new_passage_list
            #
            #         # dict_db[target_item] = passages.value
            #         # print(passages.value)
            #         pbar.update(len(batch_dialogs))
            #
            #     # passage DB format 바꾸기
            #     if isinstance(dict_db[target_item], list):
            #         for passage in dict_db[target_item]:
            #             all_passages[str(next_index)] = passage
            #             next_index += 1
            #
            #     # 저장하기
            #     with open(save_path, "w", encoding="utf-8") as fw:
            #         json.dump(all_passages, fw, ensure_ascii=False, indent=2)
            #
            # pbar.close()


# # llm engine setting
# llm_engine = tg.get_engine(args.gpt_model)
# tg.set_backward_engine(llm_engine)
#
# pbar = tqdm(total=len(train_dataset), desc="Refine corpus", unit="dialog")
#
#
# # Save refined corpus
# saved_time = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))
# save_path = os.path.join('output_refine', f"{saved_time}_{args.output_file}.jsonl")
#
# all_passages = {}
# next_index = max([int(k) for k in all_passages.keys()], default=0) + 1
#
# error_items = []
#
# items = ['Joker (2019)', 'Star Wars (1977)', 'Frozen (2019)'] # dict_train_dataset.keys()
# # Refine
# for target_item in items:
#     dialogs = dict_train_dataset[target_item]
#     random.shuffle(dialogs)
#
#     # dict_db[target_item] = '\n'.join([f"Passage {idx+1}. {passage}" for idx, passage in enumerate(dict_db[target_item])])
#     num_batch = math.ceil(len(dialogs)/args.batch_size)
#     for i in range(num_batch):
#         problem_task = refine_prompt_CRS
#         initial_solution = """{passages}"""
#
#         # dialog 처리
#         batch_dialogs = dialogs[i* args.batch_size:(i+1) * args.batch_size]
#         dialog_str = '\n\n'.join([f"Dialog {d_idx+1}\n{dialog}" for d_idx, dialog in enumerate(batch_dialogs)])
#         problem_text = problem_task.format(dialog=dialog_str, target_item=target_item)
#
#         # passages 처리
#         if target_item in dict_db:
#             passage_list = dict_db[target_item]
#         else:
#             # 없을 때 처리
#             print(f"[Warning] {target_item} not found in dict_db")
#             break
#
#         # passage_list = dict_db[target_item]
#         passages_str = '\n'.join([f"Passage {idx+1}. {passage}" for idx, passage in enumerate(passage_list)])
#         initial_solution = initial_solution.format(passages=passages_str)
#
#
#
#         # Passages is the variable of interest we want to optimize -- so requires_grad=True
#         passages = tg.Variable(value=initial_solution,
#                                requires_grad=True,
#                                role_description="passage list to optimize")
#
#         # We are not interested in optimizing the problem -- so requires_grad=False
#         problem = tg.Variable(problem_text,
#                               requires_grad=False,
#                               role_description="passage revision problem")
#
#         # Let TGD know to update code!
#         optimizer = tg.TGD(parameters=[passages])
#
#         # The system prompt that will guide the behavior of the loss function.
#         loss_system_prompt = "You are a smart language model that evaluates a passage list. You do not change or add any passages, only evaluate the existing passage list critically and give very concise feedback."
#         loss_system_prompt = tg.Variable(loss_system_prompt, requires_grad=False, role_description="system prompt to the loss function")
#
#         # The instruction that will be the prefix
#         instruction_revision = evaluation_instruction
#
#         # The format string and setting up the call
#         format_string = "{instruction}\nProblem: {{problem}}\nCurrent Passage list: {{passages}}"
#         format_string = format_string.format(instruction=instruction_revision)
#         # print(format_string)
#
#         fields = {"problem": None, "passages": None}
#         formatted_llm_call = tg.autograd.FormattedLLMCall(engine=llm_engine,
#                                                           format_string=format_string,
#                                                           fields=fields,
#                                                           system_prompt=loss_system_prompt)
#
#
#         # Let's do the forward pass for the loss function.
#         loss = loss_fn(problem, passages)
#         # print(loss.value)
#
#         # Let's look at the gradients!
#         loss.backward()
#         # print(passages.gradients)
#
#         # Let's update the code
#         optimizer.step()  # optimize 대상 업데이트
#
#
#         new_passage_list = passages_post_processing(error_items, passages.value)
#         if new_passage_list is None:
#             print(f"Skipping target_item due to formatting error: {target_item}")
#             break
#         else:
#             dict_db[target_item] = new_passage_list
#
#         # dict_db[target_item] = passages.value
#         # print(passages.value)
#         pbar.update(len(batch_dialogs))
#
#     # passage DB format 바꾸기
#     if isinstance(dict_db[target_item], list):
#         for passage in dict_db[target_item]:
#             all_passages[str(next_index)] = passage
#             next_index += 1
#
#     # 저장하기
#     with open(save_path, "w", encoding="utf-8") as fw:
#         json.dump(all_passages, fw, ensure_ascii=False, indent=2)
#
# pbar.close()
# print("error_items: ", error_items)
