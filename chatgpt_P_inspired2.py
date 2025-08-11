#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
from datetime import datetime

from openai import OpenAI
from pytz import timezone

import openai
import pickle
import time
import random

from tqdm import tqdm
import argparse

from transformers import AutoTokenizer

# from utils.parser import parse_args

# I will provide you with a dialog between a user and a system:
# User: Who is the leading actor of the movie Left Right Love Destiny?
# System: It's Cecilia Cheung
# User: You even know that. Excellent!
# System: Thank you for your praise. She's a Chinese idol. Of course I know her.
# User: Yes my idol is quite unusual.
# System: She is also an Asian Outstanding Artist in New York Chinese Film Festival. Excellent.
# User: Yes. People like her will shine wherever they go.
# System: Then don't miss her movie Failan. It shows a love tragedy in which the two people are secluded in different time spaces. However, the movie doesn't just tell a sad but beautiful love story. Through the story, the audiences can see the director's profound thinking and understanding of life and human nature.
# User: Although I really like her, I prefer to see Nicholas Tse's movies.
# System: Then you can watch Nicholas Tse's movie The Stool Pigeon. His acting is good.
# User: What kind of movie is it?

# In this context, the system provides the following response:
# System: It's an action drama, with a little bit of thriller.

# Could you rate the following three passages of relevance for generating the response to the dialog?
# 1. The Stool Pigeon Type Action plot Thriller
# 2. The Stool Pigeon is known a classic police and gangster movie
# 3. The Stool Pigeon Reputation good reputation

# You should consider the lexical and semantic similarities between the response and each passage.
# Please give me only an answer including the rating (highly relevant, relevant, neutral, irrelevant, highly irrelevant).
# 1. Highly relevant
# 2. Relevant
# 3. Neutral

MODEL = "gpt-4.1"

few_shot_refine = """<Example 1>
Pretend you are a conversational recommender system. Your task is to select the most relevant passage for generating a response to the given dialog.
First, I will provide a dialog between a user and the system, along with item of the response and the response utterance from the system.
Then, I will provide candidate passages describing the item.

Here is the dialog:
System: Hi There! What types of movies do you like to watch?
User: Hello! I'm more of an action movie or a good romance and mystery movie.

Here is the item:
Knives Out (2019)

Here is the response:
System: I just saw the trailer for Knives Out when I went to see Joker and it looked like a good mix of action and mystery!

Here are the passages:
Passage 1. Knives Out (2019) genre Comedy, Crime, Drama, Mystery, Thriller
Passage 2. Knives Out (2019) director Rian Johnson
Passage 3. Knives Out (2019) plot A detective investigates the death of a patriarch in a quirky, dysfunctional family.
Passage 4. Knives Out (2019) actors Ana de Armas, Toni Collette, Chris Evans, Daniel Craig
Passage 5. Crawl (2019) director Alexandre Aja
Passage 6. Gemini Man (2019) director Ang Lee
Passage 7. Ma (2019) genre Horror, Thriller
Passage 8. Booksmart (2019) director Olivia Wilde
Passage 9. Never Grow Old (2019) genre Western
Passage 10. 21 Bridges (2019) director Brian Kirk

Select the relevant passages from the above passages list that is relevant for generating a response to the provided dialog.
You should consider the lexical and semantic similarities between the dialog-response and each passage.
Only answer the relevant passages in order of relevance and do not add explanation. 

### Response:Passage 1. Knives Out (2019) genre Comedy, Crime, Drama, Mystery, Thriller
Passage 3. Knives Out (2019) plot A detective investigates the death of a patriarch in a quirky, dysfunctional family.
Passage 4. Knives Out (2019) actors Ana de Armas, Toni Collette, Chris Evans, Daniel Craig


<Example 2>
Pretend you are a conversational recommender system. Your task is to select the most relevant passage for generating a response to the given dialog.
First, I will provide a dialog between a user and the system, along with item of the response and the response utterance from the system.
Then, I will provide candidate passages describing the item.

Here is the dialog:
System: Hey friend, how are you today?
User: Doing great, thanks? Have you seen any good movies recently?
System: I see good movies all the time in my opinion. What types of movies do you like?
User: I enjoy comedies, thrillers, and action movies usually.
System: Are there any actors you prefer.
User: Yes! I like Ed Norton, Chris Evans, Emma Stone, Brad Pitt, & Margo Robbie a lot.
System: Are you a fan of a realistic movie or something that is definitely not something that would happen in real life?
User: I can go either way in regards to that, although probably prefer a bit of realism involved. Like comic book movies are good, because there based on at least some plausibility.
System: I love comic book movies also. Before I give suggestions, what is one of your favorite movies, and why?
User: One of my favorite movies is Se7en, due to the high level of tension involved throughout and I'll always remember how that final reveal at the end made me feel. Just awe stricken.

Here is the item:
Zodiac (2007)

Here is the response:
System: I may have a movie that you would like. This actually is based on a true story based on the Zodiac killer.

Here are the passages:
Passage 1. Zodiac (2007) plot Based on true events, journalists and detectives investigate the Zodiac Killer, a serial murderer who taunted the public and authorities in San Francisco.
Passage 2. Zodiac (2007) director David Fincher
Passage 3. Zodiac (2007) genre Crime, Drama, Mystery, Thriller
Passage 4. Zodiac (2007) actors Jake Gyllenhaal, Mark Ruffalo, Anthony Edwards, Robert Downey Jr.
Passage 5. In the Tall Grass (2019) actors Laysla De Oliveira, Avery Whitted, Patrick Wilson, Will Buie Jr.
Passage 6. Fantastic Mr. Fox (2009) actors George Clooney, Meryl Streep, Jason Schwartzman, Bill Murray
Passage 7. Houston (2013) plot A German corporate headhunter travels to Houston, Texas, in pursuit of a renowned oil company CEO, only to have his life fall apart.
Passage 8. Iron Man (2008) actors Robert Downey Jr., Gwyneth Paltrow, Jeff Bridges, Terrence Howard
Passage 9. Adventures of Captain Marvel (1941) actors Tom Tyler, Frank Coghlan Jr., William 'Billy' Benedict, Louise Currie
Passage 10. Kiss Kiss Bang Bang (2005) actors Robert Downey Jr., Val Kilmer, Michelle Monaghan, Corbin Bernsen

Select the relevant passages from the above passages list that is relevant for generating a response to the provided dialog.
You should consider the lexical and semantic similarities between the dialog-response and each passage.
Only answer the relevant passages in order of relevance and do not add explanation. 


### Response: Passage 1. Zodiac (2007) plot Based on true events, journalists and detectives investigate the Zodiac Killer, a serial murderer who taunted the public and authorities in San Francisco. 
Passage 3. Zodiac (2007) genre Crime, Drama, Mystery, Thriller


"""

few_shot_original = """<Example 1>
Pretend you are a conversational recommender system. Your task is to select the most relevant passage for generating a response to the given dialog.
First, I will provide a dialog between a user and the system, along with item of the response and the response utterance from the system.
Then, I will provide candidate passages describing the item.

Here is the dialog:
System: Hi There! What types of movies do you like to watch?
User: Hello! I'm more of an action movie or a good romance and mystery movie.

Here is the item:
Knives Out (2019)

Here is the response:
System: I just saw the trailer for Knives Out when I went to see Joker and it looked like a good mix of action and mystery!

Here are the passages:
Passage 1. Knives Out (2019) director Rian Johnson
Passage 2. Knives Out (2019) writer Rian Johnson
Passage 3. Knives Out (2019) genre Comedy, Crime, Drama, Mystery, Thriller
Passage 4. Knives Out (2019) plot A detective investigates the death of a patriarch of an eccentric, combative family.
Passage 5. Knives Out (2019) actors Ana de Armas, Toni Collette, Chris Evans, Daniel Craig
Passage 6. Knives Out (2019) rating Internet Movie Database:7.6/10,Rotten Tomatoes:99%,Metacritic:84/100
Passage 7. Bandolero! (1968) writer James Lee Barrett (screenplay), Stanley Hough (story)
Passage 8. And Now the Screaming Starts! (1973) writer Roger Marshall (screenplay by), David Case (novel)
Passage 9. Endure (2010) actors Devon Sawa, Judd Nelson, Joey Lauren Adams, Clare Kramer
Passage 10. Batman: Mystery of the Batwoman (2003) writer Bob Kane (character created by: Batman), Alan Burnett (story), Michael Reaves

Select the relevant passages from the above passages list that is relevant for generating a response to the provided dialog.
You should consider the lexical and semantic similarities between the dialog-response and each passage.
Only answer the relevant passages in order of relevance and do not add explanation. 

### Response:Passage 3. Knives Out (2019) genre Comedy, Crime, Drama, Mystery, Thriller
Passage 4. Knives Out (2019) plot A detective investigates the death of a patriarch of an eccentric, combative family.
Passage 5. Knives Out (2019) actors Ana de Armas, Toni Collette, Chris Evans, Daniel Craig


<Example 2>
Pretend you are a conversational recommender system. Your task is to select the most relevant passage for generating a response to the given dialog.
First, I will provide a dialog between a user and the system, along with item of the response and the response utterance from the system.
Then, I will provide candidate passages describing the item.

Here is the dialog:
System: Hey friend, how are you today?
User: Doing great, thanks? Have you seen any good movies recently?
System: I see good movies all the time in my opinion. What types of movies do you like?
User: I enjoy comedies, thrillers, and action movies usually.
System: Are there any actors you prefer.
User: Yes! I like Ed Norton, Chris Evans, Emma Stone, Brad Pitt, & Margo Robbie a lot.
System: Are you a fan of a realistic movie or something that is definitely not something that would happen in real life?
User: I can go either way in regards to that, although probably prefer a bit of realism involved. Like comic book movies are good, because there based on at least some plausibility.
System: I love comic book movies also. Before I give suggestions, what is one of your favorite movies, and why?
User: One of my favorite movies is Se7en, due to the high level of tension involved throughout and I'll always remember how that final reveal at the end made me feel. Just awe stricken.

Here is the item:
Zodiac (2007)

Here is the response:
System: I may have a movie that you would like. This actually is based on a true story based on the Zodiac killer.

Here are the passages:
Passage 1. Zodiac (2007) plot Based on true events, journalists and detectives investigate the Zodiac Killer, a serial murderer who taunted the public and authorities in San Francisco.
Passage 2. Zodiac (2007) director David Fincher
Passage 3. Zodiac (2007) genre Crime, Drama, Mystery, Thriller
Passage 4. Zodiac (2007) actors Jake Gyllenhaal, Mark Ruffalo, Anthony Edwards, Robert Downey Jr.
Passage 5. In the Tall Grass (2019) actors Laysla De Oliveira, Avery Whitted, Patrick Wilson, Will Buie Jr.
Passage 6. Fantastic Mr. Fox (2009) actors George Clooney, Meryl Streep, Jason Schwartzman, Bill Murray
Passage 7. Houston (2013) plot A German corporate headhunter travels to Houston, Texas, in pursuit of a renowned oil company CEO, only to have his life fall apart.
Passage 8. Iron Man (2008) actors Robert Downey Jr., Gwyneth Paltrow, Jeff Bridges, Terrence Howard
Passage 9. Adventures of Captain Marvel (1941) actors Tom Tyler, Frank Coghlan Jr., William 'Billy' Benedict, Louise Currie
Passage 10. Kiss Kiss Bang Bang (2005) actors Robert Downey Jr., Val Kilmer, Michelle Monaghan, Corbin Bernsen

Select the relevant passages from the above passages list that is relevant for generating a response to the provided dialog.
You should consider the lexical and semantic similarities between the dialog-response and each passage.
Only answer the relevant passages in order of relevance and do not add explanation. 

### Response:Passage 1. Zodiac (2007) plot Based on true events, journalists and detectives investigate the Zodiac Killer, a serial murderer who taunted the public and authorities in San Francisco.
Passage 3. Zodiac (2007) genre Crime, Drama, Mystery, Thriller


"""

reranking_template = """<Example 3>
Pretend you are a conversational recommender system. Your task is to select the most relevant passage for generating a response to the given dialog.
First, I will provide a dialog between a user and the system, along with item of the response and the response utterance from the system.
Then, I will provide candidate passages describing the item.

Here is the dialog:
%s

Here is the item:
%s

Here is the response:
%s

Here are the passages:
%s

Select the relevant passages from the above passages list that is relevant for generating a response to the provided dialog.
You should consider the lexical and semantic similarities between the dialog-response and each passage.
Only answer the relevant passages in order of relevance and do not add explanation.

### Response:"""


def execute(args,
            instructions: list = None,
            labels: list = None,
            inputs: list = None):
    # openai.api_key = args.access_token
    client = OpenAI(api_key=args.access_token)

    cnt = args.cnt
    for instruction in tqdm(instructions[cnt:], bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):

        # content = template % (instruction)
        # content = mentioned_reviews_concat + refinement_template % instruction
        # print()
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user",
                     # "content": review_summary_template % (label, instruction, label)}
                     "content": instruction}
                ],
                temperature=0,
            )

            response = response.choices[0].message.content

            args.log_file.write(json.dumps({'INPUT': instruction, 'OUTPUT': response}, ensure_ascii=False, indent=4) + '\n')
            cnt += 1

        except Exception as error:
            print("ERROR cnt: %d" % (cnt))
            print(args.log_file)
            args.cnt = int(cnt)
            time.sleep(5)
            break

        # openai.api_requestor._thread_context.session.close()
        if int(cnt) == len(instructions):
            return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--access_token', type=str, default="")
    parser.add_argument('--cnt', type=int, default=0)
    parser.add_argument('--log_name', type=str, default="")
    args = parser.parse_args()

    return args


def chatgpt_test(args,
                 instructions: list = None,
                 labels: list = None,
                 inputs: list = None,
                 ):
    print('CHATGPT_TEST_START')
    while True:
        if execute(args=args, instructions=instructions, labels=labels, inputs=inputs) == False:
            break


if __name__ == "__main__":
    # knowledgeDB_inspired = pickle.load(open("inspired2/knowledgeDB_inspired2.pkl", 'rb'))
    # all_knowledges = set(knowledgeDB_inspired)
    train_pred_aug_dataset_inspired2_know = pickle.load(open("training/crs_data/inspired2_refine/train_refined_aug_dataset_inspired2_remove_key_impressions.pkl", 'rb'))

    # ### 결과 나왔을 때, 기존 pkl 에 덮어써주는 코드 (시작)
    # pattern = r"Passage \d+. "
    # import re
    # chat_results = json.load(open("0809154513_chatgpt_result.json", 'r', encoding='utf-8'))
    # for data, chat in zip(train_pred_aug_dataset_inspired2_know, chat_results):
    #     passages = chat['OUTPUT'].split('\n')
    #     match = re.search(pattern, passages[0])
    #     if match is None:
    #         data['gpt_selection'] = ''
    #         data['candidate_knowledges_gpt'] = []
    #         data['target_knowledge'] = data['candidate_knowledges'][0]
    #     else:
    #         for idx, p in enumerate(passages):
    #             match = re.search(pattern, p)
    #
    #             p = p[match.end():]
    #             passages[idx] = p.strip()
    #
    #         gpt_selection = passages[0]
    #         candidate_knowledges_gpt = passages
    #         data['gpt_selection'] = gpt_selection
    #         data['candidate_knowledges_gpt'] = candidate_knowledges_gpt
    #         data['target_knowledge'] = gpt_selection
    # ### 결과 나왔을 때, 기존 pkl 에 덮어써주는 코드 (끝)

    args = parse_args()
    instructions = []
    labels = []
    # temp = []
    # train_pred_aug_dataset_inspired = pickle.load(open("train_error3.pkl", "rb"))
    #
    for idx, data in enumerate(train_pred_aug_dataset_inspired2_know):
        instruction = reranking_template % (data['dialog'], data['topic'], data['response'], "\n".join(["Passage %d. %s" % (idx + 1, i) for idx, i in enumerate(data['candidate_knowledges_refined_no_impressions'][:10])]))
        instructions.append(few_shot_refine + instruction)

    mdhm = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))
    if args.log_name == '':
        log_name = mdhm + '_' + 'chatgpt_result.json'
    else:
        log_name = args.log_name

    args.log_file = open(os.path.join(log_name), 'a', buffering=1, encoding='UTF-8')

    chatgpt_test(args, instructions=instructions)
