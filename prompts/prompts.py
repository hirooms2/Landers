old_prompt = """[Problem Statement]
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

### Output:
"""



second_prompt = """Problem Statement:

I will give you a few dialogs and a target item with features divided into five categories: (1) actors, (2) directors, (3) genres, (4) plot, (5) key impressions. In your response to the dialogs, you should recommend the target item.
Your task is to update the item's feature values by incorporating any information from the dialogs that is relevant to each specific category. This is intended to maximize the lexical and semantic similarity between the dialog and the item features.
For example, if the user expresses a positive preference for “thought-provoking movies” and the target item is “Inception (2010),” you should update the key impressions of Inception (2010) to reflect the aspect of “thought-provoking movies.”

Reasoning process:

Enclose your reasoning process within <think> tags.
Enclose your final answer (the updated item features) within <answer> tags.

Answer format:

- Each feature should be described in a standard JSON object, using double quotes for both keys and values, and listing all five categories as keys: {{"actors": "...", "directors": "...", "genres": "...", "plot": "...", "key impressions": "..."}}
- Keep each feature value within 50 tokens. If necessary, rephrase or truncate to stay within this limit.
- Only update the categories where relevant information from the dialogs is available. Otherwise, retain the original value.

### Input: Multiple user-system dialogs, the target item with features

[Dialogs]
{Dialogs}


[Item features for {target_item}]
{passages}

### Output: """


new_prompt = """Problem Statement:

I will give you a few dialogs and a target item with features divided into five categories: (1) actors, (2) directors, (3) genres, (4) plot, (5) key impressions. 

Some parts of the dialog may contain relevant information about the target item.
Your task is to update the item's feature values by inferring from the given dialogs.
These features will be used to help a conversational recommender system (CRS) provide accurate and generalizable recommendations.


Reasoning process:

- Enclose your reasoning process within <think> tags. In the reasoning process, you should first analyze the relevance between the dialogs and the target item. If there is an apprarent relevance in general, you write the updating points before you answer (IMPORTANT: Avoid overly specific or trivial details that are not helpful for generalization ability for CRSs).
- Enclose your final answer (the updated item features) within <answer> tags.

Answer format:

- Each feature should be described in a standard JSON object, using double quotes for both keys and values, and listing all five categories as keys: {{"actors": "...", "directors": "...", "genres": "...", "plot": "...", "key impressions": "..."}}
- Keep each feature value within 50 tokens. If necessary, rephrase or truncate to stay within this limit.
- Only update the categories where relevant information from the dialogs is available. Otherwise, retain the original value.

### Input: Multiple user-system dialogs, the target item with features

[Dialogs]
{Dialogs}


[Item features for {target_item}]
{passages}

### Output: """


labeling_prompt = """You are a conversational recommender system. Your task is to identify which feature of the given target item is most relevant to the dialog.
I will provide a dialog between a user and the system and target item features.

Here is the dialog:
{dialog}

Here are the item features about {target_item}:
{item_features} 

Reasoning process:
- You should first provide the rationale, then select the relevant item feature.
- Enclose your reasoning process within <think> tags. In the reasoning process, you should first analyze the relevance between the dialog and the target item. 

Answer format:
- Output the most relevant item feature's full original value exactly as given within <answer> tags.
- Do not change or modify any part of the content or format of the given item feature.
- Output 'No relevant item feature' inside <answer> tags, if the dialog is not related to any of the item's features.
"""

prompt_for_general_feature = """Problem Statement:

You will be given several dialogs and a target item, which includes features grouped into five categories:
(1) actors, (2) directors, (3) genres, (4) plot, and (5) key impressions.

The target item is recommended at the end of the dialogs.
Your task is to update the item's feature values by inferring relevant information from the dialogs. This will help a conversational recommender system (CRS) make more accurate and generalizable recommendations in the future.

Avoid including overly specific or trivial details tied to a particular dialog. Instead, focus on concrete and widely recognized aspects that are unique to the target item.


Reasoning process:

- Enclose your reasoning process within <think> tags. In the reasoning process, you should first analyze the relevance between the dialogs and the target item. If there is an apprarent relevance in general, you write the updating points before you answer (IMPORTANT: Avoid overly specific or trivial details that are not helpful for generalization ability for CRSs).
- Enclose your final answer (the updated item features) within <answer> tags.


Answer format:

- Each feature should be described in a standard JSON object, using double quotes for both keys and values, and listing all five categories as keys: {{"actors": "...", "directors": "...", "genres": "...", "plot": "...", "key impressions": "..."}}
- Keep each feature value within 50 tokens. If necessary, rephrase or truncate to stay within this limit.
- Only update the categories where relevant information from the dialogs is available. Otherwise, retain the original value.

### Input: Multiple user-system dialogs, the target item with features

[Dialogs]
{Dialogs}

[Item features for {target_item}]
{passages}

### Output: """
