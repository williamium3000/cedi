# CEDI examiner: dynamic conversation examiner for vision-language models.
#
# For each image, the examiner LLM:
#   1. Generates 2 first-person scene contexts (background + goal + relevant objects)
#      from a compact scene-graph encoding of the image.
#   2. For each context, runs a multi-round dialogue with the examinee VLM,
#      alternating between four question types:
#         - regular     : grounded extractive question about a visible object
#         - follow-up   : challenges/probes the examinee's previous response
#         - adversarial : presupposes a plausible-but-absent object (gt = "No")
#         - unanswerable: presupposes a non-existent entity (gt = "I can't answer ...")
#   3. The conversation-aware mode retains the examinee's history across rounds
#      within one context (calling reset_conversation/reset on the loader's
#      eval_func when supported; falls back to single-turn otherwise).
#
# Examiner LLMs:
#   - context generation: any reasoning-tier or strong chat model (default: gpt-5)
#   - per-round conversation: any chat model (default: gpt-4o)
# Both are routed through the OpenAI-compatible client in utils/llm.py.

from utils.utils import load_data
from utils.vg import format_case_vg_compact as format_case_vg
from utils.llm import LLMChat, parse_json
from examiner import prompt as PROMPT
from infer.loader import load_model
import os
import argparse
import json
import tqdm
import copy
from utils.sg import SceneGraphData
import random
import traceback
import re
import torch._dynamo 
torch._dynamo.config.cache_size_limit = 1024 * 1024 * 1024 * 1024 * 2  # 2TB


CONTEXT_PROMPT = \
"""
Your task is to create a realistic scenario in which the given image is situated in the first-person view, i.e. you should imagine the image depicts your view of the environment. This context should incorporate a background setting, the characters and objects involved, and a specific goal or objective that is relevant to the image. The context must be plausible, align with real-world experiences, and directly connect with the depicted elements in the image.

The image will be described through a list of objects, their attributes, and their spatial relationships, with each object represented by a bounding box [x1,y1,x2,y2] with values from 0 to 1, corresponding to the top-left and bottom-right corners of each object.

Image information:
{}

========================
CORE CONSTRAINTS
========================
1) First-person framing:
   - Write as if the image is what "I" am currently seeing in front of me.
2) Avoid non-visual / not inferable human interaction goals:
   - If humans are present, you may include them as part of the scene, but the goal must NOT require learning about their demographic information (names, intentions, etc.). 
3) Grounding in image instances:
    - Each context MUST involve multiple instances, attributes, or relations from the image.
    - Given sufficient diversity, you should generate contexts that naturally involve as many instances, attributes, or relations from the image as possible.
4) Diversity without redundancy:
   - The two contexts must be meaningfully different (different emphasis on objects/relations), not just rephrases of the same scenario.
   - IMPORTANT: Diversity is defined as the context/goal being able to provoke exploration of DIFFERENT objects/ attributes of objects/ relations between objects within the image.

Instructions:
1. Contextualization: Develop a background scenario that is logical and directly relevant to the visual elements in the image. The background should describe the setting, time, and possible situation in which these objects or characters might exist.
2. Goal: Identify a specific action, objective, or task that "I" am trying to accomplish, which should be coherent with the scene described.
3. Diversity: For the given image, you should generate several different and diverse contexts.

Some GOOD Examples:
```json
[
    {{
        "background": "A bustling city street during rush hour, with pedestrians walking past stores and cars honking in traffic. The image depicts the first-person view of the character.",
        "goal": "The character is trying to catch a bus before it leaves.",
        "relevant_objects": ["object_id_1", "object_id_2", "object_id_3",...]
    }},
    {{
        "background": "An open-plan corporate office during a busy afternoon, with cubicles neatly separated by dividing screens and personal photos decorating the workspace walls. The image depicts the first-person view of the character.",
        "goal": "The character is trying to send an email to his boss.",
        "relevant_objects": ["object_id_1", "object_id_2", "object_id_3",...]
    }}, 
    {{
        "background": "A bright and inviting kitchen featuring wooden cabinetry, a cozy dining area, and fresh fruit adding a vibrant touch. The image depicts the first-person view of the character",
        "goal": "The character is hungry and tries to eat something.",
        "relevant_objects": ["object_id_1", "object_id_2", "object_id_3",...]
    }}
]



Please generate two contexts based on the image information. Please make sure the contexts include only limited specific descriptions of object features in the image and stay as high-level glimpses without revealing too many observational details. Make sure the contexts are diverse and not redundant.

For each context, also select ALL the relevant object nodes from the image that are related to the background or goal. These objects will be used to generate evaluation questions. Select objects in order of relevancy, with the most relevant first.
Be comprehensive in your selection to ensure all pertinent objects are included. And include any objects that are closely related in terms of same type or same bbox location.

========================
WHAT TO PRODUCE
========================
Return EXACTLY a JSON list with TWO dictionaries. Each dictionary must contain:
- "background": <50 words, first-person situation, plausible real-world setting>
- "goal": a concrete objective that can be progressed by asking image-grounded questions
- "relevant_objects": a list of object_ids (from the image information) that are most relevant to this context, ordered by relevancy

Output format (STRICT):
```json
[
  {{"background": "...", "goal": "...", "relevant_objects": ["object_id_1", "object_id_2", "object_id_3", ...]}},
  {{"background": "...", "goal": "...", "relevant_objects": ["object_id_1", "object_id_2", "object_id_3", ...]}}
]
```
You MUST only respond in the format as described above. DO NOT RESPOND WITH ANYTHING ELSE.
"""


# SELECT_CONTEXT_NODES_PROMPT_SYSTEM = \
# """
# you will be give an image and a context (background and goal). Then given image will be presented to you as a list of objects with attributes and relation of these objects. Each objects will be presented with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
# Based on the image and context, you need to select the object nodes that are most relevant to the context, which will later be used to generate the evaluation question on this image and context.
# You can select one or multiple object nodes, but make sure they are relevant to the context or can be used to generate a evaluation question that fits the context.

# Respond format: You should return a list of selected object node in the order of their relevancy with the context. The first object node should be the most relevant object node to the context.
# you can explain, reason or perform chain-of-thought to select the object nodes. However, you MUST eventually provide a list of selected object_id in the json format as described below:
# ```json
# [
#     "object_id_1", "object_id_2", "object_id_3", ...
# ]
# ```
# """

SELECT_CONTEXT_NODES_PROMPT = \
"""
Image information:
{}

Context:
background: {}, goal: {}

Please select the object nodes that are most relevant to the context in the order of their relevance to the context. You can explain, reason, or perform chain-of-thought to select the object nodes. However, you MUST respond with the selected object_id(s) in the JSON format described above.
"""


SWITCH_PROMPT_EARLY = \
"""Based on the given conversation history, please decide which type of question to ask next. Please choose one of the following types of questions:

1. **Regular questions** – directly related to the image and context.
    *Example*: In an office setting, the goal of the conversation is to send an email to his boss. You can ask questions like "is the monitor turned on?", "where is the power button of the computer?", "I want to type in the email, what should I look for?", etc.

2. **Follow-up questions** – follow up, confirm or interrogate the model's last response to test the confidence of the model's answer or to ask for further challenging details when available.
   *Example*: If the model correctly identifies there's no fork on the table, you may ask: *"Are you sure there isn't a fork on the table to cut the cake?"*

3. **Adversarial questions** – inquire about plausible but absent objects that commonly co-occur with visible ones in the image.
    *Example*: If the image shows a cake without utensils, you may ask: *"Can I use the knife on the table to cut the cake?"* or *"Is there a fork?"*

4. **Unanswerable questions** – ask about question that cannot be answered.
   *Example*: if the image depicts a cake on the table without any utensil or people eating the cake, you can ask "What utensil is the man using to cut the cake?". This is an unanswerable question because you cannot answer with an utensil but rather you should answer "There are not any man in the image eating the cake".

Given the previously asked question types: {}, try to ask diverse types of questions.

YOU CAN ONLY SELECT ONE OF THE ABOVE FOUR TYPES OF NEXT-STEPS !!
Respond format: think thoroughly to reason or perform chain-of-thought to select the next question type. 
However, you MUST eventually provide a digit between 1 and 4 in below json format to instruct the next round:
```json
{{"type": type_id}}
```
"""

SWITCH_PROMPT_LATE = \
"""Based on the given conversation history, please decide which type of question to ask next. Please choose one of the following types of questions:

1. **Regular questions** – directly related to the image and context.
    *Example*: In an office setting, the goal of the conversation is to send an email to his boss. You can ask questions like "is the monitor turned on?", "where is the power button of the computer?", "I want to type in the email, what should I look for?", etc.

2. **Follow-up questions** – follow up, confirm or interrogate the model's last response to test the confidence of the model's answer or to ask for further challenging details when available.
   *Example*: If the model correctly identifies there's no fork on the table, you may ask: *"Are you sure there isn't a fork on the table to cut the cake?"*

3. **Adversarial questions** – inquire about plausible but absent objects that commonly co-occur with visible ones in the image.
    *Example*: If the image shows a cake without utensils, you may ask: *"Can I use the knife on the table to cut the cake?"* or *"Is there a fork?"*

4. **Unanswerable questions** – ask about question that cannot be answered.
   *Example*: if the image depicts a cake on the table without any utensil or people eating the cake, you can ask "What utensil is the man using to cut the cake?". This is an unanswerable question because you cannot answer with an utensil but rather you should answer "There are not any man in the image eating the cake".

5. **End the conversation** – to end the conversation.
   **IMPORTANT**: ONLY choose this option if you believe all the relevant objects and potential actions towards the goal have been explored by the 4 dimensions of questions above.

Given the previously asked question types: {}, try to ask diverse types of questions.

YOU CAN ONLY SELECT ONE OF THE ABOVE FIVE TYPES OF NEXT-STEPS !!
Respond format: think thoroughly to reason or perform chain-of-thought to select the next question type. 
However, you MUST eventually provide a digit between 1 and 5 (with 5 representing "End the conversation") in below json format to instruct the next round:
```json
{{"type": type_id}}
```
"""

CONV_SYSTEM_PROMPT = \
"""Your task is to **simulate a multi-round conversation with a vision-language model (VLM) about a given image**, in order to test whether the model hallucinates (i.e., produces responses inconsistent with the image) or remains faithful to it.
**Ultimate Goal:** Raise challenging visual questions and probe as many hallucinations from the responding model as possible.

You will have multiple rounds of conversations with the model. In each round, you will be provided with:

* **An image** (represented as a list of objects, their attributes, and relationships).
* **Bounding-box coordinates** for each object, given as `(x1, y1, x2, y2)` in normalized values between 0 and 1, corresponding to top-left and bottom-right corners of each object.

Your role is to **carry out a natural, open-ended, human-like conversation** with the model, asking questions about the image in the given context.

Requirements

1. **Multi-turn conversation**:

    * Ask one question per turn, then wait for the model’s response.
    * Incorporate the image, context, and dialogue history when forming your question.

2. **Natural, human-like tone**:

    * Speak as if conversing casually with another person.
    * Avoid mechanical or scripted phrasing.

3. **No disclosure of metadata**:

    * Do not reveal or reference bounding boxes, object lists, captions, or the source of your information.
    * Ask questions as if you are looking at the image with NO access to the bounding boxes, object lists, or captions.

4. **Handling mistakes**:

    * Do not correct the model if they make mistakes. 
    * However, you may ask follow-ups to probe or interrogate its response.

5. **Question style**:

    * Prefer open-ended questions over yes/no questions.
    * Ask diverse, non-redundant questions.
    * Only ask questions with **definite answers** (either clearly present in the image or confidently absent).

6. **Conversation ending**:

    * If the dialogue has naturally run its course, output **“END”** (and nothing else).

Image information:
{}


Context:
background: {}, goal: {}
"""

REGULAR_CONV_PROMPT = \
"""You are tasked with generating a *regular question* grounded in the provided image and context.

* **Target node:** `{}`
* **Context:** background: `{}`, goal: `{}`

**Instructions:**

* Ask a natural, conversational question about the specified node.
* The question must be strongly relevant to both the image and the context (background + goal).
* Avoid irrelevant, generic, or out-of-character questions.
* Keep the question consistent with the role of someone actively engaged in the situation.
* Generate the ground-truth answer based on the given image content along the question.
* **CRITICAL: Do NOT repeat or ask questions with the same meaning as any prior questions in the conversation history.**

**Previously asked questions (DO NOT repeat these or ask similar questions):**
{}

**Example:**
If the background is an office and the goal is “send an email to the boss,” suitable questions include:

* {{"question": "Is the monitor turned on?", "gt": "No"}}
* {{"question": "Where is the computer’s power button?", "gt": "It's on the computer under the table to the right of the image."}}
* {{"question": "I want to type the email—what should I look for?", "gt": "You should look for the keyboard and mouse on the table in front of the monitor."}}

**Output Requirement:**
Respond in the following json format:
```json
{{"question": "xxxx", "gt": "xxxx"}},
```

Respond with **the question and gt ONLY**. Do not include explanations, commentary, or any additional text.
"""

FOLLOW_UP_CONV_PROMPT = \
"""Based on the given image, context, and the conversation history, please ask a follow-up question about the model's last turn of conversation and the corresponding ground-truth answer.
The follow-up question can be to ask for further challenging details when available or to interrogate the model's confidence in its previous correct answer.

**Interrogate Type Requirement:**
Only apply this follow-up type if the model correctly answered the previous question. We then want to test whether the confidence of the model's answer is low despite misleading subjective or peripheral evidence.
- Challenge or verify the model's previous answer
- Probe for more details
- Test consistency
- Question contradictions

**Example follow-ups:**
- After any answer: "Are you sure about [the specific detail they mentioned]?" like "Are you sure the color is [the color they mentioned]? I see it's actually [random other color]."
- Challenging with visible evidence if last answer is non-existent: "I actually see a [the questioned item] near/around [co-occurring object/resonable location], are you sure you can't see [the questioned item]?"
- Requesting clarification: "You mentioned [X], but can you describe it more precisely?"
- Testing consistency: "Earlier you said [Y], how does that relate to what you just said?"

You should ask the question as if you are having a conversation with the model. You should also follow the above requirements.
Please respond in the following format:
```json
{{"question": "xxxx", "gt": "xxxx"}},
```
Please respond with the question and the ground-truth answer ONLY. DO NOT respond with anything else.
"""


ADVERSARIAL_CONV_PROMPT1 = \
"""Based on the given image, context, and the conversation history, please ask an adversarial question about ONE plausible scenario/object that commonly co-occur with visible feature(s) in the image but currently isn't present (the ground truth answer is always 'No').

Requirements:
1) Your generated hallucinated scenario/object and the final adversarial question should be also consistent with the context:
background: {}, goal: {}
2) Your generated hallucinated scenario/object should not be present in the image but should highly co-occur with image content.
3) ENCOURAGED if possible: Include an inductive clause that briefly mentions visible objects as part of the existence question (e.g., "To help cut the cake on the table, is there a knife present?").
4) **CRITICAL: Do NOT repeat or ask questions that are similar to any prior questions in the conversation history.**

**Previously asked questions (DO NOT repeat these or ask similar questions):**
{}

Please output the hallucinated scenario/object as scene graph elements"names"/"attributes"/"relations" and the corresponding question in the form of a json dict.

Examples:
If there is a blue banana on the table while banana should usually be yellow.
```json
{{"names": "banana", "attributes": "yellow", "question": "Is there a yellow banana present on the table?"}}
```
If there is cake on the table, which is usually accompanied by a knife but currently isn't.
```json
{{"names": "knife", "question": "Is there a knife present on the table?"}}
```

Allowed keys in the JSON output are "names", "attributes", "relations", and "question".
"""

# ADVERSARIAL_CONV_PROMPT2 = \
# """Now you should ask a adversarial question based on the generated hallucinated scenario/object and the corresponding ground-truth answer.
# Your generated adversarial question should try to flow naturally with the conversation history.
# ENCOURAGED: Include an inductive clause that briefly mentions 1 or 2 of the "co-occur_with" items as part of the existence question (e.g., "Given the cake on the table, is there a knife present?").

# Please respond in the following format:
# ```json
# {{"question": "xxxx", "gt": "No", "co-occur_with": "xxxx"}},
# ```
# Where gt is always 'No' (item is not present) and co-occur_with is the comma-separated list of visible features from the previous step.
# You should ask a question as if you are having a conversation with the model. Please respond with the question and the ground-truth answer ONLY. DO NOT respond with anything else.
# """

UNANSWERABLE_CONV_PROMPT1 = \
"""Based on the given image, context, and conversation history, generate an **unanswerable question**.

Here is the context:
background: {}, goal: {}

An *unanswerable question* refers to a query that cannot be answered using the provided information because it introduces a plausible but absent or incorrect object, attribute, or relation.
For example, if the image shows only a cake on the table, asking *"What utensil is the man using to cut the cake?"* is unanswerable since no man is present.

**Procedure:**

1. **Hallucinated Object:** Generate a plausible but absent or incorrect object that would typically co-occur with image content. Output in JSON format:

   ```json
   {{"names": "object_name"}}
   ```

2. **Hallucinated Relation/Attribute:** Generate a plausible relation or attribute involving the hallucinated object. Here the relation should link the hallucinated object to an actual object from the image. Output in JSON format:

   ```json
   {{"relations": "relation", "object": "real_object", "subject": "hallucinated_object"}}
   ```

3. **Unanswerable Question:** Formulate a natural question about this hallucinated relation or attribute. The question should sound plausible but must be unanswerable from the provided image.

**Example Walkthrough:**

* Image: A cake on the table, no people visible.
* Step 1:

  ```json
  {{"names": "man"}}
  ```
* Step 2:

  ```json
  {{"relations": "eating", "object": "cake", "subject": "man"}}
  ```
* Step 3: ask the unanswerable question about the relation with corresponding ground-truth answer: "eating" between the generated object "man" and the object "cake": “What utensil is the man using to cut the cake?”*

  ```json
  {{"question": "What utensil is the man using to cut the cake?", "gt": "There are not any man in the image eating the cake"}}
  ```
---

**CRITICAL - Avoid Repetition:**
**The SAME hallucinated object in previously asked unanswerable questions also counts as repetition. For example, if you previously asked about "the man's shirt color", do NOT ask about "the man's position" or any other feature of "the man". You MUST introduce a NEW hallucinated object.**
{}


**Your Task Now:**
Complete **Step 1** by generating a plausible but absent or incorrect object in JSON format.
"""

UNANSWERABLE_CONV_PROMPT2 = \
"""Now complete Step 2. Generate a plausible attribute or relation for the hallucinated object you created in Step 1. This relation must link the hallucinated object to one of the real objects present in the image."""

UNANSWERABLE_CONV_PROMPT3 = \
"""Now complete Step 3. You must ask an **UNANSWERABLE question** about the hallucinated relation/attribute you generated, and provide the corresponding ground-truth answer (gt).

CRITICAL GOAL (Separation Guarantee):
This must be an **UNANSWERABLE (presuppositional trap) question** that **DOES presuppose** the hallucinated object/relation/attribute exists in the image.
The question must sound plausible, but it cannot be answered from the provided image because the key premise is false/absent.

Hard Requirements (MUST follow):
1) **Strong presupposition is mandatory**
   - You MUST write the question as if the hallucinated object/relation is definitely there.
   - Use definite references and presuppositional phrasing, e.g.:
     "the", "that", "the one on the table", "the person", "the knife next to..."
   - The question should demand a **specific detail** that cannot be provided if the object is absent:
     color, exact location, count, identity, orientation, precise relation, etc.

2) **No hedging / no conditional phrasing**
   - DO NOT use: "is there any", "do you see", "if there is", "might", "maybe".
   - This is a trap question: it should *assume* existence.

3) **Ground-truth answer must explicitly correct the false premise**
   - The gt MUST point out the absence / invalid premise clearly, e.g.:
     "There is no X in the image."
     "I do not see X, so I cannot answer that."
     "No person is visible interacting with Y."
   - Do not invent details to satisfy the question.

4) **Consistency with context and conversation**
   - The question must be consistent with the context (background + goal) and conversation history, even though it is unanswerable.

**Previously asked questions (DO NOT repeat these or ask similar questions):**
{}

Good Unanswerable Question Templates (Examples):
- Definite reference + attribute detail:
  * {{"question": "What color is the handle of the knife on the table?"}}
- Definite reference + relation detail:
  * {{"question": "Which side of the cake is the fork placed on—left or right?"}}
- Definite reference + human interaction detail (only if the image has no humans; this makes it unanswerable):
  * {{"question": "What utensil is the man using to cut the cake?"}}

Bad (DISALLOWED) Examples (these would overlap with ADVERSARIAL):
- "Do you see any knife we could use?" (non-presuppositional)
- "If there is a fork, where might it be?" (conditional)

Output Requirement:
Respond in the following json format:
```json
{{"question": "xxxx"}}
```

Respond with **the question ONLY**. Do not include explanations, commentary, or any additional text.
"""

Q_TYPE_MAPPING = {
    1: "regular",
    2: "follow-up",
    3: "adversarial",
    4: "unanswerable",
    5: "end",
}
class EvalSample:
    def __init__(self, case, llm_chat_context, llm_chat_conv, eval_func):
        self.case = case
        self.image_info = format_case_vg(case)
        self.scene_graph_data = SceneGraphData.from_dict(case)

        self.llm_chat_context = llm_chat_context  # GPT-5 for context generation
        self.llm_chat_conv = llm_chat_conv  # GPT-4o for conversation
        self.eval_func = eval_func
        self.conversations = []
        # Track previously asked questions by type to avoid repetition
        self.repeat_ref_dict = {
            "regular": [],
            "adversarial": [],
            "unanswerable": []
        }

    def switch(self, conversations, swicth_history, round_num):
        conversations = copy.deepcopy(conversations)
        # Use early version (without END option) before round 6
        switch_prompt = SWITCH_PROMPT_EARLY if round_num < 6 else SWITCH_PROMPT_LATE
        conversations.append(
            {"role": "user", "content": switch_prompt.format(swicth_history)})
        type_id = self.llm_chat_conv.chat(conversations, parse_json)['type']
        # Handle case where LLM returns "END" string instead of 5
        if isinstance(type_id, str) and type_id.upper() == "END":
            type_id = 5
        return type_id
    
    def ask_regular(self, conversations, selected_nodes, context):
        conversations = copy.deepcopy(conversations)
        sampled_node = selected_nodes.pop(0)
        prev_questions_str = json.dumps(self.repeat_ref_dict["regular"], indent=2) if self.repeat_ref_dict["regular"] else "None"
        conversations.append({"role": "user", "content": REGULAR_CONV_PROMPT.format(sampled_node, context["background"], context["goal"], prev_questions_str)})
        
        # Retry up to 5 times if LLM returns None (parsing failure)
        max_retries = 5
        for attempt in range(max_retries):
            message = self.llm_chat_conv.chat(conversations, parse_json)
            if message is not None:
                # Track this question to prevent repetition
                self.repeat_ref_dict["regular"].append(message.get("question", ""))
                return message
            print(f"Warning: LLM returned None in ask_regular (attempt {attempt + 1}/{max_retries}), retrying...")
        
        # All retries failed, raise error to skip this sample
        raise ValueError(f"LLM returned None after {max_retries} retries in ask_regular, skipping sample")
    
    def ask_follow_up(self, conversations, context):
        conversations = copy.deepcopy(conversations)
        conversations.append({"role": "user", "content": FOLLOW_UP_CONV_PROMPT})
        
        # Retry up to 5 times if LLM returns None (parsing failure)
        max_retries = 5
        for attempt in range(max_retries):
            message = self.llm_chat_conv.chat(conversations, parse_json)
            if message is not None:
                return message
            print(f"Warning: LLM returned None in ask_follow_up (attempt {attempt + 1}/{max_retries}), retrying...")
        
        # All retries failed, raise error to skip this sample
        raise ValueError(f"LLM returned None after {max_retries} retries in ask_follow_up, skipping sample")
    
    def ask_unanswerable(self, conversations, context):
        conversations = copy.deepcopy(conversations)
        meta_msg = []
        prev_questions_str = json.dumps(self.repeat_ref_dict["unanswerable"], indent=2) if self.repeat_ref_dict["unanswerable"] else "None"
        conversations.append({"role": "user", "content": UNANSWERABLE_CONV_PROMPT1.format(context["background"], context["goal"], prev_questions_str)})
        message = self.llm_chat_conv.chat(conversations, None)
        meta_msg.append(message)
        conversations.append({"role": "assistant", "content": message})
        conversations.append({"role": "user", "content": UNANSWERABLE_CONV_PROMPT2})
        message = self.llm_chat_conv.chat(conversations, None)
        meta_msg.append(message)
        conversations.append({"role": "assistant", "content": message})
        prev_questions_str = json.dumps(self.repeat_ref_dict["unanswerable"], indent=2) if self.repeat_ref_dict["unanswerable"] else "None"
        conversations.append({"role": "user", "content": UNANSWERABLE_CONV_PROMPT3.format(prev_questions_str)})
        
        # Retry up to 5 times if LLM returns None (parsing failure)
        max_retries = 5
        for attempt in range(max_retries):
            message = self.llm_chat_conv.chat(conversations, parse_json)
            if message is not None:
                # Default gt to standard message for unanswerable questions
                if "gt" not in message:
                    message["gt"] = "I can't answer the question because the object doesn't exist."
                meta_msg.append(message)
                # Track this question to prevent repetition
                self.repeat_ref_dict["unanswerable"].append(message.get("question", ""))
                return message, meta_msg
            print(f"Warning: LLM returned None in ask_unanswerable (attempt {attempt + 1}/{max_retries}), retrying...")
        
        # All retries failed, raise error to skip this sample
        raise ValueError(f"LLM returned None after {max_retries} retries in ask_unanswerable, skipping sample")
    
    def ask_adversarial(self, conversations, context):
        conversations = copy.deepcopy(conversations)
        meta_msg = []
        prev_questions_str = json.dumps(self.repeat_ref_dict["adversarial"], indent=2) if self.repeat_ref_dict["adversarial"] else "None"
        conversations.append({"role": "user", "content": ADVERSARIAL_CONV_PROMPT1.format(context["background"], context["goal"], prev_questions_str)})
        
        # Retry up to 5 times if LLM returns None (parsing failure)
        max_retries = 5
        for attempt in range(max_retries):
            message = self.llm_chat_conv.chat(conversations, parse_json)
            if message is not None:
                # Default gt to 'No' for adversarial questions
                if "gt" not in message:
                    message["gt"] = "No"
                meta_msg.append(message)
                # Track this question to prevent repetition
                self.repeat_ref_dict["adversarial"].append(message.get("question", ""))
                return message, meta_msg
            print(f"Warning: LLM returned None in ask_adversarial (attempt {attempt + 1}/{max_retries}), retrying...")
        
        # All retries failed, raise error to skip this sample
        raise ValueError(f"LLM returned None after {max_retries} retries in ask_adversarial, skipping sample")
    
    def generate_context(self, case, previous_context=None):
        """Generate context(s) for the image.
        
        Args:
            case: The sample case containing image data
            previous_context: If provided, the first context that was already generated.
                            This ensures the second context stays diverse.
        
        Returns:
            List of contexts (either 2 new contexts, or 1 new context if previous_context provided)
        """
        image_info = format_case_vg(case)

        if previous_context is not None:
            # Generate only the second context, using the first context for diversity
            prompt = CONTEXT_PROMPT.format(image_info).strip()
            prompt += f"\n\nIMPORTANT: The first context has already been generated as:\n{json.dumps(previous_context, indent=2)}\n\nPlease generate ONLY ONE additional context that is meaningfully different from the above. Return a JSON list with a single dictionary."
            conversations = [
                {"role": "system", "content": "You are an expert in generating realistic and diverse contexts for images. You excel at understanding the image content and predicting the possible scenarios and context in which the image might be situated."},
                {"role": "user", "content": prompt}
            ]
            contexts = self.llm_chat_context.chat(conversations, parse_json)
            # Ensure it returns a list
            if isinstance(contexts, dict):
                contexts = [contexts]
        else:
            # Generate both contexts from scratch
            conversations = [
                {"role": "system", "content": "You are an expert in generating realistic and diverse contexts for images. You excel at understanding the image content and predicting the possible scenarios and context in which the image might be situated."},
                {"role": "user", "content": CONTEXT_PROMPT.format(image_info).strip()}
            ]
            contexts = self.llm_chat_context.chat(conversations, parse_json)

        return contexts

    def _parse_object_id(self, oid, retry_count=0, max_retries=3):
        """Parse an object id that may be an int, a numeric string, or a string like 'instance 0'.
        Returns an int or raises ValueError.
        Includes retry logic with up to 3 attempts for parsing errors.
        """
        try:
            if isinstance(oid, int):
                return oid
            # try direct int conversion
            try:
                return int(oid)
            except Exception:
                pass
            # try to extract first integer in the string
            m = re.search(r"(\d+)", str(oid))
            if m:
                return int(m.group(1))
            raise ValueError(f"Cannot parse object id: {oid}")
        except Exception as e:
            if retry_count < max_retries:
                # Retry with incremented count
                return self._parse_object_id(oid, retry_count + 1, max_retries)
            else:
                # Max retries reached, raise the error
                raise ValueError(f"Cannot parse object id after {max_retries} retries: {oid}")

    def _normalize_relevant_objects(self, raw, retry_count=0, max_retries=3):
        """Normalize various formats of relevant_objects into a list of int ids.
        Handles: list of ints/strings, a single string like "instance 0", or a comma/bracketed string.
        Includes retry logic with up to 3 attempts for parsing errors.
        """
        try:
            ids = []
            if raw is None:
                return ids
            if isinstance(raw, list):
                for oid in raw:
                    try:
                        ids.append(self._parse_object_id(oid))
                    except Exception:
                        continue
                return ids
            # if it's a string, extract all integers
            if isinstance(raw, str):
                found = re.findall(r"\d+", raw)
                return [int(x) for x in found]
            # fallback: try parsing a single value
            try:
                return [self._parse_object_id(raw)]
            except Exception:
                return []
        except Exception as e:
            if retry_count < max_retries:
                # Retry with incremented count
                return self._normalize_relevant_objects(raw, retry_count + 1, max_retries)
            else:
                # Max retries reached, return empty list
                print(f"Warning: Failed to normalize relevant objects after {max_retries} retries: {raw}")
                return []
    

    # Node selection is now integrated into context generation; standalone selection removed.
    

    def run(self, prev_contexts=None, on_context_done=None):
        """Run conversation generation.
        
        Args:
            prev_contexts: List of previously generated contexts from cache.
                          Can be None (generate both), 1-element list (generate 2nd only),
                          or 2-element list (skip this sample entirely).
        
        Returns:
            List of conversation results to save
        """
        to_save = []
        
        # Determine which contexts to generate
        if prev_contexts is None:
            # Generate both contexts from scratch
            num_to_generate = 2
            previous_context = None
        elif len(prev_contexts) == 1:
            # Generate only the second context, using first for diversity
            num_to_generate = 1
            previous_context = prev_contexts[0]
        else:
            # Both contexts already exist, nothing to do
            print("Both contexts already cached, skipping this sample.")
            return to_save
        
        # Retry context generation and parsing up to 3 times
        max_retries = 3
        for attempt in range(max_retries):
            try:
                contexts = self.generate_context(self.case, previous_context=previous_context)
                # Validate context output
                valid_contexts = []
                for context in contexts:
                    # Check if relevant_objects is a non-empty list of valid IDs
                    relevant_objects = context.get("relevant_objects", [])
                    if (isinstance(relevant_objects, list)
                        and len(relevant_objects) > 0
                        and all(isinstance(oid, (str, int)) for oid in relevant_objects)):
                        valid_contexts.append(context)
                if not valid_contexts:
                    raise ValueError("Context parsing error: No valid (non-empty) relevant_objects list.")
                break
            except Exception as e:
                print(f"Context generation or parsing failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print("Max retries reached. Skipping sample.")
                    return to_save

        for context in valid_contexts:
            # Reset repeat tracking for each new context
            self.repeat_ref_dict = {
                "regular": [],
                "adversarial": [],
                "unanswerable": []
            }
            
            # Reset conversation history for this image to ensure independent conversations per context
            if hasattr(self.eval_func, 'reset_conversation'):
                self.eval_func.reset_conversation(self.case["image"])
                print("Reset conversation history for new context")
            
            selected_nodes = [self.scene_graph_data.image_info.sg.get_object_by_id(self._parse_object_id(oid)) for oid in context.get("relevant_objects", [])]
            print("-" * 50, f"context", "-" * 50)
            print(context)
            print("selected nodes: ", selected_nodes)
            print("-" * 100)
            # TODO: can we use ICLs here?
            # sys_prompt = PROMPT.__dict__[args.p_mode]
            # loaded_icls = []
            # if args.icls is not None:
            #     loaded_icls = json.load(open(args.icls))
            
            # ICLs = []
            # for icl in loaded_icls:
            #     firstp = CONV_PROMPT.format(icl["image_info"])
            #     ICLs.append({"role": "user", "content": firstp})
            #     ICLs.extend(icl["conversations"])
            
            conversations = [
                {"role": "system", "content": CONV_SYSTEM_PROMPT.format(self.image_info, context["background"], context["goal"])}
            ]
            to_save_i = []
            switch_history = []
            # Reset conversation history for conversation-aware models
            if hasattr(self.eval_func, 'reset'):
                self.eval_func.reset()
            # Reset conversation history for specific image if method exists
            if hasattr(self.eval_func, 'reset_conversation'):
                self.eval_func.reset_conversation(self.case["image"])
            r = 0
            image_file = self.case["image"]  # Get image file path once
            while True:
                if r == 0:
                    type_id = 1 # first round always ask regular question
                else:
                    type_id = self.switch(conversations, switch_history, r)
                    # If END is selected, run switch a second time to confirm
                    if type_id == 5:
                        print("END selected, confirming...")
                        type_id_confirm = self.switch(conversations, switch_history, r)
                        if type_id_confirm == 5:
                            print("END confirmed, ending conversation")
                            switch_history.append(type_id)
                            break
                        else:
                            print(f"END not confirmed, proceeding with type {type_id_confirm}")
                            type_id = type_id_confirm
                switch_history.append(type_id)
                if type_id == 5:
                    break
                if type_id == 1:
                    if len(selected_nodes) == 0:
                        continue
                    print("asking regular question")
                    message_evaluator = self.ask_regular(conversations, selected_nodes, context)
                elif type_id == 2:
                    print("asking follow-up question")
                    message_evaluator = self.ask_follow_up(conversations, context)
                elif type_id == 3:
                    print("asking adversarial question")
                    message_evaluator, adv_meta_msg = self.ask_adversarial(conversations, context)
                elif type_id == 4:
                    print("asking unanswerable question")
                    message_evaluator, una_meta_msg = self.ask_unanswerable(conversations, context)
                # debug
                question = message_evaluator["question"] # + "/n Please also describe what we previously discussed. And if you directly see an image? Where do you see a image?"
                gt = message_evaluator["gt"]
                conversations.append({"role": "assistant", "content": question})
                
                # Call eval_func: wrapper handles conversation history automatically
                # - First round (r==0): wrapper will include image in conversation_history
                # - Later rounds (r>0): wrapper will only add text query
                # The image_file is passed every time, but the wrapper's internal conversation_history
                # determines whether to include it (only when conversation_history is empty/None)
                output = self.eval_func(image_file=image_file, query=question)
                # Handle case where eval_func returns a tuple (output_text, conversation_history)
                # Extract the output_text string from the tuple
                if isinstance(output, tuple):
                    output = output[0]  # output_text is the first element
                output = output.lower()
                conversations.append({"role": "user", "content": output})
                print("-" * 50, f"round {r}", "-" * 50)
                print(f"examiner question: {question}")
                print(f"examiner gt: {gt}")
                print(f"vlm model: {output}")
                print("-" * 100)
                r += 1
                saved_message = {"round_id": r, "prompt": question, "response":output, "q_type": Q_TYPE_MAPPING[type_id], "gt": gt}
                if type_id == 3:
                    saved_message["meta_msg"] = adv_meta_msg
                elif type_id == 4:
                    saved_message["meta_msg"] = una_meta_msg
                to_save_i.append(saved_message)
                if r > args.max_rounds:
                    print("reached max rounds")
                    break
            sample_to_save = copy.deepcopy(self.case)
            sample_to_save["conversations"] = to_save_i
            sample_to_save["context"] = context
            sample_to_save["relevant_objects"] = context.get("relevant_objects", [])
            del sample_to_save["image"]
            to_save.append(sample_to_save)
            # Let the caller persist each finished conversation right away, so a crash
            # loses at most the conversation in progress (not both contexts of the image).
            if on_context_done is not None:
                on_context_done(sample_to_save)
        return to_save
            


def load_cache(cache_file):
    """Load existing results from cache file if it exists.
    
    Returns:
        dict: Dictionary mapping (sample_id, goal) -> cached conversation result
    """
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
        print(f"Loaded {len(cached_data)} cached conversation results from {cache_file}")
        # Build index by (sample_id, goal)
        cache_index = {}
        for item in cached_data:
            sample_id = item.get("image_id", "unknown")
            context = item.get("context", {})
            goal = context.get("goal", "")
            cache_index[(sample_id, goal)] = item
        return cached_data, cache_index
    else:
        return [], {}

def save_cache(cache_file, data):
    """Save results to cache file."""
    try:
        # Write to a temp file then rename, so a killed job never leaves a half-written cache.
        tmp_file = cache_file + ".part"
        with open(tmp_file, 'w') as f:
            json.dump(data, f, indent=4)
        os.replace(tmp_file, cache_file)
        print(f"Saved {len(data)} results to cache file {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save to cache file {cache_file}: {e}")
        traceback.print_exc()

def get_sample_id(sample):
    """Generate a unique identifier for a sample based on its content."""
    # Use image_id as the primary identifier
    return sample.get("image_id", "unknown")

def get_cached_contexts_for_sample(sample_id, cached_data):
    """Get all cached contexts for a given sample_id.
    
    Args:
        sample_id: The image_id to look for
        cached_data: List of all cached conversation results
    
    Returns:
        List of context dicts that have already been processed for this sample
    """
    contexts = []
    for item in cached_data:
        if item.get("image_id") == sample_id and "context" in item:
            contexts.append(item["context"])
    return contexts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--model_path', type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument('--icls', type=str, default=None)
    parser.add_argument('--outfile', type=str)
    parser.add_argument('--cache_file', type=str, default=None, 
                       help='Cache file to store/load intermediate results for resuming')
    parser.add_argument('--max_rounds', type=int, default=20)
    parser.add_argument('--attn_implementation', type=str, default=None,
                       help='Attention backend passed to from_pretrained for Qwen2.5-VL, e.g. flash_attention_2')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode: inject "what do you see" question after first query to verify image context')
    args = parser.parse_args()
    
    # This script is for conversation-aware models, so always enable conversation history
    args.use_conversation = True
    
    # Set default cache file if not provided
    if args.cache_file is None:
        args.cache_file = args.outfile.replace('.json', '_cache.json')

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    # need to figure out how to eval on different models
    eval_func = load_model(args)
    
    samples = load_data(args)
    
    llm_chat_context = LLMChat(model_name="gpt-5")  # For context generation
    llm_chat_conv = LLMChat(model_name="gpt-4o")  # For conversation generation
    
    # Initialize cache and resume functionality
    to_save = []
    cached_data = []
    cache_index = {}
    
    # Load existing cache if provided
    if args.cache_file:
        cached_data, cache_index = load_cache(args.cache_file)
        to_save.extend(cached_data)
        print(f"Loaded {len(cached_data)} cached conversations")
    
    print("starting conversation with model...")
    total_samples = len(samples)
    total_conversations = total_samples * 2  # 2 conversations per sample
    completed_conversations = len(cached_data)
    
    for i, sample in enumerate(tqdm.tqdm(samples, desc="Processing samples")):
        sample_id = get_sample_id(sample)
        
        # Check how many contexts are already cached for this sample
        cached_contexts = get_cached_contexts_for_sample(sample_id, cached_data)
        num_cached = len(cached_contexts)
        
        if num_cached >= 2:
            print(f"Skipping sample {i+1}/{total_samples} (both conversations already processed): {sample_id}")
            continue
        elif num_cached == 1:
            print(f"Processing sample {i+1}/{total_samples}: {sample_id} (1/2 conversations cached, generating 2nd)")
        else:
            print(f"Processing sample {i+1}/{total_samples}: {sample_id} (generating both conversations)")
        
        try:
            def save_conversation(conv_item):
                global completed_conversations
                to_save.append(conv_item)
                completed_conversations += 1
                # Save cache incrementally if cache file is provided
                if args.cache_file:
                    save_cache(args.cache_file, to_save)
                    print(f"Progress: {completed_conversations}/{total_conversations} conversations completed")

            eval_sample = EvalSample(sample, llm_chat_context, llm_chat_conv, eval_func)
            # Pass cached contexts so it knows which ones to generate
            eval_sample.run(prev_contexts=cached_contexts if num_cached > 0 else None,
                            on_context_done=save_conversation)
                
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            print("Continuing with next sample...")
            traceback.print_exc()
            continue
    
    # Final save to output file
    with open(args.outfile, "w") as f:
        json.dump(to_save, f, indent=4)
    
    print(f"Completed processing {completed_conversations}/{total_conversations} conversations ({completed_conversations//2}/{total_samples} samples)")
    print(f"Results saved to {args.outfile}")
    if args.cache_file:
        print(f"Cache saved to {args.cache_file}")
