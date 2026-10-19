CONV_MODEL_PERSPECTIVE_PROMPT_VG_ICL = \
"""
Your task is to have a multi-round conversation with a vision-language model regarding a given image. The responses from the vision-language model will be later to used to evaluate whether the model is hallucinating or faithful to the image (wheteher model generate responses contradictory to the content of the image).
You need to have multiple conversations the model over multiple images. Each conversation will be based on one single image. The given image will be presented to you as a list of objects with attributes and relation of these objects. Each objects will be presented with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
Given this image, you need to perform a series of casual conversations with the vision-language model naturally and ask questions about the given detailed information of the image.
The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
3. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
4. You should end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

You will be given multiple images, and you need to have multiple conversations with the vision-language model. Once you are given the information of the image, you should start the conversation about this image. 
"""

CONV_MODEL_PERSPECTIVE_PROMPT_VG_ICL_CERTAINTY_COVERAGE = \
"""
Your task is to have a multi-round conversation with a vision-language model regarding a given image. The responses from the vision-language model will be later to used to evaluate whether the model is hallucinating or faithful to the image (wheteher model generate responses contradictory to the content of the image).
You need to have multiple conversations the model over multiple images. Each conversation will be based on one single image. The given image will be presented to you as a list of objects with attributes and relation of these objects. Each objects will be presented with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
Given this image, you need to perform a series of casual conversations with the vision-language model naturally and ask questions about the given detailed information of the image.
The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
3. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
4. The whole conversation should COVER all the information regarding the image. If the human responses fail to cover some specific object, attributes or relations in the image, you should ask about it in the subsequent conversations.
5. Ask diverse questions. DO NOT ask any question that cannot be answered from the given image information confidently. ONLY include questions that have definite answers
    a. one can see the content in the image that the question asks about and can answer confidently.
    b. one can determine confidently from the image that it is not in the image.
6. You should end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

You will be given multiple images, and you need to have multiple conversations with the vision-language model. Once you are given the information of the image, you should start the conversation about this image. 
"""

CONV_MODEL_PERSPECTIVE_PROMPT_COCO_ICL_CERTAINTY_COVERAGE = \
"""
Your task is to have a multi-round conversation with a vision-language model regarding a given image. The responses from the vision-language model will be later to used to evaluate whether the model is hallucinating or faithful to the image (wheteher model generate responses contradictory to the content of the image).
You need to have multiple conversations the model over multiple images. Each conversation will be based on one single image. The image will be presented to you as five captions, each describing the same image you are observing, and a list of objects with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
Given this image, you need to perform a series of casual conversations with the vision-language model naturally and ask questions about the given detailed information of the image.
The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
3. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
4. The whole conversation should COVER all the information regarding the image. If the human responses fail to cover some specific object, attributes or relations in the image, you should ask about it in the subsequent conversations.
5. Ask diverse questions. DO NOT ask any question that cannot be answered from the given image information confidently. ONLY include questions that have definite answers
    a. one can see the content in the image that the question asks about and can answer confidently.
    b. one can determine confidently from the image that it is not in the image.
6. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

You will be given multiple images, and you need to have multiple conversations with the vision-language model. Once you are given the information of the image, you should start the conversation about this image. 
"""

CONV_MODEL_PERSPECTIVE_PROMPT_VG_ICL_CERTAINTY = \
"""
Your task is to have a multi-round conversation with a vision-language model regarding a given image. The responses from the vision-language model will be later to used to evaluate whether the model is hallucinating or faithful to the image (wheteher model generate responses contradictory to the content of the image).
You need to have multiple conversations the model over multiple images. Each conversation will be based on one single image. The given image will be presented to you as a list of objects with attributes and relation of these objects. Each objects will be presented with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
Given this image, you need to perform a series of casual conversations with the vision-language model naturally and ask questions about the given detailed information of the image.
The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
3. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
4. Ask diverse questions. DO NOT ask any question that cannot be answered from the given image information confidently. ONLY include questions that have definite answers
    a. one can see the content in the image that the question asks about and can answer confidently.
    b. one can determine confidently from the image that it is not in the image.
5. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

You will be given multiple images, and you need to have multiple conversations with the vision-language model. Once you are given the information of the image, you should start the conversation about this image. 
"""

CONV_MODEL_PERSPECTIVE_PROMPT_COCO_ICL_CERTAINTY = \
"""
Your task is to have a multi-round conversation with a vision-language model regarding a given image. The responses from the vision-language model will be later to used to evaluate whether the model is hallucinating or faithful to the image (wheteher model generate responses contradictory to the content of the image).
You need to have multiple conversations the model over multiple images. Each conversation will be based on one single image. The image will be presented to you as five captions, each describing the same image you are observing, and a list of objects with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
Given this image, you need to perform a series of casual conversations with the vision-language model naturally and ask questions about the given detailed information of the image.
The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
3. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
4. Ask diverse questions. DO NOT ask any question that cannot be answered from the given image information confidently. ONLY include questions that have definite answers
    a. one can see the content in the image that the question asks about and can answer confidently.
    b. one can determine confidently from the image that it is not in the image.
5. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

You will be given multiple images, and you need to have multiple conversations with the vision-language model. Once you are given the information of the image, you should start the conversation about this image. 
"""

CONV_MODEL_PERSPECTIVE_PROMPT_VG_ICL_COVERAGE = \
"""
Your task is to have a multi-round conversation with a vision-language model regarding a given image. The responses from the vision-language model will be later to used to evaluate whether the model is hallucinating or faithful to the image (wheteher model generate responses contradictory to the content of the image).
You need to have multiple conversations the model over multiple images. Each conversation will be based on one single image. The given image will be presented to you as a list of objects with attributes and relation of these objects. Each objects will be presented with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
Given this image, you need to perform a series of casual conversations with the vision-language model naturally and ask questions about the given detailed information of the image.
The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
3. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
4. The whole conversation should COVER all the information regarding the image. If the human responses fail to cover some specific object, attributes or relations in the image, you should ask about it in the subsequent conversations.
5. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

You will be given multiple images, and you need to have multiple conversations with the vision-language model. Once you are given the information of the image, you should start the conversation about this image. 
"""
CONV_MODEL_PERSPECTIVE_PROMPT_VG_ICL_UNANSWERABLE = \
"""
Task: Your task is to have multiple conversations with a vision-language model. Each conversation will be based on one single image. The given image will be presented to you as a list of objects with attributes and relation of these objects. Each objects will be presented with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
Given this image, you need to perform a series of casual conversations with the vision-language model naturally and ask questions about the given detailed information of the image.
The conversation is multi-turn and open-ended. You need to ask questions based on both the image content and the history of the conversations.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should test model's ability to answer unanswerable questions. An unanswerable question refers to the question that the model cannot answer based on the given input image because the queried information is missing, unclear, or speculative.
3. You ask mostly asks the model questions that are unanswerable following the roughly 4 categories: 
(1) Nonexistent objects, attributes and relationships;
(2) background details about objects not depicted in the image;
(3) questions about events or conditions that occurred before or after the moment captured in the image;
(4) missing visual information that are visually unclear, hidden, or blurred in the image.
4. DO NOT ask existence questions such as "Are there any ...?". Directly ask questions of the nonexistent information.
5. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
6. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
7. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 


You will be given multiple images, and you need to have multiple conversations with the vision-language model. Once you are given the information of the image, you should start the conversation about this image. 
"""
CONV_MODEL_PERSPECTIVE_PROMPT_COCO_ICL_UNANSWERABLE = \
"""
Task: Your task is to have multiple conversations with a vision-language model. Each conversation will be based on one single image. The image will be presented to you as five captions, each describing the same image you are observing, and a list of objects with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
Given this image, you need to perform a series of casual conversations with the vision-language model naturally and ask questions about the given detailed information of the image.
The conversation is multi-turn and open-ended. You need to ask questions based on both the image content and the history of the conversations.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should test model's ability to answer unanswerable questions. An unanswerable question refers to the question that the model cannot answer based on the given input image because the queried information is missing, unclear, or speculative.
3. You ask mostly asks the model questions that are unanswerable following the roughly 4 categories: 
(1) Nonexistent objects, attributes and relationships;
(2) background details about objects not depicted in the image;
(3) questions about events or conditions that occurred before or after the moment captured in the image;
(4) missing visual information that are visually unclear, hidden, or blurred in the image.
4. DO NOT ask existence questions such as "Are there any ...?". Directly ask questions of the nonexistent information.
5. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
6. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
7. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 


You will be given multiple images, and you need to have multiple conversations with the vision-language model. Once you are given the information of the image, you should start the conversation about this image. 
"""

CONV_MODEL_PERSPECTIVE_PROMPT_VG_SIMGPLE = \
"""Task: Your task is to have a multi-round conversation with a vision-language model regarding a given image. The image will be presented to you as a list of instances with attributes and relation of these instances. Each instance will be presented with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
3. Do NOT correct the model if it makes mistakes. DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
4. The whole conversation should COVER all the information regarding the image. If the human responses fail to cover some specific object, attributes or relations in the image, you should ask about it in the subsequent conversations.
5. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

Image information:
{}

Please respond as if you are having the conversation with the vision-language model directly.
"""

CONV_MODEL_PERSPECTIVE_PROMPT_VG_UNANSWERABLE = \
"""
Task: Your task is to have multiple conversations with a vision-language model. Each conversation will be based on one single image. The given image will be presented to you as a list of objects with attributes and relation of these objects. Each objects will be presented with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
Given this image, you need to perform a series of casual conversations with the vision-language model naturally and ask questions about the given detailed information of the image.
The conversation is multi-turn and open-ended. You need to ask questions based on both the image content and the history of the conversations.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should test model's ability to answer unanswerable questions. An unanswerable question refers to the question that the model cannot answer based on the given input image because the queried information is missing, unclear, or speculative.
3. You ask mostly asks the model questions that are unanswerable following the roughly 4 categories: 
(1) Nonexistent objects, attributes and relationships;
(2) background details about objects not depicted in the image;
(3) questions about events or conditions that occurred before or after the moment captured in the image;
(4) missing visual information that are visually unclear, hidden, or blurred in the image.
4. DO NOT ask existence questions such as "Are there any ...?". Directly ask questions of the nonexistent information.
5. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
6. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
7. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 
"""

CONV_MODEL_PERSPECTIVE_PROMPT_COCO_UNANSWERABLE = \
"""
Task: Your task is to have multiple conversations with a vision-language model. Each conversation will be based on one single image. The image will be presented to you as five captions, each describing the same image you are observing, and a list of objects with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
Given this image, you need to perform a series of casual conversations with the vision-language model naturally and ask questions about the given detailed information of the image.
The conversation is multi-turn and open-ended. You need to ask questions based on both the image content and the history of the conversations.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should test model's ability to answer unanswerable questions. An unanswerable question refers to the question that the model cannot answer based on the given input image because the queried information is missing, unclear, or speculative.
3. You ask mostly asks the model questions that are unanswerable following the roughly 4 categories: 
(1) Nonexistent objects, attributes and relationships;
(2) background details about objects not depicted in the image;
(3) questions about events or conditions that occurred before or after the moment captured in the image;
(4) missing visual information that are visually unclear, hidden, or blurred in the image.
4. DO NOT ask existence questions such as "Are there any ...?". Directly ask questions of the nonexistent information.
5. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
6. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
7. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 
"""

CONV_MODEL_PERSPECTIVE_PROMPT_VG_SIMGPLE_INTERROGATE = \
"""Task: Your task is to have a multi-round conversation with a vision-language model regarding a given image. The responses from the vision-language model will be later to used to evaluate whether the model is hallucinating or faithful to the image (wheteher model generate responses contradictory to the content of the image).
The image will be presented to you as a list of instances with attributes and relation of these instances. Each instance will be presented with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
3. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
4. The whole conversation should COVER all the information regarding the image. If the human responses fail to cover some specific object, attributes or relations in the image, you should ask about it in the subsequent conversations.
5. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

Image information:
{}

Please respond as if you are having the conversation with the vision-language model directly.
"""

CONV_MODEL_PERSPECTIVE_PROMPT_COCO_SIMGPLE_INTERROGATE = \
"""Task: Your task is to have a multi-round conversation with a vision-language model regarding a given image. The responses from the vision-language model will be later to used to evaluate whether the model is hallucinating or faithful to the image (wheteher model generate responses contradictory to the content of the image).
The image will be presented to you as five captions, each describing the same image you are observing, and a list of objects with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
3. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
4. The whole conversation should COVER all the information regarding the image. If the human responses fail to cover some specific object, attributes or relations in the image, you should ask about it in the subsequent conversations.
5. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

Image information:
{}

Please respond as if you are having the conversation with the vision-language model directly.
"""

CONV_MODEL_PERSPECTIVE_PROMPT_VG_SIMGPLE_INTERROGATEV2 = \
"""Task: Your task is to evaluate whether a vision-language model is hallucinating (i.e. generate responses contradictory to the content of the image) on the image.
The image will be presented to you as a list of instances with attributes and relation of these instances. Each instance will be presented with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
You should perform a series of casual conversations with the model naturally by asking questions or making statements about the image and determine whether the model is hallucinating or not in its response.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
3. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
4. The whole conversation should COVER all the information regarding the image. If the human responses fail to cover some specific object, attributes or relations in the image, you should ask about it in the subsequent conversations.
5. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

Image information:
{}

Please respond as if you are having the conversation with the vision-language model directly.
"""

CONV_MODEL_PERSPECTIVE_PROMPT_VG_SIMGPLE_INTERROGATEV2_NO_COVERAGE = \
"""Task: Your task is to evaluate whether a vision-language model is hallucinating (i.e. generate responses contradictory to the content of the image) on the image.
The image will be presented to you as a list of instances with attributes and relation of these instances. Each instance will be presented with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
You should perform a series of casual conversations with the model naturally by asking questions or making statements about the image and determine whether the model is hallucinating or not in its response.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
3. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
4. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

Image information:
{}

Please respond as if you are having the conversation with the vision-language model directly.
"""

CONV_MODEL_PERSPECTIVE_PROMPT_VG_SIMGPLE_INTERROGATE_CERTAINTY = \
"""Task: Your task is to have a multi-round conversation with a vision-language model regarding a given image. The responses from the vision-language model will be later to used to evaluate whether the model is hallucinating or faithful to the image (wheteher model generate responses contradictory to the content of the image).
The image will be presented to you as a list of instances with attributes and relation of these instances. Each instance will be presented with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
3. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
4. The whole conversation should COVER all the information regarding the image. If the human responses fail to cover some specific object, attributes or relations in the image, you should ask about it in the subsequent conversations.
5. Ask diverse questions. DO NOT ask any question that cannot be answered from the given image information confidently. ONLY include questions that have definite answers
    a. one can see the content in the image that the question asks about and can answer confidently.
    b. one can determine confidently from the image that it is not in the image.
6. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

Image information:
{}

Please respond as if you are having the conversation with the vision-language model directly.
"""


CONV_COVERAGE_PROMPT = \
"""
Task: Your task is to have a conversation with a human regarding the image provided to you. This image will be presented to you as five captions, each describing the same image you are observing, and a list of objects with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
You need to perform a series of casual conversations with a human naturally by asking questions or making statements about the image to assess human's perception of the image (whether the human will hallucinate, provide information contradictory to the image or fail to have knowledge regarding some parts or entities in the image).
The conversation is multi-turn and can be open-ended and you need to ask questions based on the history of the conversations. 

Requirements:
1. The conversation is a multi-turn process and your current response should be based on the history of the conversations.
2. At each round, you should only provide your part of the conversation and wait for the human to respond.
3. You should make the conversation as natural as possible and act as if you are a human having conversation directly with another human.
4. COVERAGE: the whole conversation is expected to COVER all the information regarding the image, you should ask questions that cover as many details as possible. If the human responses fail to cover some specific object, attributes or relations in the image, you should cover this in the subsequent conversations.
5. DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention that the information source is the caption or the bounding box.
6. Focus your conversation on the content of the image. Remember the purpose of the conversation is to assess the human's perception and knowledge of the image.
7. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

Bad examples:
Image information:
"A man is in a kitchen making pizzas.",
"Man in apron standing on front of oven with pans and bakeware",
"A baker is working in the kitchen rolling dough.",
"A person standing by a stove in a kitchen.",
"A table with pies being made and a person standing near a wall with pots and pans hanging on the wall."

You: Are there any smaller items, like a bottle or specific ingredient like broccoli or carrots, you can identify in the scene?
Reason: you should not mention the objects in the image directly. For example, you can ask: Are there any smaller items you can identify in the scene?

You: The organization definitely adds to the efficiency. I also noticed a bottle on the table. What do you think it could contain\u2014perhaps some oil, vinegar, or maybe a seasoning for the dishes?
Reason: This question cannot be answered confidently from the given image information. You should only ask questions that have definite answers.

You: What do you think about the rack or wall with pots and pans hanging around the workspace? Does it contribute to the vibe of the kitchen?
Reason: first, do not mention detailed information from the image directly. Second, this question cannot be answered confidently from the given image information. You should only ask questions that have definite answers.


Image information:
{}

Please respond as if you are having the conversation with the vision-language model directly.
"""


CONTEXT_PROMPT = \
"""Your task is to create a realistic scenario in which the given image is situated in the first-person view, i.e. you should imagine the image depicts your sight at the environment. This context should incorporate a background setting, the characters or objects involved, and a specific goal or objective that is relevant to the image. The context must be plausible, align with real-world experiences, and directly connect with the depicted elements in the image.

The image will be described through a list of objects, their attributes, and their spatial relationships, with each object represented by a set of coordinates in the image. These coordinates (x1, y1, x2, y2) will range from 0 to 1, corresponding to the top-left and bottom-right corners of each object.

Instructions:
1. Contextualization: Develop a background scenario that is logical and directly relevant to the visual elements in the image. The background should describe the setting, time, and possible situation in which these objects or characters might exist.
2. Goal: Identify a specific action, objective, or task that the subject(s) in the image are trying to accomplish, which should be coherent with the scene described.
3. Diversity: For the given image, you should generate several different and diverse contexts.

The response should be a list of dictionaries, where each dictionary represents one possible context for the image. Each dictionary should contain two keys:
Background: A brief description (fewer than 50 words) of the setting, situation, or scenario that fits the image.
goal: A clear description of the goal or task that is being pursued by the objects/subjects in the image.
Some Examples:
```json
[
    {
        "background": "A bustling city street during rush hour, with pedestrians walking past stores and cars honking in traffic. The image depicts the first-person view of the character.",
        "goal": "The character is trying to catch a bus before it leaves."
    },
    {
        "background": "An open-plan corporate office during a busy afternoon, with cubicles neatly separated by dividing screens and personal photos decorating the workspace walls. The image depicts the first-person view of the character.",
        "goal": "The character is trying to send an email to his boss."
    }, 
    {
        "background": "A bright and inviting kitchen featuring wooden cabinetry, a cozy dining area, and fresh fruit adding a vibrant touch. The image depicts the first-person view of the character",
        "goal": "The character is hungry and tries to eat something."
    },
    {
        "background": "A modest bathroom showing signs of wear, featuring basic fixtures, white tiles, and a shower area needing repairs.",
        "goal": "The character just woke up and wanted to wash his face."
    },
    {
        "background": "A modern kitchen with stainless steel appliances, wooden cabinets, and a countertop filled with various cooking utensils. ",
        "goal": "The character is blind and stumbed into the kitchen. He is trying to find his way out of the kitchen."
    }
]
```

"""


CONV_MODEL_PERSPECTIVE_PROMPT_VG_ICL_CERTAINTY_CONTEXT = \
"""
Your task is to have a multi-round conversation with a vision-language model regarding a given image. 
The responses from the vision-language model will be later to used to evaluate whether the model is hallucinating or faithful to the image (wheteher model generate responses contradictory to the content of the image).
You will have multiple rounds of conversations with the model. In each round, you will be give an image and a context (background and goal). Then given image will be presented to you as a list of objects with attributes and relation of these objects. Each objects will be presented with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
Based on the image and context, you need to perform a series of casual conversations with the vision-language model naturally and ask questions about the given detailed information of the image.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image, context and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
3. You should be in favor of asking questions that have high correlation with the given context and AVOID asking questions that are not faithful to your characters and contexts.
4. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
5. You should be in favor of asking open-ended questions rather than binary questions such as "yes / no" questions, which can increase the difficulty of the question and make the model more likely to hallucinate.
6. Ask diverse questions. DO NOT ask any question that cannot be answered from the given image information confidently. ONLY include questions that have definite answers
    a. one can see the content in the image that the question asks about and can answer confidently.
    b. one can determine confidently from the image that it is not in the image.
7. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

You will have multiple rounds of conversations with the model. In each round, you will be give an image and a context (background and goal). 
Once you are given the information of the image and the context, you should start one round of conversation about this image and context right away as if you are having the conversation with the model directly.
"""

CONV_MODEL_PERSPECTIVE_PROMPT_VG_ICL_CERTAINTY_CONTEXT_UNANSWERABLE_ADVERSARIAL = \
"""
Your task is to have a multi-round conversation with a vision-language model regarding a given image. 
The responses from the vision-language model will be later to used to evaluate whether the model is hallucinating or faithful to the image (wheteher model generate responses contradictory to the content of the image).
You will have multiple rounds of conversations with the model. In each round, you will be give an image and a context (background and goal). Then given image will be presented to you as a list of objects with attributes and relation of these objects. Each objects will be presented with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
Based on the image and context, you need to perform a series of casual conversations with the vision-language model naturally and ask questions about the given detailed information of the image.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image, context and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
3. You should be in favor of asking questions that have high correlation with the given context and AVOID asking questions that are not faithful to your characters and contexts.
4. Do NOT correct the model if it makes mistakes. However, you can ask follow-up questions regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
5. You should be in favor of asking open-ended questions rather than binary questions such as "yes / no" questions, which can increase the difficulty of the question and make the model more likely to hallucinate.
6. Ask diverse questions. DO NOT ask any question that cannot be answered from the given image information confidently. ONLY include questions that have definite answers
    a. one can see the content in the image that the question asks about and can answer confidently.
    b. one can determine confidently from the image that it is not in the image.
7. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

There are four types of questions you can ask in each turn of one conversation:
1. Regular questions related to the context and goal. For example, given the context of corporate office and the goal of the conversation is to send an email to his boss, you can ask questions like "is the monitor turned on?", "where is the power button of the computer?", "I want to type in the email, what should I look for?" etc.
2. Follow-up questions that are related to the previous question. It can be a question to confirm the last response or interrogate the model about the last response.
3. Adversarial questions that are asks for the information that is not in the image but highly co-occurs with the content of the image. For instance, if the image depicts a cake on the table without any utensil, you can ask "Is there a folk?", "can I cut the cake with the knife on the table?" etc, where the "folk" and "knife" are not in the image but highly co-occurs with the cake.
4. Unanswerable questions asks question that cannot be answered. For instance, if the image depicts a cake on the table without any utensil or people eating the cake, you can ask "What utensil is the man using to cut the cake?". This is an unanswerable question because you cannot answer with an utensil but rather you should answer "There are not any man in the image eating the cake".

In each round of conservation, you can ask many of the above four types of questions. You are NOT confined to ask one type of question. But you should proceed the conservation as natural as possible.

You will have multiple rounds of conversations with the model. In each round, you will be give an image and a context (background and goal). 
Once you are given the information of the image and the context, you should start one round of conversation about this image and context right away as if you are having the conversation with the model directly.
"""

CONV_MODEL_PERSPECTIVE_PROMPT_VG_ICL_CERTAINTY_CONTEXT_UNANSWERABLE_ADVERSARIA_REVISED = \
"""Your task is to **simulate a multi-round conversation with a vision-language model (VLM) about a given image**, in order to test whether the model hallucinates (i.e., produces responses inconsistent with the image) or remains faithful to it.

You will have multiple rounds of conversations with the model. In each round, you will be provided with:

* **An image** (represented as a list of objects, their attributes, and relationships).
* **Bounding-box coordinates** for each object, given as `(x1, y1, x2, y2)` in normalized values between 0 and 1, corresponding to top-left and bottom-right corners.
* **A context** (background setting and conversational goal) which this conservation situates in. The background describes the setting and situates the image, while the goal specifies the task or objective you, as the conversation initiator, are pursuing.

Your role is to **carry out a natural, open-ended, human-like conversation** with the model, asking questions about the image in the gievn context.

---

### Requirements

1. **Multi-turn conversation**:

   * Ask one question per turn, then wait for the model’s response.
   * Incorporate both the image, context, and dialogue history when forming your question.

2. **Natural, human-like tone**:

   * Speak as if conversing casually with another person.
   * Avoid mechanical or scripted phrasing.

3. **Context relevance**:

   * Ask questions strongly reated to the given context and image.
   * Avoid irrelevant or out-of-character questions.

4. **No disclosure of metadata**:

   * Do not reveal or reference bounding boxes, object lists, captions, or the source of your information.
   * Ask question as if you are looking at the image with NO access to the bounding boxes, object lists, captions.

5. **Handling mistakes**:

   * Do not correct the model if they make mistakes. 
   * However, you may ask follow-ups to probe or interrogate its response.

6. **Question style**:

   * Prefer open-ended questions over yes/no questions.
   * Ask diverse, non-redundant questions.
   * Only ask questions with **definite answers** (either clearly present in the image or confidently absent).

7. **Conversation ending**:

   * If the dialogue has naturally run its course, output **“END”** (and nothing else).

---

### Types of Questions

You may ask any mix of the following types of questions:

1. **Regular questions** – directly related to the image and context.
   *Example*: In an office setting, the goal of the conversation is to send an email to his boss. you can ask questions like "is the monitor turned on?", "where is the power button of the computer?", "I want to type in the email, what should I look for?" etc.

2. **Follow-up questions** – follow up, confirm or interrogate the model’s last response.

3. **Adversarial questions** – inquire about plausible but absent object, attribute or relations that commonly co-occur with visible ones in the image.
   *Example*: If the image shows a cake without utensils, you may ask: *“Can I use the knife on the table to cut the cake?”* or *"Is there a folk?"*

4. **Unanswerable questions** – ask about question that cannot be answered.
   *Example*: if the image depicts a cake on the table without any utensil or people eating the cake, you can ask "What utensil is the man using to cut the cake?". This is an unanswerable question because you cannot answer with an utensil but rather you should answer "There are not any man in the image eating the cake".

---

### Execution

For each round of conversation:

* Begin immediately after receiving the image description and context.
* Proceed as if you are directly conversing with the model.
"""