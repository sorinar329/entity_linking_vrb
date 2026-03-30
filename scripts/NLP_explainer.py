import ollama
from IPython.core.display_functions import display
from PIL import Image, ImageDraw, ImageFont
import re
import requests

def filter_think_part(response):
    """
    Remove the <think> part from the OpenLlama response.

    Args:
        response (str): The OpenLlama response containing <think> tags.

    Returns:
        str: The response with the <think> part removed.
    """
    # Use regex to remove text enclosed in <think>...</think>
    filtered_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    filtered_response = filtered_response.replace("**", "")  # Remove any remaining asterisks
    return filtered_response.strip()

def generate_explanation(content,llm_model=None, show_image=False):
    """
    Generate a natural language explanation based on the provided content.

    Args:
        show_image: bool: Whether to show the image with the explanation. Defaults to False.
        llmmodel: The model to use for generating the explanation. Defaults to "deepseek-r1:7b" if None.
        content (str): The input content to be explained.

    Returns:
        str: The generated explanation.
    """
    if llm_model is None:
        llm_model = "deepseek-r1:7b"

    url = "http://192.168.200.10:11434/api/chat"

    headers = {
        "Content-Type": "application/json",
        "Authorization": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImJiMjYzNjU0LTk0NTgtNDgxZC1iN2IzLTgwNDNkOTBjZjllMiJ9.BdRhAJxbzN_tVcs_9xfd2ldJIprX-qD4KDwuyaKaGnQ"
        # If required
    }
    # Define the system message to set the personality
    system_message = {
        'role': 'system',
        'content': 'You are a friendly and helpful assistant robot who loves to provide explanations in '
                   'natural Language format without showing any code. Your adressee is a human user who wants to understand.'
                   ' You should always provide a clear and concise explanation based on the input provided.'
                    'Dont use any code or bullet points, just a fluid text.'

    }

    # Define the user message
    user_message = {
        'role': 'user',
        'content': f'Generate an natural language explanation based on this input: {content}, keep it simple and easy to understand. '
                   f'Make a fluid text without any bold text or code! (AVOID using code like "can_be_cut=True" or "obo.FOODON0000xxx" '
                   f'or bullet points.'
    }
    payload = {
        "messages": [user_message],
        "model": f"{llm_model}",  # Adjust to your model
        "stream": False  # If the API supports streaming

    }

    print()
    # Send the chat request with the system message and user message
    #response = ollama.chat(model='deepseek-r1:14b', messages=[system_message, user_message])
    response = requests.post(url, headers=headers, json=payload)
    if response.ok:
        response = response.json()
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")
    # Return the response
    return filter_think_part(response['message']['content'])

def provide_explanation_with_image(content, img_path):

    # Load image
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=34)
    # font size should be bigger
    #font = ImageFont.truetype("arial.ttf", 34)  # Adjust the font size as needed


    # Bounding box coordinates

    x1, y1, x2, y2 = 60.0, 150.0, 1020.0, 560.0  # Example coordinates, replace with actual values
    # Ensure correct rectangle orientation
    left, top = min(x1, x2), min(y1, y2)
    right, bottom = max(x1, x2), max(y1, y2)

    # Draw filled rectangle for textbox background
    # draw.rectangle((left, top, right, bottom), fill="white", outline="blue", width=2)

    # Calculate text position (with padding)
    padding = 5
    text_x = left + padding
    text_y = top + padding
    box_width = right - left - 2 * padding

    # Word-wrapping
    words = content.split()
    lines = []
    line = ""
    for word in words:
        test_line = line + (" " if line else "") + word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        w = bbox[2] - bbox[0]
        if w <= box_width:
            line = test_line
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        h = bbox[3] - bbox[1]
        draw.text((text_x, text_y), line, fill="black", font=font)
        text_y += h + 5

    display(image)