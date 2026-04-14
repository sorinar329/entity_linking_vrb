from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
import clip
import open_clip
import textwrap
from PIL import ImageFont, ImageDraw
from PIL import Image as PILImage
import cv2
from ipywidgets import HTML, VBox, Image
from ipyevents import Event
from IPython.display import display


def detect_objects(textprompt, image_path, threshold=0.5):
    # Set the model and device
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # Load the image from the path
    image = PILImage.open(image_path)

    # Convert image to RGB if not already in RGB format
    image = image.convert("RGB")

    # Prepare inputs with text and image
    inputs = processor(images=image, text=textprompt, return_tensors="pt", padding=True, truncation=True).to(device)

    # Get outputs from the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Process results and apply thresholds
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=threshold,
        text_threshold=threshold,
        target_sizes=[image.size[::-1]]
    )

    # Extract boxes, scores, and labels
    boxes = results[0]["boxes"]
    scores = results[0]["scores"]
    labels = results[0]["labels"]

    # Prepare to draw on the image
    draw = ImageDraw.Draw(image)

    # List of text queries
    text_queries = textprompt.split(". ")

    # Draw bounding boxes and labels
    for box, score, label in zip(boxes, scores, labels):
        xmin, ymin, xmax, ymax = box.tolist()

        # If label is a string, no need for `.item()`
        if isinstance(label, torch.Tensor):
            label_text = f"{text_queries[label.item()]}: {score:.2f}"
        else:
            label_text = f"{label}: {score:.2f}"

        # Draw the bounding box
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

        # Draw the label and score
        draw.text((xmin, ymin), label_text, fill="red")

    # Show the image with bounding boxes
    plt.figure(figsize=(10, 10))
    #plt.imshow(image)
    plt.axis('off')
    plt.title('Image with Bounding Boxes')
    plt.show()

    # Print results as well
    return results


def run_clip_on_bboxes(img_path, bbox, prompts, show_results=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    # to Copilot: is there a better model then ViT-B/32 for this task?
    # Convert image to PIL and to device


    # Encode text prompts
    text_prompts = list(prompts.values())
    text_inputs = clip.tokenize(text_prompts).to(device)

    image = PILImage.open(img_path)
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox.tolist(), outline="red", width=2)

    cropped_image = image.crop(bbox.tolist())
    cropped_image_tensor = preprocess(cropped_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(cropped_image_tensor)
        text_features = model.encode_text(text_inputs)
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute similarity between image features and text features
        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Map predictions back to class names
    best_match_idx = similarities[0].argmax().item()
    best_match = text_prompts[best_match_idx]
    best_key = list(prompts.keys())[best_match_idx]

    sorted_similarities, sorted_indices = torch.sort(similarities[0], descending=True)
    sorted_labels = [list(prompts.values())[i] for i in sorted_indices.tolist()]
    sorted_scores = sorted_similarities.tolist()

    if show_results:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))

                # Display the image with bounding boxes
        ax[0].imshow(cropped_image)
        ax[0].axis('off')
        ax[0].set_title('Image with Bounding Boxes')
        # Display the bar plot
        ax[1].barh(sorted_labels, sorted_scores, color='skyblue')
        ax[1].set_xlabel('Similarity Score')
        ax[1].set_ylabel('Class Labels')
        ax[1].set_title(f"Zero-Shot Image Classification - {cropped_image}")

                # Annotating bars with their scores
        for i, (score, label) in enumerate(zip(sorted_scores, sorted_labels)):
            ax[1].text(score + 0.01, i, f"{score:.2f}", va='center')

    #if show_results:
     #   plt.tight_layout()
      #  plt.show()

    return best_key, best_match

import matplotlib.image as mpimg


def show_click_coordinates(image_path, num_clicks=0):
    coords = []

    img = mpimg.imread(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Click to mark points (close plot when done)")

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            coords.append((int(event.xdata), int(event.ydata)))
            ax.plot(event.xdata, event.ydata, 'ro')
            fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Manually block execution until the plot is closed
    plt.show(block=True)  # Blocking until the window is closed

    fig.canvas.mpl_disconnect(cid)
    return coords

last_click = None
def show_clickable_image(image_path):
    global last_click

    with open(image_path, 'rb') as f:
        value = f.read()

    image = Image(value=value, format='png')
    image.layout.max_width = '100%'
    image.layout.height = 'auto'

    coords_display = HTML("<h3>Click the image to continue...</h3>")
    container = VBox([image, coords_display])
    display(container)

    # Event handler
    def handle_click(event):
        global last_click
        last_click = (event['dataX'], event['dataY'])
        coords_display.value = f"<h3>Clicked at {last_click}</h3>"

    # Setup click event
    im_events = Event(source=image, watched_events=['click'])
    im_events.on_dom_event(handle_click)

class ObjectDetectionResult:
    def __init__(self, bounding_box, predicted_class, predicted_label, ontology_instance):
        self.bounding_box = bounding_box
        self.predicted_class = predicted_class
        self.predicted_label = predicted_label
        self.ontology_instance = ontology_instance
        self.ontology_concept = None

    def __repr__(self):
        return (f"ObjectDetectionResult(bounding_box={self.bounding_box}, "
                f"predicted_class={self.predicted_class}, "
                f"predicted_label={self.predicted_label}, "
                f"ontology_instance={self.ontology_instance},)"
                f"ontology_concept={self.ontology_concept})")

    def add_semantic_annotations(self, ontology):
        # Add semantic annotations based on the ontology restrictions
        restrictions = ontology[self.predicted_class].is_a
        for restriction in restrictions:
            self.ontology_instance.is_a.append(restriction)

    def draw_annotations(self, image):
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", 40)  # Set font and size
        xmin, ymin, xmax, ymax = self.bounding_box.tolist()
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        label_text = f"{self.predicted_label}: {self.predicted_class}"
        wrapped_label_text = textwrap.fill(label_text, width=100)

        # Calculate text size using textbbox
        text_bbox = draw.textbbox((xmin, ymin), wrapped_label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Draw semi-transparent background rectangle
        shadow_color = (255, 255, 255, 51)  # 20% visibility (255 * 0.2 = 51)
        draw.rectangle([xmin, ymin - text_height, xmin + text_width, ymin], fill=shadow_color)

        # Draw text on top of the background
        draw.text((xmin, ymin - text_height), wrapped_label_text, fill="black", font=font)
        # FOR COPILOT: The text should be on top of the bounding box
        # for restriction in self.ontology_instance.is_a:
        #     # Remove prefixes from the restriction text
        #     restriction_text = str(restriction).split("#")[-1]
        #     wrapped_restriction_text = textwrap.fill(restriction_text, width=100)
        #     ymin += 20
        #     draw.text((xmin, ymin), wrapped_restriction_text, fill="blue", font=font)