import scripts.detection_concepts as detection_concepts
import scripts.ontology_concepts as ontology_concepts

from owlready2 import *

# Dynamically get path to project root (assumes src is inside the root)
this_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(this_dir, ".."))
pr2_imagepth = os.path.join(project_root, "models", "images", "pr2_speaking.png")
# Build the absolute path to the ontology
ontology_path = os.path.join(project_root, "models", "ontologies", "meals.owl")
onto = get_ontology(ontology_path).load()


SOMA = onto.get_namespace("http://www.ease-crc.org/ont/SOMA.owl#")
CUT2 = onto.get_namespace("http://www.ease-crc.org/ont/situation_awareness#")
CUT = onto.get_namespace("http://www.ease-crc.org/ont/food_cutting#")
DUL = onto.get_namespace("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#")
OBO = onto.get_namespace("http://purl.obolibrary.org/obo/")
MEALS = onto.get_namespace("http://www.ease-crc.org/ont/meals#")
with onto:
    owlready2.sync_reasoner_pellet()

leaf_classes = ontology_concepts.get_all_leaf_subclasses(MEALS.Food)

def get_prompts():
    return ontology_concepts.convert_leaf_subclasses(leaf_classes)


def get_bboxes(img_path):
   return detection_concepts.detect_objects("food.", img_path, threshold = 0.4)


def get_detection_results(img_path, prompts):
    # Create a list of food_concepts
    food_concepts = [ontology_concepts.get_food_concept(leaf_class) for leaf_class in leaf_classes]

    # Create a dictionary to map class labels to food concepts with the 'obo.' prefix
    food_concept_dict = {f"{food.namespace}": food for food in food_concepts}

    # Update the detection results with the corresponding food concepts
    detection_results = []
    for i in get_bboxes(img_path):
        for b in i["boxes"]:
            classified_class, classified_class_label = detection_concepts.run_clip_on_bboxes(img_path, bbox=b,
                                                                                             prompts=prompts,
                                                                                             show_results=False)
            if "obo." in classified_class:
                classified_class = str(classified_class).replace("obo.", "")
                classified_instance = OBO[classified_class](classified_class_label)

            if "SOMA." in classified_class:
                classified_class = str(classified_class).replace("SOMA.", "")
                classified_instance = SOMA[classified_class](classified_class_label)


            detection_result = detection_concepts.ObjectDetectionResult(
                bounding_box=b,
                predicted_class=classified_class,
                predicted_label=classified_class_label,
                ontology_instance=classified_instance
            )

            # Add the corresponding food concept to the detection result
            if f"obo.{classified_class}" in food_concept_dict:
                detection_result.ontology_concept = food_concept_dict[f"obo.{classified_class}"]
                detection_result.add_semantic_annotations(OBO)
            if f"SOMA.{classified_class}" in food_concept_dict:
                detection_result.ontology_concept = food_concept_dict[f"SOMA.{classified_class}"]
                detection_result.add_semantic_annotations(SOMA)
            # else:
            # print(f"Classified class {classified_class} not found in food_concept_dict")

            # detection_result.add_semantic_annotations(OBO)
            detection_results.append(detection_result)

    return detection_results

#
def get_clicked_obj(img_path):
    detection_results = get_detection_results(img_path, get_prompts())
    #clicked_coords = detection_concepts.show_click_coordinates(img_path)
    clicked_coords = detection_concepts.last_click
    clicked_obj = []
    for i in detection_results:
        x1, y1, x2, y2 = i.bounding_box.tolist()
        #for coord in clicked_coords:
        x, y = clicked_coords
        if x1 < x < x2 and y1 < y < y2:
            print(
                    f"Coordinate {clicked_coords} is inside the bounding box {i.bounding_box}: (label: {i.ontology_concept.name})")
            clicked_obj.append(i.ontology_concept)

    if not clicked_obj:
        print("No object clicked or no object detected in the image.")

    if len(clicked_obj) > 1:
        raise ValueError("Multiple objects detected at the clicked coordinates. Please ensure only one object is clicked.")


    return clicked_obj


def provide_explanation(clicked_obj, llm_model=None):
    explanation = NLP_explainer.generate_explanation(clicked_obj[0], llm_model=llm_model)

    return NLP_explainer.provide_explanation_with_image(explanation,
                                                        pr2_imagepth)
