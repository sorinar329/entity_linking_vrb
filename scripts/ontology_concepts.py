import owlready2
from PIL import Image, ImageDraw
import torch
from owlready2 import locstr, sync_reasoner
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
import clip
import os
# Dynamically get path to project root (assumes src is inside the root)
this_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(this_dir, ".."))

# Build the absolute path to the ontology
ontology_path = os.path.join(project_root, "models", "ontologies", "meals.owl")

def load_ontology():
    onto = owlready2.get_ontology(ontology_path).load()

    SOMA = onto.get_namespace("http://www.ease-crc.org/ont/SOMA.owl#")
    CUT2 = onto.get_namespace("http://www.ease-crc.org/ont/situation_awareness#")
    CUT = onto.get_namespace("http://www.ease-crc.org/ont/food_cutting#")
    DUL = onto.get_namespace("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#")
    OBO = onto.get_namespace("http://purl.obolibrary.org/obo/")
    MEALS = onto.get_namespace("http://www.ease-crc.org/ont/meals#")

def get_all_leaf_subclasses(owl_class):
    """
    Recursively find all leaf subclasses of a given OWL class.

    :param owl_class: The OWL class to start the search from.
    :return: A dictionary with leaf subclasses as keys and their labels (or the class itself if no labels) as values.
    """
    leaf_classes = {}
    for subclass in owl_class.subclasses():
        # If the subclass has no subclasses, it's a leaf
        if not list(subclass.subclasses()):
            # Use label if available; otherwise, use the class itself
            leaf_classes[subclass] = subclass.label[0] if subclass.label else subclass
        else:
            # Recursively check its subclasses
            leaf_classes.update(get_all_leaf_subclasses(subclass))
    return leaf_classes

def convert_leaf_subclasses(leaf_classes):
    """
    Convert the dictionary of leaf subclasses to the desired format.

    :param leaf_classes: Dictionary with leaf subclasses as keys and their labels as values.
    :return: Dictionary with string keys and values.
    """
    def extract_string(value):
        # If value is a locstr object, get the string value, otherwise return it as is
        if isinstance(value, locstr):
            return str(value)  # Ensure locstr is converted to a string
        return str(value)  # Just return the value as string if it's already a string

    return {str(key): extract_string(value) for key, value in leaf_classes.items()}


class food_concept:
    def __init__(self, namespace):
        self.name = None
        self.namespace = namespace

        self.can_be_cut = False

        self.core_can_be_removed = False
        self.stem_can_be_removed = False
        self.peel_can_be_removed = False

        self.edible_core = False
        self.edible_peel = False
        self.edible_stem = False

        self.peel_should_be_removed = False
        self.core_should_be_removed = False
        self.stem_should_be_removed = False

        self.peel_must_be_removed = False
        self.core_must_be_removed = False
        self.stem_must_be_removed = False

    def to_dict(self):
        return {
            'name': self.name,
            'namespace': self.namespace,
            'can_be_cut': self.can_be_cut,
            'core_can_be_removed': self.core_can_be_removed,
            'stem_can_be_removed': self.stem_can_be_removed,
            'peel_can_be_removed': self.peel_can_be_removed,
            'edible_core': self.edible_core,
            'edible_peel': self.edible_peel,
            'edible_stem': self.edible_stem,
            'peel_should_be_removed': self.peel_should_be_removed,
            'core_should_be_removed': self.core_should_be_removed,
            'stem_should_be_removed': self.stem_should_be_removed,
            'peel_must_be_removed': self.peel_must_be_removed,
            'core_must_be_removed': self.core_must_be_removed,
            'stem_must_be_removed': self.stem_must_be_removed
        }

    def __repr__(self):
        return (
            f"Food(\n"
            f"  name={self.name},\n"
            f"  namespace={self.namespace},\n"
            f"  can_be_cut={self.can_be_cut},\n"
            f"  core_can_be_removed={self.core_can_be_removed},\n"
            f"  stem_can_be_removed={self.stem_can_be_removed},\n"
            f"  peel_can_be_removed={self.peel_can_be_removed},\n"
            f"  edible_core={self.edible_core},\n"
            f"  edible_peel={self.edible_peel},\n"
            f"  edible_stem={self.edible_stem},\n"
            f"  peel_should_be_removed={self.peel_should_be_removed},\n"
            f"  core_should_be_removed={self.core_should_be_removed},\n"
            f"  stem_should_be_removed={self.stem_should_be_removed},\n"
            f"  peel_must_be_removed={self.peel_must_be_removed},\n"
            f"  core_must_be_removed={self.core_must_be_removed},\n"
            f"  stem_must_be_removed={self.stem_must_be_removed}\n"
            f")"
        )

def get_food_concept(owl_class):
    onto = owlready2.get_ontology(ontology_path).load()

    SOMA = onto.get_namespace("http://www.ease-crc.org/ont/SOMA.owl#")
    CUT2 = onto.get_namespace("http://www.ease-crc.org/ont/situation_awareness#")
    CUT = onto.get_namespace("http://www.ease-crc.org/ont/food_cutting#")
    DUL = onto.get_namespace("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#")
    OBO = onto.get_namespace("http://purl.obolibrary.org/obo/")
    MEALS = onto.get_namespace("http://www.ease-crc.org/ont/meals#")
    # Create an instance of the food_concept class
    food = food_concept(owl_class)
    if owl_class.label:
        # Set the name of the food concept to its label if available
        # Use the first label if there are multiple

        food.name = owl_class.label[0]
    else:
        # If no label is available, use the class name
        food.name = owl_class.name
    attributes = [
        ('can_be_cut', MEALS.Cutable),
        ('peel_can_be_removed', MEALS.Peelable),
        ('core_can_be_removed', MEALS.CoreRemovable),
        ('stem_can_be_removed', MEALS.StemRemovable),
        ('edible_core', MEALS.CoreEdible),
        ('edible_peel', MEALS.PeelEdible),
        ('edible_stem', MEALS.StemEdible),
        ('peel_should_be_removed', MEALS.PeelShouldBeRemoved),
        ('core_should_be_removed', MEALS.CoreShouldBeRemoved),
        ('stem_should_be_removed', MEALS.StemShouldBeRemoved),
        ('peel_must_be_removed', MEALS.PeelMustBeRemoved),
        ('core_must_be_removed', MEALS.CoreMustBeRemoved),
        ('stem_must_be_removed', MEALS.StemMustBeRemoved)
    ]

    for attr, meal_class in attributes:
        # Check if the class or any of its ancestors have the attribute
        if any(ancestor in meal_class.subclasses() for ancestor in owl_class.ancestors()):
            setattr(food, attr, True)

    return food
