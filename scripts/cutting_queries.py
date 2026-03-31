import pandas as pd
from IPython.display import display, HTML

from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("https://knowledgedb.informatik.uni-bremen.de/mealprepDB/MealPreparation/query")
sparql.setReturnFormat(JSON)

prefix = """
 PREFIX owl: <http://www.w3.org/2002/07/owl#>
 PREFIX cut: <http://www.ease-crc.org/ont/meals#>
 PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
 PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
 PREFIX foodon: <http://purl.obolibrary.org/obo/>
 PREFIX soma: <http://www.ease-crc.org/ont/SOMA.owl#>
 PREFIX sit_aware: <http://www.ease-crc.org/ont/situation_awareness#>
 """

def check_food_part(food, part):
    query = """
    ASK {
					  foodon:%s rdfs:subClassOf* ?dis.
					  ?dis owl:onProperty cut:hasPart.
					  ?dis owl:someValuesFrom ?tar.
					  ?tar owl:intersectionOf ?tar_int.
					  ?tar_int rdf:first cut:%s.
					  ?tar_int rdf:rest ?rest.
					  ?rest rdf:first ?first.
					  ?first owl:onProperty cut:hasEdibility.
					  {
						?first owl:someValuesFrom cut:MustBeAvoided.
					  }
					  UNION
					  {
						?first owl:someValuesFrom cut:ShouldBeAvoided.
					  }
					}
    """% (food, part)
    full_query = (prefix + query)
    sparql.setQuery(full_query)
    results = sparql.queryAndConvert()
    return results["boolean"]

def get_prior_task(verb):
    query = """
    SELECT ?res WHERE {
						%s rdfs:subClassOf* ?sub.
						?sub owl:onProperty cut:requiresPriorTask .
						?sub owl:someValuesFrom ?priortask.
						BIND(REPLACE(STR(?priortask), "^.*[#/]", "") AS ?res).
					}
					""" % (verb)
    full_query = (prefix + query)
    sparql.setQuery(prefix + query)
    results = sparql.queryAndConvert()
    return results["results"]["bindings"][0]["res"]["value"] if results["results"]["bindings"] else None

def get_cutting_tool(foodobject):
    query = """
    SELECT ?res WHERE {
							foodon:%s rdfs:subClassOf* ?peel_dis.
							?peel_dis owl:onProperty soma:hasDisposition.
							?peel_dis owl:someValuesFrom ?peel_dis_vals.
							?peel_dis_vals owl:intersectionOf ?afford_vals.
							?afford_vals rdf:first sit_aware:Cuttability.
							?afford_vals rdf:rest ?task_trigger.
							?task_trigger rdf:rest ?trigger.
							?trigger rdf:first ?trigger_wo_nil.
							?trigger_wo_nil owl:onProperty soma:affordsTrigger.
							?trigger_wo_nil owl:allValuesFrom ?trigger_tool.
							?trigger_tool owl:allValuesFrom ?tool.
							BIND(REPLACE(STR(?tool), "^.*[#/]", "") AS ?res).
						}
    """ % (foodobject)
    full_query = (prefix + query)
    sparql.setQuery(prefix + query)
    results = sparql.queryAndConvert()
    return results["results"]["bindings"][0]["res"]["value"] if results["results"]["bindings"] else "Knife"

def get_cutting_position(verb):
    query = """
    SELECT * WHERE {
						%s rdfs:subClassOf* ?pos_node.
						?pos_node owl:onProperty cut:affordsPosition.
						?pos_node owl:someValuesFrom ?pos.
						BIND(REPLACE(STR(?pos), "^.*[#/]", "") AS ?res).
					}
    """ % (verb)
    full_query = (prefix + query)
    sparql.setQuery(prefix + query)
    results = sparql.queryAndConvert()
    return results["results"]["bindings"][0]["res"]["value"] if results["results"]["bindings"] else "middle"

def get_repetition(verb):
    query = """
    SELECT ?res WHERE {
					  {
						%s rdfs:subClassOf* ?rep_node.
						?rep_node owl:onProperty cut:repetitions.
						FILTER EXISTS {
							?rep_node owl:hasValue ?val.
						}
						BIND("exactly 1" AS ?res)
					  }
					  UNION
					  {
						%s rdfs:subClassOf* ?rep_node.
						?rep_node owl:onProperty cut:repetitions.
						FILTER EXISTS {
							?rep_node owl:minQualifiedCardinality ?val.
						}
						BIND("at least 1" AS ?res)
					  }
					}
    """ % (verb, verb)
    full_query = (prefix + query)
    sparql.setQuery(prefix + query)
    results = sparql.queryAndConvert()
    return results["results"]["bindings"][0]["res"]["value"] if results["results"]["bindings"] else "1"

def get_peel_tool(foodobject):
    query = """
    SELECT ?res WHERE {
						foodon:%s rdfs:subClassOf* ?peel_dis.
						?peel_dis owl:onProperty soma:hasDisposition.
						?peel_dis owl:someValuesFrom ?peel_dis_vals.
						?peel_dis_vals owl:intersectionOf ?afford_vals.
						?afford_vals rdf:first cut:Peelability.
						?afford_vals rdf:rest ?task_trigger.
						?task_trigger rdf:rest ?trigger.
						?trigger rdf:first ?trigger_wo_nil.
						?trigger_wo_nil owl:onProperty soma:affordsTrigger.
						?trigger_wo_nil owl:allValuesFrom ?trigger_tool.
						?trigger_tool owl:allValuesFrom ?tool.
						BIND(REPLACE(STR(?tool), "^.*[#/]", "") AS ?res).
					}
    """ % (foodobject)
    full_query = (prefix + query)
    sparql.setQuery(prefix + query)
    results = sparql.queryAndConvert()
    return results["results"]["bindings"][0]["res"]["value"] if results["results"]["bindings"] else "Peeler"

def query_var(verb, foodobject):
    priorTask = get_prior_task(verb)
    cut_tool = get_cutting_tool(foodobject)
    cut_position = get_cutting_position(verb)
    repetition = get_repetition(verb)
    remove_peel = check_food_part(foodobject, "Peel")
    remove_core = check_food_part(foodobject, "Core")
    remove_stem = check_food_part(foodobject, "Stem")
    remove_shell = check_food_part(foodobject, "Shell")


    print(f"For {verb} on {foodobject}, the prior task is: {priorTask}, and the cutting tool is: {cut_tool}, and the cutting position is: {cut_position},"
          f" and the repetition is: {repetition}")

    print(f"Remove peel: {remove_peel}, Remove core: {remove_core}, Remove stem: {remove_stem}, Remove shell: {remove_shell}")

    if remove_peel or remove_shell:
        peeling_tool = get_peel_tool(foodobject)
        print(f"Peeling tool: {peeling_tool}")
    """
    Bei Stem und Core wird einfach "Remove Core/Stem" ausgegeben, aber es gibt keine Abfrage für den Tool
    """


def build_motion_table(selected_food,
                       selected_verb):


    selected_food_namespace = str(selected_food.namespace).split("obo.")[1]
    prior_task = get_prior_task(selected_verb),
    peel_tool = get_peel_tool(selected_food_namespace),
    cut_tool = get_cutting_tool(selected_food_namespace),
    reps = get_repetition(selected_verb),
    pos = get_cutting_position(selected_verb),
    shape = str(selected_verb)
    steps = []
    curr_step = 1

    def add_step(number, motion, inference):
        steps.append({"#": number, "Motions": motion, "Inference": inference})

    # Optional preconditions
    if selected_food.peel_should_be_removed or selected_food.peel_must_be_removed:
        add_step(curr_step, "Peeling using a peeling tool", f"has peel = true<br>peeling tool = {peel_tool}")
        curr_step += 1
    #if selected_food.shell_should_be_removed or selected_food.shell_must_be_removed:
    #    add_step(curr_step, "Removing the shell", "has shell = true")
    #    curr_step += 1

    if selected_food.stem_should_be_removed or selected_food.stem_must_be_removed:
        add_step(curr_step, "Removing the stem", "has stem = true")
        curr_step += 1
    if selected_food.core_should_be_removed or selected_food.core_must_be_removed:
        add_step(curr_step, "Removing the core", "has core = true")
        curr_step += 1
    if prior_task:
        add_step(curr_step, f"Execute a prior task: {prior_task}", f"has prior task = true<br>prior task = {prior_task}")
        curr_step += 1

    # Pick up tool
    add_step(curr_step, "Picking up the cutting tool by ...", f"cutting tool = {cut_tool}")
    add_step(f"{curr_step}a", "... approaching the cutting tool", f"cutting tool = {cut_tool}")
    add_step(f"{curr_step}b", "... grasping the cutting tool", f"cutting tool = {cut_tool}")
    add_step(f"{curr_step}c", "... lifting the cutting tool", f"cutting tool = {cut_tool}")
    curr_step += 1

    # Decide target
    target = selected_food.name if shape == "Food" else f"{selected_food.name} {shape}"

    # Cutting motions
    add_step(curr_step, "Cutting the target object at the position n time(s) by ...",
             f"target object = {target}<br>position = {pos}<br>n = {reps}")
    add_step(f"{curr_step}a", "... approaching the target object at the position",
             f"target object = {target}<br>position = {pos}")
    add_step(f"{curr_step}b", "... lowering the cutting tool through the target object",
             f"cutting tool = {cut_tool}<br>target object = {target}")
    add_step(f"{curr_step}c", "... lifting the cutting tool", f"cutting tool = {cut_tool}")
    curr_step += 1

    # Placing down tool
    add_step(curr_step, "Placing the cutting tool down by ...", f"cutting tool = {cut_tool}")
    add_step(f"{curr_step}a", "... approaching the target location", "")
    add_step(f"{curr_step}b", "... releasing the cutting tool", f"cutting tool = {cut_tool}")
    add_step(f"{curr_step}c", "... lifting the empty gripper", "")

    df = pd.DataFrame(steps)
    display(df)

    return df



verb = "cut:Quartering"
foodobject = "FOODON_00003523"
query_var(verb, foodobject)