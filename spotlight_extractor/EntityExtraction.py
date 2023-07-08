import matplotlib.pyplot as plt
import pandas as pd
import json, os
import sys
import networkx as nx
import time


# We want to also keep #hashtags as a token, so we modify the spaCy model's token_match
import re
# Retrieve the default token-matching regex pattern
#re_token_match = spacy.tokenizer._get_regex_pattern(nlp_spacy.Defaults.token_match)
# Add #hashtag pattern
#re_token_match = f"({re_token_match}|#\\w+)"
#nlp_spacy.tokenizer.token_match = re.compile(re_token_match).match

punctuations = ['-','#','.',',',':','!',';','£','$','%','&','?','*','_','@','|','>','<','°','§','/','rt','=']

#from news_parser import *
import time

news_id = "1"
listing_dir = "D:/GitRepos/GitRepos/skg/data-collection/twitter/output_spacy_annotations"
"""
nsubj;obj
acl:relcl;obj
acl;obj
nsubj:pass;obl
nsubj;obj;conj
"""


target_path1 = ['nsubj','dobj']       # nsubj;obj in UD
target_path2 = ['acl','relcl','dobj'] #acl:relcl;obj
target_path3 = ['acl','dobj']         #acl;obj
target_path4 = ['nsubjpass','agent','pobj']  #nsubj:pass;obl
target_path5 = ['nsubj','dobj','conj'] #nsubj;obj;conj
target_path6 = ['nsubj','conj'] #nsubj;obj;conj
target_paths = [target_path1,target_path2,target_path3,target_path4,target_path5,target_path6]

def build_graph_spacy(tokens):
    g = nx.DiGraph()
    for tok in tokens:
            gov_i = tok['head'] if  tok['head'] > 0 else 0
            dep_i = tok['id']  if  tok['id'] > 0 else 0
            #if not gov_i in inserted_nodes:
            try:
                g.add_node(tokens[gov_i]['id'],**tokens[gov_i])
            except Exception as e:
                print(tok)
            g.add_node(tokens[dep_i]['id'], **tokens[dep_i])
            g.add_edge(tok['head'], tok['id'], label=tok['dep'])
    return g



def build_path_graph_spacy(sentenceTokens):
    g = nx.Graph()
    for tok in sentenceTokens:
            gov_i = tok['head'] if  tok['head'] > 0 else 0
            dep_i = tok['id']  if  tok['id'] > 0 else 0
            #if not gov_i in inserted_nodes:
            try:
                g.add_node(sentenceTokens[gov_i]['id'],**sentenceTokens[gov_i])
            except Exception as e:
                print(tok)
            g.add_node(sentenceTokens[dep_i]['id'], **sentenceTokens[dep_i])
            g.add_edge(tok['head'], tok['id'], label=tok['dep'])
    return g


def build_path_graph_spacy_1(tokens):
    g = nx.Graph()
    inserted_nodes = set()
    for tok in tokens:
        if (tok['id'] not in inserted_nodes and tok['dep'] != 'ROOT'):
            gov_i = tok['head'] if tok['head'] > 0 else 0
            dep_i = tok['id'] if tok['id'] > 0 else 0
            # if not gov_i in inserted_nodes:
            g.add_node(tokens[gov_i]['id'], **tokens[gov_i])
            g.add_node(tokens[dep_i]['id'], **tokens[dep_i])
            g.add_edge(tok['head'], tok['id'], label=tok['dep'])
    return g


def get_entity_indices(tokens, entities, doc):
    indices = []
    foundStart=False
    foundEnd = False
    start_token = -1
    end_token = -1
    for ent in entities:
        span = doc.char_span(int(ent['@offset']), int(ent['@offset'])+len(ent['@surfaceForm']))
        if span is not None:
            ent_indices=[t['id'] for t in tokens[span.start:span.end]]
            indices.append(ent_indices)
        else:
            print(ent)
    return indices

def reorder_indices(indices):
    entities = []
    for k in indices:
        indices[k].append(k)
        indices[k].sort()
        entities.append(indices[k])
    return entities

def remove_repeated_indices(indices):
    to_remove_index = []
    for _id in indices:
        for k in indices[_id]:
            if k in indices:
                if len(indices[k]) > 0:
                    indices[_id].extend(indices[k])
                    if not k in to_remove_index: to_remove_index.append(k)
                else:
                    if not k in to_remove_index: to_remove_index.append(k)
    for k in to_remove_index:
        del indices[k]

def rebuild_spacy_entity(ids, tokens,text):
    entity = ""
    for _id in ids:
        entity += text[int(tokens[_id]['start']):int(tokens[_id]['end'])] + " "
    return entity.strip()

def rebuild_spacy_sentence(spacy_sentence,spacy_annotation):
    sentence = spacy_annotation['text'][int(spacy_sentence['start']):int(spacy_sentence['end'])]
    return sentence


def find_entity_index_set(index, ids):
    for i in range(len(ids)):
        if index in ids[i]:
            return ids[i]

def find_rel_index(node_tuples):
    for i in range(len(node_tuples[0])):
        if "VB" in node_tuples[0][i][1]:
            return node_tuples[0][i][0]


def get_path_between_entities_spacy(entities, g):
    paths = []
    for i in range(len(entities)):
        e1 = entities[i]
        for j in range(i+1, len(entities)):
            e2 = entities[j]
            #print(f"Looking for a path between {e1} and {e2}")
            #print(e1, e1.split()[-1], "-", e2, e2.split()[-1])
            try:
                sp = nx.shortest_path(g, e1[-1], e2[-1])

            except Exception as e:
                continue
            if len(sp) == 2:
                label = g.edges[(sp[0], sp[1])]['label']
                if label == "nsubj":
                    get_copula_triple(sp, g)
            containVerb = False
            #print(sp)
            for node in sp:
                if "VB" in g.nodes[node]['tag']:
                    containVerb = True
                    break
            if containVerb:
                path = [(node, g.nodes[node]['tag']) for node in sp]
                l = [g.edges[(sp[i], sp[i+1])]['label'] for i in range(len(sp)) if i < len(sp)-1]
                negation = False
                # if 'neg' in [e['label'] for e in [g.neighbors(node)] ]:
                #    negation=True
                paths.append((path, l, negation))

    return paths

def get_copula_triple(path, graph):
    node = path[-1]
    #print(f"Looking for copula relation in {path}")
    for n in graph.neighbors(node):
        label = graph.edges[(n, node)]['label']
        if label == "cop":
            #print(f"inserting copula")
            path.insert(1, n)



def filterTargetPaths(paths):
    return [path for path in paths if path[1] in target_paths]




#sentence = spacy_out['corenlp_output']['sentences'][0]