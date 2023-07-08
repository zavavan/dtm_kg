import matplotlib.pyplot as plt
import pandas as pd
import json, os
import sys
import networkx as nx
import time
from inflection import camelize
from networkx import neighbors

from corenlp_utils import utils

# We want to also keep #hashtags as a token, so we modify the spaCy model's token_match
import re
# Retrieve the default token-matching regex pattern
#re_token_match = spacy.tokenizer._get_regex_pattern(nlp_spacy.Defaults.token_match)
# Add #hashtag pattern
#re_token_match = f"({re_token_match}|#\\w+)"
#nlp_spacy.tokenizer.token_match = re.compile(re_token_match).match

punctuations = ['-','#','.',',',':','!',';','£','$','%','&','?','*','_','@','|','>','<','°','§','/','rt','=']
punctuations1 = ['-','#','.',',',':','!',';','%','&','?','*','_','@','|','>','<','°','§','/','=']
punctuations2 = ['-','.',',',':','!',';','%','&','?','*','_','|','>','<','°','§','/','=']
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
                g.add_node(tokens[gov_i]['id'],**dir(tokens[gov_i]))
            except Exception as e:
                print(tok)
            g.add_node(tokens[dep_i]['id'], **tokens[dep_i])
            g.add_edge(tok['head'], tok['id'], label=tok['dep'])
    return g

def build_graph_spacy_latest(sentence_token_ids,tokens):
    g = nx.DiGraph()
    for tok_id in sentence_token_ids:
            gov_i = tokens[tok_id].head.i if  tokens[tok_id].head.i > 0 else 0
            dep_i = tokens[tok_id].i  if  tokens[tok_id].i > 0 else 0
            #if not gov_i in inserted_nodes:
            mapping={}
            for name in dir(tokens[gov_i]):
                if not name.startswith('__') and name !='tensor':
                    mapping[name] = getattr(tokens[gov_i], name)

            g.add_node(gov_i,**mapping)
            mapping = {}
            for name in dir(tokens[dep_i]):
                if not name.startswith('__') and name != 'tensor':
                    mapping[name] = getattr(tokens[dep_i], name)
            g.add_node(dep_i,**mapping)

            g.add_edge(gov_i, dep_i, label=tokens[tok_id].dep_)
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


def build_path_graph_spacy_latest(sentence_token_ids,tokens):
    g = nx.Graph()
    for tok_id in sentence_token_ids:
        gov_i = tokens[tok_id].head.i if tokens[tok_id].head.i > 0 else 0
        dep_i = tokens[tok_id].i if tokens[tok_id].i > 0 else 0
        # if not gov_i in inserted_nodes:
        mapping = {}
        for name in dir(tokens[gov_i]):
            if not name.startswith('__') and name != 'tensor':
                mapping[name] = getattr(tokens[gov_i], name)

        g.add_node(gov_i, **mapping)
        mapping = {}
        for name in dir(tokens[dep_i]):
            if not name.startswith('__') and name != 'tensor':
                mapping[name] = getattr(tokens[dep_i], name)
        g.add_node(dep_i, **mapping)

        g.add_edge(gov_i, dep_i, label=tokens[tok_id].dep_)
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


def get_entity_indices_spacy(tokens, g):
    indices = {}
    for token in tokens:
        if ("NN" in token['tag'] or "PRP" in token['tag'] ) and token['lemma'] not in punctuations:
            _id = g.nodes[token['id']]['id']
            if not _id in indices: indices[_id] = []
            for neighbor in g.neighbors(token['id']):
                n_id = g.nodes[neighbor]['id']
                label = g.edges[(token['id'], neighbor)]['label']
                if label == "compound" or label == "amod":
                    indices[_id].append(n_id)
    return indices


def get_entity_indices_spacy_w_pronouns(tokens, g):
    indices = {}
    for token in tokens:
        if ("NN" in token.tag_ or "PRP" in token.tag_ ) and token.lemma not in punctuations:
            _id = g.nodes[token.i]['i']
            if not _id in indices: indices[_id] = []
            for neighbor in g.neighbors(token.i):
                n_id = g.nodes[neighbor]['i']
                label = g.edges[(token.i, neighbor)]['label']
                if label == "compound" or label == "amod" or (token.text=='%' and label == "nummod"):
                    indices[_id].append(n_id)
    return indices

def get_entity_indices_spacy_no_pronouns(tokens, g):
    indices = {}
    for token in tokens:
        if "NN" in token.tag_ and token.lemma not in punctuations:
            _id = g.nodes[token.i]['i']
            if not _id in indices: indices[_id] = []
            for neighbor in g.neighbors(token.i):
                n_id = g.nodes[neighbor]['i']
                label = g.edges[(token.i, neighbor)]['label']
                if label == "compound" or label == "amod" or (token.text=='%' and label == "nummod"):
                    indices[_id].append(n_id)
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

def rebuild_spacy_entity_with_coref(doc, ids, tokens):
    entity = ""
    pronoun_entity = False
    if len(ids)==1 and tokens[ids[0]].tag_=='PRP':
        #doc_token_id = char_pos_map[int(tokens[ids[0]]['start'])+ sent_start_char]
        result = resolve_coref(doc, ids, tokens)
        if result is not None:
            return result.text, tokens[ids[0]].text
        else: return None
    else:
        for _id in ids:
            entity += tokens[_id].text + " "
        return entity.strip(),''

def rebuild_spacy_entity_with_coref_lemmatize(doc, ids, tokens,camel_casesMap):
    entityString = ''
    intermediate_result = ids
    # follow the pronominal co-reference chain from spacy to get the string of the antecedent
    if len(ids)==1 and tokens[ids[0]].tag_=='PRP':
        #doc_token_id = char_pos_map[int(tokens[ids[0]]['start'])+ sent_start_char]
        result = resolve_coref(doc, ids, tokens)
        if result is None:
            return None
        else:
            intermediate_result = [result.i]

    # normalize the entity String
    for _id in intermediate_result:
        #remove leading punctuation chars
        tokenText = tokens[_id].text.lstrip(''.join(punctuations2))

        if tokenText.startswith('#') or tokenText.startswith('@'):
            entityString = " ".join([entityString, cleanString(tokenText,camel_casesMap)])
        elif tokens[_id].text=='%':
            entityString = "".join([entityString,tokens[_id].text])
        else: entityString = " ".join([entityString,tokens[_id].lemma_.strip(''.join(punctuations2)).lower() if len(re.findall('\.', tokens[_id].lemma_)) == 1 else tokens[_id].lemma_.lower()])
        #entityString += (tokens[_id].lemma_.strip(''.join(punctuations2)).lower() if len(re.findall('\.', tokens[_id].lemma_)) == 1 else tokens[_id].lemma_) + " "

    return entityString.strip()


def rebuild_spacy_entity_with_lemmas(ids, tokens,text):
    entity = ""
    for _id in ids:
        entity += tokens[_id]['lemma'] + " "
    return entity.strip()


def cleanString(input,camel_casesMap):
    result = ""
    input = input.strip(''.join(punctuations1))

    if input.lower() in camel_casesMap.keys():
        return camel_casesMap[input.lower()]

    if is_camel_case(input):
        res_list = []
        res_list = [s.lower() for s in re.findall('[A-Z][^A-Z]*', input)]
        for s in res_list:
            result = result + ' ' + s
        camel_casesMap[input.lower()] = result.strip()
        return result.strip()
    else:
        input = input.replace('_', ' ', 1)
        return input.lower().strip()



def rebuild_spacy_entity_with_lemmas_latest(ids, tokens,text):
    entity = ""
    for _id in ids:
        entity += cleanString(tokens[_id].lemma_) + " "
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


def getRelIndexes(node_tuples,g,tokens,sentStartIndex):
    resultString = ''
    result=[]
    verb_node=None
    for i in range(len(node_tuples[0])):
        if "VB" in node_tuples[0][i][1]:
            verb_node=node_tuples[0][i][0]
            result.append(verb_node - int(sentStartIndex))

    for neighbor in g.out_edges(verb_node):
       if tokens[neighbor[1]].pos_ in ['ADP'] and g.edges[(verb_node, neighbor[1])]['label'] in['prt','agent']:
            result.append(neighbor[1] - int(sentStartIndex))

    return result

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
                if "VB" in g.nodes[node]['tag_'] and (not g.nodes[node]['is_digit']) and g.nodes[node]['is_alpha'] :
                    containVerb = True
                    verbPos=g.nodes[node]['tag_']
                    break
            if containVerb:
                path = [(node, g.nodes[node]['tag_']) for node in sp]
                l = [g.edges[(sp[i], sp[i+1])]['label'] for i in range(len(sp)) if i < len(sp)-1]
                negation = False
                if 'neg' in [edge[2]['label'] for edge in list(g.edges([node],data=True))]:
                    negation=True
                paths.append((path, l, negation,verbPos))

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


#def is_camel_case(s):
#    # return True for both 'CamelCase' and 'camelCase'
#    return camelize(s) == s or camelize(s, False) == s

def is_camel_case(s):
  if s != s.lower() and s != s.upper() and "_" not in s and sum(i.isupper() for i in s[1:-1]) > 0:
      return True
  return False

def resolve_coref(doc, ids, tokens):
 if len(ids)==1 and tokens[ids[0]].tag_=='PRP':
        #doc_token_id = char_pos_map[int(tokens[ids[0]]['start'])+ sent_start_char]
        result = doc._.coref_chains.resolve(doc[ids[0]])
        if result is None:
            return None
        else:
            return result[0]
