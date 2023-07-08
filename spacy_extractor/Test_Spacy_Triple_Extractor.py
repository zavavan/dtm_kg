from datetime import time
import requests
import pandas as pd
import json
import os
import sys
from requests.auth import HTTPBasicAuth
import xml.etree.ElementTree as ET
import hashlib
from EntityExtraction import *

import xlsxwriter
__author__ = 'Vanni Zavarella'
__version__ = '0.1.0'
__maintainer__ = 'Vanni Zavarella'
__email__ = 'zavavan at yahoo dot it'
__status__ = 'Dev'


# source document folder
data_path = "/data-collection"

twitter_data_folder = os.path.join(data_path, 'twitter')

#source_doc_folder = os.path.join(data_path, 'core_api/sample2018_dt')
source_doc_folder = os.path.join(data_path, 'twitter/sample')

#spacy_annotations_folder = os.path.join(data_path, 'core_api/output_spacy')
spacy_annotations_folder = os.path.join(data_path, 'twitter\output_spacy_annotations')
spacy_annotations_triples_folder = os.path.join(data_path, 'twitter/spacy_annotations_triples')

source_docs = [ x for x in os.listdir(source_doc_folder) if os.stat(os.path.join(source_doc_folder,x)).st_size > 1024]
spacy_annotations_docs = [ x for x in os.listdir(spacy_annotations_folder) if os.stat(os.path.join(spacy_annotations_folder,x)).st_size > 1024]

file_counter = 0
startTime = time.time()
kg_df_spacy = pd.DataFrame(
        columns=['doc_id','sentence_n','triple_subj','triple_rel','triple_obj','path','dep_tree_path','sentence'])
for filename in spacy_annotations_docs:
    if file_counter % 1000 == 0:
        print("processed 1000 files!")
    with open(os.path.join(spacy_annotations_folder,filename), mode="r", encoding="utf-8") as file_name:
      #nlp_out = json.load(file_name)
      dict = json.load(file_name)
      doc_id = dict['id']
      spacy_out = dict['spacy_output']
      #if "sentences" in nlp_out['corenlp_output'].keys():
      if "sents" in spacy_out.keys():
        sent_counter=0
        sentences = spacy_out['sents']
        #sentenceTokens = getSentenceTokens(spacy_out['tokens'],sentences)
        for sentence in sentences:
            sentenceNum = str(sent_counter)
            #sentenceText = utils.rebuild_spacy_sentence(sentence,spacy_out)
            sentenceText = sentence['text']
            tokens = sentence['tokens']
            #b_dependencies = sentence['basicDependencies']
            g = build_graph_spacy(tokens)
            entities_indices = get_entity_indices_spacy(tokens, g)
            remove_repeated_indices(entities_indices)
            entities = reorder_indices(entities_indices)
            path_graph = build_path_graph_spacy(tokens)
            paths = get_path_between_entities_spacy(entities, path_graph)
            if len(paths) > 0:
                for path in paths:
                    entityString =rebuild_spacy_entity(find_entity_index_set(path[0][0][0], entities), tokens, sentenceText)
                    fromString = entityString
                    toString = rebuild_spacy_entity(find_entity_index_set(path[0][len(path[0])-1][0], entities), tokens, sentenceText)
                    relString =   tokens[find_rel_index(path)]['lemma']
                    pathString = fromString+";"+relString+";"+toString
                    deep_tree_pathString = path[1]
                    triple_subjString=fromString
                    triple_relString=relString
                    triple_objString=toString
                    kg_df_spacy = kg_df_spacy.append(
                        {'doc_id': doc_id, 'sentence_n': sentenceNum, 'path': pathString,
                         'dep_tree_path': deep_tree_pathString, 'triple_subj': triple_subjString,
                         'triple_rel': triple_relString, 'triple_obj': triple_objString, 'sentence': sentenceText},
                        ignore_index=True)
            sent_counter += 1
    file_counter += 1
kg_df_spacy.to_excel(os.path.join(twitter_data_folder, "all_spacy_triples.xls"), encoding="utf-8", engine="xlsxwriter")


print("Processed " + str(file_counter) + " files in %s seconds ---" %(time.time() - startTime ) )
