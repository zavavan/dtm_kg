import os
import time
import warnings
import tensorflow as tf

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
from tqdm import tqdm

from operator import itemgetter


import spacy
from spacy.tokens import Doc, Span
from spacy.tokens import DocBin
import coreferee
from wasabi import msg
import spacy_dbpedia_spotlight

from spacy import displacy

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')


from spacy.language import Language
import numpy as np

import re
import datetime
import html
from spacy.tokens import Token
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
import multiprocessing
from joblib import Parallel, delayed




def process_groups(groups, split_fctr, filename_triples, spacyNLP):
  counter=0

  #df = pd.DataFrame()

  split_factor=split_fctr
  parts = chunkIt([x for x in range(len(groups))], split_factor)
  indmatrixinf = []
  indmatrixsup = []
  for batch_inds in parts:
    indmatrixinf.append(batch_inds[0])
    indmatrixsup.append(batch_inds[-1])

  for i in range(0, len(indmatrixinf)):
    article_group_subset_df = pd.DataFrame()
    for df in groups[indmatrixinf[i]:indmatrixsup[i]+1]:
      article_group_subset_df = pd.concat([article_group_subset_df, df], axis=0)

    sentence_grouping = article_group_subset_df.groupby(['sentence'], sort=False)
    sentences = [sent  if not str(sent).isdigit() else group['triple_subj'].iloc[0] + ' ' + group['triple_rel'].iloc[0] + ' '+ group['triple_obj'].iloc[0] for sent,group in sentence_grouping]

    processed_senteces = []
    for doc in spacyNLP.pipe(sentences, batch_size=200):
      processed_senteces.append(doc)

    for i,(sent,group) in enumerate(sentence_grouping):
      #group_df = pd.DataFrame()

      spacyEntities1=[]
      subjEntityLinks1=[]
      subjEntityLabels1=[]
      objEntityLinks1=[]
      objEntityLabels1=[]
      subjRelatedEntityText1=[]
      subjRelatedEntityLabel1=[]
      subjRelatedEntityLinks1=[]
      objRelatedEntityText1=[]
      objRelatedEntityLabel1=[]
      objRelatedEntityLinks1=[]

      doc1 = processed_senteces[i]
      sentenceText = sentences[i]
      #print('SENTENCE: ',sentenceText)
      group_entities = doc1.ents
      #print('DOC: ',doc1)
      #print('ENTITIES: ',group_entities)

      for k,triple in group.iterrows():
        counter+=1
        found_subjEnt1 = False
        found_subjRelatedEnt1 = False
        found_objEnt1 = False
        found_objRelatedEnt1 = False

        #check if the sentences from the group do not have any DBpedia spotlight entities:
        if not group_entities:
          spacyEntities1.append(None)
        else:
          spacyEntities1.append(','.join([en.text for en in group_entities]))

          splits=triple['triple_subj'].split(' ')
          if len(splits) == 1:
            regexTripleSubj = re.compile(re.escape(triple['triple_subj']))
          else:
              regexTripleSubj = re.compile(re.escape(splits[0])+'.*'+re.escape(splits[-1]))

          splits = triple['triple_obj'].split(' ')
          if len(triple['triple_obj'].split(' '))==1:
              regexTripleObj=re.compile(re.escape(triple['triple_obj']))
          else:
              regexTripleObj = re.compile(re.escape(splits[0]) + '.*' + re.escape(splits[-1]))

          res1 =  regexTripleSubj.search(sentenceText)
          subjSpan=None
          subjHeadSpan=None
          if res1:
              subjSpan = (res1.start(), res1.end())
              reg = re.compile(re.escape(triple['triple_subj_head']))
              resHeadSubj = reg.search(sentenceText)
              subjHeadSpan = (resHeadSubj.start(), resHeadSubj.end())

          res2 =  regexTripleObj.search(sentenceText)
          objSpan=None
          objHeadSpan=None
          if res2:
              objSpan = (res2.start(), res2.end())
              reg = re.compile(re.escape(triple['triple_obj_head']))
              resHeadObj = reg.search(sentenceText)
              objHeadSpan = (resHeadObj.start(), resHeadObj.end())



          subjEntityLabels = list()
          subjEntityLinks = list()
          subjrelatedEntityTexts = list()
          subjrelatedEntityTypes = list()
          subjrelatedEntityLinks = list()

          objEntityLabels = list()
          objEntityLinks = list()
          objrelatedEntityTexts = list()
          objrelatedEntityTypes = list()
          objrelatedEntityLinks = list()


          for ent in group_entities:

              if getOverlap(subjHeadSpan, (ent.start_char,ent.end_char))>0 and not found_subjEnt1 and ent.kb_id_ != '':
                  subjEntityLinks = ent.kb_id_
                  subjEntityLabels = re.findall(r'DBpedia:[^\,]+',ent._.dbpedia_raw_result['@types'])

                  #if ent.kb_id_ and ent.kb_id_ in entityLinkDistribution.keys():
                  #    new_set = entityLinkDistribution[ent.kb_id_][1]
                  #    new_set.add(res1.group())
                  #    tuple = (entityLinkDistribution[ent.kb_id_][0] + 1, new_set)
                  #    entityLinkDistribution[ent.kb_id_] = tuple
                # else:
                #     idSet = set()
                  #    idSet.add(res1.group())
                #     entityLinkDistribution[ent.kb_id_] = (1,idSet)
                  found_subjEnt1 = True


              if includes(subjSpan,(ent.start_char,ent.end_char)) and ent.kb_id_ != '' and not found_subjEnt1 :

                  subjrelatedEntityTexts = ent.text
                  subjrelatedEntityTypes =  re.findall(r'DBpedia:[^\,]+',ent._.dbpedia_raw_result['@types'])
                  subjrelatedEntityLinks= ent.kb_id_

                  #if ent.kb_id_ and ent.kb_id_ in relatedEntityLinkDistribution.keys():
                  #    new_set = relatedEntityLinkDistribution[ent.kb_id_][1]
                  #    new_set.add(res1.group())
                  #    tuple = (relatedEntityLinkDistribution[ent.kb_id_][0] + 1, new_set)
                  #    relatedEntityLinkDistribution[ent.kb_id_] = tuple
                # else:
                  #    idSet = set()
                #     idSet.add(res1.group())
                #     relatedEntityLinkDistribution[ent.kb_id_] = (1,idSet)
                  found_subjRelatedEnt1=True





              if getOverlap(objHeadSpan, (ent.start_char,ent.end_char))>0 and not found_objEnt1 and ent.kb_id_ != '':
                  objEntityLinks = ent.kb_id_
                  objEntityLabels = re.findall(r'DBpedia:[^\,]+', ent._.dbpedia_raw_result['@types'])

                  #if ent.kb_id_ and ent.kb_id_ in entityLinkDistribution.keys():
                  #    new_set = entityLinkDistribution[ent.kb_id_][1]
                  #    new_set.add(res2.group())
                #     tuple = (entityLinkDistribution[ent.kb_id_][0] + 1, new_set)
                  #    entityLinkDistribution[ent.kb_id_] = tuple
                # else:
                #      idSet = set()
                #     idSet.add(res2.group())
                #      entityLinkDistribution[ent.kb_id_] = (1,idSet)
                  found_objEnt1 = True

              if includes(objSpan,(ent.start_char,ent.end_char)) and ent.kb_id_ != '' and not found_objEnt1:
                  objrelatedEntityTexts = ent.text
                  objrelatedEntityTypes = re.findall(r'DBpedia:[^\,]+',ent._.dbpedia_raw_result['@types'])
                  objrelatedEntityLinks = ent.kb_id_
                # if ent.kb_id_ and ent.kb_id_ in relatedEntityLinkDistribution.keys():
                #     new_set = relatedEntityLinkDistribution[ent.kb_id_][1]
                #     new_set.add(res2.group())
                #     tuple = (relatedEntityLinkDistribution[ent.kb_id_][0] + 1, new_set)
                #     relatedEntityLinkDistribution[ent.kb_id_] = tuple
                # else:
                #     idSet = set()
                #     idSet.add(res2.group())
                #     relatedEntityLinkDistribution[ent.kb_id_] = (1, idSet)
                  found_objRelatedEnt1=True




          #else:
          #    print('')
              #print('not matching subj or obj!')

        if found_subjEnt1 == False:
            subjEntityLabels1.append(None)
            subjEntityLinks1.append(None)
        else:
            subjEntityLabels1.append(subjEntityLabels)
            subjEntityLinks1.append(subjEntityLinks)

        if found_objEnt1 == False:
            objEntityLabels1.append(None)
            objEntityLinks1.append(None)
        else:
            objEntityLabels1.append(objEntityLabels)
            objEntityLinks1.append(objEntityLinks)

        if found_subjRelatedEnt1 == False:
            subjRelatedEntityText1.append(None)
            subjRelatedEntityLabel1.append(None)
            subjRelatedEntityLinks1.append(None)
        else:
            subjRelatedEntityText1.append(subjrelatedEntityTexts)
            subjRelatedEntityLabel1.append(subjrelatedEntityTypes)
            subjRelatedEntityLinks1.append(subjrelatedEntityLinks)

        if found_objRelatedEnt1 == False:
            objRelatedEntityText1.append(None)
            objRelatedEntityLabel1.append(None)
            objRelatedEntityLinks1.append(None)
        else:
            objRelatedEntityText1.append(objrelatedEntityTexts)
            objRelatedEntityLabel1.append(objrelatedEntityTypes)
            objRelatedEntityLinks1.append(objrelatedEntityLinks)

      group['spacy_entities1'] = spacyEntities1
      group['subjEntityLinks'] = subjEntityLinks1
      group['subjEntityLabels'] = subjEntityLabels1
      group['objEntityLinks'] = objEntityLinks1
      group['objEntityLabels'] = objEntityLabels1
      group['subjRelatedEntityText1'] = subjRelatedEntityText1
      group['subjRelatedEntityLabel1'] = subjRelatedEntityLabel1
      group['subjRelatedEntityLink1'] = subjRelatedEntityLinks1
      group['objRelatedEntityText1'] = objRelatedEntityText1
      group['objRelatedEntityLabel1'] = objRelatedEntityLabel1
      group['objRelatedEntityLink1'] = objRelatedEntityLinks1

      group.to_csv(filename_triples, mode='a', index=False, header=None, sep="\t", encoding="utf-8")

def getOverlap(pair_a, pair_b):
    if not pair_a:
        return 0
    return min(pair_a[-1], pair_b[-1]) - max(pair_a[0], pair_b[0]) + 1


def includes(pair_a, pair_b):
    if not pair_a:
        return False
    return  (pair_a[0] <= pair_b[0] and pair_a[-1]>=pair_b[-1])

def chunkIt(seq, num):
    #print('sequence passed to chunkIt function: ', len(seq) )
    avg = len(seq) / float(num)
    #print('avg: ',avg)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


if __name__ == '__main__':

    test_folder = 'D:/GitRepos/GitRepos/dtmkg/data-collection/dna/test'
    outputFile = os.path.join(test_folder, 'dna_Entity-Relations_Normalized_EntityLinking.tsv')
    # tests_folder = 'D:/GitRepos/GitRepos/dtmkg/data-collection/twitter/tests'

    n_jobs = multiprocessing.cpu_count()  # Count the number of cores in a computer
    print('number of processes: ', n_jobs)
    tic = time.time()

    triples = pd.read_excel(os.path.join(test_folder, 'triples_dna_all_EntityNormalized.xlsx'), engine="openpyxl")
    # triples=triples_total
    columns = triples.columns.values.tolist()
    columns.extend(['spacy_entities1', 'subjEntityLinks', 'subjEntityLabels', 'objEntityLinks', 'objEntityLabels',
         'subjRelatedEntityText1', 'subjRelatedEntityLabel1', 'subjRelatedEntityLink1', 'objRelatedEntityText1',
         'objRelatedEntityLabel1', 'subjRelatedEntityLink1'])

    triples_new = pd.DataFrame(columns=columns)
    triples_new.to_csv(outputFile, mode='w', header=columns, index=False, sep="\t", encoding="utf-8")

    # triples = triples[0:1000]

    print('number of triples: ', len(triples))



    spacyNLP = spacy.load('en_core_web_lg')
    # add the pipeline stage
    spacyNLP.add_pipe('dbpedia_spotlight', config={'dbpedia_rest_endpoint': 'http://localhost:2222/rest', 'confidence': 0.70})
    # spacyNLP.add_pipe('coreferee')
    print(spacyNLP.pipeline)

    # customizing the default tokenizer to not split hashtags
    re_token_match = spacy.tokenizer._get_regex_pattern(spacyNLP.Defaults.token_match)
    # Add #hashtag pattern
    re_token_match = f"({re_token_match}|#\\w+)"
    # Add #hashtag pattern
    re_token_match = f"({re_token_match}|@[\\w_]+)"
    re_token_match = f"({re_token_match}|\\$\\w+)"
    spacyNLP.tokenizer.token_match = re.compile(re_token_match).match

    infixes = (
            LIST_ELLIPSES
            + LIST_ICONS
            + [
                r"(?<=[0-9])[+\-\*^](?=[0-9-])",
                r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                    al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
                ),
                r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
                # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
                r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
            ]
    )
    infix_re = compile_infix_regex(infixes)
    spacyNLP.tokenizer.infix_finditer = infix_re.finditer
    toc = time.time()
    print('time to load, customize and save serialization of the spacy models = ' + str((toc - tic) / 60) + ' minutes')

    # triples = pd.read_excel(os.path.join(tests_folder, 'evaluation_92134_tweets_Entity-Relations_Normalized.xlsx'),index_col=0,engine="openpyxl")


    # triples = triples.iloc[0:100]

    tic = time.time()

    article_grouping = triples.groupby(['doc_id'], sort=False)
    print('size article grouping: ', len(article_grouping))

    latest_doc_id_processed = input('Insert Latest Processed Id: ')

    if latest_doc_id_processed:
        doc_ids_to_be_dropped=[]
        for name, group in article_grouping:
            if name!=latest_doc_id_processed:
                doc_ids_to_be_dropped.append(name)
            else:
                break

        triples= triples[~triples.doc_id.isin(doc_ids_to_be_dropped)]
        print(len(triples))




    #batch_size = int(int(len(article_grouping) / 100))
    #print('batch_size = ' + str(batch_size))
    parts = chunkIt([x for x in range(len(article_grouping))], 100)
    # print(parts)
    indmatrixinf = []
    indmatrixsup = []
    for batch_inds in parts:
        indmatrixinf.append(batch_inds[0])
        indmatrixsup.append(batch_inds[-1])

    print('inf indexes article splitting', indmatrixinf)
    print('inf indexes article splitting', indmatrixsup)

    for i in tqdm(range(0, len(indmatrixinf)),total=len(indmatrixinf)):
        process_groups([article_grouping.get_group(k) for k in list(article_grouping.indices.keys())[indmatrixinf[i]:indmatrixsup[i] + 1]], 10, outputFile, spacyNLP)