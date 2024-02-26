import os
import time
import warnings
import tensorflow as tf
from spacy_extractor.Test_Spacy_TripleExtractor_withGraphSaving import chunkIt, includes, getOverlap

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

import preprocessor as p

# p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.RESERVED, p.OPT.SMILEY)
p.set_options(p.OPT.EMOJI, p.OPT.RESERVED, p.OPT.SMILEY)


from spacy.language import Language
import numpy as np

import re
import datetime
import html
from spacy.tokens import Token
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex


if __name__ == '__main__':

    #print(tf.__version__)
    #print(tf.test.is_gpu_available())
    #print(tf.test.gpu_device_name())
    #print(tf.config.list_physical_devices('GPU'))
    #exit()
    #test_folder = 'D:/GitRepos/GitRepos/dtmkg/data-collection/twitter/tests/test'
    test_folder = 'D:/GitRepos/GitRepos/dtmkg/data-collection/dna/test'

    tests_folder = 'D:/GitRepos/GitRepos/dtmkg/data-collection/twitter/tests'


    tic = time.time()

    spacyNLP = spacy.load('en_core_web_lg')
    # add the pipeline stage
    spacyNLP.add_pipe('dbpedia_spotlight')
    spacyNLP.add_pipe('coreferee')
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

    #triples = pd.read_excel(os.path.join(tests_folder, 'evaluation_92134_tweets_Entity-Relations_Normalized.xlsx'),index_col=0,engine="openpyxl")
    triples = pd.read_excel(os.path.join(test_folder, 'triples_dna_EntityNormalized.xlsx'),index_col=0,engine="openpyxl")


    #triples = triples[0:1000]


    print('number of triples: ',len(triples))

    #triples = triples.iloc[0:100]

    tic = time.time()
    split_factor = 1
    parts = chunkIt([x for x in range(len(triples))], split_factor)
    indmatrixinf = []
    indmatrixsup = []
    for batch_inds in parts:
        indmatrixinf.append(batch_inds[0])
        indmatrixsup.append(batch_inds[len(batch_inds) - 1])

    print(indmatrixinf)
    print(indmatrixsup)

    entityLinkDistribution = dict()
    relatedEntityLinkDistribution = dict()


    #linkedTriples = []

    #subjEntityText2 = []
    #subjEntityLabel2 = []
    #subjEntityLinks2 = []
    #objEntityText2 = []
    #objEntityLabel2 = []
    #objEntityLinks2 = []

    spacyEntities1 = []
    #spacyEntities2 = []

    subjEntityLabels1 = []
    subjEntityLinks1 = []
    objEntityLabels1 = []
    objEntityLinks1 = []

    subjRelatedEntityText1 = []
    subjRelatedEntityLabel1 = []
    subjRelatedEntityLinks1 = []
    objRelatedEntityText1 = []
    objRelatedEntityLabel1 = []
    objRelatedEntityLinks1 = []

    #subjRelatedEntityText2 = []
    #subjRelatedEntityLabel2 = []
    #subjRelatedEntityLinks2 = []
    #objRelatedEntityText2 = []
    #objRelatedEntityLabel2 = []
    #objRelatedEntityLinks2 = []

    spacyRelatedEntities1 = []
    #spacyRelatedEntities2 = []

    counterNullEntities = 0

    #counterNullEntitiesAfterReplace = 0

    counterMismatchedEnts = 0
    counterMismatchedRelatedEnts = 0
    #counterMismatchedEntsAfterReplace = 0

    for i in range(0, len(indmatrixinf)):

        for k,triple in tqdm(triples.iloc[indmatrixinf[i]:indmatrixsup[i]+1].iterrows(),
                         total=round(len(triples) / split_factor)):

            tweetId = triple[0]
            # originalText = splitting[2]
            originalText = triple['sentence']
            sentenceText = triple['sentence']
            subjEntity = ''
            objEntity = ''

            regexTripleSubj=None
            regexTripleObj=None


            if not pd.isna(triple['triple_subj_pron']):
                sentenceText = re.sub(r'([^\w]?)'+ triple['triple_subj_pron'] +'([^\w]?)',r'\1' + triple['triple_subj'] + r'\2', sentenceText, 1)
            else:
                if sentenceText.find(triple['triple_subj']) != -1:
                    sentenceText = sentenceText.replace(triple['triple_subj'],triple['triple_subj'])
                    regexTripleSubj=re.compile(re.escape(triple['triple_subj']))
                else:
                    splits = triple['triple_subj'].split(' ')
                    reg = re.compile(re.escape(splits[0])+'.*'+re.escape(splits[-1]))
                    regexTripleSubj = re.compile(re.escape(splits[0])+'.*'+re.escape(splits[-1]))
                    sentenceText = re.sub(reg, triple['triple_subj'], sentenceText)


            if not pd.isna(triple['triple_obj_pron']):
                sentenceText = re.sub(r'([^\w]?)'+ triple['triple_obj_pron'] +'([^\w]?)',r'\1' + triple['triple_obj'] + r'\2', sentenceText, 1)
            else:
                if sentenceText.find(triple['triple_obj']) != -1:
                    sentenceText = sentenceText.replace(triple['triple_obj'], triple['triple_obj_lemma'])
                    regexTripleObj = re.compile(re.escape(triple['triple_obj']))
                else:
                    splits = triple['triple_obj'].split(' ')
                    reg = re.compile(re.escape(splits[0]) + '.*' + re.escape(splits[-1]))
                    regexTripleObj = re.compile(re.escape(splits[0]) + '.*' + re.escape(splits[-1]))
                    sentenceText = re.sub(reg, triple['triple_obj'], sentenceText)




            doc1 = spacyNLP(sentenceText)
            #doc2 = spacyNLP(sentenceTextReplaced)

            spacyEnts1 = doc1.ents
            #spacyEnts2 = doc2.ents

            if len(spacyEnts1)>0:
                spacyEntities1.append(','.join([en.text for en in spacyEnts1]))
            else:
                spacyEntities1.append(None)
                counterNullEntities+=1


            #if len(spacyEnts2)>0:
            #    spacyEntities2.append(','.join([en.text for en in spacyEnts2]))
            #else:
             #   spacyEntities2.append(None)
             #   counterNullEntitiesAfterReplace+=1

            found_subjEnt1 = False
            found_subjRelatedEnt1 = False
            found_objEnt1 = False
            found_objRelatedEnt1 = False


            if not regexTripleSubj:
                regexTripleSubj = re.compile(re.escape(triple['triple_subj']))
            res1 =  regexTripleSubj.search(sentenceText)

            if not regexTripleObj:
                regexTripleObj = re.compile(re.escape(triple['triple_obj']))
            res2 = regexTripleObj.search(sentenceText)

            reg = re.compile(re.escape(triple['triple_subj_head']))
            resHeadSubj = reg.search(sentenceText)
            reg = re.compile(re.escape(triple['triple_obj_head']))
            resHeadObj = reg.search(sentenceText)



            if res1 and res2 and resHeadSubj and resHeadObj:
                subjSpan = (res1.start(), res1.end())
                objSpan = (res2.start(), res2.end())
                subjHeadSpan = (resHeadSubj.start(), resHeadSubj.end())
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



                for ent in spacyEnts1:

                    if getOverlap(subjHeadSpan, (ent.start_char,ent.end_char))>0 and not found_subjEnt1 and ent.kb_id_ != '':
                        subjEntityLinks = ent.kb_id_
                        subjEntityLabels = re.findall(r'DBpedia:[^\,]+',ent._.dbpedia_raw_result['@types'])

                        if ent.kb_id_ and ent.kb_id_ in entityLinkDistribution.keys():
                            new_set = entityLinkDistribution[ent.kb_id_][1]
                            new_set.add(res1.group())
                            tuple = (entityLinkDistribution[ent.kb_id_][0] + 1, new_set)
                            entityLinkDistribution[ent.kb_id_] = tuple
                        else:
                            idSet = set()
                            idSet.add(res1.group())
                            entityLinkDistribution[ent.kb_id_] = (1,idSet)
                        found_subjEnt1 = True


                    if includes(subjSpan,(ent.start_char,ent.end_char)) and ent.kb_id_ != '' and not found_subjEnt1 :

                        subjrelatedEntityTexts = ent.text
                        subjrelatedEntityTypes =  re.findall(r'DBpedia:[^\,]+',ent._.dbpedia_raw_result['@types'])
                        subjrelatedEntityLinks= ent.kb_id_

                        if ent.kb_id_ and ent.kb_id_ in relatedEntityLinkDistribution.keys():
                            new_set = relatedEntityLinkDistribution[ent.kb_id_][1]
                            new_set.add(res1.group())
                            tuple = (relatedEntityLinkDistribution[ent.kb_id_][0] + 1, new_set)
                            relatedEntityLinkDistribution[ent.kb_id_] = tuple
                        else:
                            idSet = set()
                            idSet.add(res1.group())
                            relatedEntityLinkDistribution[ent.kb_id_] = (1,idSet)
                        found_subjRelatedEnt1=True





                    if getOverlap(objHeadSpan, (ent.start_char,ent.end_char))>0 and not found_objEnt1 and ent.kb_id_ != '':
                        objEntityLinks = ent.kb_id_
                        objEntityLabels = re.findall(r'DBpedia:[^\,]+', ent._.dbpedia_raw_result['@types'])

                        if ent.kb_id_ and ent.kb_id_ in entityLinkDistribution.keys():
                            new_set = entityLinkDistribution[ent.kb_id_][1]
                            new_set.add(res2.group())
                            tuple = (entityLinkDistribution[ent.kb_id_][0] + 1, new_set)
                            entityLinkDistribution[ent.kb_id_] = tuple
                        else:
                            idSet = set()
                            idSet.add(res2.group())
                            entityLinkDistribution[ent.kb_id_] = (1,idSet)
                        found_objEnt1 = True

                    if includes(objSpan,(ent.start_char,ent.end_char)) and ent.kb_id_ != '' and not found_objEnt1:
                        objrelatedEntityTexts = ent.text
                        objrelatedEntityTypes = re.findall(r'DBpedia:[^\,]+',ent._.dbpedia_raw_result['@types'])
                        objrelatedEntityLinks = ent.kb_id_
                        if ent.kb_id_ and ent.kb_id_ in relatedEntityLinkDistribution.keys():
                            new_set = relatedEntityLinkDistribution[ent.kb_id_][1]
                            new_set.add(res2.group())
                            tuple = (relatedEntityLinkDistribution[ent.kb_id_][0] + 1, new_set)
                            relatedEntityLinkDistribution[ent.kb_id_] = tuple
                        else:
                            idSet = set()
                            idSet.add(res2.group())
                            relatedEntityLinkDistribution[ent.kb_id_] = (1, idSet)
                        found_objRelatedEnt1=True




            else:
                print('not matching subj or obj!')

            if found_subjEnt1 == False:
                subjEntityLabels1.append(None)
                subjEntityLinks1.append(None)
                counterMismatchedEnts+=1
            else:
                subjEntityLabels1.append(subjEntityLabels)
                subjEntityLinks1.append(subjEntityLinks)

            if found_objEnt1 == False:
                objEntityLabels1.append(None)
                objEntityLinks1.append(None)
                counterMismatchedEnts += 1
            else:
                objEntityLabels1.append(subjEntityLabels)
                objEntityLinks1.append(subjEntityLinks)

            if found_subjRelatedEnt1 == False:
                subjRelatedEntityText1.append(None)
                subjRelatedEntityLabel1.append(None)
                subjRelatedEntityLinks1.append(None)
                counterMismatchedRelatedEnts+=1
            else:
                subjRelatedEntityText1.append(subjrelatedEntityTexts)
                subjRelatedEntityLabel1.append(subjrelatedEntityTypes)
                subjRelatedEntityLinks1.append(subjrelatedEntityLinks)

            if found_objRelatedEnt1 == False:
                objRelatedEntityText1.append(None)
                objRelatedEntityLabel1.append(None)
                objRelatedEntityLinks1.append(None)
                counterMismatchedRelatedEnts+=1
            else:
                objRelatedEntityText1.append(objrelatedEntityTexts)
                objRelatedEntityLabel1.append(objrelatedEntityTypes)
                objRelatedEntityLinks1.append(objrelatedEntityLinks)

            linkedTripleSubj = triple['triple_subj_lemma']
            linkedTripleObj = triple['triple_obj_lemma']
            rel = triple['triple_rel_lemma']
            #linkedTriples.append('\''+linkedTripleSubj+ '\';' +rel+';'+ '\'' + linkedTripleObj+ '\'')





    triples['spacy_entities1']=spacyEntities1
    #triples['LinkedTriples']=linkedTriples
    triples['subjEntityLinks'] = subjEntityLinks1
    triples['subjEntityLabels']=subjEntityLabels1
    triples['objEntityLinks'] = objEntityLinks1
    triples['objEntityLabels'] = objEntityLabels1
    triples['subjRelatedEntityText1'] = subjRelatedEntityText1
    triples['subjRelatedEntityLabel1'] = subjRelatedEntityLabel1
    triples['subjRelatedEntityLink1'] = subjRelatedEntityLinks1
    triples['objRelatedEntityText1'] = objRelatedEntityText1
    triples['objRelatedEntityLabel1'] = objRelatedEntityLabel1
    triples['objRelatedEntityLink1'] = objRelatedEntityLinks1



    print('Number of triples with no entities = ',counterNullEntities)
    #print('Number of triples with no entities after replacement = ',counterNullEntitiesAfterReplace)
    print('Number of triples with no matching entities = ', counterMismatchedEnts)
    #print('Number of triples with no matching entities after replacement = ', counterMismatchedEntsAfterReplace)

    entityLinkDistribution = {'EntityLink': entityLinkDistribution.keys(),
                            'Occurrence': [x for x,y in entityLinkDistribution.values()],
                            'Matches': [y for x, y in entityLinkDistribution.values()]
                              }
    df_entityLinkDistribution = pd.DataFrame(entityLinkDistribution)



    #df_entityLinkDistribution.to_excel(os.path.join(tests_folder, 'EntityLemmaFrequency_test_full_new.xlsx'),encoding="utf-8", engine="xlsxwriter")
    df_entityLinkDistribution.to_excel(os.path.join(test_folder, 'EntityLemmaFrequency_test_dna.xlsx'),encoding="utf-8", engine="xlsxwriter")

    relatedEntityLinkDistribution = {'EntityLink': relatedEntityLinkDistribution.keys(),
                            'Occurrence': [x for x,y in relatedEntityLinkDistribution.values()],
                            'Matches': [y for x, y in relatedEntityLinkDistribution.values()]
                              }
    df_relatedEntityLinkDistribution= pd.DataFrame(relatedEntityLinkDistribution)

    #df_relatedEntityLinkDistribution.to_excel(os.path.join(tests_folder, 'RelatedEntityLemmaFrequency_test_full_new.xlsx'),encoding="utf-8", engine="xlsxwriter")
    df_relatedEntityLinkDistribution.to_excel(os.path.join(test_folder, 'RelatedEntityLemmaFrequency_test_dna.xlsx'),encoding="utf-8", engine="xlsxwriter")

    #triples.to_excel(os.path.join(tests_folder, 'evaluation_92134_tweets_Entity-Relations_Normalized_EntityLinking_new.xlsx'), encoding="utf-8",engine="xlsxwriter")
    triples.to_excel(os.path.join(test_folder, 'dna_Entity-Relations_Normalized_EntityLinking.xlsx'), encoding="utf-8",engine="xlsxwriter")


