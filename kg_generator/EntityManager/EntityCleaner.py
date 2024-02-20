import re
import time

import spacy
import en_core_web_lg

from nltk.corpus import stopwords
from breame.spelling import american_spelling_exists, british_spelling_exists
from breame.spelling import get_american_spelling, get_british_spelling

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
from tqdm import tqdm

import multiprocessing
from joblib import Parallel, delayed

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





def entity_string_improvement(e):
    improved_entity_string = e.strip()
    first_character = improved_entity_string[0]
    last_character = improved_entity_string[-1]

    while (True):
        if first_character in EntityCleaner.puntuaction_reject:
            improved_entity_string = improved_entity_string[1:]
            if len(improved_entity_string) >= 1:
                first_character = improved_entity_string[0]
            else:
                break
        # print(e, '->', improved_entity_string)
        else:
            break

    while (True):
        if last_character in EntityCleaner.puntuaction_reject:
            improved_entity_string = improved_entity_string[:-1]
            if len(improved_entity_string) >= 1:
                last_character = improved_entity_string[-1]
            else:
                break
        # print(e, '->', improved_entity_string)
        else:
            break
    return improved_entity_string.strip()

def toBritishSpelling(entities):
    new_entities = []

    for e in tqdm(entities, total=len(entities)):
        if e is None:
            new_entities += [None]
        else:
            entityString = ''
            res = e.split()
            for r in res:
                bspel = get_british_spelling(r)
                #if (r != bspel):
                    #print(r, bspel)
                entityString = " ".join([entityString,bspel])


            new_entities += [entityString.strip()]

    return new_entities

def puntuaction_and_stopword(entities):
    new_entities = []
    print('len of self.entities before running puntuaction_and_stopword() : ' + str(len(entities)))
    print('len of None entities before running puntuaction_and_stopword() : ' + str(
        len([x for x in entities if x is None])))
    entities = [x for x in entities]
    for e in tqdm(entities, total=len(entities)):
        if e is None:
            new_entities += [None]

        else:
            valid_puntuaction = True

            for c in e:
                if c in EntityCleaner.puntuaction_reject:
                    valid_puntuaction = False
                    break

            if not valid_puntuaction:
                new_entities += [None]
            else:
                tmpE = EntityCleaner.regex_puntuaction_ok.sub(' ', e)
                # tmpE = tmpE.lower()
                if tmpE in EntityCleaner.stopWords:
                    new_entities += [None]
                else:
                    new_entities += [tmpE]

    return new_entities

def lemmatize(entities,sentences,nlp):

    new_entities = []
    print('len of self.entities before running lemmatize() : ' + str(len(entities)))
    print('len of None entities before running lemmatize() : ' + str(len([x for x in entities if x is None])))

    entities = [x for x in entities]
    sentences = [x for x in sentences]

    tuples = zip(entities, sentences)
    for e,sent in tqdm(tuples, total=len(entities)):
        if e is None:
            new_entities += [None]
        else:
            entityString = ''
            sent_tokens = nlp(sent)
            tokens = nlp(e)
            sent_token_strings = [t.text for t in sent_tokens]
            splits = e.split()
            #print(splits)
            for e_t in tokens:
                if e_t.text in sent_token_strings:
                    in_sent = True
                else: in_sent = False
                # remove leading punctuation chars
                tokenText = e_t.text.lstrip(''.join(EntityCleaner.punctuations2))

                if tokenText.startswith('#') or tokenText.startswith('@'):
                    entityString = " ".join([entityString, EntityCleaner.cleanString(tokenText)])
                elif tokenText == '%':
                    entityString = "".join([entityString, tokenText])

                #if the token is also part of the tokenizer output for the whole sentence, use the lemma from that tokenizer (it has more context ans is more accurate)
                elif in_sent:
                    # for verbal parts of entities (e.g. 'computing power') do not lemmatize
                    sent_token = sent_tokens[sent_token_strings.index(e_t.text)]
                    if "VB" in sent_token.tag_:
                        entityString = " ".join([entityString, sent_token.text.strip(''.join(EntityCleaner.punctuations2)).lower() if len(re.findall('\.', sent_token.lemma_)) == 1 else sent_token.text.lower()])
                    elif sent_token.tag_=='PROPN':
                        entityString = " ".join([entityString, sent_token.text.strip(''.join(EntityCleaner.punctuations2)).lower() if len(re.findall('\.', sent_token)) == 1 else sent_token.lower()])
                    else:
                        entityString = " ".join([entityString, sent_token.lemma_.strip(''.join(EntityCleaner.punctuations2)).lower() if len(re.findall('\.', sent_token.lemma_)) == 1 else sent_token.lemma_.lower()])

                # otherwise use the lemma from the spacy doc of the local entity
                else:
                    if "VB" in e_t.tag_:
                        entityString = " ".join([entityString, e_t.text.strip(''.join(EntityCleaner.punctuations2)).lower() if len(re.findall('\.', e_t.lemma_)) == 1 else e_t.text.lower()])
                    else:
                        entityString = " ".join([entityString, e_t.lemma_.strip(''.join(EntityCleaner.punctuations2)).lower() if len(re.findall('\.', e_t.lemma_)) == 1 else e_t.lemma_.lower()])


            new_entities += [entityString.strip()]

    return new_entities

def improve_entities(entities):
    new_entities = []
    print('len of self.entities before running improve_entities() : ' + str(len(entities)))
    entities = [x for x in entities]

    for e in tqdm(entities, total=len(entities)):
        if len(e) > 1:  # The entity string must have at least two characters
            e_improved = entity_string_improvement(e)
            new_entities += [e_improved]
        else:
            new_entities += [None]

    return  new_entities


class EntityCleaner:
    stopWords = set(stopwords.words('english')).difference({'it','its'})
    regex_puntuaction_ok = re.compile('[%s]' % re.escape("\"'_`"))  # possible characters
    puntuaction_reject = list("!,:;<=>?=[]^{|}~{}`") + ['\\']
    punctuations = ['-', '#', '.', ',', ':', '!', ';', '£', '$', '%', '&', '?', '*', '_', '@', '|', '>', '<', '°', '§',
                    '/', 'rt', '=']
    punctuations1 = ['-', '#', '.', ',', ':', '!', ';', '%', '&', '?', '*', '_', '@', '|', '>', '<', '°', '§', '/', '=']
    punctuations2 = ['-', '.', ',', ':', '!', ';', '&', '?', '*', '_', '|', '>', '<', '°', '§', '/', '=']

    def __init__(self, entities, sentences):
        self.entities = entities


        nlp = spacy.load('en_core_web_lg')
        # customizing the default tokenizer to not split hashtags
        re_token_match = spacy.tokenizer._get_regex_pattern(nlp.Defaults.token_match)
        # Add #hashtag pattern
        re_token_match = f"({re_token_match}|#\\w+)"
        # Add @mention pattern
        re_token_match = f"({re_token_match}|@[\\w_]+)"
        # Add @mention pattern
        re_token_match = f"({re_token_match}|\\$\\w+)"
        # Add hyphen-based pattern
        re_token_match = f"({re_token_match}|[\\w\\d]+\\-)"
        nlp.tokenizer.token_match = re.compile(re_token_match).match
        self.nlp = nlp
        self.sentences = sentences
        # this is a dict to convert non camel case hashtags to the form of previously encountered camel cases
        self.camel_casesMap = dict()

    def get_camel_casesMap(self):
        return self.camel_casesMap

    def set_camel_casesMap(self, map):
        self.camel_casesMap = map


    def is_camel_case(input):
        if input != input.lower() and input != input.upper() and "_" not in input and sum(i.isupper() for i in input[1:-1]) > 0:
            return True
        return False

    def cleanString(self,input):
        result = ""
        input = input.strip(''.join(EntityCleaner.punctuations1))
        if input.lower() in self.camel_casesMap.keys():
            return self.camel_casesMap[input.lower()]

        if EntityCleaner.is_camel_case(input):
            res_list = []
            res_list = [s.lower() for s in re.findall('[A-Z]+[^A-Z]*', input)]
            for s in res_list:
                result = result + ' ' + s
            self.camel_casesMap[input.lower()] = result.strip()
            return result.strip()
        else:
            input = input.replace('_', ' ', 1)
            return input.lower().strip()



    def puntuaction_and_stopword(self):

        new_entities = []
        print('len of self.entities before running puntuaction_and_stopword() : ' + str(len(self.entities)))
        print('len of None entities before running puntuaction_and_stopword() : ' + str(len([x for x in self.entities if x is None])))
        entities = [x for x in self.entities]
        for e in tqdm(entities, total=len(entities)):
            if e is None:
                new_entities += [None]

            else:
                valid_puntuaction = True

                for c in e:
                    if c in EntityCleaner.puntuaction_reject:
                        valid_puntuaction = False
                        break

                if not valid_puntuaction:
                    new_entities += [None]
                else:
                    tmpE = EntityCleaner.regex_puntuaction_ok.sub(' ', e)
                    #tmpE = tmpE.lower()
                    if tmpE in EntityCleaner.stopWords:
                        new_entities += [None]
                    else: new_entities += [tmpE]

        self.entities = new_entities



    def lemmatize(self):

        new_entities = []
        print('len of self.entities before running lemmatize() : ' + str(len(self.entities)))
        print('len of None entities before running lemmatize() : ' + str(len([x for x in self.entities if x is None])))

        entities = [x for x in self.entities]
        sentences = [x for x in self.sentences]

        tuples = zip(entities, sentences)
        for e,sent in tqdm(tuples, total=len(entities)):
            if e is None or sent is None:
                new_entities += [None]
            else:
                entityString = ''
                try:
                    sent_tokens = self.nlp(sent)
                except:
                    print(sent)
                    continue
                tokens = self.nlp(e)
                sent_token_strings = [t.text for t in sent_tokens]
                splits = e.split()
                #print(splits)
                for e_t in tokens:
                    if e_t.text in sent_token_strings:
                        in_sent = True
                    else: in_sent = False
                    # remove leading punctuation chars
                    tokenText = e_t.text.lstrip(''.join(EntityCleaner.punctuations2))

                    if tokenText.startswith('#') or tokenText.startswith('@'):
                        entityString = " ".join([entityString, EntityCleaner.cleanString(self,tokenText)])
                    elif tokenText == '%':
                        entityString = "".join([entityString, tokenText])

                    #if the token is also part of the tokenizer output for the whole sentence, use the lemma from that tokenizer (it has more context ans is more accurate)
                    elif in_sent:
                        # for verbal parts of entities (e.g. 'computing power') do not lemmatize
                        sent_token = sent_tokens[sent_token_strings.index(e_t.text)]
                        if "VB" in sent_token.tag_:
                            entityString = " ".join([entityString, sent_token.text.strip(''.join(EntityCleaner.punctuations2)).lower() if len(re.findall('\.', sent_token.lemma_)) == 1 else sent_token.text.lower()])
                        elif sent_token.tag_=='PROPN':
                            entityString = " ".join([entityString, sent_token.text.strip(''.join(EntityCleaner.punctuations2)).lower() if len(re.findall('\.', sent_token)) == 1 else sent_token.lower()])
                        else:
                            entityString = " ".join([entityString, sent_token.lemma_.strip(''.join(EntityCleaner.punctuations2)).lower() if len(re.findall('\.', sent_token.lemma_)) == 1 else sent_token.lemma_.lower()])

                    # otherwise use the lemma from the spacy doc of the local entity
                    else:
                        if "VB" in e_t.tag_:
                            entityString = " ".join([entityString, e_t.text.strip(''.join(EntityCleaner.punctuations2)).lower() if len(re.findall('\.', e_t.lemma_)) == 1 else e_t.text.lower()])
                        else:
                            entityString = " ".join([entityString, e_t.lemma_.strip(''.join(EntityCleaner.punctuations2)).lower() if len(re.findall('\.', e_t.lemma_)) == 1 else e_t.lemma_.lower()])


                new_entities += [entityString.strip()]

        self.entities = new_entities




    '''
	Methods entity_string_improvement() and improve_entities() remove characters that appear in some entities (e.g. 'ontology, # n #).
	They also remove entities that start with a number
	
	'''
    def entity_string_improvement(e):
        improved_entity_string = e.strip()
        first_character = improved_entity_string[0]
        last_character = improved_entity_string[-1]

        while (True):
            if first_character in EntityCleaner.puntuaction_reject:
                improved_entity_string = improved_entity_string[1:]
                if len(improved_entity_string) >= 1:
                    first_character = improved_entity_string[0]
                else:
                    break
            # print(e, '->', improved_entity_string)
            else:
                break

        while (True):
            if last_character in EntityCleaner.puntuaction_reject:
                improved_entity_string = improved_entity_string[:-1]
                if len(improved_entity_string) >= 1:
                    last_character = improved_entity_string[-1]
                else:
                    break
            # print(e, '->', improved_entity_string)
            else:
                break
        return improved_entity_string.strip()

    # keep only entities of length > 1 and strip leading/trimming punctuation chars of the whole token sequence
    def improve_entities(self):
        new_entities = []
        print('len of self.entities before running improve_entities() : ' + str(len(self.entities)))
        entities = [x for x in self.entities]

        for e in tqdm(entities, total=len(entities)):
            try:
                if len(e) > 1:  # The entity string must have at least two characters
                    e_improved = entity_string_improvement(e)
                    new_entities += [e_improved]
                else:
                    new_entities += [None]
            except:
                print(e)


        self.entities = new_entities





    def getEntitiesCleaned(self):
        return self.entities

    def getRelationsCleaned(self):
        return self.relations

    def toBritishSpelling(self):
        new_entities = []
        #print('len of self.entities before running toBritishSpelling() : ' + str(len(self.entities)))
        #print('len of None entities before running toBritishSpelling() : ' + str(len([x for x in self.entities if x is None])))
        entities = [x for x in self.entities]
        for e in tqdm(entities, total=len(entities)):
            if e is None:
                new_entities += [None]
            else:
                entityString = ''
                res = e.split()
                for r in res:
                    bspel = get_british_spelling(r)
                    #if (r != bspel):
                        #print(r, bspel)
                    entityString = " ".join([entityString,bspel])


                new_entities += [entityString.strip()]

        self.entities = new_entities


    def run(self):
        self.improve_entities()
        self.puntuaction_and_stopword()
        self.lemmatize()
        self.toBritishSpelling()







