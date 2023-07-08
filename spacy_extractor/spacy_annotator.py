import pickle
import sys

import multiprocessing as mp
import json
import os
import re

from EntityExtraction import *



from nltk import RegexpParser, tree
from nltk.tokenize import TweetTokenizer

import preprocessor as p
p.set_options(p.OPT.EMOJI, p.OPT.RESERVED, p.OPT.SMILEY, p.OPT.NUMBER)

import spacy
spacy.cli.download("en_core_web_sm")
# Load English tokenizer, tagger, parser and NER
nlp_spacy = spacy.load("en_core_web_sm")

# We want to also keep #hashtags as a token, so we modify the spaCy model's token_match
# We want to also keep #hashtags as a token, so we modify the spaCy model's token_match
import re

try:
# Python 2.6-2.7
    from HTMLParser import HTMLParser
except ImportError:
# Python 3
    from html.parser import HTMLParser

h = HTMLParser()


# Retrieve the default token-matching regex pattern
re_token_match = spacy.tokenizer._get_regex_pattern(nlp_spacy.Defaults.token_match)
# Add #hashtag pattern
re_token_match = f"({re_token_match}|#\\w+)"
nlp_spacy.tokenizer.token_match = re.compile(re_token_match).match


number_of_processes = 6
GRAMMAR = "NOUNS: {<VBG.*>*<JJ.*>*<NN.*>*<POS.*>*<NN.*>+}"
grammar_parser = RegexpParser(GRAMMAR)

#data_path = 'D:/GitRepos/GitRepos/skg/data-collection/core_api/sample2018_dt_corrected' # input json directory
data_path = '/data-collection/twitter'
split_folder = 'D:/GitRepos/GitRepos/skg/data-collection/twitter/sample'
#output_dir_spacy = 'D:/GitRepos/GitRepos/skg/data-collection/core_api/output_spacy_1/' # output json directory
output_dir_spacy = '/data-collection/twitter/output_spacy_annotations/'  # output json directory


def spacy_extraction(filename):
	print('> spacy processing:', filename)

	f = open(os.path.join(data_path,filename), 'r')
	content = f.read()
	json_content = json.loads(content)
	f.close()

	try:
		spacy_doc = nlp_spacy(json_content['title'] + " " + json_content['abstract'] + " " + json_content['fullText'])
		doc_data = spacy_doc.to_json()

	except Exception as e:
		print(e)

	fw = open(output_dir_spacy + filename, 'w+')
	json.dump({
		'id': filename.replace('.json', ''),
		'spacy_output': doc_data
	}, fw)
	fw.flush()
	fw.close()


def spacy_extraction_tweet(filename):
	try:

		f = open(os.path.join(split_folder,filename), 'r', errors='ignore')
		content = json.load(f)

		unescapedText = h.unescape(content['text'])

		preprocessed_text = preprocessCleanText(unescapedText)

		preprocessed_text = re.sub(r"(\w[\?!;:\.\#])(\w+)", r"\1 \2 ", preprocessed_text)
		spacy_doc = nlp_spacy(preprocessed_text)
		doc_data = spacy_doc.to_json()
		if 'sents' not in doc_data.keys():
			sentence = {}
			sentence['text']  = doc_data['text']
			sentence['tokens'] = doc_data['tokens']
			doc_data['sents'] = []
			doc_data['sents'].append(sentence)
		else:
			for sentence in doc_data['sents']:
				sentenceJson = {}
				sentenceText = utils.rebuild_spacy_sentence(sentence, doc_data)
				sentence_doc = nlp_spacy(sentenceText).to_json()
				sentenceTokens = sentence_doc['tokens']
				doc_data['sents'][doc_data['sents'].index(sentence)]['text']=sentenceText
				doc_data['sents'][doc_data['sents'].index(sentence)]['tokens'] = sentenceTokens

		f.close()

		fw = open(output_dir_spacy + filename, 'w+')
		json.dump({
			'id' : content['id'],
			'spacy_output': doc_data
		}, fw)
		fw.flush()
		fw.close()
	except Exception as e:
		print(e.__cause__)




def preprocessText(text):
	return p.tokenize(text)




def get_pos_tags(sentence):
    pos_tags = []
    if "tokens" in sentence:
        pos_tags = [(p['originalText'], p['pos']) for p in sentence['tokens']]
    return pos_tags

def get_entities(pos_tags):
    chunks = list()
    pos_tags_with_grammar = grammar_parser.parse(pos_tags)
    #print(pos_tags_with_grammar)
    for node in pos_tags_with_grammar:
        if isinstance(node, tree.Tree) and node.label() == 'NOUNS': # if matches our grammar
            chunk = ''
            for leaf in node.leaves():
                concept_chunk = leaf[0]
                concept_chunk = re.sub('[\=\,\…\+\-\–\“\”\"\/\[\]\®\™\%]', ' ', concept_chunk)
                concept_chunk = re.sub('[\’\'\‘]', "'", concept_chunk)
                concept_chunk = re.sub('\.$|^\.', '', concept_chunk)
                concept_chunk = concept_chunk.strip()
                chunk += ' ' + concept_chunk if concept_chunk != "'" else "" + concept_chunk
            chunk = re.sub('\.+', '.', chunk)
            chunk = re.sub('\s+', ' ', chunk)
            chunk = chunk.strip()
            chunks.append(chunk)
    return chunks



def preprocessCleanText(text):
	return p.clean(text)


def preprocessTokenizeText(text):
	return p.tokenize(text)


def spacy_extraction_tweet_string(text):
	spacy_doc = nlp_spacy(text)
	return spacy_doc.to_json()

def merge_punct(doc):
    spans = []
    for word in doc[:-1]:
        if word.is_punct or not word.nbor(1).is_punct:
            continue
        start = word.i
        end = word.i + 1
        while end < len(doc) and doc[end].is_punct:
            end += 1
        span = doc[start:end]
        spans.append((span, word.tag_, word.lemma_, word.ent_type_))
    with doc.retokenize() as retokenizer:
        for span, tag, lemma, ent_type in spans:
            attrs = {"tag": tag, "lemma": lemma, "ent_type": ent_type}
            retokenizer.merge(span, attrs=attrs)
    return doc

if __name__ == '__main__':

	try:
		os.mkdir(output_dir_spacy)
		print("Directory", output_dir_spacy, "created ")
	except FileExistsError:
		print("Directory", output_dir_spacy, "already exists")

	already_parsed = os.listdir(output_dir_spacy)
	files_to_parse = [filename for filename in os.listdir(split_folder) if filename not in already_parsed]
	print('> Start spacy')
	pool1 = mp.Pool(number_of_processes)
	result = pool1.map(spacy_extraction_tweet, files_to_parse)


