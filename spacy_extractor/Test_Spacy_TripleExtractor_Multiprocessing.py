from joblib import Parallel, delayed

from spacy_extractor.EntityExtraction import *

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
from tqdm import tqdm

import spacy
from spacy.tokens import Doc
from spacy.tokens import DocBin
import coreferee
from wasabi import msg

from spacy import displacy

import pandas as pd
import matplotlib.pyplot as plt

import preprocessor as p

# p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.RESERVED, p.OPT.SMILEY)
p.set_options(p.OPT.EMOJI, p.OPT.RESERVED, p.OPT.SMILEY)

# Import the Language object under the 'language' module in spaCy,
# and NumPy for calculating cosine similarity.
from spacy.language import Language
import numpy as np

import multiprocessing

cores = multiprocessing.cpu_count()  # Count the number of cores in a computer
print('number of cores = ', cores)


# We use the @ character to register the following Class definition
# with spaCy under the name 'tensor2attr'.
@Language.factory('tensor2attr')
# We begin by declaring the class name: Tensor2Attr. The name is
# declared using 'class', followed by the name and a colon.
class Tensor2Attr:

    # We continue by defining the first method of the class,
    # __init__(), which is called when this class is used for
    # creating a Python object. Custom components in spaCy
    # require passing two variables to the __init__() method:
    # 'name' and 'nlp'. The variable 'self' refers to any
    # object created using this class!
    def __init__(self, name, nlp):
        # We do not really do anything with this class, so we
        # simply move on using 'pass' when the object is created.
        pass

    # The __call__() method is called whenever some other object
    # is passed to an object representing this class. Since we know
    # that the class is a part of the spaCy pipeline, we already know
    # that it will receive Doc objects from the preceding layers.
    # We use the variable 'doc' to refer to any object received.
    def __call__(self, doc):
        # When an object is received, the class will instantly pass
        # the object forward to the 'add_attributes' method. The
        # reference to self informs Python that the method belongs
        # to this class.
        self.add_attributes(doc)

        # After the 'add_attributes' method finishes, the __call__
        # method returns the object.
        return doc

    # Next, we define the 'add_attributes' method that will modify
    # the incoming Doc object by calling a series of methods.
    def add_attributes(self, doc):
        # spaCy Doc objects have an attribute named 'user_hooks',
        # which allows customising the default attributes of a
        # Doc object, such as 'vector'. We use the 'user_hooks'
        # attribute to replace the attribute 'vector' with the
        # Transformer output, which is retrieved using the
        # 'doc_tensor' method defined below.
        doc.user_hooks['vector'] = self.doc_tensor

        # We then perform the same for both Spans and Tokens that
        # are contained within the Doc object.
        doc.user_span_hooks['vector'] = self.span_tensor
        doc.user_token_hooks['vector'] = self.token_tensor

        # We also replace the 'similarity' method, because the
        # default 'similarity' method looks at the default 'vector'
        # attribute, which is empty! We must first replace the
        # vectors using the 'user_hooks' attribute.
        doc.user_hooks['similarity'] = self.get_similarity
        doc.user_span_hooks['similarity'] = self.get_similarity
        doc.user_token_hooks['similarity'] = self.get_similarity

    # Define a method that takes a Doc object as input and returns
    # Transformer output for the entire Doc.
    def doc_tensor(self, doc):
        # Return Transformer output for the entire Doc. As noted
        # above, this is the last item under the attribute 'tensor'.
        # Average the output along axis 0 to handle batched outputs.
        return doc._.trf_data.tensors[-1].mean(axis=0)

    # Define a method that takes a Span as input and returns the Transformer
    # output.
    def span_tensor(self, span):
        # Get alignment information for Span. This is achieved by using
        # the 'doc' attribute of Span that refers to the Doc that contains
        # this Span. We then use the 'start' and 'end' attributes of a Span
        # to retrieve the alignment information. Finally, we flatten the
        # resulting array to use it for indexing.
        tensor_ix = span.doc._.trf_data.align[span.start: span.end].data.flatten()

        # Fetch Transformer output shape from the final dimension of the output.
        # We do this here to maintain compatibility with different Transformers,
        # which may output tensors of different shape.
        out_dim = span.doc._.trf_data.tensors[0].shape[-1]

        # Get Token tensors under tensors[0]. Reshape batched outputs so that
        # each "row" in the matrix corresponds to a single token. This is needed
        # for matching alignment information under 'tensor_ix' to the Transformer
        # output.
        tensor = span.doc._.trf_data.tensors[0].reshape(-1, out_dim)[tensor_ix]

        # Average vectors along axis 0 ("columns"). This yields a 768-dimensional
        # vector for each spaCy Span.
        return tensor.mean(axis=0)

    # Define a function that takes a Token as input and returns the Transformer
    # output.
    def token_tensor(self, token):
        # Get alignment information for Token; flatten array for indexing.
        # Again, we use the 'doc' attribute of a Token to get the parent Doc,
        # which contains the Transformer output.
        tensor_ix = token.doc._.trf_data.align[token.i].data.flatten()

        # Fetch Transformer output shape from the final dimension of the output.
        # We do this here to maintain compatibility with different Transformers,
        # which may output tensors of different shape.
        out_dim = token.doc._.trf_data.tensors[0].shape[-1]

        # Get Token tensors under tensors[0]. Reshape batched outputs so that
        # each "row" in the matrix corresponds to a single token. This is needed
        # for matching alignment information under 'tensor_ix' to the Transformer
        # output.
        tensor = token.doc._.trf_data.tensors[0].reshape(-1, out_dim)[tensor_ix]

        # Average vectors along axis 0 (columns). This yields a 768-dimensional
        # vector for each spaCy Token.
        return tensor.mean(axis=0)

    # Define a function for calculating cosine similarity between vectors
    def get_similarity(self, doc1, doc2):
        # Calculate and return cosine similarity
        return np.dot(doc1.vector, doc2.vector) / (doc1.vector_norm * doc2.vector_norm)


preprocessing = True
process_annotation = True
pronoun_resolution = True

# We want to also keep #hashtags as a token, so we modify the spaCy model's token_match
import re
import datetime
import html
from spacy.tokens import Token
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex


# Define lightweight function for resolving references in text
def resolve_references(doc: Doc) -> str:
    """Function for resolving references with the coref ouput
    doc (Doc): The Doc object processed by the coref pipeline
    RETURNS (str): The Doc string with resolved references
    """
    # token.idx : token.text
    token_mention_mapper = {}
    output_string = ""
    clusters = [
        val for key, val in doc.spans.items() if key.startswith("coref_cluster")
    ]

    # Iterate through every found cluster
    for cluster in clusters:
        first_mention = cluster[0]
        # Iterate through every other span in the cluster
        for mention_span in list(cluster)[1:]:
            # Set first_mention as value for the first token in mention_span in the token_mention_mapper
            token_mention_mapper[mention_span[0].idx] = first_mention.text + mention_span[0].whitespace_

            for token in mention_span[1:]:
                # Set empty string for all the other tokens in mention_span
                token_mention_mapper[token.idx] = ""

    # Iterate through every token in the Doc
    for token in doc:
        # Check if token exists in token_mention_mapper
        if token.idx in token_mention_mapper:
            output_string += token_mention_mapper[token.idx]
        # Else add original token text
        else:
            output_string += token.text + token.whitespace_

    return output_string


def tokens_to_spacy_dep_tree(tokens, attach_root=True):
    '''convert a dependency tree from Stanford Corenlp's JSON output format
        to Spacy's visualizer format
    params:
    '''
    words = [{'text': t['lemma'], 'tag': t['tag']} for t in tokens]
    if attach_root:
        words = [{'text': '[ROOT]', 'tag': 'ROOT'}] + words

        arcs = []
        for t in tokens:
            if attach_root:
                arc = {'start': t['head'], 'end': t['id'], 'label': t['dep'],
                       'dir': 'left' if t['head'] > t['id'] else 'right'}
                arcs.append(arc)
            else:
                if t['id'] != t['head']:
                    arc = {'start': t['head'] - 1, 'end': t['id'] - 1, 'label': t['dep'],
                           'dir': 'left' if t['head'] > t['id'] else 'right'}
                    arcs.append(arc)
    tree = {'words': words, 'arcs': arcs}
    return tree


# 1. drop sequences of entity mentions (@) at the beginning of sentence (including when prefixed by “RT” or punctuations). If size of sequence is one, it must not be followed by verb
# 2. for any sequence of n >= 2 hashtags/mentions, drop subsequence [1:n]

def processAnnotation(annotation):
    nodes_to_remove = set()
    poss = [token for i, token in enumerate(annotation)]
    sentences = [sent for sent in annotation.sents]

    sentenceMap = {sentence.text: i for i, sentence in enumerate(sentences)}

    # 1. drop sequences of entity mentions (@) at the beginning of sentence (including when prefixed by “RT” or punctuations). If size of sequence is one, it must not be followed by verb
    for sentIndex in sentenceMap.values():
        found = False
        prefix_nodes = set()
        startToken_index = sentences[sentIndex].start
        endToken_index = sentences[sentIndex].end
        for i in range(min(2, endToken_index - startToken_index - 1)):
            # if (poss[startToken_index+i].is_punct or poss[startToken_index+i].text in ['RT','rt','VIA','via']):
            prefix_nodes.add(startToken_index + i)
            if poss[startToken_index + i].text.startswith('@') and (
                    poss[startToken_index + i].is_sent_end or 'VB' not in poss[startToken_index + i + 1].tag_):
                found = True
                break
        if found:
            nodes_to_remove.update(prefix_nodes)
            nodes_to_remove.add(startToken_index + i)

            for j, token in enumerate(poss[startToken_index + i + 1: endToken_index]):
                if token.text.startswith('@') or token.tag_ == ':':
                    nodes_to_remove.add(token.i)
                else:
                    break

    # 1. drop sequences of entity mentions (@) at the beginning of sentence (including when prefixed by “RT” or punctuations). If size of sequence is one, it must not be followed by verb
    # sentStart = False
    found = False
    #
    # for token in poss:
    #   if token.is_sent_start:
    #        sentStart=True
    #    else: sentStart=False

    #     if sentStart:
    #        start_token = sentences[sentenceMap[token.sent.text]].start
    #        end_token = sentences[sentenceMap[token.sent.text]].end
    #        sent_len = end_token-start_token
    #        prefix_nodes=set()
    #        for i in range(min(2,sent_len)):
    #            if (poss[token.i+i].is_punct or poss[token.i+i].text in ['RT','rt','VIA','via']):
    #                prefix_nodes.add(token.i+i)
    #            try:
    #                if poss[token.i+i].text.startswith('@') and ( poss[token.i+i].is_sent_end or 'VB' not in poss[token.i+i+1].tag_):
    #                    found = True
    #                    break
    #            except IndexError:
    #                print('sentence: ' + token.sent.text)
    #        if found:
    #           nodes_to_remove.add(prefix_nodes)
    #           nodes_to_remove.add(token.i+i)
    #            found=False
    # if i > 0:
    #    nodes_to_remove.add(token.i+i-1)

    #            length_mentions_prefix = 1
    #           for j, token in enumerate(poss[token.i+i+1: end_token]):
    #                if token.text.startswith('@') or token.tag_ == ':':
    #                    nodes_to_remove.add(token.i)
    #                else:
    #                    break

    # if length_mentions_prefix > 1:
    #   for k in range(token.i+i+1, token.i+i+1+j):
    #      nodes_to_remove.add(k)
    # edges_to_remove.update(g.in_edges(k+1))

    # 2. for any sequence of n >= 2 hashtags/mentions/URL, drop subsequence [1:n]
    length_hashtag_mention_url_sequence = 0
    node_list = []
    for i, tok in enumerate(poss):
        if tok.text.startswith('#') or tok.text.startswith('@') or tok.like_url:
            node_list.append(i)
            length_hashtag_mention_url_sequence += 1
        else:
            if length_hashtag_mention_url_sequence >= 2:
                nodes_to_remove.update(node_list[1:])
            length_hashtag_mention_url_sequence = 0
            node_list = []

    if length_hashtag_mention_url_sequence >= 2:
        nodes_to_remove.update(node_list[1:])

    # 3. remove all URLs
    for i, tok in enumerate(poss):
        if tok.like_url:
            nodes_to_remove.add(i)

    # if some edges were to be removed, modify the text and re-annotate the whole doc:
    if len(nodes_to_remove) > 0:

        # modify the text
        nodes_to_remove = list(nodes_to_remove)
        nodes_to_remove.sort()
        out_text = annotation.text
        for n in nodes_to_remove:
            try:
                out_text = out_text.replace(poss[n].text, '', 1)
            except IndexError:
                print(token.sent.text)

        out_text = out_text.strip()
        if len(out_text) <= 3:
            return None

        else:
            annotation = spacyNLP(out_text)
            return annotation, [token for i, token in enumerate(annotation)]
    else:
        return annotation, [token for i, token in enumerate(annotation)]


# 1. drop sequences of entity mentions (@) at the beginning of sentence (including when prefixed by “RT” or punctuations). If size of sequence is one, it must not be followed by verb
# 2. for any sequence of n >= 2 hashtags/mentions, drop subsequence [1:n]

def processSentenceAnnotation(text, sentence_doc):
    nodes_to_remove = set()

    # 1. drop sequences of entity mentions (@) at the beginning of sentence (including when prefixed by “RT” or punctuations). If size of sequence is one, it must not be followed by verb
    found = False
    poss = [t for t in sentence_doc]

    for i, token in enumerate(poss[0:min(3, len(poss) - 3)]):
        # for i,token in range(min(2, len(poss) - 2)):
        if token.text.startswith('@') and 'VB' not in poss[poss.index(token) + 1].tag_:
            found = True
            break
    if found:
        nodes_to_remove.add(i)
        if i > 0:
            nodes_to_remove.add(i - 1)

        length_mentions_prefix = 1
        for j, token in enumerate(poss[i + 1: len(poss) - 1]):
            if token.text.startswith('@') or token.tag == ':':
                length_mentions_prefix += 1
            else:
                break

        if length_mentions_prefix > 1:
            for k in range(i + 1, j):
                nodes_to_remove.add(k)
                # edges_to_remove.update(g.in_edges(k+1))

    # 2. for any sequence of n >= 2 hashtags/mentions/URL, drop subsequence [1:n]
    length_hashtag_mention_url_sequence = 0
    node_list = []
    for i, pos in enumerate(poss):
        if pos.text.startswith('#') or pos.text.startswith('@') or pos.like_url:
            node_list.append(i)
            length_hashtag_mention_url_sequence += 1
        else:
            if length_hashtag_mention_url_sequence >= 2:
                nodes_to_remove.update(node_list[1:])
            length_hashtag_mention_url_sequence = 0
            node_list = []
    if length_hashtag_mention_url_sequence >= 2:
        nodes_to_remove.update(node_list[1:])

    # 3. remove all URLs
    for i, pos in enumerate(poss):
        if pos.like_url:
            nodes_to_remove.add(i)

    # if some edges were to be removed, modify the text and re-annotate the sentence:
    if len(nodes_to_remove) > 0:

        # modify the text
        nodes_to_remove = list(nodes_to_remove)
        nodes_to_remove.sort()
        out_text = text
        for n in nodes_to_remove:
            out_text = out_text.replace(poss[n].text, '', 1)

        out_text = out_text.strip()
        if len(out_text) <= 3:
            return None

        else:
            sentence_doc = spacyNLP(out_text)
            # tokens = sentence_doc['tokens']
            if not sentence_doc.to_json()['sents']:
                return None
            else:
                # poss = annotation['sents'][0]['tokens']
                # depparses = annotation['sentences'][0]['basicDependencies']
                return sentence_doc, [token for i, token in enumerate(sentence_doc)]
    else:
        return sentence_doc, [token for i, token in enumerate(sentence_doc)]


def preprocessCleanText(text, p):
    return p.clean(text)


def preprocessTweet(text, p):
    unescapedText = html.unescape(text)
    preprocessed_text = preprocessCleanText(unescapedText, p)
    preprocessed_text = re.sub(r"(\w[\?!\);])(\w+)", r"\1 \2 ", preprocessed_text)
    preprocessed_text = re.sub(r"(\.)(#)", r"\1 \2 ", preprocessed_text)
    preprocessed_text = re.sub(r"(\w)(http)", r"\1 \2 ", preprocessed_text)
    preprocessed_text = re.sub(r"(\w)([\?!])", r"\1 \2 ", preprocessed_text)
    return preprocessed_text


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def process(indmatrixinf, indmatrixsup):
    tic = time.time()

    db =DocBin()
    kg_df = pd.DataFrame(
        columns=['doc_id', 'sentence_n', 'triple_subj', 'triple_subj_lemma', 'triple_subj_pron', 'triple_rel',
                 'triple_rel_lemma', 'triple_obj', 'triple_obj_lemma', 'triple_obj_pron', 'path', 'dep_tree_path',
                 'pos_path', 'neg', 'pronoun', 'sentence', 'original_tweet', 'doc_counter'])
    # tweet_id:token_length
    accepted_tweets = {}
    # tweet_id:token_length
    discarded_tweets = {}
    # tweet_id:token_length
    accepted_tweets_processed = {}
    # tweet_id:token_length
    discarded_tweets_processed = {}
    entity_lemma_distr = dict()
    relation_lemma_distr = dict()
    # rel-->(lemma, count)
    relations = list()
    counter=0
    counter_used_docs = 0
    coreferences = 0

    for line in tqdm(sliced_lines[indmatrixinf:(indmatrixsup + 1)]):
        if counter > 0 and  counter % 100 == 0:
            print('Processed 100 docs!')
        tweet_length = 0
        found_pattern = False
        splitting = line.split('\t')
        tweetId = splitting[0]
        originalText = splitting[2]
        if preprocessing:
            text = preprocessTweet(originalText, p)
        else:
            text = originalText

        annotation = spacyNLP(text)

        # check tweet token length before tweet processing in order to collect statistics on tweets from which triples are extracted
        tweet_length = len([token for i, token in enumerate(annotation)])

        if process_annotation:
            result = processAnnotation(annotation)

        # for k in range(tweet_length):
        #   print(annotation[k].vector)

        if result is not None:
            db.add(result[0])
            counter_used_docs += 1
            tweet_length_processed = len([token for i, token in enumerate(annotation)])
            tokens = result[1]
            sent_counter = 0
            for sentence in result[0].sents:
                sentenceNum = str(sent_counter)
                sent_start_char = sentence.start
                sentenceText = sentence.text
                sentence_tokens = [token for i, token in enumerate(result[0]) if token.sent == sentence]
                sentence_tokens_ids = [token.i for i, token in enumerate(result[0]) if token.sent == sentence]
                g = build_graph_spacy_latest(sentence_tokens_ids, tokens)
                if pronoun_resolution:
                    entities_indices = get_entity_indices_spacy_w_pronouns(sentence_tokens, g)
                else:
                    entities_indices = get_entity_indices_spacy_no_pronouns(sentence_tokens, g)
                remove_repeated_indices(entities_indices)
                entities = reorder_indices(entities_indices)
                path_graph = build_path_graph_spacy_latest(sentence_tokens_ids, tokens)
                intermediate_paths = get_path_between_entities_spacy(entities, path_graph)
                paths = filterTargetPaths(intermediate_paths)
                if len(paths) > 0:

                    for path in paths:
                        pronoun_entity = False
                        result1 = rebuild_spacy_entity_with_coref(result[0],
                                                                  find_entity_index_set(path[0][0][0], entities),
                                                                  tokens)
                        result2 = rebuild_spacy_entity_with_coref(result[0], find_entity_index_set(
                            path[0][len(path[0]) - 1][0], entities), tokens)
                        if result1 is not None and result2 is not None:

                            found_pattern = True
                            entityString = result1[0]
                            toString = result2[0]
                            if result1[1] != '' or result2[1] != '':
                                coreferences += 1
                                pronoun_entity = True
                            subjPron = result1[1]
                            objPron = result2[1]
                            entitySubjPOS = path[0][0][1]
                            relPOS = path[3]
                            entityObjPOS = path[0][len(path[0]) - 1][1]

                            entityLemmatizedString = rebuild_spacy_entity_with_coref_lemmatize(result[0],
                                                                                               find_entity_index_set(
                                                                                                   path[0][0][0],
                                                                                                   entities),
                                                                                               tokens)

                            if entityLemmatizedString in entity_lemma_distr.keys():
                                entity_lemma_distr[entityLemmatizedString] = entity_lemma_distr[
                                                                                 entityLemmatizedString] + 1
                            else:
                                entity_lemma_distr[entityLemmatizedString] = 1

                            fromString = entityString

                            toStringLemmatized = rebuild_spacy_entity_with_coref_lemmatize(result[0],
                                                                                           find_entity_index_set(
                                                                                               path[0][len(
                                                                                                   path[0]) - 1][0],
                                                                                               entities), tokens)
                            if toStringLemmatized in entity_lemma_distr.keys():
                                entity_lemma_distr[toStringLemmatized] = entity_lemma_distr[toStringLemmatized] + 1
                            else:
                                entity_lemma_distr[toStringLemmatized] = 1

                            relString = ' '.join(
                                [tokens[x].text for x in getRelIndexes(path, g, tokens)]).strip().lower()

                            relStringLemmatized = tokens[find_rel_index(path)].lemma_.lower()

                            relations.append(
                                [relString, relStringLemmatized, counter_used_docs, getRelIndexes(path, g, tokens)])

                            if relStringLemmatized in relation_lemma_distr.keys():
                                relation_lemma_distr[relStringLemmatized] = relation_lemma_distr[
                                                                                relStringLemmatized] + 1
                            else:
                                relation_lemma_distr[relStringLemmatized] = 1

                            pathString = fromString + ";" + relString + ";" + toString
                            deep_tree_pathString = path[1]
                            negation = path[2]
                            triple_subjString = fromString
                            triple_relString = relString
                            triple_objString = toString
                            kg_df = kg_df.append(
                                {'doc_id': tweetId, 'sentence_n': sentenceNum, 'path': pathString,
                                 'dep_tree_path': deep_tree_pathString,
                                 'pos_path': entitySubjPOS + ',' + relPOS + ',' + entityObjPOS,
                                 'triple_subj': triple_subjString, 'triple_subj_lemma': entityLemmatizedString,
                                 'triple_subj_pron': subjPron,
                                 'triple_rel': triple_relString,
                                 'triple_rel_lemma': relStringLemmatized, 'triple_obj': triple_objString,
                                 'triple_obj_lemma': toStringLemmatized, 'triple_obj_pron': objPron,
                                 'neg': negation, 'pronoun': pronoun_entity, 'sentence': sentence.text,
                                 'original_tweet': originalText, 'doc_counter': counter
                                 # 'dep_tree': tokens_to_spacy_dep_tree(tokens)
                                 },
                                ignore_index=True)
                # else:
                # discarded_tweets+=1
                sent_counter += 1

    if found_pattern:
        if tweetId in accepted_tweets.keys():
            accepted_tweets[tweetId + '_1'] = tweet_length
        else:
            accepted_tweets[tweetId] = tweet_length
    else:
        if tweetId in discarded_tweets.keys():
            discarded_tweets[tweetId + '_1'] = tweet_length
        else:
            discarded_tweets[tweetId] = tweet_length

    if found_pattern:
        if tweetId in accepted_tweets_processed.keys():
            accepted_tweets_processed[tweetId + '_1'] = tweet_length_processed
        else:
            accepted_tweets_processed[tweetId] = tweet_length_processed
    else:
        if tweetId in discarded_tweets_processed.keys():
            discarded_tweets_processed[tweetId + '_1'] = tweet_length_processed
        else:
            discarded_tweets_processed[tweetId] = tweet_length_processed
    counter += 1

    toc = time.time()
    print('time process slice of tweets = ' + str((toc - tic) / 60) + ' minutes')

    return db,kg_df,accepted_tweets,discarded_tweets,relation_lemma_distr,relations


if __name__ == '__main__':

    test_folder = 'D:/GitRepos/GitRepos/skg/data-collection/twitter/tests'
    preprocessing_folder = 'D:/GitRepos/GitRepos/skg/data-collection/twitter/tests/spacy/results_preprocessing_heuristics_100'
    no_preprocessing_folder = 'D:/GitRepos/GitRepos/skg/data-collection/twitter/tests/spacy/results_preprocessing_100'

    tic = time.time()
    # spacyNLP = spacy.load("en_core_web_lg")

    spacyNLP = spacy.load("en_core_web_trf")
    if pronoun_resolution:
        spacyNLP.add_pipe('coreferee')
    spacyNLP.add_pipe('tensor2attr')

    print(spacyNLP.pipeline)
    db = DocBin()

    # spacyNLP = spacy.load("en_core_web_trf")

    # spacyNLPCoref = spacy.load("en_coreference_web_trf")

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

    spacyNLP.to_disk(os.path.join(test_folder, 'serializedSpacy'))

    toc = time.time()
    print('time to load, customize and save serialization of the spacy models = ' + str((toc - tic) / 60) + ' minutes')

    kg_df_corenlp = pd.DataFrame(
        columns=['doc_id', 'sentence_n', 'triple_subj', 'triple_subj_lemma', 'triple_subj_pron', 'triple_rel',
                 'triple_rel_lemma', 'triple_obj', 'triple_obj_lemma', 'triple_obj_pron', 'path', 'dep_tree_path',
                 'pos_path', 'neg', 'pronoun', 'sentence', 'original_tweet', 'doc_counter'])

    file = open(os.path.join('D:/GitRepos/GitRepos/skg/data-collection/twitter', 'sample100k.tsv'), mode="r",
                encoding="utf-8")
    # Get a dependency tree from a Stanford CoreNLP pipeline

    lines = [line for line in file]
    start = 0
    end = 1000
    sliced_lines = lines[start:end]
    file_lines = sum(1 for line in sliced_lines)

    cores = multiprocessing.cpu_count()  # Count the number of cores in a computer
    n_jobs = cores

    batch_size = int(int(len(sliced_lines) / n_jobs))
    print('batch_size = ' + str(batch_size))
    parts = chunkIt([x for x in range(file_lines)], n_jobs)
    indmatrixinf = []
    indmatrixsup = []
    for batch_inds in parts:
        indmatrixinf.append(batch_inds[0])
        indmatrixsup.append(batch_inds[len(batch_inds) - 1])

    print(indmatrixinf)
    print(indmatrixsup)
    startTime = time.time()
    # executor = Parallel(n_jobs=n_jobs)
    executor = Parallel(n_jobs=n_jobs, batch_size=batch_size,
                        verbose=15)  # "loky"  "threading",  "multiprocessing", backend="multiprocessing"
    do = delayed(process)
    tasks = (do(indmatrixinf[i], indmatrixsup[i]) for i in range(0, len(indmatrixinf)))
    rv = executor(tasks)

    docs_lemma = []
    docs_lemma_sentiment = []
    docs_spanssentences = []
    docs_textsentences = []
    docs_tensesentences = []
    docs_locatsentences = []
    for r in rv:
        docs_lemma.extend(r[0])
        docs_lemma_sentiment.extend(r[1])

    print('Time to execute all the batches of tweets = '  + str(time.time() - startTime))


    counter = 1
    counter_used_docs = 0
    # tweet_id:token_length
    accepted_tweets = {}
    # tweet_id:token_length
    discarded_tweets = {}

    # tweet_id:token_length
    accepted_tweets_processed = {}
    # tweet_id:token_length
    discarded_tweets_processed = {}

    discarded_sent = 0

    entity_lemma_distr = dict()
    relation_lemma_distr = dict()
    # rel-->(lemma, count)
    relations = list()
    entity_head_lemma_distr = dict()

    coreferences = 0



    print('Merging lemmatized forms:')
    tic = time.time()
    merge_map = dict()
    for key in [k for k in entity_lemma_distr.keys() if " " in k]:
        if key.replace(" ", "") in entity_lemma_distr.keys():
            merge_map[key.replace(" ", "")] = key

    updated_column = []
    for lemma in kg_df_corenlp['triple_subj_lemma']:
        if lemma in merge_map.keys():
            updated_column.append(merge_map[lemma])
        else:
            updated_column.append(lemma)

    kg_df_corenlp['triple_subj_lemma'] = updated_column

    updated_column = []
    for lemma in kg_df_corenlp['triple_obj_lemma']:
        if lemma in merge_map.keys():
            updated_column.append(merge_map[lemma])
        else:
            updated_column.append(lemma)
    kg_df_corenlp['triple_obj_lemma'] = updated_column

    for key, value in entity_lemma_distr.copy().items():
        if key in merge_map.keys():
            entity_lemma_distr[merge_map[key]] = entity_lemma_distr[merge_map[key]] + value
            del entity_lemma_distr[key]

    toc = time.time()
    print('time to merge lemmatized forms = ' + str(toc - tic) + ' seconds')

    kg_df_corenlp.to_excel(os.path.join(test_folder,
                                        'en_core_web_lg_preprocessing_depTrees_coref1000k.xlsx' if process_annotation == True else 'en_core_web_lg_no_preprocessing_depTrees_coref.xlsx'),
                           encoding="utf-8", engine="xlsxwriter")
    # kg_df_corenlp.to_excel(os.path.join(test_folder,'en_core_web_lg_preprocessing_depTrees.xlsx' if process_annotation == True else 'en_core_web_lg_no_preprocessing_depTrees.xlsx'), encoding="utf-8", engine="xlsxwriter")

    # dataframes tweet token length distibution (length before processing)
    data_accepted = {'tweet': accepted_tweets.keys(),
                     'tokens': accepted_tweets.values()}
    df_accepted = pd.DataFrame(data_accepted)
    df_accepted.to_csv(os.path.join(test_folder, 'TokenLengthDistributionTweetExtractedPath_coref100k.tsv'), sep=' ')
    # df_accepted.to_csv(os.path.join(test_folder, 'TokenLengthDistributionTweetExtractedPath.tsv'), sep=' ')

    data_discarded = {'tweet': discarded_tweets.keys(),
                      'tokens': discarded_tweets.values()}
    df_discarded = pd.DataFrame(data_discarded)
    df_discarded.to_csv(os.path.join(test_folder, 'TokenLengthDistributionTweetDiscardedPath_coref100k.tsv'), sep=' ')
    # df_discarded.to_csv(os.path.join(test_folder, 'TokenLengthDistributionTweetDiscardedPath.tsv'), sep=' ')

    # dataframes tweet token length distibution (length after processing)
    data_accepted_processed = {'tweet': accepted_tweets_processed.keys(),
                               'tokens': accepted_tweets_processed.values()}
    df_accepted_processed = pd.DataFrame(data_accepted_processed)
    data_discarded_processed = {'tweet': discarded_tweets_processed.keys(),
                                'tokens': discarded_tweets_processed.values()}
    df_discarded_processed = pd.DataFrame(data_discarded_processed)

    print('Accepted tweets = ' + str(len(accepted_tweets)))
    print('Discarded tweets = ' + str(len(discarded_tweets)))

    f1 = plt.figure()
    plt.hist(df_accepted['tokens'], bins=20, color='blue', edgecolor='black')
    plt.xlabel('Tweet Token Size')
    plt.ylabel('Number of Tweets')
    plt.title('Token Length Distribution for triple-extracting tweets(coref)')
    # ax.set_xlabel('Tweet Token Size', labelpad=20, weight='bold', size=12)
    # ax.set_ylabel('Number of Tweets', labelpad=20, weight='bold', size=12)
    # ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    f1.savefig(os.path.join(test_folder, 'TokenLengthDistributionTweetExtractedPath_coref100k.png'))
    # f1.savefig(os.path.join(test_folder, 'TokenLengthDistributionTweetExtractedPath.png'))
    plt.clf()

    f2 = plt.figure()
    plt.hist(df_discarded['tokens'], bins=20, color='blue', edgecolor='black')
    plt.xlabel('Tweet Token Size')
    plt.ylabel('Number of Tweets')
    plt.title('Token Length Distribution for triple-discarded tweets(coref)')
    # ax.set_xlabel('Tweet Token Size', labelpad=20, weight='bold', size=12)
    # ax.set_ylabel('Number of Tweets', labelpad=20, weight='bold', size=12)
    # ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    f2.savefig(os.path.join(test_folder, 'TokenLengthDistributionTweetDiscardedPath_coref100k.png'))
    # f2.savefig(os.path.join(test_folder, 'TokenLengthDistributionTweetDiscardedPath.png'))
    plt.clf()

    # print(entity_lemma_distr)
    entity_lemma_distr = {'EntityLemma': entity_lemma_distr.keys(),
                          'Occurrence': entity_lemma_distr.values()}
    df_entity_lemma_distr = pd.DataFrame(entity_lemma_distr)
    df_entity_lemma_distr.to_excel(os.path.join(test_folder, 'LemmatizedEntityDistribution_coref100k.xlsx'),
                                   encoding="utf-8", engine="xlsxwriter")
    # df_entity_lemma_distr.to_excel(os.path.join(test_folder, 'LemmatizedEntityDistribution.xlsx'),encoding="utf-8", engine="xlsxwriter")

    relation_lemma_distr = {'RelationHeadLemma': relation_lemma_distr.keys(),
                            'Occurrence': relation_lemma_distr.values()}
    df_relation_lemma_distr = pd.DataFrame(relation_lemma_distr)
    df_relation_lemma_distr.to_excel(os.path.join(test_folder, 'LemmatizedRelationDistribution_coref100k.xlsx'),
                                     encoding="utf-8", engine="xlsxwriter")
    # df_relation_lemma_distr.to_excel(os.path.join(test_folder, 'LemmatizedRelationDistribution.xlsx'),encoding="utf-8", engine="xlsxwriter")

    print('Total number of coreferences: ' + str(coreferences))

    relations = {'RelationHead': [x[0] for x in relations],
                 'RelationHeadLemma': [x[1] for x in relations],
                 'DocIndex': [x[2] for x in relations],
                 'TokenIndex': [x[3] for x in relations]}
    df_relations = pd.DataFrame(relations)
    # df_relation_lemma_distr.to_excel(os.path.join(test_folder, 'LemmatizedRelationDistribution_coref.xlsx'),encoding="utf-8", engine="xlsxwriter")
    df_relations.to_csv(os.path.join(test_folder, 'Relations100k.csv'), encoding="utf-8")

    print('Total number of coreferences: ' + str(coreferences))

    db.to_disk(os.path.join(test_folder, 'doc_bin100k'))


