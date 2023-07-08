from spotlight_extractor.EntityExtraction import *
import requests
import spacy
from spacy import displacy
import preprocessor as p
p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.RESERVED, p.OPT.SMILEY)
preprocessing = True
process_annotation = True
# We want to also keep #hashtags as a token, so we modify the spaCy model's token_match
import re
import html


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

def processAnnotation(g, poss, text, sentence_doc):
    nodes_to_remove = set()

    # 1. drop sequences of entity mentions (@) at the beginning of sentence (including when prefixed by “RT” or punctuations). If size of sequence is one, it must not be followed by verb
    found = False
    for i in range(min(2, len(poss) - 2)):
        if text[int(poss[i]['start']):int(poss[i]['end'])].startswith('@') and 'VB' not in poss[i + 1]['tag']:
            found = True
            break
    if found:
        nodes_to_remove.add(int(poss[i]['id']))
        if i > 0:
            nodes_to_remove.add(int(poss[i - 1]['id']))

        length_mentions_prefix = 1
        for j in range(i + 1, len(poss) - 1):
            if text[int(poss[j]['start']):int(poss[j]['end'])].startswith('@') or poss[j]['tag'] == ':':
                length_mentions_prefix += 1
            else:
                break

        if length_mentions_prefix > 1:
            for k in range(i + 1, j):
                nodes_to_remove.add(int(poss[k]['id']))
                # edges_to_remove.update(g.in_edges(k+1))

    # 2. for any sequence of n >= 2 hashtags/mentions, drop subsequence [1:n]
    length_hashtag_mention_sequence = 0
    node_list = []
    for pos in poss:
        if text[int(pos['start']):int(pos['end'])].startswith('#') or text[int(pos['start']):int(
                pos['end'])].startswith('@'):
            node_list.append(int(pos['id']))
            length_hashtag_mention_sequence += 1
        else:
            if length_hashtag_mention_sequence >= 2:
                nodes_to_remove.update(node_list[1:])
            length_hashtag_mention_sequence = 0
            node_list = []
    if length_hashtag_mention_sequence >= 2:
        nodes_to_remove.update(node_list[1:])

    # 3. drop any hashtag tagged as verb if preceded by another hashtag
    """
    for pos in poss:
      if 'VB' in pos['tag'] and text[int(pos['start']):int(pos['end'])].startswith('#') \
              and   text[int(poss[poss.index(pos) - 1]['start']):int(poss[poss.index(pos) - 1]['end'])].startswith('#'):
          nodes_to_remove.add(int(pos['id']))
          #edges_to_remove.update(g.in_edges(int(pos['index'])))




    # 3. drop tails of > 2 hashtags/mentions
    length_hashtag_tail = 0
    node_list = []
    for pos in reversed(poss):
        if text[int(pos['start']):int(pos['end'])].startswith('#') or text[int(pos['start']):int(pos['end'])].startswith('@'):
            node_list.append(int(pos['id']))
            length_hashtag_tail += 1
        else:
            break

    if length_hashtag_tail > 2:
        node_list.sort()
        nodes_to_remove.update(node_list[1:])
       # for n in nodes_to_remove:
       #     edges_to_remove.update(g.in_edges(n))






    #4. drop dependent node of a dep edge if it is a hashtag, and remove all dependent nodes
   # for dep in depparses:
   #     if dep['dep']=='dep' and dep['dependentGloss'].startswith('#'):
    #        nodes_to_remove.add(int(dep['dependent']))
     #       #edges_to_remove.add((int(dep['governor']),int(dep['dependent'])))
      #      nodes_to_remove.update(list(nx.dfs_preorder_nodes(g, source=int(dep['dependent']))))

    """

    # if some edges were to be removed, modify the text and re-annotate the sentence:
    if len(nodes_to_remove) > 0:

        # modify the text
        nodes_to_remove = list(nodes_to_remove)
        nodes_to_remove.sort()
        out_text = text
        for n in nodes_to_remove:
            out_text = out_text.replace(text[int(poss[n]['start']):int(poss[n]['end'])], '', 1)

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
                return sentence_doc
    else:
        return sentence_doc


def preprocessCleanText(text, p):
    return p.clean(text)


def preprocessTweet(text, p):
    unescapedText = html.unescape(text)
    preprocessed_text = preprocessCleanText(unescapedText, p)
    preprocessed_text = re.sub(r"(\w[\?!;\.])(\w+)", r"\1 \2 ", preprocessed_text)
    return preprocessed_text

api_endpoint = "https://api.dbpedia-spotlight.org/en/annotate"
parameters={}
parameters['confidence']=0.35
Headers = {"Accept" :"application/json"}

spacyNLP = spacy.load("en_core_web_lg")
#spacyNLP = spacy.load("en_core_web_trf")

# customizing the default tokenizer to not split hashtags
re_token_match = spacy.tokenizer._get_regex_pattern(spacyNLP.Defaults.token_match)
# Add #hashtag pattern
re_token_match = f"({re_token_match}|#\\w+)"
spacyNLP.tokenizer.token_match = re.compile(re_token_match).match

test_folder = 'D:/GitRepos/GitRepos/skg/data-collection/twitter/dbpedia-spotlight_tests'
preprocessing_folder = 'D:/GitRepos/GitRepos/skg/data-collection/twitter/spotlight_tests/results_preprocessing_heuristics_100'
no_preprocessing_folder = 'D:/GitRepos/GitRepos/skg/data-collection/twitter/spotlight_tests/results_preprocessing_100'
kg_df_corenlp = pd.DataFrame(
    columns=['doc_id', 'sentence_n', 'triple_subj', 'triple_rel', 'triple_obj', 'path', 'dep_tree_path', 'neg',
             'sentence', 'original_tweet', 'dep_tree'])

file = open(os.path.join('D:/GitRepos/GitRepos/skg/data-collection/twitter', 'sample.tsv'), mode="r", encoding="utf-8")
# Get a dependency tree from a Stanford CoreNLP pipeline

counter = 1
for line in file:
    if counter < 1000:
        splitting = line.split('\t')
        tweetId = splitting[0]
        originalText = splitting[2]
        if preprocessing:
            text = preprocessTweet(originalText, p)
        else:
            text = originalText

        annotation = spacyNLP(text)
        doc_data = annotation.to_json()
        sent_counter = 0
        if 'sents' not in doc_data.keys():
            sentence = {}
            sentence['text'] = doc_data['text']
            sentence['tokens'] = doc_data['tokens']
            doc_data['sents'] = []
            doc_data['sents'].append(sentence)

        for sentence in doc_data['sents']:
            sentenceNum = str(sent_counter)
            # sentenceText = utils.rebuild_spacy_sentence(sentence, doc_data)
            sentenceText = rebuild_spacy_sentence(sentence, doc_data)
            sentence_doc = spacyNLP(sentenceText)
            tokens = sentence_doc.to_json()['tokens']
            g = build_graph_spacy(tokens)
            if process_annotation:
                sentence_doc = processAnnotation(g, tokens, sentenceText, sentence_doc)

            if sentence_doc is not None:
                tokens = sentence_doc.to_json()['tokens']
                g = build_graph_spacy(tokens)
                parameters['text'] = sentence_doc.text
                response = requests.get(api_endpoint, params=parameters, headers=Headers)
                if response.status_code == 200:
                    if 'Resources' in response.json().keys() and len(response.json()['Resources'])>0:
                        entities_indices = get_entity_indices(tokens, response.json()['Resources'],sentence_doc)
                        path_graph = build_path_graph_spacy(tokens)
                        paths = filterTargetPaths(get_path_between_entities_spacy(entities_indices, path_graph))
                        for path in paths:
                            entityString = rebuild_spacy_entity(find_entity_index_set(path[0][0][0], entities_indices), tokens,
                                                                sentence_doc.text)
                            fromString = entityString
                            toString = rebuild_spacy_entity(find_entity_index_set(path[0][len(path[0]) - 1][0], entities_indices),
                                                            tokens, sentence_doc.text)
                            relString = sentence_doc.text[
                                        int(tokens[find_rel_index(path)]['start']):int(tokens[find_rel_index(path)]['end'])]
                            pathString = fromString + ";" + relString + ";" + toString
                            deep_tree_pathString = path[1]
                            negation = path[2]
                            triple_subjString = fromString
                            triple_relString = relString
                            triple_objString = toString
                            kg_df_corenlp = kg_df_corenlp.append(
                                {'doc_id': tweetId, 'sentence_n': sentenceNum, 'path': pathString,
                                 'dep_tree_path': deep_tree_pathString,
                                 'triple_subj': triple_subjString, 'triple_rel': triple_relString,
                                 'triple_obj': triple_objString,
                                 'neg': negation, 'sentence': sentence_doc.text, 'original_tweet': originalText,
                                 'dep_tree': tokens_to_spacy_dep_tree(tokens)},
                                ignore_index=True)

            sent_counter += 1
            counter += 1

kg_df_corenlp.to_excel(os.path.join(test_folder,
                                    'en_core_web_trf_preprocessing_depTrees.xlsx' if process_annotation == True else 'en_core_web_trf_no_preprocessing_depTrees.xlsx'),
                       encoding="utf-8", engine="xlsxwriter")