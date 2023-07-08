import re
import spacy
from spacy import displacy
from spacy.lang.tokenizer_exceptions import URL_PATTERN
from spacy.language import Language
from spacy.tokens import Token
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
import sys


def processAnnotation(text,sentence_doc):
    nodes_to_remove = set()

    # 1. drop sequences of entity mentions (@) at the beginning of sentence (including when prefixed by “RT” or punctuations). If size of sequence is one, it must not be followed by verb
    found = False
    poss = [t for t in sentence_doc]

    for i, token in enumerate(poss[0:min(3, len(poss) - 3)]):
        # for i,token in range(min(2, len(poss) - 2)):
        if token.text.startswith('@') and 'VB' not in poss[poss.index(token)+1].tag_:
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
            #tokens = sentence_doc['tokens']
            if not sentence_doc.to_json()['sents']:
                return None
            else:
                #poss = annotation['sents'][0]['tokens']
                #depparses = annotation['sentences'][0]['basicDependencies']
                return sentence_doc
    else:
        return sentence_doc


def create_tokenizer(nlp):
    # contains the regex to match all sorts of urls:

    # spacy defaults: when the standard behaviour is required, they
    # need to be included when subclassing the tokenizer
    prefix_re = spacy.util.compile_prefix_regex(Language.Defaults.prefixes)
    infix_re = spacy.util.compile_infix_regex(Language.Defaults.infixes)
    suffix_re = spacy.util.compile_suffix_regex(Language.Defaults.suffixes)

    # extending the default url regex with regex for hashtags with "or" = |
    hashtag_pattern = r'''|^(#[\w\W_-]+)$'''
    url_and_hashtag = URL_PATTERN + hashtag_pattern
    url_and_hashtag_re = re.compile(url_and_hashtag)

    # set a custom extension to match if token is a hashtag
    hashtag_getter = lambda token: token.text.startswith('#')
    Token.set_extension('is_hashtag', getter=hashtag_getter)

    return spacy.Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                     suffix_search=suffix_re.search,
                     infix_finditer=infix_re.finditer,
                     token_match=url_and_hashtag_re.match
                     )

def custom_tokenizer(nlp):
    # spacy defaults: when the standard behaviour is required, they
    # need to be included when subclassing the tokenizer
    special_cases = {":)": [{"ORTH": ":)"}]}
    prefix_re = spacy.util.compile_prefix_regex(Language.Defaults.prefixes)
    infix_re = spacy.util.compile_infix_regex(Language.Defaults.infixes)
    suffix_re = spacy.util.compile_suffix_regex(Language.Defaults.suffixes)
    simple_url_re = re.compile(r'''^https?://''')

    return spacy.Tokenizer(nlp.vocab, rules=special_cases,
                                prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                url_match=simple_url_re.match)

#this is a pipeline filter function for  spacy, to allow it tokenize hashtags expressions, instead of splitting them
def hashtag_pipe(doc):
    merged_hashtag = False
    while True:
        for token_index,token in enumerate(doc):
            if token.text == '#':
                if token.head is not None:
                    start_index = token.idx
                    end_index = start_index + len(token.head.text) + 1
                    if doc.merge(start_index, end_index) is not None:
                        merged_hashtag = True
                        break
        if not merged_hashtag:
            break
        merged_hashtag = False
    return doc

if __name__ == '__main__':
    spacyNLP = spacy.load("en_core_web_trf")
    spacyNLP.add_pipe('coreferee')

    #print(spacy.explain("PART"))
    #print(spacy.explain("pobj"))
    #sys.exit('')
    #prefixes = list(spacyNLP.tokenizer.prefixes)
    #print(prefixes)
    #prefixes.remove("#")
    #prefix_regex = spacyNLP.util.compile_prefix_regex(prefixes)
    #spacy.tokenizer.prefix_search = prefix_regex.search
    #spacyNLP.tokenizer = create_tokenizer(spacyNLP)
    # Retrieve the default token-matching regex pattern
    #re_token_match = spacy.tokenizer._get_regex_pattern(spacyNLP.Defaults.token_match)

    re_token_match = spacy.tokenizer._get_regex_pattern(spacyNLP.Defaults.token_match)
    # Add #hashtag pattern
    re_token_match = f"({re_token_match}|#\\w+)"
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

    text = "Would you dare having Simplicity, Accountability, and Collaboration impacting 50% of the performance review criteria for managers ? #dubaitechtalk"
    doc = spacyNLP(text)
    for i, token in enumerate(doc):
        print(token.like_url, ':\t', token.lemma_)

    #doc = processAnnotation(text, doc)
    displacy.serve(doc)

    #for i, token in enumerate(doc):
    #    print(token.like_url, ':\t', token.lemma_)

    #tok_exp = spacyNLP.tokenizer.explain("#digitaltransformation #Infographic via Survey: 80% of companies in your industry are developing new digitally-enabled processes :)")
    #assert [t.text for t in doc if not t.is_space] == [t[1] for t in tok_exp]
    #for t in tok_exp:
    #    print(t[1], "\\t", t[0])
    #for token in doc:
    #   print(token.text)
    #spacyNLP.tokenizer = create_tokenizer(spacyNLP)
    #spacyNLP.add_pipe(hashtag_pipe)
    # get default pattern for tokens that don't get split
    #add your patterns (here: hashtags and in-word hyphens)
    #re_token_match = f"({re_token_match}|#\w+|\w+-\w+)"

    #displacy.serve(spacyNLP("Microsoft releases out-of-band update to fix Kerberos auth issues caused by a patch for CVE-2022-37966 https://t.co/DfhGGzOzPO #Cybersecurity #DigitalTransformation #CyberAttack #Privacy #Dataprotection #DataScience #FAANG #BigTech #Tech #Technology #Blockchain #Innovation #Sci…"), style="dep")


