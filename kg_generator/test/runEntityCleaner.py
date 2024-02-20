import numpy as np

from kg_generator.EntityManager.EntityCleaner import *
from tqdm.auto import tqdm
import csv
import pandas as pd
import os
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from spacy.language import Language
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

def process_groups(groups, split_fctr, filename_triples):
    #print('inside process_groups!')
    #print('number of document groups: ', str(len(groups)))
    n_jobs = split_fctr
    batch_size = int(int(len(groups) / n_jobs))
    #print('batch_size = ' + str(batch_size))
    parts = chunkIt([x for x in range(len(groups))], n_jobs)
    indmatrixinf = []
    indmatrixsup = []
    for batch_inds in parts:
        indmatrixinf.append(batch_inds[0])
        indmatrixsup.append(batch_inds[len(batch_inds) - 1])
        #print(indmatrixinf)
        #print(indmatrixsup)
    startTime = time.time()
    # executor = Parallel(n_jobs=n_jobs)
    # require='sharedmem'
    executor = Parallel(n_jobs=n_jobs, backend='threading', batch_size=batch_size, prefer="threads",
                  verbose=15)  # "loky"  "threading",  "multiprocessing", backend="multiprocessing"
    do = delayed(processSubGroups)
    tasks = (do(indmatrixinf[i], indmatrixsup[i]+1, groups, filename_triples) for i in range(0,len(indmatrixinf)))
    rv = executor(tasks)
    toc = time.time()
    print('Time to process ' + str(len(groups)) + ' article groups = ' + str((toc - startTime) / 60) + ' minutes')


def processSubGroups(x,y,groups, outFile):
    #print('inside process sub_groups!')

    article_group_subset_df = pd.DataFrame()
    for df in groups[x:y]:
      article_group_subset_df = pd.concat([article_group_subset_df, df], axis=0)

    #print('size of article_group_subset_df: ' + str(len(article_group_subset_df)))
    sentence_grouping = article_group_subset_df.groupby(['sentence'], sort=False)
    sentences = [sent  if not str(sent).isdigit() else group['triple_subj'].iloc[0] + ' ' + group['triple_rel'].iloc[0] + ' '+ group['triple_obj'].iloc[0] for sent,group in sentence_grouping]

    processed_senteces = []
    for doc in nlp.pipe(sentences, batch_size=50):
      processed_senteces.append(doc)

    #print('process sentence grouping of size: ', str(sentence_grouping))
    for i,(sent,group) in enumerate(sentence_grouping):
      doc = processed_senteces[i]

      group['triple_subj_lemma'] = process(group['triple_subj'],sent,doc)
      group.dropna(subset=['triple_subj_lemma'], inplace=True)
      group['triple_obj_lemma'] =  process(group['triple_obj'],sent,doc)
      group.dropna(subset=['triple_obj_lemma'], inplace=True)

      group.to_csv(outFile, mode='a', index=False, header=None, sep="\t", encoding="utf-8")



if __name__ == '__main__':

    global nlp
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe('tensor2attr')

    print(nlp.pipeline)
    # customizing the default tokenizer to not split hashtags
    re_token_match = spacy.tokenizer._get_regex_pattern(nlp.Defaults.token_match)
    # Add #hashtag pattern
    re_token_match = f"({re_token_match}|#\\w+)"
    # Add #hashtag pattern
    re_token_match = f"({re_token_match}|@[\\w_]+)"
    re_token_match = f"({re_token_match}|\\$\\w+)"
    nlp.tokenizer.token_match = re.compile(re_token_match).match

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
    nlp.tokenizer.infix_finditer = infix_re.finditer


    '''
    text = 'Do-i and in-house products'
    entityString='Do-i'
    regex = re.compile(r'\b' +  re.escape(entityString) + r'\b')
    match = re.search(regex, text)

    
    sent_tokens = nlp(text)
    e='human adaptive air-conditioner'
    splits = e.split(' ')
    if len(splits) == 1:
        regex = re.compile(re.escape(e))
    else:
        regex = re.compile(re.escape(splits[0]) + '.*' + re.escape(splits[-1]))
    match = re.search(regex, text)
    e_span = sent_tokens.char_span(match.start(), match.end(), alignment_mode='strict')
    e_span
    '''



    def cleanString(input,camel_casesMap):
        result = ""
        input = input.strip(''.join(EntityCleaner.punctuations1))
        if input.lower() in camel_casesMap.keys():
            return camel_casesMap[input.lower()]

        if EntityCleaner.is_camel_case(input):
            res_list = []
            res_list = [s.lower() for s in re.findall('[A-Z]+[^A-Z]*', input)]
            for s in res_list:
                result = result + ' ' + s
            camel_casesMap[input.lower()] = result.strip()
            return result.strip()
        else:
            input = input.replace('_', ' ', 1)
            return input.lower().strip()

    def find_files(filename, search_path):
        result = []

        # Wlaking top-down from the root
        for root, dir, files in os.walk(search_path):
            if filename in files:
                result.append(os.path.join(root, filename))
        return result


    def process(entities,sent,doc):
        global camelCaseMap
        local_entities = improve_entities(entities)
        local_entities = puntuaction_and_stopword(local_entities)
        local_entities = lemmatize_new(local_entities, sent, doc)
        local_entities = toBritishSpelling(local_entities)
        return local_entities


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


    def toBritishSpelling(local_entities):
        new_entities = []
        for e in local_entities:
            if e is None:
                new_entities += [None]
            else:
                entityString = ''
                res = e.split()
                for r in res:
                    bspel = get_british_spelling(r)
                    # if (r != bspel):
                    # print(r, bspel)
                    entityString = " ".join([entityString, bspel])

                new_entities += [entityString.strip()]

        return new_entities


    def puntuaction_and_stopword(local_entities):
        new_entities = []
        for e in local_entities:
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


    def lemmatize(local_entities, local_sentences, nlp, camelCaseMap):

        new_entities = []
        print('len of entities before running lemmatize() : ' + str(len(local_entities)))
        print('len of sentences before running lemmatize() : ' + str(len(local_sentences)))


        tuples = zip(local_entities, local_sentences)
        with tqdm(total=len(local_entities), position=0, leave=True) as pbar:
            for e, sent in tqdm(tuples, position=0, leave=True, total=len(local_entities)):
                pbar.update()
                if e is None:
                    new_entities += [None]
                else:
                    entityString = ''
                    sent_tokens = nlp(sent)
                    tokens = nlp(e)
                    sent_token_strings = [t.text for t in sent_tokens]
                    splits = e.split()
                    # print(splits)
                    for e_t in tokens:
                        if e_t.text in sent_token_strings:
                            in_sent = True
                        else:
                            in_sent = False
                        # remove leading punctuation chars
                        tokenText = e_t.text.lstrip(''.join(EntityCleaner.punctuations2))

                        if tokenText.startswith('#') or tokenText.startswith('@'):
                            entityString = " ".join([entityString, cleanString(tokenText,camelCaseMap)])
                        elif tokenText == '%':
                            entityString = "".join([entityString, tokenText])

                        # if the token is also part of the tokenizer output for the whole sentence, use the lemma from that tokenizer (it has more context ans is more accurate)
                        elif in_sent:
                            # for verbal parts of entities (e.g. 'computing power') do not lemmatize
                            sent_token = sent_tokens[sent_token_strings.index(e_t.text)]
                            if "VB" in sent_token.tag_:
                                entityString = " ".join([entityString, sent_token.text.strip(
                                    ''.join(EntityCleaner.punctuations2)).lower() if len(
                                    re.findall('\.', sent_token.lemma_)) == 1 else sent_token.text.lower()])
                            elif sent_token.tag_ == 'PROPN':
                                entityString = " ".join([entityString, sent_token.text.strip(
                                    ''.join(EntityCleaner.punctuations2)).lower() if len(
                                    re.findall('\.', sent_token)) == 1 else sent_token.lower()])
                            else:
                                entityString = " ".join([entityString, sent_token.lemma_.strip(
                                    ''.join(EntityCleaner.punctuations2)).lower() if len(
                                    re.findall('\.', sent_token.lemma_)) == 1 else sent_token.lemma_.lower()])

                        # otherwise use the lemma from the spacy doc of the local entity
                        else:
                            if "VB" in e_t.tag_:
                                entityString = " ".join([entityString, e_t.text.strip(
                                    ''.join(EntityCleaner.punctuations2)).lower() if len(
                                    re.findall('\.', e_t.lemma_)) == 1 else e_t.text.lower()])
                            else:
                                entityString = " ".join([entityString, e_t.lemma_.strip(
                                    ''.join(EntityCleaner.punctuations2)).lower() if len(
                                    re.findall('\.', e_t.lemma_)) == 1 else e_t.lemma_.lower()])

                    new_entities += [entityString.strip()]

        print('len of entities after running lemmatize() : ' + str(len(local_entities)))
        print('len of sentences after running lemmatize() : ' + str(len(local_sentences)))
        return new_entities

    def lemmatize_new(local_entities, sentenceText, sent_tokens):
        new_entities = []
        for e in local_entities:
            if e is None:
                new_entities += [None]
            else:
                entityString = ''
                #tokens = nlp(e)
                sent_token_strings = [t.text for t in sent_tokens]
                regex = re.compile(r'\b' + re.escape(e) + r'\b')
                match = re.search(regex, sentenceText)

                if not match:
                    if re.match(r'[\d.]+\s%',e):
                        e = re.sub(r'\s+', "", e)
                        regex = re.compile(re.escape(e))
                        match = re.search(regex, sentenceText)
                    elif len(e.split(' '))>1:
                        regex = re.compile(r'\b' + re.escape(e.split(' ')[0]) + '.*' + re.escape(e.split(' ')[-1]) + r'\b')
                        match = re.search(regex, sentenceText)

                    if not match:
                        new_entities += [None]
                        continue

                e_span = sent_tokens.char_span(match.start(), match.end(), alignment_mode='strict')

                if e_span:
                    for e_t in e_span:
                    # remove leading punctuation chars
                        tokenText = e_t.text.lstrip(''.join(EntityCleaner.punctuations2))

                        if tokenText.startswith('#') or tokenText.startswith('@'):
                            entityString = " ".join([entityString, cleanString(tokenText,camelCaseMap)])
                        elif tokenText == '%':
                            entityString = "".join([entityString, tokenText])

                        # for verbal parts of entities (e.g. 'computing power') do not lemmatize
                        if "VB" in e_t.tag_:
                            entityString = " ".join([entityString, e_t.text.strip(
                                ''.join(EntityCleaner.punctuations2)).lower() if len(
                                re.findall('\.', e_t.lemma_)) == 1 else e_t.text.lower()])
                        elif e_t.tag_ == 'PROPN':
                            entityString = " ".join([entityString, e_t.text.strip(
                                ''.join(EntityCleaner.punctuations2)).lower() if len(
                                re.findall('\.', e_t)) == 1 else e_t.lower()])
                        elif '%' not in entityString:
                            entityString = " ".join([entityString, e_t.lemma_.strip(
                                ''.join(EntityCleaner.punctuations2)).lower() if len(
                                re.findall('\.', e_t.lemma_)) == 1 else e_t.lemma_.lower()])


                    new_entities += [entityString.strip()]
                    #this is the case when the regex of the entity matches a wrong segment of the sent, maybe because there are more occurrences of the enttiy string in text...
                    # in this case we re-run spacy on the entity only, to get pos
                else:
                    new_entities += [None]
                    print('still dunno what to do...')

        #if len(new_entities)!=len(local_entities):
        #   print('stop here')

        return new_entities

    def improve_entities(local_entities):
        new_entities = []
        for e in local_entities:
            if len(e) > 1:  # The entity string must have at least two characters
                e_improved = entity_string_improvement(e)
                new_entities += [e_improved]
            else:
                new_entities += [None]

        return new_entities


    #tests_test_folder = 'D:/GitRepos/GitRepos/dtmkg/data-collection/twitter/tests/test'
    #tests_folder = 'D:/GitRepos/GitRepos/dtmkg/data-collection/twitter/tests'


    multiproc = True
    #tests_test_folder = 'D:/GitRepos/GitRepos/dtmkg/data-collection/dna/test'
    #tests_folder = 'D:/GitRepos/GitRepos/dtmkg/data-collection/dna/test'




    test_folder = 'D:/GitRepos/GitRepos/dtmkg/data-collection/AEC_project/test'
    triples = pd.read_csv(os.path.join(test_folder, 'triples_openalex_all.tsv'), header=0, lineterminator='\n', low_memory=False, sep='	')

    outputFile = os.path.join(test_folder, 'triples_openalex_all_EntityNormalized.tsv')



    global camelCaseMap
    camelCaseMap = dict()
    result = find_files('camelCaseMap.csv', test_folder)
    if len(result)>0:
        with open(result[0], newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                camelCaseMap[row['camelcase']]=row['multitok']





    #triples = triples[0:5000]
    print('number of triples: ',len(triples))

    triples = triples[triples['triple_subj'].notnull()]
    triples = triples[triples['triple_obj'].notnull()]

    print('number of triples: ',len(triples))

    if multiproc:
        #triples = triples[0:100000]
        if not os.path.isfile(outputFile):
            columns = triples.columns.values.tolist()
            columns.extend(['triple_subj_lemma', 'triple_obj_lemma'])

            triples_new = pd.DataFrame(columns=columns)
            triples_new.to_csv(outputFile, mode='w', header=columns, index=False, sep="\t", encoding="utf-8")


        global article_grouping
        global article_ids
        article_grouping = triples.groupby(['doc_id'], sort=False)
        article_ids = list(article_grouping.indices.keys())
        print('size article grouping: ', len(article_grouping))

        latest_doc_id_processed = input('Insert Latest Processed Id: ')

        if latest_doc_id_processed:
            doc_ids_to_be_dropped = []
            for name, group in article_grouping:
                if name != latest_doc_id_processed:
                    doc_ids_to_be_dropped.append(name)
                else:
                    break

            triples = triples[~triples.doc_id.isin(doc_ids_to_be_dropped)]
            print(len(triples))

        last_doc_id_to_be_processed = input('Insert Last to be Processed Id: ')

        if last_doc_id_to_be_processed:
            doc_ids_to_be_dropped_tail = []
            for name, group in reversed(article_grouping):
                if name != latest_doc_id_processed:
                    doc_ids_to_be_dropped_tail.append(name)
                else:
                    break
            print('Number of docs already processing in the other code: ', len(doc_ids_to_be_dropped_tail))
            triples = triples[~triples.doc_id.isin(doc_ids_to_be_dropped_tail)]
            print(len(triples))


        # redo the article grouping after filtering the triples:
        article_grouping = triples.groupby(['doc_id'], sort=False)
        article_ids = list(article_grouping.indices.keys())
        print('size article grouping after filtering triples: ', len(article_grouping))

        parts = chunkIt([x for x in range(len(article_grouping))], 500)
        # print(parts)
        indmatrixinf = []
        indmatrixsup = []
        for batch_inds in parts:
            indmatrixinf.append(batch_inds[0])
            indmatrixsup.append(batch_inds[-1])

        #print('inf indexes article splitting', indmatrixinf)
        #print('inf indexes article splitting', indmatrixsup)

        for i in tqdm(range(0, len(indmatrixinf)), total=len(indmatrixinf)):
            process_groups([article_grouping.get_group(k) for k in
                            article_ids[indmatrixinf[i]:indmatrixsup[i] + 1]], 10, outputFile)



         ################################################################################################################

        ''' 
        relations = pd.read_csv(os.path.join(tests_folder, 'Relations_100k_tweets_test.csv'), sep='\t')
        normalization_df = pd.DataFrame(columns=['triple_subj', 'triple_subj_lemma','triple_obj','triple_obj_lemma'])

        triples = triples
        subj_entities = triples['triple_subj']
        obj_entities = triples['triple_obj']
        sentences = triples['sentence']
        print(len(subj_entities))
        print(len(obj_entities))
        print(len(sentences))
        #print(len(relations))


        normalization_df['triple_subj'] = subj_entities
        normalization_df['triple_obj'] = obj_entities

       # perform a first run on EntityCleaner on subj and obj columns in order to update the camelCaseMap
        #entityCleaner1 = EntityCleaner(subj_entities,sentences)
        #if the camelCaseMap read from file is not empty
        #if camelCaseMap:
        #    entityCleaner1.set_camel_casesMap(camelCaseMap)
        #entityCleaner1.run()

        #entityCleaner2 = EntityCleaner(obj_entities, sentences)
        # if the camelCaseMap read from file is not empty
        #if camelCaseMap:
        #    entityCleaner2.set_camel_casesMap(camelCaseMap)
        #entityCleaner2.set_camel_casesMap(entityCleaner1.get_camel_casesMap())
        #entityCleaner2.run()

        #print the updated camelCaseMap to file
        #camelCaseMap = entityCleaner2.get_camel_casesMap()
        #print(camelCaseMap)
       # camelCaseMap_df = pd.DataFrame(columns=['camelcase','multitok'])
        #camelCaseMap_df['camelcase']=camelCaseMap.keys()
        #camelCaseMap_df['multitok'] =camelCaseMap.values()
        #camelCaseMap_df.to_csv(os.path.join(test_folder, 'camelCaseMap.csv'))

        entityCleaner1 = EntityCleaner(subj_entities,sentences)


        nlp = entityCleaner1.nlp
        entities = entityCleaner1.entities
        sentences = entityCleaner1.sentences

        # if the camelCaseMap read from file is not empty
        if camelCaseMap:
            entityCleaner1.set_camel_casesMap(camelCaseMap)
        # apply EntityCleaner on subj column
        #entityCleaner1 = EntityCleaner(subj_entities, sentences)
        #entityCleaner1.set_camel_casesMap(camelCaseMap)

        if multiproc:
            global_entities = []
            tic = time.time()
            #n_jobs = multiprocessing.cpu_count()  # Count the number of cores in a computer

            parts = chunkIt([x for x in range(len(article_grouping))], 100)
            # print(parts)
            indmatrixinf = []
            indmatrixsup = []
            for batch_inds in parts:
                indmatrixinf.append(batch_inds[0])
                indmatrixsup.append(batch_inds[-1])

            print('inf indexes article splitting', indmatrixinf)
            print('inf indexes article splitting', indmatrixsup)




            rv = executor(tasks)

            for r in rv:
                global_entities.extend(r)

            entityCleaner1.entities = global_entities

        else:
            entityCleaner1.run()

        print('Number of None in triple_subj_lemma: ', sum(x is None for x in entityCleaner1.getEntitiesCleaned()))

        # apply EntityCleaner on obj column
        entityCleaner2 = EntityCleaner(obj_entities, sentences)
        entities = entityCleaner2.entities
        if camelCaseMap:
            entityCleaner2.set_camel_casesMap(camelCaseMap)

        if multiproc:
            global_entities = []
            tic = time.time()
            n_jobs = 8 # Count the number of cores in a computer
            batch_size = int(int(len(entityCleaner2.entities) / n_jobs))
            print('batch_size = ' + str(batch_size))
            parts = chunkIt([x for x in range(len(entityCleaner2.entities))], n_jobs)
            indmatrixinf = []
            indmatrixsup = []
            for batch_inds in parts:
                indmatrixinf.append(batch_inds[0])
                indmatrixsup.append(batch_inds[len(batch_inds) - 1])
            print(indmatrixinf)
            print(indmatrixsup)
            startTime = time.time()
            # executor = Parallel(n_jobs=n_jobs)
            # require='sharedmem'
            executor = Parallel(n_jobs=n_jobs, backend='threading', batch_size=batch_size, prefer="threads",
                                verbose=15)  # "loky"  "threading",  "multiprocessing", backend="multiprocessing"
            do = delayed(process)
            tasks = (do(indmatrixinf[i], indmatrixsup[i]) for i in range(0, len(indmatrixinf)))
            rv = executor(tasks)

            for r in rv:
                global_entities.extend(r)

            entityCleaner2.entities = global_entities

        else:
            entityCleaner1.run()


        print('Number of None in triple_obj_lemma: ', sum(x is None for x in entityCleaner2.getEntitiesCleaned()))

        normalization_df['triple_subj_lemma'] = entityCleaner1.getEntitiesCleaned()
        triples['triple_subj_lemma'] = entityCleaner1.getEntitiesCleaned()
        normalization_df['triple_obj_lemma'] = entityCleaner2.getEntitiesCleaned()
        triples['triple_obj_lemma'] = entityCleaner2.getEntitiesCleaned()

        print('Number of Nan in the triples df : ', int(triples['triple_subj_lemma'].isna().sum()) + int(triples['triple_obj_lemma'].isna().sum()))

        # update the Triple master file and Relation file by cutting rows after Entity Normalization:
        mask = list(np.where(triples["triple_subj_lemma"].isna() | triples["triple_obj_lemma"].isna())[0])

        print('mask size = ', len(mask))

        print(len(triples))
        #print(len(relations))
       # triples.drop(mask,inplace=True)
        triples.drop(triples.index[triples["triple_subj_lemma"].isna() | triples["triple_obj_lemma"].isna()], inplace=True)
        #relations.drop(mask, inplace=True)
        print(len(triples))

        #exit()
        #print(len(relations))

         #triples.to_excel(os.path.join(tests_folder, 'evaluation_489_tweets_EntityNormalized.xlsx'), encoding="utf-8",engine="xlsxwriter")
        #triples.to_excel(os.path.join(tests_folder, 'triples_dna_all_EntityNormalized.xlsx'), encoding="utf-8",engine="xlsxwriter")
        triples.to_excel(os.path.join(test_folder, 'triples_openalex_all_EntityNormalized.xlsx'), encoding="utf-8",engine="xlsxwriter")


        #relations.to_csv(os.path.join(tests_folder, 'evaluation_tweets_99_Relations.csv'), na_rep='NaN', sep='	')

        #normalization_df.to_csv(os.path.join(tests_folder, 'evaluation_489_tweets_EntityNormalizations.tsv'), na_rep='NaN', sep='	')
        #normalization_df.to_csv(os.path.join(tests_folder, 'evaluation_dna_all_EntityNormalizations.tsv'), na_rep='NaN', sep='	')

        normalization_df.to_csv(os.path.join(test_folder, 'evaluation_openalex_all_EntityNormalizations.tsv'), na_rep='NaN', sep='	')

   '''



