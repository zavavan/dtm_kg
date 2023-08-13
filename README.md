# Digital Transformation Monitor Graph Generator



This repository contains the source code developed to build a pipeline for generating a Knowledge Graph about Digital Transformation from an heterogeneus set of textual documents, including scientific papers, EU project reports, and EU patents. The results of this research work have been published in: 

This work has been described in:

## Repository Description

- **data-preparation/** 

- **spacy-extractor/** contains code to extract triples using dependency parse tree patterns from spacy annotations of sentences

- **dtmkg-generator/** contains the scripts for performing all operations to clean entities and relations and making triples.

- **evaluation/** contains the scripts we used to generate the sample of triples about Digital Transformation and evaluate our approach.

## Usage
Please follow this guide to run the code and reproduce our results.

### Environments
This project uses Python 3.10


### Downloads 
1. Clone the repository on your local environment


### Requirements
1. Go to the main folder dtmkg/
2. Install Python3.10 requirements by running:
```
pip3 install -r requirements.txt
```
3. Download English package for spaCy using 
```
python3 -m spacy download "en_core_web_trf"
```

### Data preparation


### dtmkg-generator

5. At the end the files *selected_triples.csv* and *kg.graphml* will be generated.  The file *selected_triples.csv* contains all triples with additional information generated with our method. The file *triples.csv* contains all triples generated without details. The script *to_rdf.py* can be used to generate the rdf and nt files.


### Evaluation

The directory evalution contains the scripts we used for performing our evaluation and the manually annotated gold standard.

### Other info










