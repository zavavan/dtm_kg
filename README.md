# Digital Transformation Monitor Graph Generator



This repository contains the source code developed to build a pipeline for generating a Knowledge Graph about Digital Transformation from an heterogeneus set of textual documents, including scientific papers, EU project reports, and EU patents. The results of this research work have been published in: 

This work has been described in:

![Digital Transformation Monitor Graph Generator Schema](https://github.com/zavavan/dtmkg/blob/master/skg_schema.png)
**Figure 1**: Digital Transformation Monitor Graph Generator Schema

## Repository Description

- **data-preparation/** contains the scripts used to model the data downloaded from MAG dataset about the Semantic Web into a format that can be mined by the Luan Yi et al. tool. 

- **cso-openie-extractor/** contains the scripts that have been used to enrich the Luan Yi et al. result wirh CSO topics and Stanford Core NLP relations (i.e., OpenIE and verbs detected by the PoS tagger).

- **dtmkg-generator/** contains the scripts for performing all operations to clean entities and relations and making triples.

- **evaluation/** contains the scripts we used to generate the sample of triples about Digital Transformation and evaluate our approach.

## Usage
Please follow this guide to run the code and reproduce our results. Please contact us because we need to provide extra files that cannot be pushed into the github reporsitory for files limit of 100 MB.

### Environments
This project uses Python 3.7 (ensure you have Python 3.6 or above installed.).


### Downloads 
1. Clone the repository on your local environment


### Requirements
1. Go to the main folder dtmkg/
2. Install Python3.7 requirements by running:
```
pip3 install -r requirements.txt
```
3. Download English package for spaCy using 
```
python3 -m spacy download "en_core_web_trf"
```

### Data preparation
1. Go to the directory data-preparation/. It contains the abstracts coming from the MAG datasets, a script to parse them and produce the input files that will be fed to the Luan Yi et al. tool.

2. To prepare the data you need to run:

```
python3 parse_input.py
```

3. The script produces:
- **data.csv** a file that contains the id, title, abstract, keywords, and doi (when available) of publications
- **all_abstracts.txt** a textual file that contains all abstracts and that will be used later by the pipeline


4. The result is a csv file called *csv_e_r_full.csv* which contains all entities and relations extracted by the used tools


### Toward the SKG
This code generates heristic based relations through the window of verbs, and validates entities based on CSO topics, Semantic Web Keywords and statistics. Finally it maps all relations following the taxonomy "SKG_predicates" we defined. 

1. Go to skg-generator
2. Download and unzip the archive we provided in this directory.
3. Copy the *csv_e_r_full.csv* in this directory
4. Run
```
python3 run.py
```
5. At the end the files *selected_triples.csv* and *kg.graphml* will be generated.  The file *selected_triples.csv* contains all triples with additional information generated with our method. The file *triples.csv* contains all triples generated without details. The script *to_rdf.py* can be used to generate the rdf and nt files.


### Evaluation

The directory evalution contains the scripts we used for performing our evaluation and the manually annotated gold standard.

### Other info

All the code has been developed on a server which mounts Ubuntu 17.10. Physical components were:
-  GB RAM memory
- TB HDD
-  X GPU
- Intel(R) Core(TM) i3-7100 CPU @ 3.90GHz

The execution of all modules required about  days.








