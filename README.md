# BioNLP_analysis

NLP preprocessing pipeline: sentence split, term detection, part-of-speech tagging , lemmatization, biological name entity recognition and transformation to internal representation.

This pipeline run only in ccg tepeu server

1. Preprocessing_data

In file nlp-preprocessing-pipeline.sh, change CORPUS_PATH and TERM_PATH by your counts paths 

ORIGINAL_CORPUS_PATH=/export/space1/users/compu2/bionlp/conditional-random-fields/data-sets/original

CORPUS_PATH=/export/space1/users/compu2/bionlp/conditional-random-fields/data-sets

TERM_PATH=/export/space1/users/compu2/bionlp/nlp-preprocessing-pipeline/dictionaries

POST_PATH=/export/space1/users/compu2/bionlp/stanford-postagger-2018-02-27

LEMMA_PATH=/export/space1/users/compu2/bionlp/biolemmatizer

Make five directories in /nlp-preprocessing-pipeline/corpus
term pre pos transformed lemma preprocessed 

2. Conditional_random_fields

python3 training-validation-v1.py --inputPath /export/space1/users/guest02/conditional-random-fields/data-sets --trainingFile training-data-set-70.txt --testFile test-data-set-30.txt --outputPath /export/space1/users/guest02/conditional-random-fields > output-training.txt 
