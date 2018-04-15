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

3.Output file example from training-validation-v1.py

********** TRAINING AND TESTING REPORT **********

Training file: training-data-set-70.txt

best params:{'c2': 0.07914565142816338, 'c1': 0.21725964976439643}

best CV score:0.8106074404273954

model size: 0.11M

Flat F1: 0.8356545961

             precision    recall  f1-score   support

       GENE      0.915     0.769     0.836       390

avg / total      0.915     0.769     0.836       390

Performance: 0.836


