# -*- coding: UTF-8 -*-

import os
from itertools import chain
from optparse import OptionParser
from time import time
from collections import Counter

import nltk
import sklearn
import scipy.stats
import sys

from sklearn.externals import joblib
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

from nltk.corpus import stopwords
from trainingTesting_Sklearn_crfsuite import word2features
from trainingTesting_Sklearn_crfsuite import sent2features
# from trainingTesting_Sklearn_crfsuite import hasNonAlphaNum
# from trainingTesting_Sklearn_crfsuite import hasDigit

# Objective
# Tagging transformed file with CRF model with sklearn-crfsuite.
#
# Input parameters
# --inputPath=PATH    Path of transformed files x|y|z
# --modelPath        Path to CRF model
# --modelName    Model name
# --outputPath=PATH    Output path to place output files
# --filteringStopWords   Filtering stop words
# --filterSymbols      Filtering punctuation marks

# Output
# 1) Tagged files in transformed format

# Examples
# Sentences
# C:\Anaconda2\python tagging_Sklearn_crfsuite.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\classifying_TFSentences\corpus\ECK120011394_FhlA\transformed --modelName aspectsTraining.fStopWords_False.fSymbols_True --modelPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\trainingTest_CRF_TERM_TAGS --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\classifying_TFSentences\corpus\ECK120011394_FhlA\transformed_CRFtagged --filterSymbols > output.taggingCRF.20161107.txt
# C:\Anaconda2\python tagging_Sklearn_crfsuite.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\classifying_TFSentences\corpus\ECK120011394_FhlA\transformed --modelName sentencesTraining.fStopWords_False.fSymbols_False --modelPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\trainingTest_CRF_TERM_TAGS --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\classifying_TFSentences\corpus\ECK120011394_FhlA\transformed_CRFtagged > output.taggingCRF.20161107.txt

#################################
#           FUNCTIONS           #
#################################
# def hasDigit(text):
#     has = False
#     if len(text) < 3:
#         return False
#     myRegex = nltk.re.compile('[0-9]')
#     if myRegex.search(text) != None:
#         has = True
#     return has
#
#
# def hasNonAlphaNum(text):
#     has = False
#     if len(text) < 3:
#         return False
#     myRegex = nltk.re.compile('\W')
#     if myRegex.search(text) != None:
#         has = True
#     return has

# IMPORTED FROM TRAINING SCRIPT
# def word2features(sent, i):
#     # print "i: " + str(i)
#     # print "sent[i]" + sent[i]
#     listElem = sent[i].split('|')
#     word = listElem[0]
#     lemma = listElem[1]
#     postag = listElem[2]
#
#     features = {
#         # Names of TF and genes change by lower and upper characters: 'word.lower()': word.lower(),
#         # Suffixes
#         'word[-3:]': word[-3:],
#         'word[-2:]': word[-2:],
#         'word[-1:]': word[-1:],
#         'word.isupper()': word.isupper(),
#         'word.istitle()': word.istitle(),
#         'word.hasDigit()': hasDigit(word),
#         'word.hasNonAlphaNum': hasNonAlphaNum(word),
#         # 'word.isdigit()': word.isdigit(),
#         'word': word,
#         'lemma': lemma,
#         'lemma[-3:]': lemma[-3:],
#         'lemma[-2:]': lemma[-2:],
#         'lemma[-1:]': lemma[-1:],
#         'postag': postag,
#         # Prefixes
#         'postag[:2]': postag[:2],
#         'postag[:1]': postag[:1],
#     }
#     if i > 0:
#         listElem = sent[i - 1].split('|')
#         word1 = listElem[0]
#         lemma1 = listElem[1]
#         postag1 = listElem[2]
#         features.update({
#             '-1:word.lower()': word1.lower(),
#             '-1:word.istitle()': word1.istitle(),
#             '-1:word.isupper()': word1.isupper(),
#             '-1:word.hasDigit()': hasDigit(word1),
#             '-1:word.hasNonAlphaNum': hasNonAlphaNum(word1),
#             '-1:word': word1,
#             '-1:lemma': lemma1,
#             '-1:postag': postag1,
#             '-1:postag[:2]': postag1[:2],
#             '-1:postag[:1]': postag1[:1],
#         })
#     # else:
#     #    features['BOS'] = True
#
#     if i < len(sent) - 1:
#         listElem = sent[i + 1].split('|')
#         word1 = listElem[0]
#         lemma1 = listElem[1]
#         postag1 = listElem[2]
#         features.update({
#             '+1:word.lower()': word1.lower(),
#             '+1:word.istitle()': word1.istitle(),
#             '+1:word.isupper()': word1.isupper(),
#             '+1:word.hasDigit()': hasDigit(word1),
#             '+1:word.hasNonAlphaNum': hasNonAlphaNum(word1),
#             '+1:word': word1,
#             '+1:lemma': lemma1,
#             '+1:postag': postag1,
#             '+1:postag[:2]': postag1[:2],
#             '+1:postag[:1]': postag1[:1],
#         })
#     # else:
#     #    features['EOS'] = True
#     if i > 1:
#         listElem = sent[i - 2].split('|')
#         word2 = listElem[0]
#         lemma2 = listElem[1]
#         postag2 = listElem[2]
#         features.update({
#             '-2:word.lower()': word2.lower(),
#             '-2:word.istitle()': word2.istitle(),
#             '-2:word.isupper()': word2.isupper(),
#             '-2:word.hasDigit()': hasDigit(word2),
#             '-2:word.hasNonAlphaNum': hasNonAlphaNum(word2),
#             '-2:word': word2,
#             '-2:lemma': lemma2,
#             '-2:postag': postag2,
#             '-2:postag[:2]': postag2[:2],
#             '-2:postag[:1]': postag2[:1],
#         })
#
#     if i < len(sent) - 2:
#         listElem = sent[i + 2].split('|')
#         word2 = listElem[0]
#         lemma2 = listElem[1]
#         postag2 = listElem[2]
#         features.update({
#             '+2:word.lower()': word2.lower(),
#             '+2:word.istitle()': word2.istitle(),
#             '+2:word.isupper()': word2.isupper(),
#             '+2:word.hasDigit()': hasDigit(word2),
#             '+2:word.hasNonAlphaNum': hasNonAlphaNum(word2),
#             '+2:word': word2,
#             '+2:lemma': lemma2,
#             '+2:postag': postag2,
#             '+2:postag[:2]': postag2[:2],
#             '+2:postag[:1]': postag2[:1],
#         })
#
#     trigrams = False
#     if trigrams:
#         if i > 2:
#             listElem = sent[i - 3].split('|')
#             word3 = listElem[0]
#             lemma3 = listElem[1]
#             postag3 = listElem[2]
#             features.update({
#                 '-3:word.lower()': word3.lower(),
#                 '-3:word.istitle()': word3.istitle(),
#                 '-3:word.isupper()': word3.isupper(),
#                 '-3:word.hasDigit()': hasDigit(word3),
#                 '-3:word.hasNonAlphaNum': hasNonAlphaNum(word3),
#                 '-3:word': word3,
#                 '-3:lemma': lemma3,
#                 '-3:postag': postag3,
#                 '-3:postag[:2]': postag3[:2],
#                 '-3:postag[:1]': postag3[:1],
#             })
#
#         if i < len(sent) - 3:
#             listElem = sent[i + 3].split('|')
#             word3 = listElem[0]
#             lemma3 = listElem[1]
#             postag3 = listElem[2]
#             features.update({
#                 '+3:word.lower()': word3.lower(),
#                 '+3:word.istitle()': word3.istitle(),
#                 '+3:word.isupper()': word3.isupper(),
#                 '+3:word.hasDigit()': hasDigit(word3),
#                 '+3:word.hasNonAlphaNum': hasNonAlphaNum(word3),
#                 '+3:word': word3,
#                 '+3:lemma': lemma3,
#                 '+3:postag': postag3,
#                 '+3:postag[:2]': postag3[:2],
#                 '+3:postag[:1]': postag3[:1],
#             })
#
#     return features


# def sent2features(sent):
#     return [word2features(sent, i) for i in range(len(sent))]


__author__ = 'CMendezC'

##########################################
#               MAIN PROGRAM             #
##########################################

if __name__ == "__main__":
    # Defining parameters
    parser = OptionParser()
    parser.add_option("--inputPath", dest="inputPath",
                      help="Path of training data set", metavar="PATH")
    parser.add_option("--outputPath", dest="outputPath",
                      help="Output path to place output files",
                      metavar="PATH")
    parser.add_option("--modelPath", dest="modelPath",
                      help="Path to read CRF model",
                      metavar="PATH")
    parser.add_option("--modelName", dest="modelName",
                      help="Model name", metavar="TEXT")
    parser.add_option("--filterStopWords", default=False,
                      action="store_true", dest="filterStopWords",
                      help="Filtering stop words")
    parser.add_option("--filterSymbols", default=False,
                      action="store_true", dest="filterSymbols",
                      help="Filtering punctuation marks")

    (options, args) = parser.parse_args()
    if len(args) > 0:
        parser.error("Any parameter given.")
        sys.exit(1)

    print('-------------------------------- PARAMETERS --------------------------------')
    print("Path to read input files: " + options.inputPath)
    print("Mode name: " + str(options.modelName))
    print("Model path: " + options.modelPath)
    print("Path to place output files: " + options.outputPath)
    print("Filtering stop words: " + str(options.filterStopWords))
    symbols = ['.', ',', ':', ';', '?', '!', '\'', '"', '<', '>', '(', ')', '-', '_', '/', '\\', '¿', '¡', '+', '{',
               '}', '[', ']', '*', '%', '$', '#', '&', '°', '`', '...']
    # symbols = [sym.decode('utf-8') for sym in ['.', ',', ':', ';', '?', '!', '\'', '"', '<', '>', '(', ')', '-', '_', '/', '\\', '¿', '¡', '+', '{',
    #            '}', '[', ']', '*', '%', '$', '#', '&', '°']]
    # symbols = [u'.', u',', u':', u';', u'?', u'!', u'\'', u'"', u'<', u'>', u'(', u')', u'-', u'_', u'/', u'\\', u'¿', u'¡', u'+', u'{',
    #             u'}', u'[', u']', u'*', u'%', u'$', u'#', u'&', u'°', u'`']
    print("Filtering symbols " + str(symbols) + ': ' + str(options.filterSymbols))

    print('-------------------------------- PROCESSING --------------------------------')

    stopwords = [word.decode('utf-8') for word in stopwords.words('english')]

    # Read CRF model
    t0 = time()
    print('Reading CRF model...')
    crf = joblib.load(os.path.join(options.modelPath, 'models', options.modelName + '.mod'))
    print("Reading CRF model done in: %fs" % (time() - t0))

    print('Processing corpus...')
    t0 = time()
    labels = list(['MF', 'TF', 'DFAM', 'DMOT', 'DPOS', 'PRO'])
    # Walk directory to read files
    for path, dirs, files in os.walk(options.inputPath):
        # For each file in dir
        for file in files:
            print("   Preprocessing file..." + str(file))
            sentencesInputData = []
            sentencesOutputData = []
            with open(os.path.join(options.inputPath, file), "r") as iFile:
                lines = iFile.readlines()
                for line in lines:
                    listLine = []
                    # line = line.decode("utf-8")
                    for token in line.strip('\n').split():
                        if options.filterStopWords:
                            listToken = token.split('|')
                            lemma = listToken[1]
                            # Original if lemma in stopwords.words('english'):
                            if lemma in stopwords:
                                continue
                        if options.filterSymbols:
                            listToken = token.split('|')
                            lemma = listToken[1]
                            if lemma in symbols:
                                if lemma == ',':
                                    print "Coma , identificada"
                                continue
                        listLine.append(token)
                    sentencesInputData.append(listLine)
                print "   Sentences input data: " + str(len(sentencesInputData))
                # print sentencesInputData[0]
                # print(sent2features(sentencesInputData[0])[0])
                # print(sent2labels(sentencesInputData[0]))
                X_input = [sent2features(s) for s in sentencesInputData]
                print(sent2features(sentencesInputData[0])[0])
                # y_test = [sent2labels(s) for s in sentencesInputData]
                # Predicting tags
                t1 = time()
                print "   Predicting tags with model"
                y_pred = crf.predict(X_input)
                print y_pred[0]
                print("      Prediction done in: %fs" % (time() - t1))
                # Tagging with CRF model
                print "   Tagging file"
                for line, tagLine in zip(lines, y_pred):
                    outputLine = ''
                    idx_tagLine = 0
                    line = line.strip('\n')
                    print "\nLine: " + str(line)
                    print "CRF tagged line: " + str(tagLine)
                    for token in line.split():
                        listToken = token.split('|')
                        word = listToken[0]
                        lemma = listToken[1]
                        tag = listToken[2]
                        if options.filterStopWords:
                            if lemma in stopwords:
                                outputLine += token + ' '
                                continue
                        if options.filterSymbols:
                            if lemma in symbols:
                                if lemma == ',':
                                    print "Coma , identificada"
                                outputLine += token + ' '
                                continue
                        CRFtag = tagLine[idx_tagLine]
                        if (tag not in labels) and (CRFtag != 'O'):
                            print "*** CRF change token {} to {}".format(token, CRFtag)
                            outputLine += word + '|' + lemma + '|' + CRFtag + ' '
                        else:
                            outputLine += word + '|' + lemma + '|' + tag + ' '
                        idx_tagLine += 1
                    sentencesOutputData.append(outputLine.rstrip())
            with open(os.path.join(options.outputPath, file), "w") as oFile:
                for line in sentencesOutputData:
                    oFile.write(line + '\n')

    print("Processing corpus done in: %fs" % (time() - t0))
