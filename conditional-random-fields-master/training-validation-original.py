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


# Objective
# Training and evaluation of CRFs with sklearn-crfsuite.
#
# Input parameters
# --inputPath=PATH    Path of training and test data set
# --trainingFile        File with training data set
# --testFile        File with test data set
# --outputPath=PATH    Output path to place output files
# --filteringStopWords   Filtering stop words
# --filterSymbols      Filtering punctuation marks

# Output
# 1) Best model

# Examples
# Sentences
# C:\Anaconda2\python trainingTesting_Sklearn_crfsuite.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\trainingTest_CRF_TERM_TAGS\trainingTest_Datasets --trainingFile sentencesTraining.txt --testFile sentencesTest.txt --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\trainingTest_CRF_TERM_TAGS > output.TrainingTestingCRF.20161106_1.txt
# C:\Anaconda2\python trainingTesting_Sklearn_crfsuite.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\trainingTest_CRF_TERM_TAGS\trainingTest_Datasets --trainingFile sentencesTraining.txt --testFile sentencesTest.txt --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\trainingTest_CRF_TERM_TAGS --filterStopWords > output.TrainingTestingCRF.20161106_2.txt
# C:\Anaconda2\python trainingTesting_Sklearn_crfsuite.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\trainingTest_CRF_TERM_TAGS\trainingTest_Datasets --trainingFile sentencesTraining.txt --testFile sentencesTest.txt --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\trainingTest_CRF_TERM_TAGS --filterSymbols > output.TrainingTestingCRF.20161106_3.txt
# C:\Anaconda2\python trainingTesting_Sklearn_crfsuite.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\trainingTest_CRF_TERM_TAGS\trainingTest_Datasets --trainingFile sentencesTraining.txt --testFile sentencesTest.txt --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\trainingTest_CRF_TERM_TAGS --filterStopWords --filterSymbols > output.TrainingTestingCRF.20161106_4.txt

# Aspects
# C:\Anaconda2\python trainingTesting_Sklearn_crfsuite.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\trainingTest_CRF_TERM_TAGS\trainingTest_Datasets --trainingFile aspectsTraining.txt --testFile aspectsTest.txt --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\trainingTest_CRF_TERM_TAGS > output.TrainingTestingCRF.20161106_5.txt
# C:\Anaconda2\python trainingTesting_Sklearn_crfsuite.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\trainingTest_CRF_TERM_TAGS\trainingTest_Datasets --trainingFile aspectsTraining.txt --testFile aspectsTest.txt --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\trainingTest_CRF_TERM_TAGS --filterStopWords > output.TrainingTestingCRF.20161106_6.txt
# C:\Anaconda2\python trainingTesting_Sklearn_crfsuite.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\trainingTest_CRF_TERM_TAGS\trainingTest_Datasets --trainingFile aspectsTraining.txt --testFile aspectsTest.txt --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\trainingTest_CRF_TERM_TAGS --filterSymbols > output.TrainingTestingCRF.20161106_7.txt
# C:\Anaconda2\python trainingTesting_Sklearn_crfsuite.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\trainingTest_CRF_TERM_TAGS\trainingTest_Datasets --trainingFile aspectsTraining.txt --testFile aspectsTest.txt --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\trainingTest_CRF_TERM_TAGS --filterStopWords --filterSymbols > output.TrainingTestingCRF.20161106_8.txt

#################################
#           FUNCTIONS           #
#################################

def wordSize(text):
    lWord = len(text)
    if lWord == 1:
        return '1'
    elif lWord == 2:
        return '2'
    elif lWord == 3:
        return '3'
    elif lWord == 4:
        return '4'
    elif lWord == 5:
        return '5'
    elif 6 <= lWord <= 10:
        return '6-10'
    elif 11 <= lWord <= 15:
        return '11-15'
    elif 16 <= lWord <= 20:
        return '16-20'
    elif 21 <= lWord <= 30:
        return '21-30'
    else:
        return '>30'

def hasUpperLower(text):
    has = False
    if len(text) < 3:
        return False
    regexUp = nltk.re.compile('[A-Z]')
    regexLo = nltk.re.compile('[a-z]')
    if (regexUp.search(text) != None) and (regexLo.search(text) != None):
        has = True
    return has

def hasDigit(text):
    has = False
    if len(text) < 3:
        return False
    myRegex = nltk.re.compile('[0-9]')
    if myRegex.search(text) != None:
        has = True
    return has


def hasNonAlphaNum(text):
    has = False
    if len(text) < 3:
        return False
    myRegex = nltk.re.compile('\W')
    if myRegex.search(text) != None:
        has = True
    return has

def word2features(sent, i):
    # print "i: " + str(i)
    # print "sent[i]" + sent[i]
    listElem = sent[i].split('|')
    word = listElem[0]
    lemma = listElem[1]
    postag = listElem[2]

    features = {
        # Names of TF and genes change by lower and upper characters: 'word.lower()': word.lower(),
        # Suffixes
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[-1:]': word[-1:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.hasDigit()': hasDigit(word),
        'word.hasNonAlphaNum': hasNonAlphaNum(word),
        # 'word.hasUpperLower': hasUpperLower(word),
        #'wordSize': wordSize(word),
        # 'word.isdigit()': word.isdigit(),
        'word': word,
        'lemma': lemma,
        'lemma[-3:]': lemma[-3:],
        'lemma[-2:]': lemma[-2:],
        'lemma[-1:]': lemma[-1:],
        'postag': postag,
        # Prefixes
        'postag[:2]': postag[:2],
        'postag[:1]': postag[:1],
    }
    if i > 0:
        listElem = sent[i - 1].split('|')
        word1 = listElem[0]
        lemma1 = listElem[1]
        postag1 = listElem[2]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.hasDigit()': hasDigit(word1),
            '-1:word.hasNonAlphaNum': hasNonAlphaNum(word1),
            # '-1:word.hasUpperLower': hasUpperLower(word1),
            '-1:word': word1,
            '-1:lemma': lemma1,
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:postag[:1]': postag1[:1],
        })
    # else:
    #    features['BOS'] = True

    if i < len(sent) - 1:
        listElem = sent[i + 1].split('|')
        word1 = listElem[0]
        lemma1 = listElem[1]
        postag1 = listElem[2]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.hasDigit()': hasDigit(word1),
            '+1:word.hasNonAlphaNum': hasNonAlphaNum(word1),
            # '+1:word.hasUpperLower': hasUpperLower(word1),
            '+1:word': word1,
            '+1:lemma': lemma1,
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:postag[:1]': postag1[:1],
        })
    # else:
    #    features['EOS'] = True
    if i > 1:
        listElem = sent[i - 2].split('|')
        word2 = listElem[0]
        lemma2 = listElem[1]
        postag2 = listElem[2]
        features.update({
            '-2:word.lower()': word2.lower(),
            '-2:word.istitle()': word2.istitle(),
            '-2:word.isupper()': word2.isupper(),
            '-2:word.hasDigit()': hasDigit(word2),
            '-2:word.hasNonAlphaNum': hasNonAlphaNum(word2),
            # '-2:word.hasUpperLower': hasUpperLower(word2),
            '-2:word': word2,
            '-2:lemma': lemma2,
            '-2:postag': postag2,
            '-2:postag[:2]': postag2[:2],
            '-2:postag[:1]': postag2[:1],
        })

    if i < len(sent) - 2:
        listElem = sent[i + 2].split('|')
        word2 = listElem[0]
        lemma2 = listElem[1]
        postag2 = listElem[2]
        features.update({
            '+2:word.lower()': word2.lower(),
            '+2:word.istitle()': word2.istitle(),
            '+2:word.isupper()': word2.isupper(),
            '+2:word.hasDigit()': hasDigit(word2),
            '+2:word.hasNonAlphaNum': hasNonAlphaNum(word2),
            # '+2:word.hasUpperLower': hasUpperLower(word2),
            '+2:word': word2,
            '+2:lemma': lemma2,
            '+2:postag': postag2,
            '+2:postag[:2]': postag2[:2],
            '+2:postag[:1]': postag2[:1],
        })

    trigrams = False
    if trigrams:
        if i > 2:
            listElem = sent[i - 3].split('|')
            word3 = listElem[0]
            lemma3 = listElem[1]
            postag3 = listElem[2]
            features.update({
                '-3:word.lower()': word3.lower(),
                '-3:word.istitle()': word3.istitle(),
                '-3:word.isupper()': word3.isupper(),
                '-3:word.hasDigit()': hasDigit(word3),
                '-3:word.hasNonAlphaNum': hasNonAlphaNum(word3),
                # '-3:word.hasUpperLower': hasUpperLower(word3),
                '-3:word': word3,
                '-3:lemma': lemma3,
                '-3:postag': postag3,
                '-3:postag[:2]': postag3[:2],
                '-3:postag[:1]': postag3[:1],
            })

        if i < len(sent) - 3:
            listElem = sent[i + 3].split('|')
            word3 = listElem[0]
            lemma3 = listElem[1]
            postag3 = listElem[2]
            features.update({
                '+3:word.lower()': word3.lower(),
                '+3:word.istitle()': word3.istitle(),
                '+3:word.isupper()': word3.isupper(),
                '+3:word.hasDigit()': hasDigit(word3),
                '+3:word.hasNonAlphaNum': hasNonAlphaNum(word3),
                # '+3:word.hasUpperLower': hasUpperLower(word3),
                '+3:word': word3,
                '+3:lemma': lemma3,
                '+3:postag': postag3,
                '+3:postag[:2]': postag3[:2],
                '+3:postag[:1]': postag3[:1],
            })

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [elem.split('|')[3] for elem in sent]
    # return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


def print_transitions(trans_features, f):
    for (label_from, label_to), weight in trans_features:
        # f.write("%-6s -> %-7s %0.6f\n" % (label_from, label_to, weight))
        # f.write("label_from :" + label_from)
        # f.write("label_to :" + label_to)
        # f.write("label_weight :" + weight)
        # f.write("{} -> {} {:0.6f}\n".format(label_from.encode("utf-8"), label_to.encode("utf-8"), weight))
        f.write("{:6} -> {:7} {:0.6f}\n".format(label_from, label_to, weight))


def print_state_features(state_features, f):
    for (attr, label), weight in state_features:
        # f.write("%0.6f %-8s %s\n" % (weight, label, attr))
        # f.write(attr.encode("utf-8"))
        # '{:06.2f}'.format(3.141592653589793)
        f.write("{:0.6f} {:8} {}\n".format(weight, label, attr.encode("utf-8")))


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
    parser.add_option("--trainingFile", dest="trainingFile",
                      help="File with training data set", metavar="FILE")
    parser.add_option("--testFile", dest="testFile",
                      help="File with test data set", metavar="FILE")
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
    print("Path of training data set: " + options.inputPath)
    print("File with training data set: " + str(options.trainingFile))
    print("Path of test data set: " + options.inputPath)
    print("File with test data set: " + str(options.testFile))
    print("Filtering stop words: " + str(options.filterStopWords))
    symbols = ['.', ',', ':', ';', '?', '!', '\'', '"', '<', '>', '(', ')', '-', '_', '/', '\\', '¿', '¡', '+', '{',
               '}', '[', ']', '*', '%', '$', '#', '&', '°', '`', '...']
    print("Filtering symbols " + str(symbols) + ': ' + str(options.filterSymbols))

    print('-------------------------------- PROCESSING --------------------------------')
    print('Reading corpus...')
    t0 = time()

    sentencesTrainingData = []
    sentencesTestData = []

    stopwords = [word.decode('utf-8') for word in stopwords.words('english')]

    with open(os.path.join(options.inputPath, options.trainingFile), "r") as iFile:
        # with open(os.path.join(options.inputPath, options.trainingFile), "r", encoding="utf-8", errors='replace') as iFile:
        for line in iFile.readlines():
            listLine = []
            line = line.decode("utf-8")
            for token in line.strip('\n').split():
                if options.filterStopWords:
                    listToken = token.split('|')
                    lemma = listToken[1]
                    # Original: if lemma in stopwords.words('english'):
                    # trainingTesting_Sklearn_crfsuite.py:269:
                    # UnicodeWarning: Unicode equal comparison failed to
                    # convert both arguments to Unicode -
                    # interpreting them as being unequal
                    if lemma in stopwords:
                        continue
                if options.filterSymbols:
                    listToken = token.split('|')
                    lemma = listToken[1]
                    if lemma in symbols:
                        # if lemma == ',':
                        #     print "Coma , identificada"
                        continue
                listLine.append(token)
            sentencesTrainingData.append(listLine)
        print "   Sentences training data: " + str(len(sentencesTrainingData))
        # print sentencesTrainingData[0]

    with open(os.path.join(options.inputPath, options.testFile), "r") as iFile:
        # with open(os.path.join(options.inputPath, options.testFile), "r", encoding="utf-8", errors='replace') as iFile:
        for line in iFile.readlines():
            listLine = []
            line = line.decode("utf-8")
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
                        # if lemma == ',':
                         #    print "Coma , identificada"
                        continue
                listLine.append(token)
            sentencesTestData.append(listLine)
        print "   Sentences test data: " + str(len(sentencesTestData))
        # print sentencesTestData[0]

    print("Reading corpus done in: %fs" % (time() - t0))

    print(sent2features(sentencesTrainingData[0])[0])
    print(sent2features(sentencesTestData[0])[0])
    # print(sent2labels(sentencesTrainingData[0]))
    # print(sent2labels(sentencesTestData[0]))
    t0 = time()

    X_train = [sent2features(s) for s in sentencesTrainingData]
    y_train = [sent2labels(s) for s in sentencesTrainingData]

    X_test = [sent2features(s) for s in sentencesTestData]
    # print X_test
    y_test = [sent2labels(s) for s in sentencesTestData]

    # Fixed parameters
    # crf = sklearn_crfsuite.CRF(
    #     algorithm='lbfgs',
    #     c1=0.1,
    #     c2=0.1,
    #     max_iterations=100,
    #     all_possible_transitions=True
    # )

    # Hyperparameter Optimization
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=100,
        all_possible_transitions=True
    )
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
    }

    # Original: labels = list(crf.classes_)
    # Original: labels.remove('O')
    labels = list(['GENE'])

    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='weighted', labels=labels)

    # search
    rs = RandomizedSearchCV(crf, params_space,
                            cv=10,
                            verbose=3,
                            n_jobs=-1,
                            n_iter=20,
                            # n_iter=50,
                            scoring=f1_scorer)
    rs.fit(X_train, y_train)

    # Fixed parameters
    # crf.fit(X_train, y_train)

    # Best hiperparameters
    # crf = rs.best_estimator_
    nameReport = options.trainingFile.replace('.txt', '.fStopWords_' + str(options.filterStopWords) + '.fSymbols_' + str(
        options.filterSymbols) + '.txt')
    with open(os.path.join(options.outputPath, "reports", "report_" + nameReport), mode="w") as oFile:
        oFile.write("********** TRAINING AND TESTING REPORT **********\n")
        oFile.write("Training file: " + options.trainingFile + '\n')
        oFile.write('\n')
        oFile.write('best params:' + str(rs.best_params_) + '\n')
        oFile.write('best CV score:' + str(rs.best_score_) + '\n')
        oFile.write('model size: {:0.2f}M\n'.format(rs.best_estimator_.size_ / 1000000))

    print("Training done in: %fs" % (time() - t0))
    t0 = time()

    # Update best crf
    crf = rs.best_estimator_

    # Saving model
    print("     Saving training model...")
    t1 = time()
    nameModel = options.trainingFile.replace('.txt', '.fStopWords_' + str(options.filterStopWords) + '.fSymbols_' + str(
        options.filterSymbols) + '.mod')
    joblib.dump(crf, os.path.join(options.outputPath, "models", nameModel))
    print("        Saving training model done in: %fs" % (time() - t1))

    # Evaluation against test data
    y_pred = crf.predict(X_test)
    print("*********************************")
    name = options.trainingFile.replace('.txt', '.fStopWords_' + str(options.filterStopWords) + '.fSymbols_' + str(
        options.filterSymbols) + '.txt')
    with open(os.path.join(options.outputPath, "reports", "y_pred_" + name), "w") as oFile:
        for y in y_pred:
            oFile.write(str(y) + '\n')

    print("*********************************")
    name = options.trainingFile.replace('.txt', '.fStopWords_' + str(options.filterStopWords) + '.fSymbols_' + str(
        options.filterSymbols) + '.txt')
    with open(os.path.join(options.outputPath, "reports", "y_test_" + name), "w") as oFile:
        for y in y_test:
            oFile.write(str(y) + '\n')

    print("Prediction done in: %fs" % (time() - t0))

    # labels = list(crf.classes_)
    # labels.remove('O')

    with open(os.path.join(options.outputPath, "reports", "report_" + nameReport), mode="a") as oFile:
        oFile.write('\n')
        oFile.write("Flat F1: " + str(metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)))
        oFile.write('\n')
        # labels = list(crf.classes_)
        sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
        )
        oFile.write(metrics.flat_classification_report(
            y_test, y_pred, labels=sorted_labels, digits=3
        ))
        oFile.write('\n')

        oFile.write("\nTop likely transitions:\n")
        print_transitions(Counter(crf.transition_features_).most_common(50), oFile)
        oFile.write('\n')

        oFile.write("\nTop unlikely transitions:\n")
        print_transitions(Counter(crf.transition_features_).most_common()[-50:], oFile)
        oFile.write('\n')

        oFile.write("\nTop positive:\n")
        print_state_features(Counter(crf.state_features_).most_common(200), oFile)
        oFile.write('\n')

        oFile.write("\nTop negative:\n")
        print_state_features(Counter(crf.state_features_).most_common()[-200:], oFile)
        oFile.write('\n')
