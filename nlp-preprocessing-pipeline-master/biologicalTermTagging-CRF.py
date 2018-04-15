# -*- coding: UTF-8 -*-
import json
from optparse import OptionParser
import os
import sys
from time import time
from nltk.corpus import words
import re

__author__ = 'CMendezC'

# Objective: Tagging biological terms from lists of terms related to aspects of interest:
#   1) Changing POS tag by term tag

# Parameters:
#   1) --inputPath Path to read input files.
#   2) --outputPath Path to place output files.
#   3) --termPath Path to read term lists
#   4) --termFiles JSON file with terms files and tags
#   5) --crf Let POS tag instead of substituting it by term or freq tag

# Output:
#   1) Files with biological terms tagged

# Execution:
# python biologicalTermTagging.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT\ECK120012096_GntR\lemma --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT\ECK120012096_GntR\term --termPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\resources\termLists --termFiles termFilesTag.json

# FhlA
# python biologicalTermTagging.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011394_FhlA\lemma --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011394_FhlA\term --termPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\resources\termLists --termFiles termFilesTag.json

# MarA
# python biologicalTermTagging.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011412_MarA\lemma --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011412_MarA\term --termPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\resources\termLists --termFiles termFilesTag.json

# ArgR
# python biologicalTermTagging.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011670_ArgR\lemma --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011670_ArgR\term --termPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\resources\termLists --termFiles termFilesTag.json

# CytR
# python biologicalTermTagging.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120012407_CytR\lemma --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120012407_CytR\term --termPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\resources\termLists --termFiles termFilesTag.json

# Rob
# python biologicalTermTagging.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011190_Rob\lemma --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011190_Rob\term --termPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\resources\termLists --termFiles termFilesTag.json

###########################################################
#                       MAIN PROGRAM                      #
###########################################################

if __name__ == "__main__":
    # Parameter definition
    parser = OptionParser()
    parser.add_option("--inputPath", dest="inputPath",
                      help="Path to read input files", metavar="PATH")
    parser.add_option("--outputPath", dest="outputPath",
                      help="Path to place transformed files", metavar="PATH")
    parser.add_option("--termPath", dest="termPath",
                      help="Path to read term files", metavar="PATH")
    parser.add_option("--termFiles", dest="termFiles",
                      help="JSON file with terms files and tags", metavar="FILE")
    parser.add_option("--crf", default=False,
                      action="store_true", dest="crf",
                      help="Let POS tag instead of substituting it by term or freq tag?")
    parser.add_option("--termLower", default=False,
                      action="store_true", dest="termLower",
                      help="Compare with terms in lower case?")
    parser.add_option("--termCapitalize", default=False,
                      action="store_true", dest="termCapitalize",
                      help="Compare with capitalize terms?")

    (options, args) = parser.parse_args()

    if len(args) > 0:
        parser.error("None parameters indicated.")
        sys.exit(1)

    # Printing parameter values
    print('-------------------------------- PARAMETERS --------------------------------')
    print("Path to read input files: " + str(options.inputPath))
    print("Path to place transformed files: " + str(options.outputPath))
    print("Path to read term files: " + str(options.termPath))
    print("Let POS tag instead of substituting it by term or freq tag? " + str(options.crf))
    print("Compare with terms in lower case? " + str(options.termLower))
    print("Compare with capitalize terms? " + str(options.termCapitalize))

    print('Loading biological term files...')
    with open(os.path.join(options.termPath, options.termFiles)) as data_file:
        lists = json.load(data_file)

    hashTermFiles = lists["hashTermFiles"]
    hashTerms = lists["hashTerms"]
    hashTermsOrig = {}

    for key in hashTermFiles.keys():
        if key not in hashTermsOrig:
            hashTermsOrig[key] = []
        for f in hashTermFiles[key]:
            # print('File: ' + f)
            with open(os.path.join(options.termPath, f), "r", encoding="utf-8", errors="replace") as iFile:
                for line in iFile:
                    line = line.strip('\n')
                    lineHyp = line.replace(' ', '-')
                    if lineHyp not in hashTerms[key]:
                        hashTerms[key].append(lineHyp)
                        hashTermsOrig[key].append(line)
                        if options.termLower:
                            hashTerms[key].append(lineHyp.lower())
                            hashTermsOrig[key].append(line.lower())
                        if options.termCapitalize:
                            hashTerms[key].append(lineHyp.capitalize())
                            hashTermsOrig[key].append(line.capitalize())
        print('   Terms read {} size: {}'.format(key, len(hashTerms[key])))
        print('   Terms read {} size: {}'.format(key, len(hashTermsOrig[key])))
        #print("hashTermsOrig: {}".format(hashTermsOrig))

    #regularWords =  words.words('en')
    print()

    filesPreprocessed = 0
    t0 = time()
    reHyphen = re.compile('-')
    print("Biological term tagging files...")
    # Walk directory to read files
    for path, dirs, files in os.walk(options.inputPath):
        # For each file in dir
        for file in files:
            print("   Biological term tagging file..." + str(file))
            with open(os.path.join(path, file), "r", encoding="utf-8", errors="replace") as iFile:
                # Create output file to write
                with open(os.path.join(options.outputPath, file.replace('lem.txt', 'term.txt')), "w", encoding="utf-8") as oFile:
                    for line in iFile:
                        if line == '\n':
                            oFile.write(line)
                        else:
                            line = line.strip('\r\n')
                            listLine1 = line.split('\t')
                            if len(listLine1) < 3:
                                continue
                            word = listLine1[0]
                            pos = listLine1[1]
                            listLine2 = listLine1[2].split(' ')
                            lemma = listLine2[0]
                            if len(word) > 1:
                                for termTag in hashTerms:
                                    if word in hashTerms[termTag]:
                                        if word.find('-') > -1:
                                            found = False
                                            repetitions = word.count('-')
                                            print("repetitions: {}".format(repetitions))
                                            wordOrig = word
                                            for i in range(0, repetitions):
                                                wordOrig = wordOrig.replace('-', ' ', 1)
                                                print("Word: {}".format(wordOrig))
                                                if wordOrig in hashTermsOrig[termTag]:
                                                    print("WordOrig: {}".format(wordOrig))
                                                    found = True
                                                    line = ''
                                                    for w, l in zip(word.split('-'), lemma.split('-')):
                                                        line += w + '\t' + listLine1[1] + '\t' + l + ' ' + termTag + ' TermTag' + '\n'
                                                    line = line.rstrip('\n')
                                                    break
                                                    #print("Line: {}".format(line))
                                            if not found:
                                                line = listLine1[0] + '\t' + listLine1[1] + '\t' + listLine2[0] + ' ' + termTag + ' TermTag'
                                        else:
                                            line = listLine1[0] + '\t' + listLine1[1] + '\t' + listLine2[0] + ' ' + termTag + ' TermTag'
                                    else:
                                        line = listLine1[0] + '\t' + listLine1[1] + '\t' + listLine2[0] + ' ' + 'O' + ' TermTag'
                                        # line = listLine1[0] + '\t' + termTag + '\t' + listLine2[0] + ' ' + termTag + ' TermTag'
                                        #line = word + '\t' + termTag + '\t' + lemma + ' ' + termTag + ' TermTag'
                            oFile.write(line + '\n')
            filesPreprocessed += 1

    # Imprime archivos procesados
    print()
    print("Files preprocessed: " + str(filesPreprocessed))
    print("In: %fs" % (time() - t0))
