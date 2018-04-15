# -*- coding: UTF-8 -*-

from optparse import OptionParser
import os
import sys
from time import time
from subprocess import call

__author__ = 'CMendezC'

# Objective: Part-of-Speech Tagging of several files with Stanford POS Tagger.

# Parameters:
#   1) --inputPath Path to read TXT files.
#   2) --outputPath Path to place POST files.
#   3) --taggerPath Path POS Tagger command.
#   4) --biolemmatizer Format for biolemmatizer?.

# Output:
#   1) POS Tagged files.
#   2) If --biolemmatizer with format:
#   Rob	NNP
#   is	VBZ
#   a	DT
#   transcriptional	JJ
#   dual	JJ
#   regulator	NN
#   .	.
#
#   Its	PRP$
#   N-terminal	JJ ...

# Execution:
# GntR
# python posTaggingStanford.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT\ECK120012096_GntR\preprocessed --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT\ECK120012096_GntR\post --taggerPath C:\Users\cmendezc\Documents\GENOMICAS\STANFORD_POSTAGGER\stanford-postagger-2015-12-09 --biolemmatizer

# FhlA
# python posTaggingStanford.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011394_FhlA\preprocessed --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011394_FhlA\post --taggerPath C:\Users\cmendezc\Documents\GENOMICAS\STANFORD_POSTAGGER\stanford-postagger-2015-12-09 --biolemmatizer

# MarA
# python posTaggingStanford.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011412_MarA\preprocessed --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011412_MarA\post --taggerPath C:\Users\cmendezc\Documents\GENOMICAS\STANFORD_POSTAGGER\stanford-postagger-2015-12-09 --biolemmatizer

# ArgR
# python posTaggingStanford.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011670_ArgR\preprocessed --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011670_ArgR\post --taggerPath C:\Users\cmendezc\Documents\GENOMICAS\STANFORD_POSTAGGER\stanford-postagger-2015-12-09 --biolemmatizer

# CytR
# python posTaggingStanford.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120012407_CytR\preprocessed --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120012407_CytR\post --taggerPath C:\Users\cmendezc\Documents\GENOMICAS\STANFORD_POSTAGGER\stanford-postagger-2015-12-09 --biolemmatizer

# Rob
# python posTaggingStanford.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011190_Rob\preprocessed --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011190_Rob\post --taggerPath C:\Users\cmendezc\Documents\GENOMICAS\STANFORD_POSTAGGER\stanford-postagger-2015-12-09 --biolemmatizer

# EXTRACTING REGULATORY INTERACTIONS
# python posTaggingStanford.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\EXTRACTING_REGULATORY_INTERACTIONS\corpus_ecoli\preprocessed --outputPath C:\Users\cmendezc\Documents\GENOMICAS\EXTRACTING_REGULATORY_INTERACTIONS\corpus_ecoli\post  --taggerPath C:\Users\cmendezc\Documents\GENOMICAS\STANFORD_POSTAGGER\stanford-postagger-2015-12-09 --biolemmatizer

###########################################################
#                       MAIN PROGRAM                      #
###########################################################

if __name__ == "__main__":
    # Parameter definition
    parser = OptionParser()
    parser.add_option("-i", "--inputPath", dest="inputPath",
                      help="Path to read TXT files", metavar="PATH")
    parser.add_option("-o", "--outputPath", dest="outputPath",
                      help="Path to place POST files", metavar="PATH")
    parser.add_option("-a", "--taggerPath", dest="taggerPath", default="",
                      help="Path FreeLing analyzer files", metavar="PATH")
    parser.add_option("-p", "--biolemmatizer", default=False,
                      action="store_true", dest="biolemmatizer",
                      help="Format for biolemmatizer?")

    (options, args) = parser.parse_args()
    if len(args) > 0:
        parser.error("None parameters indicated.")
        sys.exit(1)

    # Printing parameter values
    print('-------------------------------- PARAMETERS --------------------------------')
    print("Path to read input files: " + str(options.inputPath))
    print("Path to place output files: " + str(options.outputPath))
    print("Path POS Tagger command: " + str(options.taggerPath))
    print("Format for biolemmatizer?: " + str(options.biolemmatizer))

    filesTagged = 0
    t0 = time()
    print("Tagging corpus...")
    # Walk directory to read files
    for path, dirs, files in os.walk(options.inputPath):
        # For each file in dir
        for file in files:
            print("   Tagging file..." + str(file))
            try:
                # FREELING: taggerPath = os.path.join(options.taggerPath, "analyzer.ex")
                # FREELING: command = taggerPath + " -f " + os.path.join("%FREELINGSHARE%", "config", "en.cfg") + " <" + os.path.join(path, file) + "> " + os.path.join(options.outputPath, file) + ".post.txt"

                # stanford-postagger models\english-left3words-distsim.tagger
                # C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TFsummaries_tagged_SGC_aspectRP-DOM\ECK120011190.Rob.sum.txt
                # >
                # C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\aspectsOfInterest_TrainingSet\testingTaggers\ECK120011190.Rob.sum.txt

                import platform
                plat = platform.system()
                if plat == 'Linux':
                    # FOR LINUX
                    # java -mx300m -cp 'stanford-postagger.jar:lib/*' edu.stanford.nlp.tagger.maxent.MaxentTagger
                    # -model $1 -textFile $2
                    command = "java -mx300m -cp " + os.path.join(options.taggerPath, 'stanford-postagger.jar:') + \
                              os.path.join(options.taggerPath, 'lib/*') + \
                              ' edu.stanford.nlp.tagger.maxent.MaxentTagger -model ' + \
                              os.path.join(options.taggerPath, 'models', 'english-left3words-distsim.tagger') + \
                              ' -textFile ' + os.path.join(options.inputPath, file) + \
                              ' > ' + os.path.join(options.outputPath, file.replace('pre.txt', 'pos.txt'))
                else:
                    # C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\preprocessingCorpus>java -mx300m
                    # -cp "C:\Users\cmendezc\Documents\GENOMICAS\STANFORD_POSTAGGER\stanford-postagger-2015-12-09\stanford-postagger.jar;
                    # C:\Users\cmendezc\Documents\GENOMICAS\STANFORD_POSTAGGER\stanford-postagger-2015-12-09\lib/*"
                    # edu.stanford.nlp.tagger.maxent.MaxentTagger -model
                    # C:\Users\cmendezc\Documents\GENOMICAS\STANFORD_POSTAGGER\stanford-postagger-2015-12-09\models\english-left3words-distsim.tagger
                    # -textFile C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\aspectClassificationDatasets\preprocessed\ECK120011190.Rob.sum.pre.txt
                    #taggerPath = os.path.join('java')
                    command = "java -mx300m -cp " + os.path.join(options.taggerPath, 'stanford-postagger.jar;') + \
                              os.path.join(options.taggerPath, 'lib/*') + \
                              ' edu.stanford.nlp.tagger.maxent.MaxentTagger -model ' + \
                              os.path.join(options.taggerPath, 'models', 'english-left3words-distsim.tagger') + \
                              ' -textFile ' + os.path.join(options.inputPath, file) + \
                              ' > ' + os.path.join(options.outputPath, file.replace('pre.txt', 'pos.txt'))  #print(command)

                retcode = call(command, shell=True)
                if retcode < 0:
                    print("   Child was terminated by signal", -retcode, file=sys.stderr)
                else:
                    print("   Child returned", retcode, file=sys.stderr)
                    filesTagged += 1
            except OSError as e:
                print("   Execution failed:", e, file=sys.stderr)

            text = ""
            if options.biolemmatizer:
                with open(os.path.join(options.outputPath, file.replace('pre.txt', 'pos.txt')), "r", encoding="utf-8", errors="replace") as iFile:
                    text = iFile.read()
                    # -LRB-_-LRB- PTS_NN -RRB-_-RRB-
                    # for_IN Mlc_NN inactivation_NN ._.
                    text = text.replace('-LRB-', '(')
                    text = text.replace('-RRB-', ')')

                    text = text.replace('-LSB-', '[')
                    text = text.replace('-RSB-', ']')

                    text = text.replace('_', '\t')
                    text = text.replace(' ', '\n')
                    text = text.replace('.\n', '.\n\n')
                with open(os.path.join(options.outputPath, file.replace('pre.txt', 'pos.txt')), "w", encoding="utf-8", errors="replace") as oFile:
                    oFile.write(text)

    # Imprime archivos procesados
    print()
    print("Files POS Tagged: " + str(filesTagged))
    print("Files POS Tagged in: %fs" % (time() - t0))
