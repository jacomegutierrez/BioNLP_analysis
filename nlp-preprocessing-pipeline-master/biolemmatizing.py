# -*- coding: UTF-8 -*-

from optparse import OptionParser
import os
import sys
from time import time
from subprocess import call

__author__ = 'CMendezC'

# Objective: Lemmatizing several files with BIOLemmatizer.

# Parameters:
#   1) --inputPath Path to read TXT files.
#   2) --outputPath Path to place POST files.
#   3) --biolemmatizerPath Path BIOLemmatizer command.

# Input:
#   1) POS Tagged files in format:
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

# Output:
#   1) BIOLemmatized files.

# Execution:
# python biolemmatizing.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT\ECK120012096_GntR\post --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT\ECK120012096_GntR\lemma --biolemmatizerPath C:\Users\cmendezc\Documents\GENOMICAS\BIO_LEMMATIZER

# FhlA
# python biolemmatizing.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011394_FhlA\post --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011394_FhlA\lemma --biolemmatizerPath C:\Users\cmendezc\Documents\GENOMICAS\BIO_LEMMATIZER

# MarA
# python biolemmatizing.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011412_MarA\post --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011412_MarA\lemma --biolemmatizerPath C:\Users\cmendezc\Documents\GENOMICAS\BIO_LEMMATIZER

# ArgR
# python biolemmatizing.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011670_ArgR\post --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011670_ArgR\lemma --biolemmatizerPath C:\Users\cmendezc\Documents\GENOMICAS\BIO_LEMMATIZER

# CytR
# python biolemmatizing.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120012407_CytR\post --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120012407_CytR\lemma --biolemmatizerPath C:\Users\cmendezc\Documents\GENOMICAS\BIO_LEMMATIZER

# Rob
# python biolemmatizing.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011190_Rob\post --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011190_Rob\lemma --biolemmatizerPath C:\Users\cmendezc\Documents\GENOMICAS\BIO_LEMMATIZER

# EXTRACTING REGULATORY INTERACTIONS
# python biolemmatizing.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\EXTRACTING_REGULATORY_INTERACTIONS\corpus_ecoli\post --outputPath C:\Users\cmendezc\Documents\GENOMICAS\EXTRACTING_REGULATORY_INTERACTIONS\corpus_ecoli\lemma  --biolemmatizerPath C:\Users\cmendezc\Documents\GENOMICAS\BIO_LEMMATIZER

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
    parser.add_option("-a", "--biolemmatizerPath", dest="biolemmatizerPath", default="",
                      help="Path BIOLemmatizer", metavar="PATH")

    (options, args) = parser.parse_args()
    if len(args) > 0:
        parser.error("None parameters indicated.")
        sys.exit(1)

    # Printing parameter values
    print('-------------------------------- PARAMETERS --------------------------------')
    print("Path to read input files: " + str(options.inputPath))
    print("Path to place output files: " + str(options.outputPath))
    print("Path BIOLemmatizer command: " + str(options.biolemmatizerPath))

    filesTagged = 0
    t0 = time()
    print("Lemmatizing corpus...")
    # Walk directory to read files
    for path, dirs, files in os.walk(options.inputPath):
        # For each file in dir
        for file in files:
            print("   Lemmatizing file..." + str(file))
            try:

                #java -Xmx1G -jar biolemmatizer-core-1.2-jar-with-dependencies.jar
                # -i C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\aspectsOfInterest_TrainingSet\sentences_POST_Test.Stanford.post.biolemm.txt
                # -o C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\aspectsOfInterest_TrainingSet\sentences_POST_Test.Stanford.post.biolemm.lemm.txt

                taggerPath = os.path.join('java')
                command = taggerPath + " -Xmx1G -jar " + os.path.join(options.biolemmatizerPath, 'biolemmatizer-core-1.2-jar-with-dependencies.jar') + \
                                    ' -i ' + os.path.join(options.inputPath, file) + \
                                    ' -o ' + os.path.join(options.outputPath, file.replace('pos.txt', 'lem.txt'))
                #print(command)
                retcode = call(command, shell=True)
                if retcode < 0:
                    print("   Child was terminated by signal", -retcode, file=sys.stderr)
                else:
                    print("   Child returned", retcode, file=sys.stderr)
                    filesTagged += 1
            except OSError as e:
                print("   Execution failed:", e, file=sys.stderr)

    # Imprime archivos procesados
    print()
    print("Files BIOLemmatized: " + str(filesTagged))
    print("Files BIOLemmatized in: %fs" % (time() - t0))
