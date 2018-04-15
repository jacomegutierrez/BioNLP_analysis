# -*- coding: UTF-8 -*-

from optparse import OptionParser
import os
import sys
from time import time

__author__ = 'CMendezC'

# Objective: Join transformed files for obtaining training and test data sets

# Parameters:
#   1) --inputPath      Path to read files.
#   2) --trainingFile   File name for training data.
#   3) --testFile       File name for test data.
#   4) --outputPath     Path to write files.

# Ouput:
#   1) Files created.

# Execution:
# python3.4 prepare-training-test.py
# --inputPath /export/space1/users/compu2/bionlp/conditional-random-fields/data-sets/transformed
# --trainingFile training-data-set-70.txt
# --testFile test-data-set-30.txt
# --outputPath /export/space1/users/compu2/bionlp/conditional-random-fields/data-sets
# python3.4 prepare-training-test.py --inputPath /export/space1/users/compu2/bionlp/conditional-random-fields/data-sets/transformed --trainingFile training-data-set-70.txt --testFile test-data-set-30.txt --outputPath /export/space1/users/compu2/bionlp/conditional-random-fields/data-sets

###########################################################
#                       MAIN PROGRAM                      #
###########################################################

if __name__ == "__main__":
    # Parameter definition
    parser = OptionParser()
    parser.add_option("--inputPath", dest="inputPath",
                      help="Path to read files", metavar="PATH")
    parser.add_option("--trainingFile", dest="trainingFile",
                      help="File for training examples", metavar="FILE")
    parser.add_option("--testFile", dest="testFile",
                      help="File for test examples", metavar="FILE")
    parser.add_option("--outputPath", dest="outputPath",
                      help="Path to write output file, feature parameter is concatenated to file name.", metavar="PATH")

    (options, args) = parser.parse_args()
    if len(args) > 0:
        parser.error("None parameters indicated.")
        sys.exit(1)

    # Printing parameter values
    print('-------------------------------- PARAMETERS --------------------------------')
    print("Path to read files: " + str(options.inputPath))
    print("File for training examples", str(options.trainingFile))
    print("File for test examples", str(options.testFile))
    print("Path to write output files: " + str(options.outputPath))

    t0 = time()
    trainingDataset = []
    testDataset = []

    counter = 1
    for path, dirs, files in os.walk(options.inputPath):
        # For each file in dir
        for file in files:
            if counter <= 70:
                print("   Joining file {} {} to training data set".format(counter, file))
                with open(os.path.join(path, file), "r", encoding="utf-8", errors="replace") as iFile:
                    for line in iFile:
                        line = line.strip('\r\n')
                        trainingDataset.append(line)
            elif counter > 70 and counter <= 100:
                print("   Joining file {} {} to test data set".format(counter, file))
                with open(os.path.join(path, file), "r", encoding="utf-8", errors="replace") as iFile:
                    for line in iFile:
                        line = line.strip('\r\n')
                        testDataset.append(line)
            counter += 1
    with open(os.path.join(options.outputPath, options.trainingFile), "w", encoding="utf-8", errors="replace") as oFile:
        for line in trainingDataset:
            oFile.write("{}\n".format(line))
    with open(os.path.join(options.outputPath, options.testFile), "w", encoding="utf-8", errors="replace") as oFile:
        for line in testDataset:
            oFile.write("{}\n".format(line))

