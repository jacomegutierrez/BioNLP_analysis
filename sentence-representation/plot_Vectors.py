# -*- coding: UTF-8 -*-
import os
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt
import re
import numpy as np
import matplotlib
from sklearn.metrics.pairwise import cosine_similarity

matplotlib.use('Qt4Agg')


__author__ = 'CMendezC'

# Objective: Plot vectors into 2D and 3D
# with a color for vectors using different transformations

# Parameters:
#   1) --vectorPath Path to read vectors.
#   2) --vectorFile File to read vectors.
#   3) --outputPath Path to place plot files.
#   4) --outputFormat Plot file format: pdf, png
#   5) --absoluteValue Employ absolute values in vectors

# Ouput:
#   1) Plots

# Execution:
# C:\Anaconda3\python plot_Vectors_LSA.py
# --outputPath C:\Users\cmendezc\Documents\GENOMICAS\GENSOR_UNITS\wordEmbeddings\lsa\plots
# --vectorPath C:\Users\cmendezc\Documents\GENOMICAS\GENSOR_UNITS\wordEmbeddings\lsa
# --vectorFile GU_lsa_local_vectors_2T.txt
# --absoluteValue
# --outputFormat pdf

# C:\Anaconda3\python plot_Vectors_LSA.py --outputPath C:\Users\cmendezc\Documents\GENOMICAS\GENSOR_UNITS\wordEmbeddings\lsa\plots --vectorPath C:\Users\cmendezc\Documents\GENOMICAS\GENSOR_UNITS\wordEmbeddings\lsa --vectorFile GU_lsa_local_vectors_2T.txt --absoluteValue --outputFormat pdf
# C:\Anaconda3\python plot_Vectors_LSA.py --outputPath C:\Users\cmendezc\Documents\GENOMICAS\GENSOR_UNITS\wordEmbeddings\lsa\plots --vectorPath C:\Users\cmendezc\Documents\GENOMICAS\GENSOR_UNITS\wordEmbeddings\lsa --vectorFile GU_lsa_local_vectors_10T.txt --absoluteValue --outputFormat pdf
# C:\Anaconda3\python plot_Vectors_LSA.py --outputPath C:\Users\cmendezc\Documents\GENOMICAS\GENSOR_UNITS\wordEmbeddings\lsa\plots --vectorPath C:\Users\cmendezc\Documents\GENOMICAS\GENSOR_UNITS\wordEmbeddings\lsa --vectorFile GU_lsa_local_vectors_36T.txt --absoluteValue --outputFormat pdf
# C:\Anaconda3\python plot_Vectors_LSA.py --outputPath C:\Users\cmendezc\Documents\GENOMICAS\GENSOR_UNITS\wordEmbeddings\lsa\plots --vectorPath C:\Users\cmendezc\Documents\GENOMICAS\GENSOR_UNITS\wordEmbeddings\lsa --vectorFile GU_lsa_local_vectors_88T.txt --absoluteValue --outputFormat pdf
# C:\Anaconda3\python plot_Vectors_LSA.py --outputPath C:\Users\cmendezc\Documents\GENOMICAS\GENSOR_UNITS\wordEmbeddings\lsa\plots --vectorPath C:\Users\cmendezc\Documents\GENOMICAS\GENSOR_UNITS\wordEmbeddings\lsa --vectorFile GU_lsa_local_vectors_120T.txt --absoluteValue --outputFormat pdf

# python3.4 plot_Vectors.py
# --outputPath /home/cmendezc/gitlab_repositories/sentence-representation-word-embeddings/sentence-representation
# --vectorPath /home/cmendezc/gitlab_repositories/sentence-representation-word-embeddings/sentence-representation
# --vectorFile test.vec --absoluteValue --outputFormat pdf
# python3.4 plot_Vectors.py --outputPath /home/cmendezc/gitlab_repositories/sentence-representation-word-embeddings/sentence-representation --vectorPath /home/cmendezc/gitlab_repositories/sentence-representation-word-embeddings/sentence-representation --vectorFile test.vec --absoluteValue --outputFormat pdf

###########################################################
#                       MAIN PROGRAM                      #
###########################################################

if __name__ == "__main__":
    # Parameter definition
    parser = OptionParser()
    parser.add_option("--vectorPath", dest="vectorPath",
                      help="Path to read vector file", metavar="PATH")
    parser.add_option("--vectorFile", dest="vectorFile",
                      help="File to read vectors", metavar="FILE")
    parser.add_option("--outputPath", dest="outputPath",
                      help="Path to place clustering classified files", metavar="PATH")
    parser.add_option("--outputFormat", dest="outputFormat", choices=('pdf', 'png'),
                      help="Plot output format", metavar="PATH")
    parser.add_option("--absoluteValue", default=False,
                      action="store_true", dest="absoluteValue",
                      help="Use vector absolute values?")

    (options, args) = parser.parse_args()
    if len(args) > 0:
        parser.error("None parameters indicated.")
        sys.exit(1)

    # Printing parameter values
    print('-------------------------------- PARAMETERS --------------------------------')
    print("Path to read vector file: " + str(options.vectorPath))
    print("File to read vectors: " + str(options.vectorFile))
    print("Path to write plots: " + str(options.outputPath))
    print("Plot output format: " + str(options.outputFormat))
    print("Use vector absolute values? " + str(options.absoluteValue))

    #regexLen = re.compile(r'_(?P<vectorLen>[0-9]+)T')
    listVectors = []
    listLabels = []
    print("Reading vectors...")
    #result = regexLen.search(options.vectorFile)
    #vectorLen = 0
    #if result:
    #    vectorLen = int(result.group('vectorLen'))
    #    print("Vector vectorLen: {}".format(vectorLen))
    #else:
    #    print("None vectorLen mentioned within name file!")
    #    quit()
    with open(os.path.join(options.vectorPath, options.vectorFile), mode="r", encoding='utf8') as iFile:
        for line in iFile.readlines():
            line = line.strip('\r\n')
            listLine = line.split()
            # print("Len listLine: {}".format(len(listLine)))
            label = listLine[0][:12]
            # print("   Label: {}".format(label))
            vector = []
            listValues = listLine[1:]
            # print("   Len listValues: {}".format(len(listValues)))
            #if len(listValues) != vectorLen:
            #    print("Vector vectorLen does not match: {}".format(label))
            #    continue
            for elem in listValues:
                if options.absoluteValue:
                    vector.append(abs(float(elem)))
                else:
                    vector.append(float(elem))
            listLabels.append(label)
            listVectors.append(vector)
    print("   Reading vectors done!")
    print("    Len vectors: " + str(len(listVectors)))
    print("    Len labels: " + str(len(listLabels)))

    similarityMatrix = cosine_similarity(np.array(listVectors))
    print("similarityMatrix shape: {}".format(similarityMatrix.shape))

    t0 = time()
    print("Plotting heatmap...")
    # fig, ax = plt.subplots()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # heatmap = ax.pcolor(similarityMatrix, cmap=plt.cm.Reds, alpha=0.8)
    heatmap = ax.pcolor(similarityMatrix, cmap=plt.cm.Reds)
    fig = plt.gcf()
    fig.set_size_inches(16, 16)
    ax.set_frame_on(False)
    ax.set_yticks(np.arange(similarityMatrix.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(similarityMatrix.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticklabels(listLabels, minor=False, size='xx-small')
    ax.set_yticklabels(listLabels, minor=False, size='xx-small')
    plt.xticks(rotation=90)
    ax.grid(False)

    # Turn off all the ticks
    ax = plt.gca()

    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    fig.tight_layout()
    if options.absoluteValue:
        fileName = options.vectorFile+ '.abs.' + options.outputFormat
    else:
        fileName = options.vectorFile + '.' + options.outputFormat
    fig.savefig(os.path.join(options.outputPath, fileName))

    # plt.axis('tight')
    # plt.show()
    # plt.savefig('test.png', bbox_inches='tight')

    print("   Plotting heatmap done in %fs" % (time() - t0))
