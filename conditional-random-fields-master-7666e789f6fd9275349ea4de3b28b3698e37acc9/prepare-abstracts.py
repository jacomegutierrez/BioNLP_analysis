# -*- coding: UTF-8 -*-

from optparse import OptionParser
import os
import sys
from time import time
import re

__author__ = 'CMendezC'

# Objective: Take text-annotated-abstracts-original.txt as input
# for obtaining abstracts separated in files without tags and collecting dictionary of genes
# for tagging after NLP pipeline.

# Parameters:
#   1) --inputPath      Input path.
#   2) --inputFile   Input file.
#   3) --outputPath     Output path

# Execution:
# python3 prepare-abstracts.py
# --inputPath /export/space1/users/compu2/bionlp/conditional-random-fields/data-sets
# --inputFile text-annotated-abstracts.txt
# --outputPath /export/space1/users/compu2/bionlp/conditional-random-fields/data-sets/original
# --dicPath /export/space1/users/compu2/bionlp/nlp-preprocessing-pipeline/dictionaries
# --dicFile genes.txt
# python3 prepare-abstracts.py --inputPath /export/space1/users/compu2/bionlp/conditional-random-fields/data-sets --inputFile text-annotated-abstracts-original.txt --outputPath /export/space1/users/compu2/bionlp/conditional-random-fields/data-sets/original --dicPath /export/space1/users/compu2/bionlp/nlp-preprocessing-pipeline/dictionaries --dicFile genes.txt


if __name__ == "__main__":
    # Parameter definition
    parser = OptionParser()
    parser.add_option("--inputPath", dest="inputPath",
                      help="Input path", metavar="PATH")
    parser.add_option("--inputFile", dest="inputFile",
                      help="Input file", metavar="FILE")
    parser.add_option("--outputPath", dest="outputPath",
                      help="Output path", metavar="PATH")
    parser.add_option("--dicPath", dest="dicPath",
                      help="Dictionary path", metavar="PATH")
    parser.add_option("--dicFile", dest="dicFile",
                      help="Dictionary file", metavar="FILE")

    (options, args) = parser.parse_args()
    if len(args) > 0:
        parser.error("None parameters indicated.")
        sys.exit(1)

    # Printing parameter values
    print('-------------------------------- PARAMETERS --------------------------------')
    print("Input path: " + str(options.inputPath))
    print("Input file", str(options.inputFile))
    print("Output path: " + str(options.outputPath))
    print("Dictionary path: " + str(options.dicPath))
    print("Dictionary file", str(options.dicFile))

    filesWritten = 0
    t0 = time()
    hashGenes = {}

    rePmid = re.compile(r'([\d]+)\|t\|')
    reGene = re.compile(r'<g>([^<]+)</g>')
    reTags = re.compile(r'(<g>|</g>|<d>|</d>|<i>|</i>)')
    with open(os.path.join(options.inputPath, options.inputFile), "r", encoding="utf-8", errors="replace") as iFile:
        print("Reading file..." + options.inputFile)
        for line in iFile:
            line = line.strip('\r\n')
            for gene in reGene.findall(line):
                # print("genes: {}".format(gene))
                if gene not in hashGenes:
                    hashGenes[gene] = 1
                else:
                    hashGenes[gene] += 1
            line = reTags.sub('', line)
            result = rePmid.match(line)
            if result:
                line = rePmid.sub('', line)
                with open(os.path.join(options.outputPath, result.group(1) + ".txt"), "w", encoding="utf-8", errors="replace") as oFile:
                    oFile.write(line)
            else:
                print("Warning: line without PMID")
    with open(os.path.join(options.dicPath, options.dicFile), "w", encoding="utf-8", errors="replace") as dFile:
        for gene in hashGenes.keys():
            dFile.write("{}\n".format(gene))


