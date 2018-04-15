# -*- coding: UTF-8 -*-
import json
import re
from optparse import OptionParser
import os
import sys
from time import time

import nltk

__author__ = 'CMendezC'


# Objective: Preprocessing paper files:
#     Eliminate lines beginning with:
#           Copyright � 1997
#           © 1997 Elsevier
#           Copyright © 1998,
#           Keywords: GntR; cAMP-CRP; GntP family
#           Received 21 October 1996/Accepted 27 December 1996
#           Received 6 January 1997; accepted 5 June 1997; Received by A. Nakazawa
#           (Received 29 June 1998/Accepted 3 August 1998)
#           REFERENCES: Eisenberg, R.C., Dobrogosz, W.J., 1967 | Hung, A., Orozco, A., Zwaig, N., 1970.
#                       Shine, J. & Dalgarno, L. (1974).
#                       34. Saier, M. H., T. M. Ramseier, and J. Reizer. 1996.
#           * Corresponding author. Mailing address: Department of Microbiology,
#           Phone: (614) 688-3518.
#           Fax: (614) 688-3519.
#           E-mail: conway.51@osu.edu.
#           Downloaded from
#     Selecting lines until ACKNOWLEDGMENTS or REFERENCES or Acknowledgements or References
#     Biological term detection

# Parameters:
#   1) --inputPath Path to read TXT files.
#   2) --outputPath Path to place POST files.
#   3) --termPath Path to read term lists
#   4) --termFiles JSON file with terms files and length
#   5) --termDetection If term detection is performed
#   6) --multiDocument  Processing multidocuments within input file?
#   7) --tabFormat  File with format PMID\tNUMSENT\tSENT\tCLASS?
#   8) --joinPunctuation Join separated punctuation (it comes separated from ODIN-XML files)

# Output:
#   1) preprocessed files with biological term detection

# Execution:
# GntR
# python preprocessingTermDetection.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT\ECK120012096_GntR\original --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT\ECK120012096_GntR\preprocessed --termPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\resources\termLists  --termFiles termFilesLength.json

# FhlA
# python preprocessingTermDetection.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011394_FhlA\original --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011394_FhlA\preprocessed --termPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\resources\termLists  --termFiles termFilesLength.json

# MarA
# python preprocessingTermDetection.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011412_MarA\original --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011412_MarA\preprocessed --termPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\resources\termLists  --termFiles termFilesLength.json

# ArgR
# python preprocessingTermDetection.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011670_ArgR\original --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011670_ArgR\preprocessed --termPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\resources\termLists  --termFiles termFilesLength.json

# CytR
# python preprocessingTermDetection.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120012407_CytR\original --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120012407_CytR\preprocessed --termPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\resources\termLists  --termFiles termFilesLength.json

# Rob
# python preprocessingTermDetection.py --inputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011190_Rob\original --outputPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\corpus\TF_PMIDs_TXT_ECK120011190_Rob\preprocessed --termPath C:\Users\cmendezc\Documents\GENOMICAS\AUTOMATIC_SUMMARIZATION_TFS\resources\termLists  --termFiles termFilesLength.json

# EXTRACTING REGULATORY INTERACTIONS
# python preprocessingTermDetection.py
# --inputPath C:\Users\cmendezc\Documents\GENOMICAS\EXTRACTING_REGULATORY_INTERACTIONS\corpus_ecoli\original
# --outputPath C:\Users\cmendezc\Documents\GENOMICAS\EXTRACTING_REGULATORY_INTERACTIONS\corpus_ecoli\preprocessed
# --termPath C:\Users\cmendezc\Documents\GENOMICAS\preprocessingTermTagging_v1.0\termLists
# --termFiles termFilesLength.json

# def addEndPeriod(cad):
#     if cad.endswith('.'):
#         return cad
#     else:
#         return cad + '.'


###########################################################
#                       MAIN PROGRAM                      #
###########################################################

if __name__ == "__main__":
    # Parameter definition
    parser = OptionParser()
    parser.add_option("--inputPath", dest="inputPath",
                      help="Path to read input files", metavar="PATH")
    parser.add_option("--outputPath", dest="outputPath",
                      help="Path to place output files", metavar="PATH")
    parser.add_option("--termPath", dest="termPath",
                      help="Path of term files", metavar="PATH")
    parser.add_option("--termFiles", dest="termFiles",
                      help="JSON file with terms files and length", metavar="PATH")
    parser.add_option("--termDetection", default=False,
                      action="store_true", dest="termDetection",
                      help="Perform term detection?")
    parser.add_option("--multiDocument", default=False,
                      action="store_true", dest="multiDocument",
                      help="Processing multidocuments within input file?")
    parser.add_option("--tabFormat", default=False,
                      action="store_true", dest="tabFormat",
                      help="File with format PMID\tNUMSENT\tSENT\tCLASS?")
    parser.add_option("--joinPunctuation", default=False,
                      action="store_true", dest="joinPunctuation",
                      help="Join separated punctuation?")
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
    print("Path to place output files: " + str(options.outputPath))
    print("Perform term detection?: " + str(options.termDetection))
    if options.termDetection:
        print("Path to read terminological resources: " + str(options.termPath))
        print("JSON file with terms files and length: " + str(options.termFiles))
    print("Processing multidocuments within input file?: " + str(options.multiDocument))
    print("File with format PMID\tNUMSENT\tSENT\tCLASS?: " + str(options.tabFormat))
    print("Join separated punctuation?: " + str(options.joinPunctuation))

    # ####       REGEX DEFINITION FOR UNNECESSARY LINES    #####
    regexEmptyLine = re.compile('^\s*$')
    #           Copyright � 1997
    #           © 1997 Elsevier
    #           Copyright © 1998,
    #           Keywords: GntR; cAMP-CRP; GntP family
    #           Received 21 October 1996/Accepted 27 December 1996
    #           Received 6 January 1997; accepted 5 June 1997; Received by A. Nakazawa
    #           (Received 29 June 1998/Accepted 3 August 1998)
    #           * Corresponding author. Mailing address: Department of Microbiology,
    #           Phone: (614) 688-3518.
    #           Fax: (614) 688-3519.
    #           E-mail: conway.51@osu.edu.
    #           Downloaded from
    #           www.sciencedirect.com Current Opinion in Microbiology 2008, 11:87–9388 Cell regulation
    #           DOI 10.1016 / j.mib .2008.02.007
    #           Correspondence to J

    #           journal homepage: www.elsevier.com/locate/biotechadv
    #           Research review paper
    #           Article history:
    #           Accepted 18 April 2014
    #           Available online 26 April 2014
    #           Abbreviations : ROS ,
    #           JOURNAL OF
    #           0021-9193/02

    #           Mailing address : CSIC - Estación Experimental del Zaidín , Apdo .
    #           Correos 419 , E - 18008 Granada , Spain .
    #           Phone : 34 - 58 - 121011 .
    #           Fax : 34 - 58 - 129600 .
    #           Present address : Department of Biology , Imperial College of Science ,

    expression = '^(Copyright|© [0-9][0-9][0-9][0-9]|Keywords:|\(?Received [0-9]?[0-9]|\*?\s?Corresponding author|' + \
                 'Phone:|Fax:|E-mail:|Phone\s:|Fax\s:|E-mail\s:|Mailing\saddress\s:|Present\saddress\s:|' + \
                 'Downloaded\sfrom|DOI|www\.sciencedirect\.com|Correspondence to [A-ZÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛ]|' + \
                 'journal homepage:|Research review paper|Article history:|\(?Accepted [0-9]?[0-9]|' + \
                 'Available online|Abbreviations:|ACKNOWLEDGMENTS\s|REFERENCES\s|' + \
                 'All rights reserved|Published by Elsevier|' + \
                 'Verbatim copying and redistribution of this article|J Bacteriol [0-9][0-9][0-9][0-9]|' + \
                 'Mol Microbiol [0-9][0-9][0-9][0-9]|Nucleic Acids Res [0-9][0-9][0-9][0-9]|' + \
                 'JOURNAL OF|[0-9][0-9][0-9][0-9]\-[0-9][0-9][0-9]/[0-9][0-9]|[0-9][0-9][0-9] – [0-9][0-9][0-9] Vol)'
    regexUnnecessaryLines = re.compile(expression)
    #regexUnnecessaryLines = re.compile('^(Copyright)')
    #           REFERENCES: Eisenberg, R.C., Dobrogosz, W.J., 1967
    #                       Hung, A., Orozco, A., Zwaig, N., 1970.
    #                       Shine, J. & Dalgarno, L. (1974).
    #                       34. Saier, M. H., T. M. Ramseier, and J. Reizer. 1996.
    #                       1. Pesavento, C. & Hengge, R. Bacterial nucleotide-based
    #                       Battesti , N .
    #                       Aiba , H . , T .
    #                       Yamamoto , and M .
    # regexReferences = re.compile('^([0-9]?[0-9]\.\s)?[A-ZÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛ][a-záéíóúàèìòùüâêîôû\-]+\s?,\s([A-ZÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛ]\s?\.\s?)+.*([0-9][0-9][0-9][0-9])')
    # regexReferences = re.compile('^([0-9]?[0-9]\.\s)?[A-ZÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛ][a-záéíóúàèìòùüâêîôû\-]+\s?,\s([A-ZÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛ]\s?\.\s?)+')
    regexReferences = re.compile('^([0-9]?[0-9]\.\s)?[A-ZÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛ][a-záéíóúàèìòùüâêîôû\-]+\s?,\s([A-ZÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛ]\s?\.\s?)+($|.*\(\s?[0-9][0-9][0-9][0-9]\s?\))')
    # Lines without words, with only symbols
    # --.-,.;....a...........c....
    # .........
    # 2.;
    # ..~......: ........................
    # ::..:.< -.;-.:.;L.:.5 %..-.-...;..;..,:
    # ?........., .....,: ........,,::, , ...
    # ..
    # .J
    # L,.
    # 2
    # i
    # regexLinesNoText = re.compile('^[^a-zA-Z0-9]')

    # regexUnderscoreWord = re.compile(r'\b_\b')

    # 40 o more dots which appear in index lines
    regexIndexLine = re.compile('\.{40}')

    # e-mails
    regexEmail = re.compile(
        '(e-mail : |e-mail: |e-mail )?([a-zA-Z0-9\._\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-]+ |[a-zA-Z0-9\._\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-]+ )')

    ### DETECTAR CONTENTS Y ELIMINAR HASTA INTRODUCTION (?): Overview of oxidative stress response ... ... 28 2 .
    ### SI ES INTRODUCTION, AKNOLEDGMENTS U OTRO TÍTULO, PONERLE PUNTO O ELIMINARLO SI ES A INICIO DE PALABRA Y NO HAY OTRO PALABRA DESPUÉS.
    # A VECES SE USA Summary

    # Join separated punctuation
    if options.joinPunctuation:
        # 1) join to right: (, [, “, ‘, ±, ~
        regexPuncRigth = re.compile('(?P<punct>[\(\[“‘±~])\s')
        # 2) join to left: ), ], ., ,, ”, ´, ;, %, :, ’, '
        regexPuncLeft = re.compile('\s(?P<punct>[\)\]\.,”´;%:’\'])')
        # 3) join both sides: -, /, –, —
        regexPuncBoth = re.compile('\s(?P<punct>[-/–—])\s')
        # 4) genitive: ArgP ’ s
        regexPuncGenitive = re.compile('(?P<before>[a-zA-Z])\s’\ss\s')

    # ####       LOADING BIOLOGICAL TERM FILES    #####
    if options.termDetection:
        with open(os.path.join(options.termPath, options.termFiles)) as data_file:
            hashes = json.load(data_file)

        hashTermFiles = hashes["hashTermFiles"]
        hashTerms = hashes["hashTerms"]

        for key in hashTermFiles.keys():
            for f in hashTermFiles[key]:
                with open(os.path.join(options.termPath, f), "r", encoding="utf-8", errors="replace") as iFile:
                    for line in iFile:
                        line = line.strip('\n')
                        if line not in hashTerms[key]:
                            hashTerms[key].append(line)
                            if options.termLower:
                                hashTerms[key].append(line.lower())
                            if options.termCapitalize:
                                hashTerms[key].append(line.capitalize())
            print('   Terms read {} size: {}'.format(key, len(hashTerms[key])))

    filesProcessed = 0
    t0 = time()
    print("Preprocessing files...")
    # Walk directory to read files
    for path, dirs, files in os.walk(options.inputPath):
        # For each file in dir
        for file in files:
            print("   Preprocessing file..." + str(file))
            text = ''
            listSentences = []
            references = 0
            with open(os.path.join(path, file), "r", encoding="utf-8", errors="replace") as iFile:
                # Create output file to write
                # with open(os.path.join(options.outputPath, file.replace('.txt', '.pre.txt')), "w", encoding="utf-8") as oFile:
                for line in iFile:
                    originalLine = line.strip('\n')
                    if options.joinPunctuation:
                        originalLine = regexPuncGenitive.sub(r'\g<before>’s', originalLine)
                        originalLine = regexPuncRigth.sub(r'\g<punct>', originalLine)
                        originalLine = regexPuncLeft.sub(r'\g<punct>', originalLine)
                        originalLine = regexPuncBoth.sub(r'\g<punct>', originalLine)
                    if options.tabFormat:
                        listLine = originalLine.split('\t')
                        line = listLine[2]
                    ### DETECTAR AKNOWLEDGMENTS Y ELIMINAR TODO LO QUE SIGA
                    # This eliminate usefull part of pepers if line.upper().startswith('ACKNOWLEDGMENT') or line.upper().startswith('REFERENCES') or references > 2:

                    # Do not eliminate references because within them there are RIs
                    # if not options.multiDocument:
                    #     if line.upper() == 'ACKNOWLEDGMENTS' or line.upper() == 'REFERENCES' or references > 2:
                    #         break
                    if not options.multiDocument:
                        if line.upper() == 'ACKNOWLEDGMENTS':
                            break
                    # if line == '' or line == None:
                    if regexEmptyLine.match(line) != None:
                        print('Empty line ' + line)
                        continue
                    # Do not eliminate references because within them there are RIs
                    #  if regexReferences.match(line) != None:
                    #    print('Reference line ' + str(line.encode(encoding='UTF-8', errors='replace')))
                    #    references += 1
                    #    continue
                    # if regexUnnecessaryLines.match(line) != None:
                    if regexUnnecessaryLines.search(line) != None:
                        print('Unnecessary line ' + str(line.encode(encoding='UTF-8', errors='replace')))
                        continue
                    if regexIndexLine.search(line) != None:
                        print('Index line ' + line)
                        continue
                    if regexEmail.search(line) != None:
                         print('Line with email: ' + line)
                         line = regexEmail.sub(' ', line)
                    #     print(line)

                    text += originalLine + '\n'

            if options.termDetection:
                # ####       BIOLOGICAL TERM DETECTION    #####
                print('     Detecting biological terms...')
                for key in sorted(hashTerms.keys(), reverse=True):
                    #print('     length: ' + str(key))
                    for term in hashTerms[key]:
                        #print(str(term.encode(encoding='UTF-8', errors='replace')))
                        text = text.replace(term, term.replace(' ', '-'))
                        #regexTerm = re.compile(r'' + term)
                        #regexTerm.sub(term.replace(' ', '_TERM_'), text)

            filesProcessed += 1
            with open(os.path.join(options.outputPath, file.replace(' ', '').replace('.txt', '.pre.txt')), "w", encoding="utf-8") as oFile:
                oFile.write(text)

    # Imprime archivos procesados
    print()
    print("Files preprocessed: " + str(filesProcessed))
    print("In: %fs" % (time() - t0))
