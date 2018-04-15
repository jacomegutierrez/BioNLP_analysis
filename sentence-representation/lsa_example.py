"""Pirated example from Gensim library (a NLP specialized tool):
https://radimrehurek.com/gensim/tut2.html
https://radimrehurek.com/gensim/wiki.html#latent-semantic-analysis

Ignacio Arroyo
"""
# run example tepeu: python3.4 lsa_example.py --input lsa_example.csv

import gensim
import logging
from six import iteritems
from gensim import corpora
import argparse

from pdb import set_trace as st # Debug the program step by step calling st()
                                # anywhere.
class corpus_streamer(object):
    """ This Object streams the input raw text file row by row.
    """
    def __init__(self, file_name, dictionary=None, strings=None):
        self.file_name=file_name
        self.dictionary=dictionary
        self.strings=strings

    def __iter__(self):
        for line in open(self.file_name):
        # assume there's one document per line, tokens separated by whitespace
            if self.dictionary and not self.strings:
                yield self.dictionary.doc2bow(line.lower().split())
            elif not self.dictionary and self.strings:
                yield line.strip().lower()
# Logging all our program
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--n_topics", help="Number of eigenvectors picked up.",
                    default=2, type=int)
parser.add_argument("--input", help="Input file to perform LSA.",
                    required=True)

args = parser.parse_args()

n_topics=args.n_topics
n_docs=0
input_file=args.input
#input_file='/medargsia/iarroyof/Volumen de 384 GB/data/GUs_textform_noPeriods.txt'
#input_file='lsa_example.csv'
#input_file='wiki_sample/wiki_75_AA.txt.cln'
#input_file='wiki_sample/wiki_77_AA.txt'

# A little stopwords list
stoplist = set('for a of the and to in _ [ ]'.split())
# Do not load the text corpus into memory, but stream it!
fille=corpus_streamer(input_file, strings=True)
dictionary=corpora.Dictionary(line.lower().split() for line in fille)#open(input_file))
# remove stop words and words that appear only once
stop_ids=[dictionary.token2id[stopword] for stopword in stoplist
                                             if stopword in dictionary.token2id]
once_ids=[tokenid for tokenid, docfreq in iteritems(dictionary.dfs)
                                                            if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids)
# remove gaps in id sequence after words that were removed
dictionary.compactify()
# Store the dictionary
dictionary.save('lsa_mini.dict')
# Reading sentences from file into a list of strings.
# Use instead streaming objects:
# Load stored word-id map (dictionary)
stream_it = corpus_streamer(input_file, dictionary=dictionary)
#for vector in stream_it:  # load one vector into memory at a time
#    print vector
# Convert to sparse matrix
sparse_corpus = [text for text in stream_it]
# Store to disk, for later use collect statistics about all tokens
corpora.MmCorpus.serialize('lsa_mini.mm',
                            sparse_corpus)
## LSA zone
# load the dictionary saved before
id2word = dictionary.load('lsa_mini.dict')
# Now load the sparse matrix corpus from file into a (memory friendly) streaming
# object.
corpus=corpora.MmCorpus('lsa_mini.mm')

## IF TfidfModel
tfidf = gensim.models.TfidfModel(corpus) # step 1 -- initialize a model
corpus = tfidf[corpus]
## FI TfidfModel
# Compute the LSA vectors
lsa=gensim.models.lsimodel.LsiModel(corpus, id2word=dictionary,
                                                     num_topics=n_topics)
# Print the n topics in our corpus:
#lsa.print_topics(n_topics)
f=open("topics_file.txt","w")
f.write("-------------------------------------------------\n")
for t in lsa.show_topics():
    f.write("%s\n" % str(t))
    
f.write("-------------------------------------------------\n")
f.close()
# create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
corpus_lsa = lsa[corpus]
# Stream sentences from file into a list of strings called "sentences"
sentences=corpus_streamer(input_file, strings=True)
n=0
for pertenence, sentence in zip(corpus_lsa, sentences):
    if n_docs <= 0:
    	#print "%s\t\t%s" % (pertenence, sentence.split("\t")[0])
    	p=[dict(pertenence)[x] if x in dict(pertenence) else 0.0 
                                            for x in range(n_topics)]
    	print("%s %s" % ("".join(sentence.split("\t")[0].split()),
                            "".join(str(p)[1:].strip("]").split(",")) ))
    else:
        if n<n_docs:
            pertenence=[dict(pertenence)[x] if x in dict(pertenence) else 0.0 
                                                    for x in range(n_topics)]
            print("%s\t\t%s" % (pertenence, sentence))
            n+=1
        else:
            break



# ============================== Homework ======================================
# Modify the program for doing this for a sample of the English Wikipedia.
# Compute LSA for 20 topics and print the fist 10 topics.
# Take care of avoiding loading and printing documents of a large corpus, so
# change the number of documents to print or sample the entire set randomly and
# print a subset.
# ==============================================================================
