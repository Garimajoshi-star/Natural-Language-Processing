
#pre-required libraries
#pip install nltk
#pip install pattern
#pip install genism
#pip install spaCy
#pip install TextBlob
#import nltk
#nltk.download()

# steps to implement tokenisation of words and sentences using the NLTK module:

import nltk
nltk.download()
from nltk.tokenize import sent_tokenize, word_tokenize

data = "Delhi is the capital of India. It covers an area of 1,484 square kilometres."

phrases = sent_tokenize(data)
words = word_tokenize(data)
 
print("\n\n", phrases)
print("\n\n", words)

# steps to predict Parts of Speech (POS) using NLTK libraries:

import nltk
from nltk.tokenize import PunktSentenceTokenizer

data = "Delhi is the capital of India. It covers an area of 1,484 square kilometres."
sentences = nltk.sent_tokenize(data)   
for sent in sentences:
    print(nltk.pos_tag(nltk.word_tokenize(sent)))

# steps to implement lemmatisation using NLTK libraries, PorterStemmer and WordNetLemmatizer modules:
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

data= "Delhi is the capital of India. It covers an area of 1,484 square kilometres."
punctuations=" ; , . ! ? : "

sentence_words = nltk.word_tokenize(data)
for word in sentence_words:
    if word in punctuations:
        sentence_words.remove(word)

sentence_words
print("{0:40}{1:40}".format("Word","Lemma"))
for word in sentence_words:
    print ("{0:40}{1:40}".format(word,wordnet_lemmatizer.lemmatize(word, pos="v")))

# steps to identify stop words using NLTK and Corpus modules:

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

data = "Delhi is the capital of India. It covers an area of 1,484 square kilometres."
stopWords = set(stopwords.words('english'))
words = word_tokenize(data)
wordsFiltered = []
for w in words:
    if w not in stopWords:
        wordsFiltered.append(w)
print("\n\n", wordsFiltered, "\n\n")
print(len(stopWords))
print("\n\n", stopWords)


# steps to create dependency parsing and chunking using the NLTK library:
import nltk
data = "Delhi is the capital of India." 
tokens = nltk.word_tokenize(data)
print(tokens)
tag = nltk.pos_tag(tokens)
print(tag)
grammar = "NP: {<DT>?<JJ>*<NN>}"
cp  =nltk.RegexpParser(grammar)
parse_tree= cp.parse(tag)
print(parse_tree)
parse_tree.draw()   

# steps to implement the Name Entity Recognition (NER) process using NLTK libraries and spaCy module:
#pip install spacy
#python -m spacy download en

import nltk
import spacy
import en_core_web_sm
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags

spacy_nlp = spacy.load('en')
data = "Delhi is the capital of India. It covers an area of 1,484 square kilometres.”
document = spacy_nlp(data)
print('Input data: %s' % (data))
for token in document.ents:
    print('Type: %s, Value: %s' % (token.label_, token))






















