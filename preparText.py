#/* this scrip prepares the text in order to feed into lda model */

from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 're', 'edu', 'use','need','away','also','come','go',
                   'take','get','sure','want','example','include','first',
                   'second','next','may','make'])

import spacy
nlp = spacy.load('en', disable=['parser', 'ner'])

def prepareDoc(doc):
	
	docy = " ".join([wrd for wrd in simple_preprocess(doc) if wrd not in stop_words])
	return [token.lemma_ for token in nlp(docy)]
	
