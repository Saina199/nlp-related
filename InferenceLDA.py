""" inference using LDA model """

from gensim.models.ldamodel import LdaModel
import preparText as px
from gensim.corpora import Dictionary


def inference(doc,model_loc):

	lda = LdaModel.load(model_loc+'/ldamodel.model')
	unseen_doc = px.prepareDoc(doc)
	dictionary = Dictionary.load(model_loc+'/dictionary.dic')

	unseen_corpus = dictionary.doc2bow(unseen_doc)
	topics = lda[unseen_corpus]

	return topics,lda



