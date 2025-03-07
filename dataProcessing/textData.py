import gensim.downloader as api

# GloVe 
glove = api.load("glove-wiki-gigaword-300")

# Word2Vec
word2vec = api.load("word2vec-google-news-300")

def get_text_embedding(word, model_type="word2vec"):
    model = word2vec if model_type == "word2vec" else glove
    return model[word] if word in model else [0] * 300  