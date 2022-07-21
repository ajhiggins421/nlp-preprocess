from gensim.models import Word2Vec


def create(sentences):
    model = Word2Vec(sentences, min_count=1)
    model.save("word2vec.model")


def get_common(amount=10):
    model = Word2Vec.load("word2vec.model")
    common = model.wv.index_to_key
    return common[:amount]


def get_similar(word, amount=10):
    model = Word2Vec.load("word2vec.model")
    similar = model.wv.most_similar(word)
    return similar[:amount]


def get_vector(word):
    model = Word2Vec.load("word2vec.model")
    return model.wv.get_vector(word)
