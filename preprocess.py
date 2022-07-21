import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def regex(text):
    text = re.sub(r'\[[0-9]*\]', ' ', text)  # [0-9]* --> Matches zero or more repetitions of any digit from 0 to 9
    text = text.lower()  # everything to lowercase
    text = re.sub(r'\W^.?!', ' ', text)  # \W --> Matches any character which is not a word character except (.?!)
    text = re.sub(r'\d', ' ', text)  # \d --> Matches any decimal digit
    text = re.sub(r'\s+', ' ', text)  # \s --> Matches any characters that are considered whitespace (Ex: [\t\n\r\f\v].)
    return text


def remove_stopwords(sentences):
    for i in range(len(sentences)):
        words = nltk.word_tokenize(sentences[i])
        words = [word for word in words if word not in stopwords.words("english")]
        sentences[i] = " ".join(words)
    return sentences


def remove_punctuation(sentences):
    for i in range(len(sentences)):
        words = nltk.word_tokenize(sentences[i])
        words = [word for word in words if word.isalnum()]
        sentences[i] = " ".join(words)
    return sentences


def stem_sentences(sentences):
    for i in range(len(sentences)):
        words = nltk.word_tokenize(sentences[i])
        words = [stemmer.stem(word) for word in words]
        sentences[i] = " ".join(words)
    return sentences


def lem_sentences(sentences):
    for i in range(len(sentences)):
        words = nltk.word_tokenize(sentences[i])
        words = [lemmatizer.lemmatize(word) for word in words]
        sentences[i] = " ".join(words)
    return sentences


def pos_tag(text):
    all_words = nltk.word_tokenize(text)
    tagged_words = nltk.pos_tag(all_words)
    word_tags = []
    for tw in tagged_words:
        word_tags.append(tw[0] + "_" + tw[1])
    tagged_paragraph = ' '.join(word_tags)

    return tagged_paragraph


def preprocess_text(text, stem=False, lem=False):
    text = regex(text)
    text = text.lower()
    sentences = nltk.sent_tokenize(text)
    sentences = remove_stopwords(sentences)
    sentences = remove_punctuation(sentences)
    if stem:
        sentences = stem_sentences(sentences)
    if lem:
        sentences = lem_sentences(sentences)
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    return sentences
