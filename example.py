import requests
import bs4 as bs
import preprocess
import visualize
import word2vec

url = 'https://en.wikipedia.org/wiki/Elon_Musk'
source = requests.get(url).content
soup = bs.BeautifulSoup(source, "html.parser")

text = ""
for paragraph in soup.find_all('p'):
    text += paragraph.text

sentences = preprocess.preprocess_text(text)
word2vec.create(sentences)

print(word2vec.get_common(10))
print(word2vec.get_similar("climate", 5))

visualize.word_cloud(sentences)