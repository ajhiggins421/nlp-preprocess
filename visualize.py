import matplotlib.pyplot as plt
from wordcloud import WordCloud


def word_cloud(sentences):
    wordcloud = WordCloud(
                            background_color='white',
                            max_words=100,
                            max_font_size=50,
                            random_state=42
                            ).generate(str(sentences))
    plt.figure(figsize=(10,10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()