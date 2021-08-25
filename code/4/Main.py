import random
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# sample_number = 20
group_number = 4
if group_number == 3:
    list_of_words = ["orange", "lemon", "banana", "melon", "watermelon",
                     "grape", "tomato", "potato", "carrot", "cucumber",
                     "wifi", "technology", "laptop", "computer", "tablet",
                     "mobile", "phone", "internet", "pc", "modem",
                     "football", "volleyball", "basketball", "hockey", "karate",
                     "curling", "wrestling", "badminton", "tennis", "waterpolo"]
elif group_number == 4:
    list_of_words = ["orange", "lemon", "banana", "melon", "watermelon",
                     "grape", "tomato", "potato", "carrot", "cucumber",
                     "wifi", "technology", "laptop", "computer", "tablet",
                     "mobile", "phone", "internet", "pc", "modem",
                     "football", "volleyball", "basketball", "hockey", "karate",
                     "curling", "wrestling", "badminton", "tennis", "waterpolo",
                     "iran", "usa", "canada", "uk", "spain",
                     "france", "italy", "brazil", "iraq", "australia"]


def get_line_number():
    file = open("glove.txt", encoding="utf8")
    line_num = 0
    for line in file:
        if line != "\n":
            line_num += 1
    file.close()
    return line_num


def get_data(words_index):
    file = open("glove.txt", encoding="utf8")
    words = []
    data = []
    idx = 0
    for line in file:
        if idx in words_index:
            data_str = line.replace("\n", "").split(" ")
            words.append(data_str[0])
            data_num = []
            for str in data_str[1:]:
                data_num.append(float(str))
            data.append(data_num)
        idx += 1
    return words, data


def get_index_by_word(word_str):
    file = open("glove.txt", encoding="utf8")
    index = 0
    for line in file:
        word = line.replace("\n", "").split(" ")[0]
        if word == word_str:
            return index
        index += 1


# line_num = get_line_number()
# words_index = random.sample(range(0, line_num), sample_number)

words_index = []
for item in list_of_words:
    words_index.append(get_index_by_word(item))

words, data = get_data(words_index)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data)
principalDf = pd.DataFrame(data=principalComponents
                           , columns=['pc 1', 'pc 2'])

principalDf.plot(x='pc 1', y='pc 2', kind='scatter')
for idx in range(len(words)):
    plt.text(principalComponents[idx][0], principalComponents[idx][1], words[idx])
plt.show()
