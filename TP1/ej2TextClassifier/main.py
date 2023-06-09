import pandas as pd
import unidecode
from probability_helper import *
import numpy as np

def remove_stop_words_from(list,use_unidecode):
    counters = {}
    # FIXME: regex
    symbols = [".", ",","¡", "!", "¿", "?", "(", ")", ":", ";", "-", "\"", "\'", "%", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    stop_words = symbols + ["de","la","el","en","y","que","los","un","del","una", "a", "de", "la", "en", "el", "y"]

    for idx, title in enumerate(list):
        title_words = title.split(" ")
        mod_words = []
        for word in title_words:
            if use_unidecode:
                word = unidecode.unidecode(word.lower())
            word = word.lower()
            if word not in stop_words:
                mod_word = "".join(w for w in word if w not in symbols)
                mod_words.append(mod_word)

        list[idx] = " ".join(mod_words)
        title_words = list[idx].split(" ")
    return list

def get_freq_table(df):
    freq_table = {}
    for cat in df["categoria"].unique():
        freq_table[cat] = {}

    for row_idx, title in enumerate(df["titular"]):
        words = title.split(" ")
        title_cat = df["categoria"][row_idx]
        for word in words:
            if not word in freq_table[title_cat]: # if word is not in the dict
                freq_table[title_cat][word] = 1
            else:
                freq_table[title_cat][word] += 1

    return freq_table


#P(ai/vj) * P(vj)
def get_conditional_probs(instance, class_probability, class_qty, freq_table, categories):
    probabilities = {}
    for cat in categories:
        probabilities[cat] = 0

    for name, class_first_prob in class_probability.items():
        total_freq = sum(freq_table[name].values())
        curr_prob = class_first_prob
        words = instance["titular"].split(" ")
        for word in words:
            numerator = freq_table[name][word] if word in freq_table[name] else 0
            curr_prob *= (numerator + 1) / (total_freq + class_qty) # TODO: only apply correction if numerator is 0


        probabilities[name] = curr_prob

    total = sum(probabilities.values())
    for key in probabilities:
        probabilities[key] /= total

    return probabilities

def partition_dataset(df, partition_percentage):
    # shuffle dataframe rows
    df = df.sample(frac=1).reset_index(drop=True)

    partition_size = int(np.floor(len(df) * partition_percentage))
    partitions = []

    bottom = 0
    up = partition_size
    while bottom < len(df):

        partitions.append(df[bottom:up].copy())
        bottom += partition_size
        up += partition_size
        if up > len(df):
            up = len(df)

    if  (up - bottom) != partition_size:
        partitions[-2] = pd.concat([partitions[-2], partitions[-1]], ignore_index=True)

        partitions = partitions[:-1]

    return partitions

def main():
    use_unidecode = True

    df = pd.read_csv("Noticias_argentinas.csv", usecols=["fecha","titular","fuente","categoria"])

    df["titular"] = remove_stop_words_from(df["titular"].tolist(), use_unidecode)
    df = df.loc[(df["categoria"] != "Noticias destacadas") & (df["categoria"] != "Destacadas")]

    partitions = partition_dataset(df, 0.2) # TODO: capaz recibirlo de config
    print(len(partitions))

    categories = sorted(df["categoria"].unique())

    with open("post_processing/classification.csv", "w") as classifier_file:
        for partition in partitions:
            test = partition
            train = pd.concat([df for df in partitions if df is not partition]) # TODO: check and verify if a copy is needed

            train.reset_index(drop=True, inplace=True)
            test.reset_index(drop=True, inplace=True)

            freq_table = get_freq_table(train)

            # TODO: apply Laplace if category is not present ?
            class_probability = get_class_probability(train, 'categoria')

            class_qty = len(class_probability)

            classifier_file.write("prediction," + ",".join(cat for cat in categories)+ ",real\n")

            for idx, instance in test.iterrows():
                probabilites = get_conditional_probs(instance, class_probability, class_qty, freq_table, categories)
                predicted_cat = max(probabilites, key=probabilites.get)
                real_cat = instance["categoria"]

                # TODO: check order
                classifier_file.write(predicted_cat + "," + ",".join(str(prob) for prob in probabilites.values())+
                                      "," + real_cat + "\n")


if __name__ == "__main__":
    main()