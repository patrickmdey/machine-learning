from matplotlib import pyplot as plt
import pandas as pd
import unidecode
from probability_helper import *
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
from graphs import get_graphs


def remove_stop_words_from(list, has_to_replace, print_most_used_words, use_unidecode):
    counters = {}
    # FIXME: regex
    symbols = [".", ",","¡", "!", "¿", "?", "(", ")", ":", ";", "-", "\"", "\'", "%", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    stop_words = symbols + ["de","la","el","en","y","que","los","un","del","una", "a", "de", "la", "en", "el", "y"]

    for idx, title in enumerate(list):
        title_words = title.split(" ")
        mod_words = []

        if has_to_replace:
            for word in title_words:
                if use_unidecode:
                    word = unidecode.unidecode(word.lower())
                word = word.lower()
                if word not in stop_words:
                    mod_word = "".join(w for w in word if w not in symbols)
                    mod_words.append(mod_word)
            list[idx] = " ".join(mod_words)
            title_words = list[idx].split(" ")
        
        if print_most_used_words:
            for word in title_words:
                if word in counters:
                    counters[word] += 1
                else:
                    counters[word] = 1
    
    # if has_to_replace and not print_most_used_words:
    #     print("\n".join(list[:30]))

    if print_most_used_words:
        print("Las 10 palabras mas usadas son:")
        sorted_counters = sorted(counters.items(), key=lambda x: x[1], reverse=True)[:10]
        output_string = "\n".join([f"{item[0]}: {item[1]}" for item in sorted_counters])
        print(output_string)

    return list

def get_freq_table(df):
    freq_table = {}
    for cat in df["categoria"].unique():
        freq_table[cat] = {}
    
    for row_idx, title in enumerate(df["titular"]):
        words = title.split(" ")
        title_cat = df["categoria"][row_idx]
        for word in words:
            if not word in freq_table[title_cat]: # Si no esta en el diccionario
                freq_table[title_cat][word] = 1
            else:
                freq_table[title_cat][word] += 1
    
    return freq_table

def get_conditional_probs(instance, class_probability, class_qty, freq_table):
    probabilities = {}
    
    for name, class_first_prob in class_probability.items():
        total_freq = sum(freq_table[name].values())
        curr_prob = class_first_prob
        words = instance["titular"].split(" ")
        for word in words:
            if word in freq_table[name]:
                curr_prob *= freq_table[name][word] / total_freq
            else:
                curr_prob *= 1/(total_freq + class_qty)

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
    
    # TODO: pass to function
    bottom = 0
    up = partition_size
    while bottom < len(df):
        partitions.append(df[bottom:up].copy())
        bottom += partition_size
        up += partition_size
        if up > len(df):
            up = partition_size
    return partitions


def main():
    print_most_used_word = False
    has_to_replace = True
    use_unidecode = True

    df = pd.read_csv("Noticias_argentinas.csv", 
                     usecols=["fecha","titular","fuente","categoria"])
    
    df["titular"] = remove_stop_words_from(df["titular"].tolist(), has_to_replace, print_most_used_word, use_unidecode)
    df = df.loc[(df["categoria"] != "Noticias destacadas")]

    partitions = partition_dataset(df, 0.2)

    categories = df["categoria"].unique()

    result_confusion_matrix = pd.DataFrame({real_cat: {pred_cat: 0 for pred_cat in categories} for real_cat in categories})

    step = 1 / len(partitions)
    threshold = step


    metrics_per_class = {real_cat: {"tp": 0, "tn":0, "fp": 0, "fn": 0} for real_cat in categories}

    for partition in partitions:
        test = partition
        train = pd.concat([df for df in partitions if df is not partition]) # TODO: check and verify if a copy is needed
        
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)

        freq_table = get_freq_table(train)

        # TODO: apply Laplace if category is not present ?
        class_probability = get_class_probability(train, 'categoria')

        class_qty = len(class_probability)

        confusion_matrix = {real_cat: {pred_cat: 0 for pred_cat in categories} for real_cat in categories}

        total = 0    
        true_positive = 0

        for idx, instance in test.iterrows():
            total += 1
            probabilites = get_conditional_probs(instance, class_probability, class_qty, freq_table)
            predicted_cat = max(probabilites, key=probabilites.get)

            real_cat = instance["categoria"]

            if predicted_cat == real_cat: 
                metrics_per_class[predicted_cat]["tp"] += 1 #it really is a hit
                other_cats = [cat for cat in categories if cat != predicted_cat]
                for cat in other_cats:
                    metrics_per_class[cat]["tn"] += 1 #for all the other cat wont be a hit for sure
            else:
                metrics_per_class[predicted_cat]["fp"] += 1 #for the other cat shouldn't be a hit but it is
                metrics_per_class[real_cat]["fn"] += 1 #for the cat its a hit but it shouldn't be

            confusion_matrix[test["categoria"][idx]][predicted_cat] += 1
                

        for cat in categories:
            true_positive += confusion_matrix[cat][cat]
        
        print("Precision: ", true_positive / total)  

        result_confusion_matrix += pd.DataFrame(confusion_matrix)

        threshold += step
    pd.DataFrame(metrics_per_class).to_csv("post_processing/metrics_per_class_.csv")

    result_confusion_matrix.to_csv("post_processing/confusion_matrix.csv")
    # get_graphs()     
        
if __name__ == "__main__":
    main()