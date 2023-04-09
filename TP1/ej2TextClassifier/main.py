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
    use_unidecode = True

    df = pd.read_csv("Noticias_argentinas.csv", usecols=["fecha","titular","fuente","categoria"])
    
    df["titular"] = remove_stop_words_from(df["titular"].tolist(), use_unidecode)
    df = df.loc[(df["categoria"] != "Noticias destacadas")]

    partitions = partition_dataset(df, 0.2)

    categories = sorted(df["categoria"].unique())

    result_confusion_matrix = pd.DataFrame({real_cat: {pred_cat: 0 for pred_cat in sorted(categories)} for real_cat in categories})

    step = 1 / len(partitions)
    threshold = step

    # with open("out/roc.txt", "w") as roc_file:

    metrics_per_class = {real_cat: {"tp": 0, "tn":0, "fp": 0, "fn": 0} for real_cat in categories} #TODO: esto debería ir en el for ?

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
            #TODO: Aca otro for que recorre todo el umbral sería ?
            # roc_file.write(f"threshold: {threshold}\n", threshold)

            #confusion_matrix = {real_cat: {pred_cat: 0 for pred_cat in categories} for real_cat in categories}

            #total = 0    
            #true_positive = 0

            classifier_file.write("prediction," + ",".join(cat for cat in categories)+ ",real\n")

            for idx, instance in test.iterrows():
                #total += 1
                probabilites = get_conditional_probs(instance, class_probability, class_qty, freq_table, categories)
                predicted_cat = max(probabilites, key=probabilites.get)
                real_cat = instance["categoria"]
                
                # TODO: check order
                classifier_file.write(predicted_cat + "," + ",".join(str(prob) for prob in probabilites.values())+ 
                                      "," + real_cat + "\n")
            
                # pred   cat1 cat2 cat3 real
                # ca1    0.1   0.5  0.4 cat2


                # if predicted_cat == real_cat: 
                #     metrics_per_class[predicted_cat]["tp"] += 1 #it really is a hit
                #     other_cats = [cat for cat in categories if cat != predicted_cat]
                #     for cat in other_cats:
                #         metrics_per_class[cat]["tn"] += 1 #for all the other cat, it wont be a hit for sure
                # else:
                #     metrics_per_class[predicted_cat]["fp"] += 1 #for the other cat shouldn't be a hit but it is
                #     metrics_per_class[real_cat]["fn"] += 1 #for the cat its a hit but it shouldn't be

                # confusion_matrix[test["categoria"][idx]][predicted_cat] += 1
                    
            #for cat in categories:
            #    true_positive += confusion_matrix[cat][cat]
            
            # print("Precision: ", true_positive / total)  

            #result_confusion_matrix += pd.DataFrame(confusion_matrix)

            threshold += step
    pd.DataFrame(metrics_per_class).to_csv("post_processing/metrics_per_class_.csv")

    #result_confusion_matrix.to_csv("post_processing/confusion_matrix.csv")
        
if __name__ == "__main__":
    main()