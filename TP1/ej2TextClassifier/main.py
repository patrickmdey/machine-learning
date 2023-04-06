import pandas as pd
import unidecode
from probability_helper import *
from sklearn.model_selection import train_test_split


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

def get_most_probable_class(df, instance, class_probability, class_qty, freq_table):
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
    
    return max(probabilities, key=probabilities.get)
        

def main():
    print_most_used_word = False
    has_to_replace = True
    use_unidecode = True

    df = pd.read_csv("Noticias_argentinas.csv", 
                     usecols=["fecha","titular","fuente","categoria"])
    
    df["titular"] = remove_stop_words_from(df["titular"].tolist(), has_to_replace, print_most_used_word, use_unidecode)
    df = df.loc[(df["categoria"] != "Noticias destacadas")]
    
    

    #shuffle dataframe rows
    # df = df.sample(frac=1).reset_index(drop=True)
    train, test = train_test_split(df, test_size=0.2)

    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    freq_table = get_freq_table(train)

    # TODO: apply Laplace if category is not present ?
    class_probability = get_class_probability(train, 'categoria')

    class_qty = len(class_probability)


    total = 0
    positive = 0
    for idx, instance in test.iterrows():
        total += 1
        # TODO: take into account the frequency of the words that appears in instance
        most_prob = get_most_probable_class(train, instance, class_probability, class_qty, freq_table)
        # print(most_prob + " -> " + instance["categoria"])
        if most_prob == instance["categoria"]:
            positive +=1
        
    print("Precision: ", positive/total)    
        
        
if __name__ == "__main__":
    main()