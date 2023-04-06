import pandas as pd

def remove_stop_words_from(list, has_to_replace, print_most_used_words):
    counters = {}
    symbols = [".", ",", "!", "?", "(", ")", ":", ";", "-"]
    stop_words = symbols + ["de","la","el","en","y","que","qué","los","un","del","una", "a", "de", "la", "en", "el", "y"]

    for idx, title in enumerate(list):
        title_words = title.split(" ")
        mod_words = []

        if has_to_replace:
            for word in title_words:
                # word = unidecode.unidecode(word.lower())
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
    
    if has_to_replace and not print_most_used_words:
        print("\n".join(list[:30]))

    if print_most_used_words:
        print("Las 10 palabras mas usadas son:")
        sorted_counters = sorted(counters.items(), key=lambda x: x[1], reverse=True)[:10]
        output_string = "\n".join([f"{item[0]}: {item[1]}" for item in sorted_counters])
        print(output_string)

    return list

def main():
    print_most_used_word = False
    has_to_replace = True

    df = pd.read_csv("Noticias_argentinas.csv", 
                     usecols=["fecha","titular","fuente","categoria"])
    
    df["titular"] = remove_stop_words_from(df["titular"].tolist(), has_to_replace, print_most_used_word)
        
if __name__ == "__main__":
    main()