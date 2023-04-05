import pandas as pd
def main ():
    df = pd.read_csv("Noticias_argentinas.csv", 
                     usecols=["fecha","titular","fuente","categoria"])

    title_list = df["titular"].tolist()
    counters = {}
    for title in title_list:
        title = title.lower()
        replaceables = [".", ",", "!", "?", "(", ")", ":", ";", "-"]
        for char in replaceables:
            title = title.replace(char, "")
        words = title.split(" ")
        for word in words:
            # if len(word) <= 3:
            #     continue
            if word in counters:
                counters[word] += 1
            else:
                counters[word] = 1

    print("Las 10 palabras mas usadas son:")
    sorted_counters = sorted(counters.items(), key=lambda x: x[1], reverse=True)[:10]
    output_string = "\n".join([f"{item[0]}: {item[1]}" for item in sorted_counters])
    print(output_string)
    
        
if __name__ == "__main__":
    main()