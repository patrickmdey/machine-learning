import pandas as pd
import numpy as np
import sys
import json

#Vector a = "x": [1, -1, -1, 0],
#Vector b = "x": [2, 450, 3.5, 1],

def main():

    new_instance = [1, 1, 1, 1]

    with open(sys.argv[1], 'r') as config_file:
        config = json.load(config_file)
        if 'x' in config:
            if len(config['x']) == 4:
                new_instance = config['x']
                
            else:
                print('Wrong configuration')
                config_file.close()
                exit()
        
    
    new_instance[1] = 1 if new_instance[1] >= 500 else 0 if new_instance[1] >= 0 else -1
    new_instance[2] = 1 if new_instance[2] >= 3 else 0 if new_instance[2] >= 0 else -1

    df = pd.read_csv("./binary.csv")

    # Categorize columns
    df['gpa'] = pd.cut(x=df['gpa'], bins=[0, 3, np.inf],
                     labels=[0,1])
    df['gre'] = pd.cut(x=df['gre'], bins=[0, 500, np.inf],
                     labels=[0,1])
    
    rank_probs = []
    gre_probs = []
    gpa_probs = []
    admit_probs = []

    rank_values = sorted(set(df['rank']))
    gre_values = sorted(set(df['gre']))
    gpa_values = sorted(set(df['gpa']))
    admit_values = sorted(set(df['admit']))
    
    for value in rank_values:
        rank_probs.append(len(df[df["rank"] == value]) / len(df)) # P(rank)

    for idx, gre in enumerate(gre_values):
        gre_probs.append([])
        for rank in rank_values:
            rank_len = len(df[df["rank"] == rank])
            prob = len(df[(df["rank"] == rank) & (df["gre"] == gre)]) / rank_len
            gre_probs[idx].append(prob)
    
    for idx, gpa in enumerate(gpa_values):
        gpa_probs.append([])
        for rank in rank_values:
            rank_len = len(df[df["rank"] == rank])
            prob = len(df[(df["rank"] == rank) & (df["gpa"] == gpa)]) / rank_len
            gpa_probs[idx].append(prob)

    for a_idx, admit in enumerate(admit_values):
        admit_probs.append([])
        for gpa_idx, gpa in enumerate(gpa_values):
            admit_probs[a_idx].append([])
            for gre_idx, gre in enumerate(gre_values):
                admit_probs[a_idx][gpa_idx].append([])
                for rank in rank_values:
                    init_len = len(df[(df["rank"] == rank) & (df["gpa"] == gpa) & (df["gre"] == gre)])
                    conditional_len = len(df[(df['admit'] == admit) & (df["rank"] == rank) &
                                     (df["gpa"] == gpa) & (df["gre"] == gre)])
                    # TODO: capaz no hacerlo siempre
                    prob =  (conditional_len + 1) / (init_len + 2)
                    admit_probs[a_idx][gpa_idx][gre_idx].append(prob)
    
    ranks = [new_instance[0]] if new_instance[0] != -1 else rank_values
    gres = [new_instance[1]] if new_instance[1] != -1 else gre_values
    gpas = [new_instance[2]] if new_instance[2] != -1 else gpa_values

    numerator = 0
    denominator = 0
    admit = new_instance[3]
    for rank in ranks:
        rank_prob = rank_probs[rank - 1]
        for gre in gres:
            gre_prob = gre_probs[gre][rank - 1]
            for gpa in gpas:
                gpa_prob = gpa_probs[gpa][rank - 1]
                admit_prob = admit_probs[admit][gpa][gre][rank - 1] 

                denominator += gre_prob * gpa_prob * rank_prob
                numerator += gre_prob * gpa_prob * rank_prob * admit_prob
    
    print("Probability: " + str(numerator / denominator))    
    
if __name__ == "__main__":
    main()