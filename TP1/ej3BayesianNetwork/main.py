import pandas as pd
import numpy as np
import sys
import json

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
        
    
    new_instance[1] = 1 if new_instance[1] >= 500 else 0 
    new_instance[2] = 1 if new_instance[2] >= 3 else 0 
    print(new_instance)

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
        rank_probs.append(len(df[df["rank"] == value]) / len(df))

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
                    prob = len(df[(df['admit'] == admit) & (df["rank"] == rank) &
                                     (df["gpa"] == gpa) & (df["gre"] == gre)]) / init_len
                    admit_probs[a_idx][gpa_idx][gre_idx].append(prob)

    
    
    denominator = rank_probs[new_instance[0] - 1]
    numerator = rank_probs[new_instance[0] - 1]

    if new_instance[1] != -1 and new_instance[2] != -1:
        # TODO: capaz armar un diccionario para no hacer el -1
        denominator *= gre_probs[new_instance[1]][new_instance[0] - 1] * gpa_probs[new_instance[2]][new_instance[0] - 1]

        numerator *= admit_probs[new_instance[3]][new_instance[2]][new_instance[1]][new_instance[0] - 1] * gre_probs[new_instance[1]][new_instance[0] - 1] * gpa_probs[new_instance[2]][new_instance[0] - 1]
    
    elif new_instance[1] != -1:
        denominator *= gre_probs[1][new_instance[new_instance[0] - 1]]
        numerator *= gre_probs[1][new_instance[new_instance[0] - 1]]
        aux_den = 0
        aux_num = 0
        for idx, value in enumerate(gpa_probs):
            aux_num += value[new_instance[0] - 1] * admit_probs[new_instance[3]][idx][new_instance[1]][new_instance[0] - 1]
            aux_den += value
        denominator *= aux_den
        numerator *= aux_num
    
    elif new_instance[2] != -1:
        denominator *= gpa_probs[2][new_instance[0] - 1]
        numerator *= gpa_probs[2][new_instance[0] - 1]
        aux_den = 0
        aux_num = 0
        for idx, value in enumerate(gre_probs):
            aux_num += value[new_instance[0] - 1] * admit_probs[new_instance[3]][new_instance[2]][idx][new_instance[0] - 1]
            aux_den += value[new_instance[0] - 1]
        denominator *= aux_den
    
    else:
        # r = 1 AND gpa = 0 AND gre = 0 ++++ r = 1 AND gpa = 0 AND gre = 1 ++++++
        aux_den = 0
        aux_num = 0
        # print(gpa_probs)
        for gpa_index, gpa_p in enumerate(gpa_probs):
            for gre_index, gre_p in enumerate(gre_probs):
                # aux_num = gpa_p * gre_p * admit_probs[new_instance[3]][gpa_index][gre_index][new_instance[0] - 1]
                aux_num += gpa_p[new_instance[0] - 1] * gre_p[new_instance[0] - 1] * admit_probs[new_instance[3]][gpa_index][gre_index][new_instance[0] - 1]

                aux_den += gpa_p[new_instance[0] - 1] * gre_p[new_instance[0] - 1]
        denominator *= aux_den
        numerator *= aux_num

    print(numerator/denominator)

    # TODO: falta LAPLACE, cual seria la cantidad de clases?

    
    
if __name__ == "__main__":
    main()