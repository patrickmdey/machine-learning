import json
import sys
import pandas as pd
from probability_helper import *
import numpy as np

def main():
    with open(sys.argv[1], 'r') as config_file:
        config = json.load(config_file)
        category = "Nacionalidad"
        if 'class' in config:
            category = config['class']

        x = 11111

        if not 'file' in config:
            print('Must specify dataset in config file')
            config_file.close()
            exit()

        new_instance = [1, 1, 1, 1, 1]
        if 'x' in config:
            if len(config['x']) == 5 and config['x'].isnumeric():
                new_instance = [int(x) for x in config['x']]
                
            else:
                print('Wrong configuration')
                config_file.close()
                exit()

        print("Input x: " + str(new_instance))

        df = pd.read_csv(config['file'])

        class_probability = get_class_probability(df, category)
        print("P(vj)" + str(class_probability))
        
        value_conditional_probability = {}

        for key in class_probability:
            value_conditional_probability[key] = [] # TODO: capaz np array con el len de new_instance

        # Obtenemos P(ai/vj). 
        class_qty = len(value_conditional_probability)
        for key in class_probability:
            for (idx, col) in enumerate(df.loc[:, df.columns != category]): #all columns except class_column
                value_conditional_probability[key].append(get_value_conditional_prob(df, col, new_instance[idx], key, category, class_qty))
        
        # TODO: preguntar si hay que hacer Laplace siempre o no
        print("P(ai/vj)" + str(value_conditional_probability))

        results = {}
        # Obtenemos vnb para cada clase
        for key in class_probability:
            results[key] = class_probability[key] * np.prod(value_conditional_probability[key])

        print("vj/ai = " + str(results))
if __name__ == "__main__":
    main()