
def get_class_probability(df, column_name):
    class_probability = {}

    total = 0
    for value in df[column_name]:
        if value not in class_probability:
            class_probability[value] = 0

        class_probability[value] += 1
        total += 1
    
        
    for key in class_probability:
        class_probability[key] /= total 
    
    return class_probability

def get_value_conditional_prob(df, column_name, value, category, class_column, class_qty):
    # Nos quedamos con todos los registros donde la clase tiene el valor category (nacionalidad = I ) 
    cat_df = df[df[class_column] == category]
    value_df_len = len(cat_df[cat_df[column_name] == value])

    # Correccion de Laplace
    return (value_df_len + 1) / (len(cat_df) + class_qty)

