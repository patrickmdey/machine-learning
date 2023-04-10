
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
    #we get all the rows where the class has the value category (nacionalidad = I)
    cat_df = df[df[class_column] == category]
    value_df_len = len(cat_df[cat_df[column_name] == value]) #amount of rows where the value is the one we are looking for

    # Correccion de Laplace
    return (value_df_len + 1) / (len(cat_df) + class_qty)