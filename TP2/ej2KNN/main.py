import json
from KNN import *
from sklearn.preprocessing import MinMaxScaler
from post_processing import *

def calculate_average_word_count(df, rating):
    df_one_star = df[df['Star Rating'] == rating]
    word_count = df_one_star['wordcount'].sum()
    print("Average word count for 1 star reviews:", word_count / len(df_one_star))

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
            up = len(df)

    if (up - bottom) != partition_size:
        partitions[-2] = pd.concat([partitions[-2], partitions[-1]], ignore_index=True)

        partitions = partitions[:-1]

    return partitions

def classify_using_knn(df, k, test_size, weighted, normalize, remove_missing):
    df = df[['wordcount', 'titleSentiment', 'sentimentValue', 'Star Rating']]
    if remove_missing:
        df = df.dropna()
    
    else:
        df.loc[:, 'titleSentiment'] = df.loc[:, 'titleSentiment'].fillna(0)
        # TODO: mencionar por que usamos esto en fillna y capaz hacer varias pruebas CAPAZ ELIMINARLOS Y LISTO
        # TODO: cambiar a 0.5 quizas
        df.loc[(df['titleSentiment'] == 0) & (df['Star Rating'] >= 3), 'titleSentiment'] = 'positive'
        df.loc[(df['titleSentiment'] == 0) & (df['Star Rating'] < 3), 'titleSentiment'] = 'negative'

    df.loc[df['titleSentiment'] == 'positive', 'titleSentiment'] = 1
    df.loc[df['titleSentiment'] == 'negative', 'titleSentiment'] = 0

    if normalize:
        scaler = MinMaxScaler()
        df.loc[:, ['wordcount', 'titleSentiment', 'sentimentValue']] = scaler.fit_transform(df[['wordcount', 'titleSentiment', 'sentimentValue']].values)

    partitions = partition_dataset(df, test_size)
    knn = KNN(k)

    idx = 0
    for partition in partitions:
        test = partition
        train = pd.concat([df for df in partitions if df is not partition])
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        knn.fit(train[['wordcount', 'titleSentiment', 'sentimentValue']].to_numpy(),
                train[['Star Rating']].to_numpy())

        df_to_csv = pd.DataFrame(columns=['predicted', 'real'])
        new_row = {}
        for instance in test.to_numpy():
            attributes = instance[0:-1]
            y = instance[-1]

            prediction = knn.predict(attributes, weighted)

            new_row['predicted'] = prediction
            new_row['real'] = int(y)
            df_to_csv = pd.concat([df_to_csv, pd.DataFrame([new_row])], ignore_index=True)

        df_to_csv.to_csv("post_processing/knn/classification" + str(idx) + ".csv")
        idx += 1

def main():
    csv_file = ""
    k = 3
    test_size = 0.2
    weighted = False
    normalize = True
    remove_missing = False
    with open("config.json") as config_file: # open(sys.argv[1], 'r') as config_file:
        config = json.load(config_file)
        csv_file = config["file"]
        k = config["k"]
        test_size = config["test_size"]
        weighted = config["weighted"]
        normalize = config["normalize"]
        remove_missing = config["remove_missing"]
    config_file.close()

    df = pd.read_csv(csv_file, sep=';')

    # Ej a)
    calculate_average_word_count(df, 1)

    # Ej c)
    classify_using_knn(df, k, test_size, weighted, normalize, remove_missing)
    
if __name__ == "__main__":
    main()
