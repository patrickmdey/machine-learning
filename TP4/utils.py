from matplotlib import pyplot as plt
import seaborn as sns

def replace_missing_values(df, column, value, method="mean"):
    mean_df = df.copy()
    median_df = df.copy()
    df.drop(df[df[column] >= value].index, inplace=True) # drops the rows that have Alcohol >= value
    df_mean = df[column].mean()
    df_median = df[column].median()

    # print()
    if (method == "mean"):
        # print("Replacing for mean...")
        mean_df[column].replace(value, df_mean, inplace=True)
        print("New " + column +" mean:", mean_df[column].mean())
        print("New " + column +" median:", mean_df[column].median())
        return mean_df
    elif (method == "median"):
        # print("Replacing for median...")
        median_df[column].replace(value, df_median, inplace=True)
        print("New " + column +" mean:", median_df[column].mean())
        print("New " + column +" median:", median_df[column].median())
        return median_df
    
def cov_analysis(df):
    print(df.groupby("Sexo").cov())

def bar_column(df, column_name):
    plt.clf()
    y = df[column_name].value_counts()
    plt.bar(y.index, y)
    plt.xlabel(column_name)
    plt.ylabel("Cantidad")
    plt.savefig("./out/"+column_name + "_bar.png")

def kurtosis_column(df, column):
    print("Kurtosis:", df[column].kurtosis())

def skewness_column(df, column):
    print("Skewness:", df[column].skew())


def boxplot_column(df, column_name, method=""):
    plt.clf()
    boxplot = df.boxplot(column=column_name, grid=False)
    boxplot.set_ylabel("Cantidad")
    boxplot.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    str = column_name + "_with_" + method if method != "" else column_name
    boxplot.figure.savefig("./out/data_analysis/"+str + "_boxplot.png")

def boxplot_grouped_column(df, column_name, group_by):
    plt.clf()
    boxplot = sns.boxplot(x=group_by, y=column_name, data=df)
    boxplot.set_ylabel(column_name)
    
    # boxplot.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    boxplot.figure.savefig("./out/"+column_name + "_by_" + group_by + "_boxplot.png")


# Define a function to categorize the values based on the conditions
def categorize(val):
    if val <= 1100:
        return 'CAT1'
    elif val <= 1700:
        return 'CAT2'
    else:
        return 'CAT3'

def categorize_calories_col(df):
    df_copy = df.copy()

    df_copy['Categorías'] = df['Calorías'].apply(categorize)

    # df_copy = df.assign(Category=lambda x: "CATE1" if x <=["Calorías"] 1100 else ("CATE2" if x["Calorías"] <= 1700 else "CATE3"))

    return df_copy


def scatter_category(df, x, y):
    plt.clf()

    scatter = df.plot.scatter(x, y)

    scatter.set_xlabel(x)
    scatter.set_ylabel("Calorías")
    scatter.figure.savefig("./out/"+x + "_vs_" + y + "_scatter.png")