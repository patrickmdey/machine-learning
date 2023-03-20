from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def corr_analysis(df):
    corr = df.corr()
    heatmap = sns.heatmap(corr)
    heatmap.figure.savefig("./out/corr_heatmap.png")
    # print(df.corr())

def linear_model(X, y, multiple=False, x_label="null", y_label="null"):
    plt.clf()
    reg = LinearRegression().fit(X, y) # te calcula la recta de regresion
    print("R^2 for " + ("multiple" if multiple else "linear"  + x_label) + ": " + str(reg.score(X, y))) # R^2
    print("Bi for " + ("multiple" if multiple else "linear"  + x_label) + ": " + str(reg.coef_)) # coeficiente de la recta (b_i)
    if not multiple:
        plt.scatter(X, y)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.plot(X, reg.predict(X), color='red', linewidth=2)
        plt.savefig("./out/"+x_label + "_vs_" + y_label + "_linear.png")

# TODO: faltaria hacerlo para cada variable por separado
def test_linear_model(df, test_size=0.2):
    train, test = train_test_split(df, test_size=0.2)
    X_train = train.drop("Sales", axis=1)
    y_train = train["Sales"]
    X_test = test.drop("Sales", axis=1)
    y_test = test["Sales"]
    reg = LinearRegression().fit(X_train, y_train) # te calcula la recta de regresion
    y_pred = reg.predict(X_test)
    print("MSE for multiple linear model: " + str(mean_squared_error(y_test, y_pred)))
