import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_heatmap(df):
    plt.clf()
    sns.heatmap(df, annot=True, fmt='.0f')
    plt.tight_layout()
    plt.savefig("post_processing/heatmap.png")

def get_graphs():
    df = pd.read_csv("post_processing/confusion_matrix.csv")
    get_heatmap(df)
