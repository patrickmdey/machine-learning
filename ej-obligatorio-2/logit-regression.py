from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


df = pd.read_csv("german_credit.csv")

# df.loc[:, float_analysis_cols] = StandardScaler().fit_transform(df[float_analysis_cols].values)

response = df["Creditability"]
df = df.drop("Creditability", axis=1)

#cols = df[:, df.columns != "Creditability"]

#cols = cols.columns

df = StandardScaler().fit_transform(df.values)

X_train, X_test, y_train, y_test = train_test_split(df, response, test_size=0.2)


#train_x = train.loc[:, df.columns != "Creditability"]
#test_x = test.loc[:, df.columns != "Creditability"]

model = LogisticRegression()
#X_train = train_x.values.reshape(len(train_x), 1)
#X_train = train_x
#y_train = train["Creditability"]
#X_test = test_x
#X_test = test[train_x].values.reshape(len(train_x), 1)
#y_test = test["Creditability"]
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

y_pred = model.predict(X_test)
confussion_matrix = confusion_matrix(y_test, y_pred)
print(confussion_matrix)
print(confusion_matrix)

# precision_score = precision_score(y_test, y_pred)
# print("Precision", precision_score)

coefficients = logistic_regression.coef_

coef_df = pd.DataFrame({'Variable': X.columns, 'Coeficiente': coefficients[0]})

coef_df['Coeficiente_Abs'] = coef_df['Coeficiente'].abs()
coef_df = coef_df.sort_values('Coeficiente_Abs', ascending=False)

print(coef_df)