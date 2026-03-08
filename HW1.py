import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sklearn.metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(train.head())
print(test.head())

print(train.shape)
print(test.shape)

train.info()
test.info()

X = train.drop("SalePrice", axis=1)
y = train["SalePrice"]

# hist = sns.histplot(y)
# plt.show()

mean_price = np.mean(y)
min_price = np.min(y)
max_price = np.max(y)

print(mean_price)
print(min_price)
print(max_price)


corr = train.corr(numeric_only=True)

print(corr["SalePrice"].sort_values(ascending=False))
print("---")
print(train.isnull().sum().sort_values(ascending=False))

train = train.drop(["PoolQC", "MiscFeature"], axis=1)

X = train.drop("SalePrice", axis=1)
y = train["SalePrice"]

print(train.isnull().sum().sort_values(ascending=False))

train = train.drop(["Alley", "Fence"], axis=1)

X = train.drop("SalePrice", axis=1)
y = train["SalePrice"]

print(train.isnull().sum().sort_values(ascending=False))

train["MasVnrType"] = train["MasVnrType"].fillna("None")
train["FireplaceQu"] = train["FireplaceQu"].fillna("None")
median = train["LotFrontage"].median()
train["LotFrontage"] = train["LotFrontage"].fillna(median)

print(train.isnull().sum().sort_values(ascending=False))

train["GarageFinish"] = train["GarageFinish"].fillna("None")
train["GarageCond"] = train["GarageCond"].fillna("None")
train["GarageType"] = train["GarageType"].fillna("None")
median = train["GarageYrBlt"].median()
train["GarageYrBlt"] = train["GarageYrBlt"].fillna(median)
train["GarageQual"] = train["GarageQual"].fillna("None")

print(train.isnull().sum().sort_values(ascending=False))

train["BsmtFinType2"] = train["BsmtFinType2"].fillna("None")
train["BsmtExposure"] = train["BsmtExposure"].fillna("None")
train["BsmtQual"] = train["BsmtQual"].fillna("None")
train["BsmtCond"] = train["BsmtCond"].fillna("None")
train["BsmtFinType1"] = train["BsmtFinType1"].fillna("None")

print(train.isnull().sum().sort_values(ascending=False))

median = train["MasVnrArea"].median()
train["MasVnrArea"] = train["MasVnrArea"].fillna(median)
mode = train["Electrical"].mode()[0]
train["Electrical"] = train["Electrical"].fillna(mode)

print(train.isnull().sum().sort_values(ascending=False))

train = pd.get_dummies(train)
print(train.shape)

X = train.drop("SalePrice", axis=1)
y = train["SalePrice"]
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,y, random_state=42, test_size=0.2
)
print(X_test.shape)
print(y_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE: ", mae)
print("MSE: ", mse)
print("R2: ", r2)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred= ridge.predict(X_test)
mae = mean_absolute_error(y_test, ridge_pred)
mse = mean_squared_error(y_test, ridge_pred)
r2 = r2_score(y_test, ridge_pred)
print("Ridge MAE: ", mae)
print("Ridge MSE: ", mse)
print("Ridge R2: ", r2)

lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
mae = mean_absolute_error(y_test, lasso_pred)
mse = mean_squared_error(y_test, lasso_pred)
r2 = r2_score(y_test, lasso_pred)
print("Lasso MAE: ", mae)
print("Lasso MSE: ", mse)
print("Lasso R2: ", r2)