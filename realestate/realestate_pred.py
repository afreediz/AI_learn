import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin

def load_housing_data():
    csv_path = os.path.join("../datasets", "housing", "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()

# need to split training and test set based on income_cat as its most influnetial on dataset
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set_instance in (strat_train_set , strat_test_set):
    set_instance.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"] / 100, label="population", c="median_house_value",cmap=plt.get_cmap("jet"), colorbar=True)

# splitting the training set into features and labels to train model
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# seperating numerical and categorical features
housing_num = housing.drop("ocean_proximity", axis=1)

# sample custom transformer
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        
num_pipeline = Pipeline([
    # fills the null values with median
        ('imputer', SimpleImputer(strategy="median")),
    # adds new attributes
        ('attribs_adder', CombinedAttributesAdder()),
    # scales the data to have similar range
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

num_attrs = list(housing_num)
cat_attrs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attrs),
    # makes the categorical data to numerical repn eg : [1,0,0,0],[0,1,0,0] ( 4 categories ) ...as sparse matrix
    ("cat", OneHotEncoder(), cat_attrs)
])

housing_prepared = full_pipeline.fit_transform(housing)

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)


housing_predictions = lin_reg.predict(housing_prepared)
# print(housing.head(10))

plt.figure(figsize=(10, 6))
plt.scatter(housing_labels, housing_predictions, alpha=0.4)
plt.xlabel("Actual prices")
plt.title("Realestate Prediction")
plt.ylabel("Predicted prices")
plt.legend()
plt.show()