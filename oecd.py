import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import time

oecd_bli = pd.read_csv("./datasets/oecd_bli_2015.csv", thousands=",")
gdp_per_capita = pd.read_csv("./datasets/gdp_per_capita.csv", thousands=",", delimiter="\t", encoding="latin1", na_values="n/a")

def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)

country_stats.to_csv("country_stats.csv", index=False)

X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

country_stats.plot(kind="scatter", x="GDP per capita", y="Life satisfaction")

plt.show()

model = sklearn.linear_model.LinearRegression()

model.fit(X, y)


# Generating predictions
X_pred = np.linspace(np.min(X), np.max(X), 100).reshape(-1, 1)  # Generate points for predictions
y_pred = model.predict(X_pred)

# Plotting the model line
plt.plot(X_pred, y_pred, color='red', linewidth=2, label='Linear Regression Model')

# Enhance plot with labels, title, etc.
plt.xlabel('GDP per capita')
plt.ylabel('Life satisfaction')
plt.title('Linear Regression Model')
plt.legend()

plt.show()

print(model.predict([[22587]]))