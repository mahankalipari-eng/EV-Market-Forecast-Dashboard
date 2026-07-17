# Install required libraries
!pip -q install pandas numpy matplotlib scikit-learn plotly

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression

# -------------------------------
# Sample EV Sales Dataset
# -------------------------------
df = pd.DataFrame({
    "Year":[2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025],
    "Sales":[1200,1800,2600,4100,6300,9500,14500,21500,31000,43000,59000]
})

print("EV Sales Dataset")
display(df)

# -------------------------------
# Train Linear Regression Model
# -------------------------------
X = df[['Year']]
y = df['Sales']

model = LinearRegression()
model.fit(X, y)

# -------------------------------
# Predict Future Sales
# -------------------------------
future_years = pd.DataFrame({
    "Year":[2026,2027,2028,2029,2030]
})

future_years["Predicted Sales"] = model.predict(future_years).astype(int)

print("\nForecast Data")
display(future_years)

# -------------------------------
# Growth Rate
# -------------------------------
growth = ((future_years.iloc[-1]["Predicted Sales"] -
            df.iloc[-1]["Sales"]) /
            df.iloc[-1]["Sales"]) * 100

print(f"\nEstimated Growth Rate : {growth:.2f}%")

# -------------------------------
# Matplotlib Graph
# -------------------------------
plt.figure(figsize=(10,5))

plt.plot(df["Year"], df["Sales"],
         marker='o',
         linewidth=3,
         label="Historical Sales")

plt.plot(future_years["Year"],
         future_years["Predicted Sales"],
         marker='o',
         linestyle='--',
         linewidth=3,
         label="Forecast Sales")

plt.xlabel("Year")
plt.ylabel("EV Sales")
plt.title("EV Market Forecast")
plt.grid(True)
plt.legend()
plt.show()

# -------------------------------
# Interactive Plotly Graph
# -------------------------------
history = df.copy()
history["Type"] = "Historical"

forecast = future_years.rename(columns={"Predicted Sales":"Sales"})
forecast["Type"] = "Forecast"

combined = pd.concat([history, forecast], ignore_index=True)

fig = px.line(
    combined,
    x="Year",
    y="Sales",
    color="Type",
    markers=True,
    title="EV Market Forecast Dashboard"
)

fig.show()

# -------------------------------
# Predict Any Year
# -------------------------------
year = int(input("Enter a year (2026-2050): "))
prediction = model.predict([[year]])

print(f"\nPredicted EV Sales in {year}: {int(prediction[0])}")
