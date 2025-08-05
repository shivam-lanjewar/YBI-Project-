# Step 1: Import library and function
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Sample data (replace with your actual data)
data = {'size': [1000, 1500, 1200, 1800, 2000],
        'bedrooms': [2, 3, 2, 4, 4],
        'age': [10, 5, 8, 2, 1],
        'price': [250000, 350000, 300000, 450000, 500000]}
df = pd.DataFrame(data)

# Step 3: Define features (X) and target (y)
X = df[['size', 'bedrooms', 'age']]
y = df['price']

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Coefficients and Intercept
model.intercept_ , model.coef_

# Make predictions on the test set
y_pred = model.predict(X_test)

# Step 7: Calculate Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Example prediction for a new house
new_house = pd.DataFrame({'size': [1600], 'bedrooms': [3], 'age': [7]})
predicted_price = model.predict(new_house)
print(f"Predicted price for new house: {predicted_price[0]}")
