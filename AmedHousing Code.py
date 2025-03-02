import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Title of the app
st.title("Ames Housing Price Prediction")

# Load Data (cached for performance)
@st.cache_data
def load_data():
    # Ensure 'AmesHousing.xlsx' is in the same directory as this file.
    df = pd.read_excel("AmesHousing.xlsx")
    return df

df = load_data()

# Display a brief overview of the dataset
st.subheader("Dataset Overview")
st.write(df.head())

# Preprocessing: select features and target variable
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
target = 'SalePrice'

# Drop rows with missing values in selected columns
df = df.dropna(subset=features + [target])
X = df[features]
y = df[target]

# Fill any remaining missing values with the median
X = X.fillna(X.median())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"Mean Squared Error: {mse:,.2f}")
st.write(f"R-squared: {r2:.2f}")

# Visualization: How Overall Quality affects Sale Price
st.subheader("Feature Analysis: Overall Quality vs Sale Price")
fig, ax = plt.subplots()
sns.boxplot(x=df['OverallQual'], y=df['SalePrice'], ax=ax)
ax.set_xlabel("Overall Quality")
ax.set_ylabel("Sale Price")
st.pyplot(fig)

# Sidebar: User inputs for prediction
st.sidebar.header("Input Features for Prediction")

def user_input_features():
    OverallQual = st.sidebar.slider("Overall Quality", 1, 10, 5)
    GrLivArea = st.sidebar.number_input("Above Ground Living Area (sq ft)", min_value=500, max_value=5000, value=1500)
    GarageCars = st.sidebar.slider("Garage Cars", 0, 4, 2)
    TotalBsmtSF = st.sidebar.number_input("Total Basement Area (sq ft)", min_value=0, max_value=3000, value=1000)
    FullBath = st.sidebar.slider("Number of Full Bathrooms", 1, 5, 2)
    YearBuilt = st.sidebar.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
    
    data = {
        "OverallQual": OverallQual,
        "GrLivArea": GrLivArea,
        "GarageCars": GarageCars,
        "TotalBsmtSF": TotalBsmtSF,
        "FullBath": FullBath,
        "YearBuilt": YearBuilt
    }
    features_df = pd.DataFrame(data, index=[0])
    return features_df

input_df = user_input_features()

st.subheader("User Input Parameters")
st.write(input_df)

# Make Prediction with error handling
try:
    prediction = model.predict(input_df)[0]
    st.subheader("Predicted Housing Price ($)")
    st.write(f"${prediction:,.2f}")
except Exception as e:
    st.error("An error occurred during prediction. Please review your input values.")
