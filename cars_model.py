
import pandas as pd
import numpy as np
import re
import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint

def load_and_clean_data(filepath):
    """Loads data, cleans the price column, and finds price and year columns."""
    df = pd.read_csv(filepath)
    
    # Find price column
    price_col = next((col for col in df.columns if "price" in col.lower()), None)
    if not price_col:
        raise ValueError("No 'price' column found in the dataset.")
    
    # Find year column dynamically
    year_col = next((col for col in df.columns if "year" in col.lower()), None)
    if not year_col:
        print("Warning: No 'year' column found. Proceeding without 'car_age' feature.")

    # Drop rows where essential information is missing
    subset_to_drop = [price_col]
    if year_col:
        subset_to_drop.append(year_col)
    df = df.dropna(subset=subset_to_drop)
    df = df.reset_index(drop=True)

    def clean_price(x):
        x = str(x)
        x = re.sub(r"[^\d.]", "", x)
        return float(x) if x != "" else np.nan

    df[price_col] = df[price_col].apply(clean_price)
    df = df.dropna(subset=[price_col])
    
    return df, price_col, year_col

def feature_engineering(df, year_col):
    """Creates new features to improve model performance if year column exists."""
    if year_col:
        current_year = datetime.datetime.now().year
        # Ensure year column is numeric, coercing errors to NaN
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
        df = df.dropna(subset=[year_col]) # Drop rows where year could not be converted
        df[year_col] = df[year_col].astype(int)

        df['car_age'] = current_year - df[year_col]
        df = df.drop(year_col, axis=1)
    return df

def build_pipeline(numeric_features, categorical_features):
    """Builds a full preprocessing and modeling pipeline."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    return pipeline

def train_model(X, y, pipeline):
    """Splits data and trains the model using RandomizedSearchCV for hyperparameter tuning."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the parameter distribution for RandomizedSearchCV
    param_dist = {
        'regressor__n_estimators': randint(100, 500),
        'regressor__max_depth': [None] + list(np.arange(10, 31, 5)),
        'regressor__min_samples_split': randint(2, 11),
        'regressor__min_samples_leaf': randint(1, 5)
    }

    # Use RandomizedSearchCV to find the best hyperparameters
    # n_iter controls how many different combinations to try. Increase for better results.
    # n_jobs is set to 1 to avoid potential multiprocessing issues on Windows.
    search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_dist, 
        n_iter=10, # Reduced n_iter for faster debugging
        cv=3,      # Reduced cv folds for faster debugging
        verbose=1, 
        n_jobs=1, 
        random_state=42
    )
    
    print("Starting model tuning with RandomizedSearchCV...", flush=True)
    search.fit(X_train, y_train)
    
    print(f"Best parameters found: {search.best_params_}", flush=True)
    best_model = search.best_estimator_
    
    print("Evaluating model on the test set...", flush=True)
    score = best_model.score(X_test, y_test)
    print(f"Optimized Model R^2 Score on Test Set: {score:.4f}", flush=True)
    
    return best_model, X_train, X_test, y_train, y_test

def evaluate_predictions(model, X_test, y_test):
    """Makes predictions and evaluates them based on a percentage difference."""
    predictions = model.predict(X_test)

    results = X_test.copy()
    results["Actual Price"] = y_test
    results["Predicted Price"] = predictions
    results["Difference"] = results["Actual Price"] - results["Predicted Price"]

    def evaluate(row):
        # Avoid division by zero
        if row["Actual Price"] == 0:
            return "Cannot Evaluate"
        
        diff_percent = row["Difference"] / row["Actual Price"]
        
        if diff_percent > 0.15:  # Predicted price is >15% lower than actual
            return "Good Deal 🔥"
        elif diff_percent < -0.15: # Predicted price is >15% higher than actual
            return "Overpriced ❌"
        else:
            return "Fair Price 👍"

    results["Evaluation"] = results.apply(evaluate, axis=1)

    # Note: 'Good Deal' here means the actual price is higher than predicted, implying it might be a good find for a buyer.
    # The logic is inverted compared to the original script for clarity.
    best_deals = results[results['Evaluation'] == 'Good Deal 🔥'].sort_values(by="Difference", ascending=False).head(10)

    print("\n--- Top 10 Good Deals ---")
    print(best_deals[["Actual Price", "Predicted Price", "Difference", "Evaluation", "brand", "model", "car_age"]])
    return results

def predict_new_car(model, train_columns, car_details):
    """Predicts the price of a new car, ensuring columns match the training data."""
    new_car_df = pd.DataFrame([car_details])
    
    # Align columns with the training data, filling missing ones with NaN
    new_car_df = new_car_df.reindex(columns=train_columns, fill_value=np.nan)
    
    predicted_price = model.predict(new_car_df)
    
    print(f"\n--- New Car Prediction ---")
    print(f"Car Details: {car_details}")
    print(f"Predicted Price: {predicted_price[0]:,.2f}")
    
    return predicted_price[0]

if __name__ == "__main__":
    print("Script started...", flush=True)

    # 1. Load and prepare data
    print("Step 1: Loading and cleaning data...", flush=True)
    df, target_col, year_col = load_and_clean_data("car_ads_details_kaggle.csv")
    print("Data loaded. Starting feature engineering...", flush=True)
    df = feature_engineering(df.copy(), year_col)
    print("Step 1 complete.", flush=True)

    # 2. Define features and target
    print("Step 2: Defining features and target...", flush=True)
    features = [c for c in df.columns if c != target_col]
    X = df[features]
    y = df[target_col]
    print("Step 2 complete.", flush=True)

    # 3. Identify feature types
    print("Step 3: Identifying feature types...", flush=True)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    print(f"Found {len(numeric_cols)} numeric features and {len(categorical_cols)} categorical features.", flush=True)
    print("Step 3 complete.", flush=True)

    # 4. Build and train model
    print("Step 4: Building pipeline and training model...", flush=True)
    pipeline = build_pipeline(numeric_cols, categorical_cols)
    model, X_train, X_test, y_train, y_test = train_model(X, y, pipeline)
    print("Step 4 complete.", flush=True)

    # 5. Evaluate model predictions
    print("Step 5: Evaluating model predictions...", flush=True)
    results_df = evaluate_predictions(model, X_test, y_test)
    print("Step 5 complete.", flush=True)

    # 6. Predict price for a new car example
    print("Step 6: Predicting price for a new car example...", flush=True)
    new_car = {
        'brand': 'Toyota',
        'model': 'Corolla',
        'mileage': 80000,
        'fuel': 'Petrol',
        'transmission': 'Automatic',
    }
    # Add car_age to the example only if it was used as a feature
    if 'car_age' in X_train.columns:
        new_car['car_age'] = 5 # Example: 2026 - 5 = 2021 model
    
    # Use X_train.columns to ensure consistency
    predicted_price = predict_new_car(model, X_train.columns, new_car)

    # Example of comparing with an actual price
    actual_price = 450000
    difference = actual_price - predicted_price
    
    print(f"Actual Price: {actual_price:,.2f}", flush=True)
    if actual_price > 0:
        diff_percent = difference / actual_price
        if diff_percent > 0.15:
            print("Conclusion: Good Deal 🔥", flush=True)
        elif diff_percent < -0.15:
            print("Conclusion: Overpriced ❌", flush=True)
        else:
            print("Conclusion: Fair Price 👍", flush=True)
    print("Step 6 complete.", flush=True)
    print("Script finished.", flush=True)
