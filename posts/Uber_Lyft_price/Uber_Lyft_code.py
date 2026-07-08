import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.utils import resample


#KNN and Random Forest

#Load the dataset
df = pd.read_csv("cab_rides.csv")
#Create new features
df['datetime'] = pd.to_datetime(df['time_stamp'], unit='ms')
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
#Change categorical variables into numeric variables
df = pd.get_dummies(df,columns=["cab_type","name","source","destination"])
#Drop unnecessary features and missing data
df.drop(columns=["id","product_id","time_stamp","datetime"],inplace=True)
df = df.dropna()
print(df.columns)

#Define input (X) and output (y) features
X = df.drop(columns=["price"])
y = df["price"]
#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
#KNN sample of 200000 from training set
X_train_knn_sample = X_train.sample(n=200000, random_state=123)
y_train_knn_sample = y_train[X_train_knn_sample.index]

#Scale numeric variables for KNN
scaler = StandardScaler()
X_train_knn_scaled = scaler.fit_transform(X_train_knn_sample)
X_test_knn_scaled  = scaler.transform(X_test)

#print(df.head(10))
#print(df.columns)

#KNN model
print("--------- KNN ---------\n")
k_values = range(1, 13, 2)
rmse_scores = []

for k in k_values:
    print("Running KNN model with k =",k)
    knn = KNeighborsRegressor(n_neighbors=k, n_jobs=-1)
    scores = cross_val_score(knn, X_train_knn_scaled, y_train_knn_sample,
                             cv=5, scoring='neg_mean_squared_error')
    rmse_scores.append(np.sqrt(-scores.mean()))

best_k = k_values[np.argmin(rmse_scores)]
print(f"Best K: {best_k}")

knn_best = KNeighborsRegressor(n_neighbors=best_k)
knn_best.fit(X_train_knn_scaled, y_train_knn_sample)
y_pred_knn = knn_best.predict(X_test_knn_scaled)


knn_rmse = np.sqrt(mean_squared_error(y_test, y_pred_knn))
knn_mae  = mean_absolute_error(y_test, y_pred_knn)
knn_r2   = r2_score(y_test, y_pred_knn)

print(f"\n--- KNN Results (k={best_k}) ---")
print(f"RMSE: {knn_rmse:.3f}")
print(f"MAE:  {knn_mae:.3f}")
print(f"R²:   {knn_r2:.3f}")

#Residuals plot for KNN
residuals_knn = y_test - y_pred_knn

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_knn, residuals_knn, alpha=0.3, color='blue')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title(f"KNN: Residual Plot (k={best_k})")
plt.tight_layout()
plt.show()

#Random Forest Regressor
print("\n--------- Random Forest Regressor ---------\n")
rtree = RandomForestRegressor(random_state=123)
rtree.fit(X_train, np.ravel(y_train))
y_pred_rf = rtree.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_mae  = mean_absolute_error(y_test, y_pred_rf)
rf_r2   = r2_score(y_test, y_pred_rf)

print(f"\n--- Random Forest Results ---")
print(f"RMSE: {rf_rmse:.3f}")
print(f"MAE:  {rf_mae:.3f}")
print(f"R²:   {rf_r2:.3f}")

#Residuals plot for RF
residuals_rf = y_test - y_pred_rf

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_rf, residuals_rf, alpha=0.3, color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title(f"Random Forest: Residual Plot")
plt.tight_layout()
plt.show()

#Feature Importance Table
print("\n--- Feature Importance Table ---")
print(
    pd.DataFrame(
        data={
            "feature": rtree.feature_names_in_ ,
            "importance": rtree.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
)

print("\n--- Manual Prediction Check ---")
sample_ride = X_test.iloc[0:1]  # take one ride from test set
sample_scaled = scaler.transform(sample_ride)
manual_pred_knn = knn_best.predict(sample_scaled)
manual_pred_rf = rtree.predict(sample_ride)

print(f"Predicted Price (KNN): ${manual_pred_knn[0]:.2f}")
print(f"Predicted Price (Random Forest): ${manual_pred_rf[0]:.2f}")
print(f"Actual Price:    ${y_test.iloc[0]:.2f}")

#Confidence Intervals
def compute_ci(model, X_test, y_test, n_iterations=20):
    rmse_list = []

    for i in range(n_iterations):
        X_resample, y_resample = resample(X_test, y_test)
        y_pred = model.predict(X_resample)
        rmse = np.sqrt(mean_squared_error(y_resample, y_pred))
        rmse_list.append(rmse)

    lower = np.percentile(rmse_list, 2.5)
    upper = np.percentile(rmse_list, 97.5)

    return lower, upper

rf_lower, rf_upper = compute_ci(rtree, X_test, y_test)
knn_lower, knn_upper = compute_ci(knn_best, X_test_knn_scaled, y_test)

print("\n--- Confidence Intervals (95%) ---")
print(f"RF RMSE CI: [{rf_lower:.3f}, {rf_upper:.3f}]")
print(f"KNN RMSE CI: [{knn_lower:.3f}, {knn_upper:.3f}]")
