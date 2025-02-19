import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Load data
file_path = "mpu6050_data_merged.csv"
df = pd.read_csv(file_path)

# Debugging: Check if the file is loaded correctly
if df.empty:
    raise ValueError("CSV file is empty or could not be loaded correctly.")

# Print column names for debugging
print("Columns in CSV:", df.columns)

# Convert necessary columns to numeric and check for errors
cols_to_convert = ['Accel X (m/s^2)', 'Accel Y (m/s^2)', 'Accel Z (m/s^2)',
                   'Gyro X (deg/s)', 'Gyro Y (deg/s)', 'Gyro Z (deg/s)', 'Timestamp']

for col in cols_to_convert:
    if col not in df.columns:
        raise ValueError(f"Missing expected column: {col}")

    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric

# Remove rows with NaN values
df.dropna(inplace=True)

# Ensure data is sorted by timestamp
df = df.sort_values(by="Timestamp")

# Compute acceleration magnitude
df['accel_magnitude'] = np.sqrt(df['Accel X (m/s^2)']**2 + df['Accel Y (m/s^2)']**2 + df['Accel Z (m/s^2)']**2)

# Compute time difference
df['time_diff'] = df['Timestamp'].diff().replace(0, np.nan)

# Compute jerk (rate of change of acceleration)
df['jerk'] = df['accel_magnitude'].diff() / df['time_diff']

# Compute gyroscope magnitude
df['gyro_magnitude'] = np.sqrt(df['Gyro X (deg/s)']**2 + df['Gyro Y (deg/s)']**2 + df['Gyro Z (deg/s)']**2)

# Fill NaN values
df.fillna(0, inplace=True)

# Extract features for ML model
features = df[['accel_magnitude', 'jerk', 'gyro_magnitude']]

# Debugging: Ensure features are not empty
if features.empty:
    print("⚠️ Feature DataFrame is empty! Here’s the dataset after preprocessing:")
    print(df.head())
    raise ValueError("Feature DataFrame is empty. Check preprocessing steps.")

# Train Isolation Forest for anomaly detection
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
df['fatigue_alert'] = model.fit_predict(features)

# Convert predictions to readable format
df['fatigue_alert'] = df['fatigue_alert'].apply(lambda x: 'Fatigue Detected' if x == -1 else 'Normal')

# Save processed data
df.to_csv("highway_hypnosis_detection.csv", index=False)

# Visualize results
plt.figure(figsize=(10,5))
plt.plot(df['Timestamp'], df['accel_magnitude'], label="Acceleration Magnitude")
plt.scatter(df['Timestamp'][df['fatigue_alert'] == 'Fatigue Detected'], 
            df['accel_magnitude'][df['fatigue_alert'] == 'Fatigue Detected'], 
            color='red', label="Fatigue Alert")
plt.xlabel("Time")
plt.ylabel("Acceleration Magnitude")
plt.legend()
plt.title("Highway Hypnosis Detection")
plt.show()
