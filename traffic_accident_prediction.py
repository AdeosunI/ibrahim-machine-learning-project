import pandas as pd

# Load the dataset
df = pd.read_csv("US_Accidents_March23.csv")

# Display the first few rows
print("Initial dataset preview:")
print(df.head())

# Show dataset summary
print("\nInitial dataset info:")
print(df.info())

# Convert 'Start_Time' to datetime format and drop rows with invalid dates
df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
df.dropna(subset=["Start_Time"], inplace=True)

# Broad cleaning step without filtering for specific years
df_filtered = df.copy()
df_filtered.dropna(
    subset=[
        "Severity",
        "Temperature(F)",
        "Wind_Speed(mph)",
        "Visibility(mi)",
        "Weather_Condition",
    ],
    inplace=True,
)

# Select and rename relevant columns
df_final = df_filtered[
    [
        "Start_Time",
        "Severity",
        "Temperature(F)",
        "Wind_Speed(mph)",
        "Visibility(mi)",
        "Weather_Condition",
    ]
].copy()

# Extract date and year parts
df_final["Date"] = df_final["Start_Time"].dt.date
df_final["Year"] = df_final["Start_Time"].dt.year

# Display the preprocessed data
print("\nCleaned and filtered dataset preview:")
print(df_final.head())
print("\nCleaned dataset info:")
print(df_final.info())

import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Separate data by year for time-based splitting
train_data = df_final[df_final["Year"].isin([2022, 2023])]
test_data = df_final[df_final["Year"].isin([2020, 2021])]

# Regroup and scale each set
daily_train_counts = train_data.groupby("Date").size()
daily_test_counts = test_data.groupby("Date").size()

daily_train_counts.index = pd.DatetimeIndex(daily_train_counts.index)
daily_train_counts = daily_train_counts.asfreq("D", fill_value=0)

scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(daily_train_counts.values.reshape(-1, 1))
if daily_test_counts.empty:
    raise ValueError(
        "Test dataset for years 2020 and 2021 is empty. Please check if data from those years exists."
    )
else:
    scaled_test = scaler.transform(daily_test_counts.values.reshape(-1, 1))


# Create sequences
def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


seq_length = 30
X_train, y_train = create_sequences(scaled_train, seq_length)
X_test, y_test = create_sequences(scaled_test, seq_length)

# Reshape for LSTM

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# --- Multiple Simulation & Model Comparison ---
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def compute_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))


# Store RMSE results
arma_rmses = []
lstm_rmses = []

num_trials = 5
for trial in range(num_trials):
    print(f"\n--- Trial {trial + 1} ---")

    # --- ARMA Trial ---
    # Refit ARMA each time (simulated randomness by subsampling if needed)
    try:
        arma_model = ARIMA(daily_train_counts.diff().dropna(), order=(2, 0, 2))
        arma_result = arma_model.fit()
        arma_forecast = arma_result.forecast(steps=len(daily_test_counts))

        # Clip to same shape if needed
        arma_rmse = compute_rmse(
            daily_test_counts.values[: len(arma_forecast)], arma_forecast
        )
        arma_rmses.append(arma_rmse)
        print(f"ARMA RMSE (Trial {trial + 1}): {arma_rmse:.4f}")
    except Exception as e:
        print(f"ARMA failed on trial {trial + 1}: {e}")
        arma_rmses.append(np.nan)

    # --- LSTM Trial ---
    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(seq_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(
        X_train, y_train, epochs=5, batch_size=32, verbose=0
    )  # Reduced epochs for faster testing
    y_pred = model.predict(X_test)

    lstm_rmse = compute_rmse(
        scaler.inverse_transform(y_test.reshape(-1, 1)),
        scaler.inverse_transform(y_pred),
    )
    lstm_rmses.append(lstm_rmse)
    print(f"LSTM RMSE (Trial {trial + 1}): {lstm_rmse:.4f}")

# Final comparison
avg_arma_rmse = np.nanmean(arma_rmses)
avg_lstm_rmse = np.mean(lstm_rmses)

print("\n--- Summary ---")
print(f"Average ARMA RMSE over {num_trials} trials: {avg_arma_rmse:.4f}")
print(f"Average LSTM RMSE over {num_trials} trials: {avg_lstm_rmse:.4f}")

if avg_arma_rmse < avg_lstm_rmse:
    print("ARMA model performed better on average.")
else:
    print("LSTM model performed better on average.")


# --- Visualization: RMSE Comparison ---
models = ["ARMA", "LSTM"]
avg_rmses = [avg_arma_rmse, avg_lstm_rmse]

plt.figure(figsize=(6, 4))
plt.bar(models, avg_rmses, color=["skyblue", "salmon"])
plt.title("Average RMSE Comparison")
plt.ylabel("RMSE")
plt.tight_layout()
plt.show()

# --- Visualization: Actual vs Predicted (last trial) ---
plt.figure(figsize=(10, 5))
# Align x and y values properly for actual and LSTM forecast
x_vals = np.arange(len(y_pred))
plt.plot(
    x_vals,
    daily_test_counts.values[seq_length : seq_length + len(y_pred)],
    label="Actual",
    color="black",
    linewidth=2,
)
plt.plot(
    x_vals,
    arma_forecast[: len(x_vals)],
    label="ARMA Forecast",
    linestyle="--",
    color="blue",
)
plt.plot(
    x_vals,
    scaler.inverse_transform(y_pred),
    label="LSTM Forecast",
    linestyle="--",
    color="red",
)
plt.title("Actual vs Predicted Accident Counts: ARMA and LSTM (Last Trial)")
plt.xlabel("Days")
plt.ylabel("Accident Count")
plt.legend()
plt.tight_layout()
plt.show()
