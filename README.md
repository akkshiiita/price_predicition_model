
Stock Price Prediction using LSTM

📌 Project Overview

This project predicts stock prices using a Stacked Long Short-Term Memory (LSTM) model. The model is trained on historical stock data and forecasts future prices based on past trends. The dataset used in this project is Apple Inc. (AAPL) stock data.

🔧 Tech Stack

Python

TensorFlow/Keras

NumPy & Pandas

Scikit-Learn

Matplotlib

Pandas DataReader

📂 Dataset

The dataset consists of historical stock prices of Apple (AAPL) and includes columns like Open, High, Low, Close, and Volume. We primarily use the Close price for prediction.

🛠 Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/yourusername/stock-price-prediction-lstm.git
cd stock-price-prediction-lstm

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run the Project

python train.py

📊 Data Preprocessing

Load stock price data using Pandas.

Normalize the Close prices using MinMaxScaler.

Convert the time series data into a supervised learning problem using a sliding window approach.

Split data into training (65%) and testing (35%) sets.

🏗️ Model Architecture

The model consists of a stacked LSTM with:

Three LSTM layers (each with 50 units)

Dense layer with 1 neuron (output layer)

Optimizer: adam

Loss function: mean_squared_error

🚀 Training the Model

The model is trained for 100 epochs with a batch size of 64.

Uses Mean Squared Error (MSE) as the loss function.

Training data is reshaped into (samples, time_steps, features) format.

🔍 Performance Evaluation

Uses Root Mean Squared Error (RMSE) to evaluate model performance.

Predictions are transformed back to the original scale using inverse_transform().

📈 Results & Visualization

The model predicts the next 30 days of stock prices and visualizes:

Actual vs. Predicted Stock Prices for training and test data.

Future Stock Price Predictions.

Example Plot:

plt.plot(actual_prices, label='Actual Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.legend()
plt.show()

🔮 Future Improvements

Add GRU and compare with LSTM.

Use technical indicators as additional features.

Deploy the model as an API using Flask/FastAPI.

📌 Contributors

Akshita

📜 License

This project is licensed under the MIT License.
