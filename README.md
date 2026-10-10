
# LSTM-Based Temperature Forecasting for Crop Recommendation

## 1. Project Overview

This project demonstrates two machine learning applications:

1. **Temperature Forecasting using LSTM:** Predicts daily temperature using the previous 30 days of historical temperature data.
2. **Crop Recommendation using Neural Networks:** Predicts a suitable crop using soil nutrients and weather-related features.

Note: These are two separate models. The crop classifier does not automatically use the LSTM forecast.

## 2. Problem Statement

Historical weather data contains useful patterns, but predicting temperature from previous observations can be challenging.

Similarly, selecting a suitable crop depends on soil nutrients and environmental conditions.

This project demonstrates how deep learning can help forecast temperature and classify crops using existing datasets.

## 3. Datasets Used

### A. Weather Dataset

- Source: NASA POWER
- Location: Warangal
- Period: 2001–2020
- Records: 7,305 daily observations
- Features: Temperature (T2M), relative humidity (RH2M), and precipitation (PRECTOTCORR)
- Target: Daily temperature in degrees Celsius
- Input sequence: Previous 30 days of temperature

### B. Crop Recommendation Dataset

- File: `data/Crop_recommendation.csv`
- Records: 2,200
- Features: N, P, K, temperature, humidity, pH, rainfall
- Target: Crop label
- Classes: 22 crop categories

## 4. Model Architectures

### A. LSTM Temperature Forecasting

- Input: 30 consecutive daily temperature values
- LSTM layer 1: 64 units
- LSTM layer 2: 32 units
- Dropout: 0.2
- Dense layer: 16 neurons with ReLU
- Output layer: 1 neuron for temperature prediction

### B. Crop Classification Neural Network

- Input layer: 7 features
- Dense layer: 64 neurons with ReLU
- Dropout: 0.2
- Dense layer: 32 neurons with ReLU
- Dropout: 0.2
- Output layer: 22 neurons with Softmax

## 5. Training Configuration

### Temperature Model

- Framework: TensorFlow/Keras
- Optimizer: Adam
- Loss function: Mean Squared Error
- Training epochs: 30
- Sequence length: 30 days

### Crop Classifier

- Framework: TensorFlow/Keras
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- Maximum epochs: 100
- Early stopping: Patience of 10 epochs

## 6. Results

### Temperature Forecasting

- Test MAE: 0.67°C
- Test RMSE: 0.90°C

The model achieved these results on the chronological held-out test period.

### Crop Classification

- Test accuracy: 99.32%
- Test samples: 440 out of 2,200 records

These results apply to the supplied dataset and do not guarantee real-world performance.

## 7. Graphs

Temperature model:
- `data/training_loss.png`
- `data/actual_vs_predicted.png`

Crop classifier:
- `data/crop_accuracy.png`
- `data/crop_loss.png`

## 8. Project Structure

```text
LSTM_Crop_Prediction/
├── data/
├── models/
├── inspect_weather.py
├── visualize_weather.py
├── prepare_weather.py
├── create_sequences.py
├── train_lstm.py
├── plot_training.py
├── evaluate_lstm.py
├── inspect_crop_data.py
├── prepare_crop_data.py
├── train_crop_model.py
├── evaluate_crop_model.py
├── plot_crop_training.py
├── predict_crop.py
├── predict_temperature.py
└── README.md
```

## 9. Installation Requirements

Python 3.11 is used for this project.

Install the required libraries with:

```bash
py -3.11 -m pip install numpy pandas matplotlib scikit-learn tensorflow joblib
```

## 10. How to Run

Open the project folder in VS Code.

Run the crop prediction program:

```bash
py -3.11 predict_crop.py
```

Run the temperature test prediction program:

```bash
py -3.11 predict_temperature.py
```

## 11. Limitations

- Temperature forecasts depend on historical patterns.
- The temperature model uses only previous temperature values.
- Crop classification depends on the quality and distribution of its training dataset.
- The two models operate independently.
- Predictions should not replace expert agricultural advice or local weather forecasts.

## 12. Future Improvements

- Forecast multiple future days.
- Include humidity and rainfall in temperature forecasting.
- Develop a user-friendly web application.
- Evaluate the models on additional locations and datasets.
- Explore a validated method for combining weather forecasts with crop recommendations.

## 13. Conclusion

This project demonstrates the use of an LSTM network for daily temperature forecasting and a neural network for crop classification.

The temperature model achieved a test MAE of 0.67°C and RMSE of 0.90°C. The crop classifier achieved 99.32% accuracy on its held-out test set.

The project provides practical experience in data preprocessing, sequence generation, deep learning, model evaluation, and prediction.
