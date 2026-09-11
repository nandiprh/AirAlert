AI based air quality predictor and pollution hotspot

Urban air pollution changes based on traffic, weather, industrial activity and seasonal patterns. AQI monitoring stations provide measurements only at specific location.
This system predicts future air quality and identify potential pollution hotspots using historical pollutant and methodological data.

5 hybrid technologies are used here ------------>

1. LSTM + CNN
2. CNN + BiLSTM
3. LSTM + Attention
4. Transformer + LSTM
5. convLSTM + Attention

The datasets are collected from 
OpenAQ
UCI air quality dataset
India Open Govt. Data air quality datasets

The output is =====>

AQI forecast -> pollutant forecast -> hotspot msp -> public warning

=====================================================================================>

For CSV file classification problem and class imbalance handling, a sequence is:
```txt
                                             Load CSV Dataset
                                                    ↓
                                            Data Preprocessing
                                                    ↓
                                            Data Leakage Checks

                          
                          Target leakage • duplicate leakage • suspicious features
                                                    ↓
                            1. Train / Validation / Test Split or Train / Test Split
                            2. 10-fold cross validation
                                                    ↓
                        SMOTE Algorithm (balancing dataset on TRAINING data only)
                                                    ↓
                                            Initial Model Training
                                                    ↓
                            Overfitting / Underfitting Detection and Rectification
                                                    ↓
                                Apply Correction Technique and Retrain
                                                    ↓
                                            Final Test Evaluation
```
