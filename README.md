# **Hand Movement Classification with Random Forest**
This repository contains code for training a Random Forest model to classify hand movements based on sensor data. The hand movements correspond to five different finger actions, each labeled with specific categories.

## **Dataset**
The dataset consists of sensor readings for five hand movements:

* bas_parmak
* isaret_parmak
* orta_parmak
* yuzuk_parmak
* serce_parmak
  
Each dataset file corresponds to one of these movements and contains 8 sensor columns (sensor1 to sensor8) and a label column indicating the hand movement.

## Training the Model
**To train the Random Forest model, follow these steps:**

Place your dataset files in the specified directory (/content/sample_data/data).
Run the training script.
The Random Forest model is trained with the following parameters:

* n_estimators=150
* max_depth=10

## Results
**After training, the following metrics are printed:**

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
