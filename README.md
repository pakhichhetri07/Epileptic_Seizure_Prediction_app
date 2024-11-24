### Model Overview: EEG-Based Epileptic Seizure Prediction

### Model Type:
- **Voting Classifier** with a soft voting mechanism.
- Combines predictions from five different classifiers:
  - XGBoost
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - Multi-Layer Perceptron (MLP)

---

### Input Data:
- **Source:** EEG signal data.
- **UCI-Machine learning learning:** Begleiter, H. (1995). EEG Database [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TS3D
-**EEG Dataset:** https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition
  
---

### Preprocessing Steps:
1. **Class Imbalance Handling:**
   - **SMOTE** (Synthetic Minority Over-sampling Technique) is used to balance the dataset by creating synthetic features based on statistical measures(such as mean, median, etc).
2. **Feature Scaling:**
   - **StandardScaler** is applied to standardize the features to scale data to have mean=0 and standard deviation=1.
3. **Wavelet Transform:**
   - **Discrete Wavelet Transform (DWT)** is used to perform signal processing for extracting features from EEG signals.
4. **Dimensionality Reduction:**
   - **PCA** reduces the feature dimensions while maintaining 95% of the data variance.

---

### Prediction Process:
- Each of the five classifiers makes independent predictions on the PCA-transformed test data.
- The **Voting Classifier** aggregates these predictions using **soft voting**.
  - **Example:** If Model_1 predicts class 0 (probality=0.6), Mode_2 predicts class 1 (probality=0.7), and Model_3 predicts class                   0 (probality=0.8), then the final prediction is class 0 based on averaged probabilities .

---

### Target Variable:
- The model predicts **binary class labels**:
  - **1:** Epileptic (y-label 1)
  - **0:** Non-epileptic (all other non-epileptic conditions (y-label: 2,3,4,5) are combined into class 0).

---

### Performance Metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix:**
  - Evaluates true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).

---

### Model Saving:
- The trained Voting Classifier model is serialized and saved as **`Model.pkl`** using **pickle** for later use.

---

### Feature Extraction Process:
- Each EEG signal undergoes **DWT (Wavelet Decomposition)**.
  - Signals are transformed into wavelet coefficients (features).
  - These features are then flattened into a single feature vector, which is fed into the model.

---

### Test Data Flow:
1. **Scaling:** Test data is standardized using **StandardScaler**.
2. **Feature Extraction:** Features are extracted from the EEG signals using **DWT**.
3. **Dimensionality Reduction:** **PCA** reduces the feature dimensions.
4. **Prediction:** The PCA-transformed data is passed into the trained classifiers for final predictions.
