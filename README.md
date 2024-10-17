### Model Overview: EEG-Based Epileptic Seizure Prediction

### Model Type:
- **Voting Classifier** with a hard voting mechanism.
- Combines predictions from five different classifiers:
  - XGBoost
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - Multi-Layer Perceptron (MLP)

---

### Input Data:
- **Source:** EEG signal data.
- **Feature Extraction:** 
  - Discrete Wavelet Transform (DWT) is applied to extract relevant features from the EEG signals.
- **Dimensionality Reduction:**
  - Principal Component Analysis (PCA) is used to reduce dimensionality, retaining components that explain 95% of the variance.

---

### Preprocessing Steps:
1. **Class Imbalance Handling:**
   - **SMOTE** (Synthetic Minority Over-sampling Technique) is used to balance the dataset.
2. **Feature Scaling:**
   - **StandardScaler** is applied to standardize the features.
3. **Wavelet Transform:**
   - **Discrete Wavelet Transform (DWT)** is used for extracting features from EEG signals.
4. **Dimensionality Reduction:**
   - **PCA** reduces the feature dimensions while maintaining 95% of the data variance.

---

### Prediction Process:
- Each of the five classifiers makes independent predictions on the PCA-transformed test data.
- The **Voting Classifier** aggregates these predictions using **hard voting**.
  - **Example:** If 3 out of 5 classifiers predict a sample as class 0 (non-epileptic), the final prediction will be class 0.

---

### Target Variable:
- The model predicts **binary class labels**:
  - **1:** Epileptic
  - **0:** Non-epileptic (all other non-epileptic conditions are combined into class 0).

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
