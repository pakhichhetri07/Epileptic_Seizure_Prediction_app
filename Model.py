############################
# Import necessary libraries
############################
import pandas as pd                                     
import numpy as np                                                                
from sklearn.preprocessing import StandardScaler                    
from sklearn.model_selection import train_test_split     
import matplotlib.pyplot as plt                          
import seaborn as sns                                   
from sklearn.metrics import confusion_matrix            
from imblearn.over_sampling import SMOTE                 
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score     
from sklearn.decomposition import PCA                    
from sklearn.manifold import TSNE                      
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
import joblib
import pickle
import os

############################
# Loading the dataset
############################
data = pd.read_csv("/home/pakhi/PAKHI/Project/data/Signal_Data.csv")

print("Dataset Information:")
print(data.info())  

# Check for missing and null values
missing_values = data.isnull().sum().sum()
print("\nMissing Values in dataset:", missing_values)

null_values = data.isna().sum().sum()
print("\nNull Values in dataset:", null_values)

# Data splitting into input (X) and target (y)
X = data.drop(columns=['column_a', 'y'])  
y = data['y']

# Combine non-epileptic classes (2, 3, 4, 5) into a single class (0)
y_combined = y.copy()
y_combined[y_combined != 1] = 0  # Class 1 is epileptic, class 0 is Non-epileptic

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y_combined, test_size=0.2, random_state=42)

####################################
#Balance the dataset
####################################
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f'Original dataset shape: {X_train.shape}')
print(f'Resampled dataset shape: {X_train_resampled.shape}')
print(f'Class distribution after SMOTE: {pd.Series(y_train_resampled).value_counts()}')

######################################
# Feature scaling 
######################################
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)
print(f'Scaled Resampled Data Shape: {X_train_scaled.shape}')
print(f'Test Data Shape: {X_test_scaled.shape}')

######################################################
# Feature Extraction
######################################################
import pywt
def extract_dwt_features(data):
    features = []
    for signal in data:
        coeffs = pywt.wavedec(signal, wavelet='db4', level=5)
        flattened_coeffs = np.hstack(coeffs)  
        features.append(flattened_coeffs)
    return np.array(features)

# Extract DWT features from the scaled training and test data
X_train_dwt = extract_dwt_features(X_train_scaled)
X_test_dwt = extract_dwt_features(X_test_scaled)
print(f'DWT Feature Shape for Resampled Data: {X_train_dwt.shape}')
print(f'DWT Feature Shape for Test Data: {X_test_dwt.shape}')

######################################################
# Dimensionality reduction 
######################################################
pca = PCA(n_components=0.95)  # Retain components that explain 95% of the variance
X_train_pca = pca.fit_transform(X_train_dwt)
X_test_pca = pca.transform(X_test_dwt)
print(f'PCA Transformed Shape for Resampled Data: {X_train_pca.shape}')
print(f'PCA Transformed Shape for Test Data: {X_test_pca.shape}')


######################################################
# Train different classifiers
######################################################
classifiers = {
    'XGBoost': XGBClassifier(eval_metric='mlogloss'),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'MLP': MLPClassifier(max_iter=500)
}

# Train each classifier and display performance metrics
for name, clf in classifiers.items():
    clf.fit(X_train_pca, y_train_resampled) 
    y_pred = clf.predict(X_test_pca)  

    # Calculate and display evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Display confusion matrix and metrics
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nClassifier: {name}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))

###########################################################
# Hard Voting Classifier (Ensemble of multiple classifiers)
###########################################################
voting_clf = VotingClassifier(estimators=[
    ('XGBoost', XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)),
    ('KNN', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('MLP', MLPClassifier(max_iter=500))],
    voting='hard'
)

# Train the voting classifier
voting_clf.fit(X_train_pca, y_train_resampled)
y_pred = voting_clf.predict(X_test_pca)

# Evaluate the voting classifier
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

####################################################################
# Confusion matrix and performance metrics for the voting classifier
####################################################################
cm = confusion_matrix(y_test, y_pred)
print(f"\nVoting Classifier (Hard) Results:")
print(f"Confusion Matrix:\n{cm}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(classification_report(y_test, y_pred))

# Save the trained model to a file using pickle
model_filename = 'Model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(voting_clf, model_file) 

# Print model save confirmation with absolute path
absolute_path = os.path.abspath(model_filename)
print("Model saved successfully!")
print("Model file path:", absolute_path)
