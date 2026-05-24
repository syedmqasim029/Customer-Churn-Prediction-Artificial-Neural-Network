# Customer-Churn-Prediction-Artificial-Neural-Network

Artificial Neural Network for predicting customer churn.
A full DL project from end to end predicting if a customer of a bank will churn or not, with an Artificial Neural Network (ANN) trained using TensorFlow and deployed with Streamlit.

Live at: https://customer-churn-prediction-artificial-neural-network.streamlit.app/

## Project Structure

```
├── app.py                    # Streamlit web application
├── experiments.ipynb         # Data preprocessing & model training notebook
├── prediction.ipynb          # Model testing notebook
├── model.h5                  # Saved trained ANN model
├── scaler.pkl                # Saved StandardScaler object
├── label_encoder_gender.pkl  # Saved LabelEncoder for Gender
├── onehot_encoder_geo.pkl    # Saved OneHotEncoder for Geography
├── Churn_Modelling.csv       # Dataset
└── requirements.txt          # Project dependencies
```

---

## Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/syedmqasim029/Customer-Churn-Prediction-Artificial-Neural-Network

# 2. Create virtual environment
python -m venv myvenv

# 3. Activate virtual environment
source myvenv/bin/activate        # Mac/Linux
myvenv\Scripts\activate           # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the app
streamlit run app.py
```

---

## Pipeline Overview

### 1. Data Preprocessing
- Dropped irrelevant columns: `RowNumber`, `CustomerId`, `Surname`
- Applied **LabelEncoder** on `Gender` column (Male=1, Female=0)
- Applied **OneHotEncoder** on `Geography` column (France, Germany, Spain → 3 binary columns)
- Saved both encoder objects as `.pkl` files

### 2. Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 3. Feature Scaling
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
```
Scaler saved as `scaler.pkl`

### 4. Model Architecture
```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1,  activation='sigmoid')
])
```

| Layer | Neurons | Activation |
|-------|---------|------------|
| Hidden Layer 1 | 64 | ReLU |
| Hidden Layer 2 | 32 | ReLU |
| Output Layer | 1 | Sigmoid |

### 5. Model Compilation
```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### 6. Callbacks
```python
EarlyStopping(monitor='val_loss', patience=10)
TensorBoard(log_dir='logs')
```

### 7. Model Saved
```python
model.save('model.h5')
```
### Streamlit App
The app loads all saved .pkl files and the model, takes the user input for all the features and calculates the churn probability in real-time.

Prediction Logic:
```python
if prediction > 0.5:
    "Customer is likely to Churn"
else:
    "Customer is NOT likely to Churn"
```
---

##  Sample Test Cases

| Geography | Gender | Age | Balance | Credit Score | Active | Result |
|-----------|--------|-----|---------|--------------|--------|--------|
| Germany | Female | 45 | 125000 | 400 | No |  Churn (0.69) |
| France | Male | 30 | 50000 | 750 | Yes |  No Churn (0.01) |
| Germany | Female | 58 | 180000 | 350 | No |  Churn (1.00) |

---
##  Tech Stack

- **Python 3.12**
- **TensorFlow**
- **Scikit-learn**
- **Pandas**
- **Streamlit**
- **Pickle**

---

## Author

**Syed Muhammad Qasim**
GitHub: [@syedmqasim029](https://github.com/syedmqasim029)
