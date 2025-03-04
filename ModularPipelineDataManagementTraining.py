import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from config_loader import load_config

# Load configuration settings
config = load_config("config.yaml")

# Data Management: Load data and split into train/test
data = pd.read_csv(config['data']['path'])
X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config['data']['test_size'],
    random_state=config['data']['random_state']
)

# Model Training: Create and train the model using hyperparameters from config
model_params = config['model']['hyperparameters']
model = RandomForestClassifier(**model_params)
model.fit(X_train, y_train)

# Model Evaluation: Assess model performance on test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# Save the trained model for later deployment
joblib.dump(model, "models/random_forest_model.pkl")
