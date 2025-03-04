from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model (ensure the model file exists)
model = joblib.load("models/random_forest_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    # Expect JSON with a key "features" containing a list of feature dictionaries or arrays
    data = request.get_json()
    # Convert input to a pandas DataFrame
    features = pd.DataFrame(data["features"])
    # Make predictions
    predictions = model.predict(features)
    return jsonify({"predictions": predictions.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
