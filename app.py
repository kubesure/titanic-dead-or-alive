from flask import Flask, request, jsonify
import pandas as pd
from catboost import CatBoostClassifier

# Load the model
model = CatBoostClassifier()
model.load_model("titanic_catboost.cbm")

# Flask app
app = Flask(__name__)

# Feature engineering function
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Handle missing age
    if pd.isnull(df['Age'].values[0]):
        df['Age'] = 30  # median age or your strategy

    return df

# API route
@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.json
    df = preprocess_input(input_data)

    # Predict probability
    prob = model.predict_proba(df[model.feature_names_])[0][1]
    prediction = int(prob >= 0.5)

    return jsonify({
        "prediction": prediction,
        "confidence": round(prob, 4)
    })

# Run the app
if __name__ == "__main__":
    app.run(debug=True)