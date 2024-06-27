import logging
import joblib
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('/home/raj/Documents/project/phishing website/phishing_model.pkl')

# Set up logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.info('Info message: The application has started')
logging.warning('Warning message: This is a warning')
logging.error('Error message: An error occurred')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        logging.info(f"Received data for prediction: {data}")
        
        # Convert data to DataFrame
        df = pd.DataFrame(data, index=[0])
        logging.info(f"Data converted to DataFrame: {df}")
        
        # Make prediction
        prediction = model.predict(df)
        logging.info(f"Model prediction: {prediction[0]}")
        
        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8050)
