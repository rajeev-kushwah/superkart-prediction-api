import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify


# Initialize Flask app
superkart_api = Flask("superkart_sales_api")


# Load the trained model pipeline (preprocessing + model)
model = joblib.load("superkart_model.joblib")

# Health check route
@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart Sales Prediction API"

# Define Endpoint to predict sales for a single product
@superkart_api.post('/v1/predict')
def predict_sales():
    try:
        # Parse JSON payload
        data = request.get_json()
        print("Raw incoming data:", data)

        # Validate expected fields
        required_fields = [
            'Product_Weight',
            'Product_Sugar_Content',
            'Product_Allocated_Area',
            'Product_MRP',
            'Store_Size',
            'Store_Location_City_Type',
            'Store_Type',
            'Store_Age_Years',
            'Product_Type_Category'
        ]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({'error': f"Missing fields: {missing_fields}"}), 400

        # Convert and transform input
        sample = {
            'Product_Weight': float(data['Product_Weight']),
            'Product_Sugar_Content': data['Product_Sugar_Content'],
            'Product_Allocated_Area_Log': np.log1p(float(data['Product_Allocated_Area'])),  # transform here
            'Product_MRP': float(data['Product_MRP']),
            'Store_Size': data['Store_Size'],
            'Store_Location_City_Type': data['Store_Location_City_Type'],
            'Store_Type': data['Store_Type'],
            'Store_Age_Years': int(data['Store_Age_Years']),
            'Product_Type_Category': data['Product_Type_Category']
        }

        input_df = pd.DataFrame([sample])
        print("Transformed input for model:\n", input_df)

        # Make prediction
        prediction = model.predict(input_df).tolist()[0]
        return jsonify({'Predicted_Sales': prediction})

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500



# Define an endpoint to predict sales for a batch of products
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    try:
        # Check that a file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        # Read CSV
        input_data = pd.read_csv(file)

        print("Raw batch data:\n", input_data.head())

        # Apply same transformation used for single prediction
        input_data['Product_Allocated_Area_Log'] = np.log1p(
            input_data['Product_Allocated_Area'].astype(float)
        )

        # Remove original column because model expects the log column
        input_data.drop(columns=['Product_Allocated_Area'], inplace=True)

        print("Transformed batch data:\n", input_data.head())

        # Make predictions
        predictions = model.predict(input_data).tolist()

        # Return predictions
        output_dict = {
            str(i): round(float(pred), 2)
            for i, pred in enumerate(predictions)
        }

        return jsonify(output_dict)

    except Exception as e:
        print("Error during batch prediction:", str(e))
        return jsonify({
            'error': f"Batch prediction failed: {str(e)}"
        }), 500



# Run the app (for local testing only)
if __name__ == '__main__':
    superkart_api.run(debug=True)
