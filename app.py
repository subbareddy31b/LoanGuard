from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

#Load the pickled model
with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('/index.html')

@app.route('/index')
def retIndex():
    return render_template('/index.html')

@app.route('/form')
def form():
    return render_template('/form.html')

@app.route('/upload')
def upload():
    return render_template('/upload.html')

@app.route('/visualization')
def visualization():
    return render_template('/visualization.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Data ::: ",data)
        loantoincomeRatio = float(data['LoanAmount'])/float(data['Income'])
        input_data = np.array([loantoincomeRatio,float(data['InterestRate']),float(data['CreditScore']),
                               float(data['Income']),float(data['MonthsEmployed']),float(data['LoanAmount']),
                               float(data['DTIRatio']),float(data['Age'])])
        input_data = input_data.reshape(1,-1)
        print("input :: ",input_data)
        result = loaded_model.predict(input_data)
        print("Result :: ",result)
        ans = "No"
        if result == 1:
            ans = "Yes"
        return jsonify({'result': ans})
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/process_csv', methods=['POST'])
def process_csv():
    try:
        # Check if the 'file' key is in the request files
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file provided'})

        file = request.files['file']
        print("File Content:")
        if file.filename.endswith('.csv'):
            print("inside if")
            try:
                df = pd.read_csv(file)
            except Exception as read_csv_error:
                return jsonify({'success': False, 'message': f'Error reading CSV: {str(read_csv_error)}'})
            columns_to_check = ['InterestRate', 'CreditScore','Income','MonthsEmployed','LoanAmount','DTIRatio','Age']
            if all(col in df.columns for col in columns_to_check):
                print("All columns are present")
                temp_df = pd.DataFrame()
                temp_df['LoanToIncomeRatio'] = df['LoanAmount']/df['Income']
                for col in columns_to_check:
                    temp_df[col] = df[col]
                
                print(temp_df)
                predictions = []
                for _, row in temp_df.iterrows():
                    iData = row.values
                    input_data_reshaped = iData.reshape(1, -1)
                    prediction = loaded_model.predict(input_data_reshaped)
                    predictions.append(prediction.tolist()[0])
                print("Predictions from model")
                print(predictions[:20])
                df['Default'] = predictions
                df['Default'] = df['Default'].replace({0: 'No', 1: 'Yes'})

                output_file = BytesIO()
                df.to_csv(output_file, index=False)
                output_file.seek(0)
                print("CSV SUCCESS")
                # Send the processed file to the frontend
                return send_file(output_file, download_name='output.csv', as_attachment=True)
            else:
                return jsonify({'success': False, 'message': 'One or more columns are missing. Provide correct data'})
        else:
            print("inside else")
            return jsonify({'success': False, 'message': 'Invalid file format. Please upload a CSV file.'})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/visualization1')
def visualization1():
    #Income, LoanAmount,CreditScore, InterestRate,MonthsEmployed
    bar_chart_dataI = {
        'labels': ['Model Data', 'User Data'],
        'data': [77905.60,float(request.args.get('Income'))]
    }
    bar_chart_dataLA = {
        'labels': ['Model Data', 'User Data'],
        'data': [135016.95,float(request.args.get('LoanAmount'))]
    }
    bar_chart_dataCS = {
        'labels': ['Model Data', 'User Data'],
        'data': [567.635,float(request.args.get('CreditScore'))]
    }    
    bar_chart_dataIR = {
        'labels': ['Model Data', 'User Data'],
        'data': [14.52,float(request.args.get('InterestRate'))]
    }
    bar_chart_dataME = {
        'labels': ['Model Data', 'User Data'],
        'data': [55.52,float(request.args.get('MonthsEmployed'))]
    }
    return render_template('visualization1.html', bar_chart_dataI=bar_chart_dataI,bar_chart_dataLA=bar_chart_dataLA,bar_chart_dataCS=bar_chart_dataCS,bar_chart_dataIR=bar_chart_dataIR,bar_chart_dataME=bar_chart_dataME)

if __name__ == '__main__':
    app.run(debug=True)
