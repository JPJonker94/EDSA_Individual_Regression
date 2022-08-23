# Import all modules used to create web application
from flask import Flask
from flask import request
from flask import render_template
from flask import send_from_directory
from model_for_prediction import upload_data
import os

# instance of Flask app
app = Flask(__name__)


# Create route for saving raw data and downloading prediction made
@app.route('/', methods=['GET', 'POST'])
def upload():
    """
    A web based application allowing user to upload data and calculate the response variable.
    input: Raw EDSA Individual | Electricity Shortfall Challenge train or test data.
    :return downloads a csv file with all predictions as per the submission.
    """

    # For POST method request the file from user
    if request.method == 'POST':
        # Requesting file from user
        file = request.files['file']
        # Create a variable to store csv to the file_save path including file name
        filename_save = f"{os.path.join('file_saved', file.filename)}"
        # Save uploaded file to path created
        file.save(filename_save)
        # Calculate the response variables using the upload_data function from web_deployment
        result = upload_data(filename_save)
        # Automatically download the response data calculated from the file results directory
        return send_from_directory('file_results', result)
    # Create a html template that will be shown the input and download results tabs on the server
    return render_template('upload_data.html')


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
