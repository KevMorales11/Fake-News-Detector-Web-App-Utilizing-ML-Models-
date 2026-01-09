Setup Instructions

# Clone the repository

git clone <repository-url>
cd fake-news-web

# Create and activate the virtual environment 

python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

# Install required dependencies

pip install -r requirements.txt

# Add Databases

True.csv
Fake.csv

# Train the model

Run the script to prepare the data, train the logistic Regression model, and the save the vectorizer and model for the 
backend server

python backend/train.py

Expected success message:

Model training complete and saved in backend folder!

# Run the web application 
# Dependencies

This project requires the following Python packages:

Flask
pandas
scikit-learn
joblib
PyPDF2
python-docx
numpy
werkzeug

# Install the dependencies with: 

pip install -r requirements.txt


# License

This project currently does not have a formal license. Please contact the project owner for permissions regarding use or contributions.
