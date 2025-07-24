from flask import Flask, request, jsonify
from flask_cors import CORS
from select_data import get_all_properties

app = Flask(__name__)
CORS(app)

@app.route('/api/properties', methods=['GET'])
def get_properties():
    properties = get_all_properties()
    api_response = properties
    return api_response

if __name__ == '__main__':
    app.run(debug=True)
    