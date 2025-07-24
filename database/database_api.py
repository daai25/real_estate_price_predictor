from flask import Flask, request, jsonify
from flask_cors import CORS
from select_data import get_all_properties

app = Flask(__name__)
CORS(app)

@app.route('/api/properties', methods=['GET'])
def get_properties():
    print("Received request to get properties")
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    query = data['query']
    api_response = get_all_properties(query)
    
    return jsonify(api_response)

if __name__ == '__main__':
    app.run(debug=True)
    