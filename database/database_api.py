from flask import Flask, request, jsonify
from flask_cors import CORS
from select_data import get_all_properties

app = Flask(__name__)
CORS(app)

@app.route('/api/properties', methods=['GET'])
def get_properties():
    print("Received request to get properties")
    
    try:
        api_response = get_all_properties()  # Remove the query parameter since function doesn't expect it
        # Ensure we return a list
        if not isinstance(api_response, list):
            api_response = []
        print(f"Returning {len(api_response)} properties")
        return jsonify(api_response)
    except Exception as e:
        print(f"Error getting properties: {e}")
        return jsonify([]), 200  # Return empty array instead of error

if __name__ == '__main__':
    app.run(debug=True, port=5001)
    