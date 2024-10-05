from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import joblib
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

encoder = joblib.load('onehot_encoder.pkl')

def load_data(user_location):
    df = pd.read_csv('cleaned_data.csv',
                     usecols=['location', 'rest_type', 'cuisines', 'listed_in(type)', 'cost', 'rate', 'votes', 'name'],
                     dtype={'location': 'category', 'rest_type': 'category', 'cuisines': 'category',
                            'listed_in(type)': 'category',
                            'cost': 'float32', 'rate': 'float32', 'votes': 'int32'})
    return df[df['location'] == user_location]

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Restaurant Recommender API!"})

@app.route('/recommend', methods=['POST', 'OPTIONS'])
def recommend_restaurants():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'OK'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response

    input_data = request.json

    user_location = input_data.get('location', '')
    user_cost = input_data.get('cost', 0)
    user_cuisines = input_data.get('cuisines', '')
    user_dining_type = input_data.get('listed_in(type)', '')

    filtered_df = load_data(user_location)

    filtered_df = filtered_df[(filtered_df['votes'] > 0) & (filtered_df['cost'] <= user_cost)]

    if user_cuisines:
        filtered_df = filtered_df[filtered_df['cuisines'].str.contains(user_cuisines, case=False)]
    if user_dining_type:
        filtered_df = filtered_df[filtered_df['listed_in(type)'].str.contains(user_dining_type, case=False)]

    if filtered_df.empty:
        return jsonify({'message': 'No restaurants found matching your criteria.'}), 404

    recommended_restaurants = filtered_df[['name', 'location', 'cost', 'rate', 'cuisines', 'votes', 'listed_in(type)']]
    response = jsonify(recommended_restaurants.to_dict(orient='records'))
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5500))
    app.run(host='0.0.0.0', port=port, debug=True)
