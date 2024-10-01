from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)

# Allow CORS for all routes
CORS(app)

df = pd.read_csv('cleaned_data.csv')
encoder = joblib.load('onehot_encoder.pkl')

# Preprocess data
df_encoded = encoder.transform(df[['location', 'rest_type', 'cuisines', 'listed_in(type)']])
df_encoded = pd.DataFrame(df_encoded, columns=encoder.get_feature_names_out())
df_final = pd.concat([df_encoded, df[['cost', 'rate', 'votes', 'book_table', 'online_order']]], axis=1)

@app.route('/recommend', methods=['POST', 'OPTIONS'])  # Handle POST and OPTIONS for preflight
def recommend_restaurants():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'OK'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response

    input_data = request.json
    page = input_data.get('page', 1)
    results_per_page = 4

    user_location = input_data.get('location', '')
    user_cost = input_data.get('cost', 0)
    user_cuisines = input_data.get('cuisines', '')
    user_dining_type = input_data.get('listed_in(type)', '')

    input_df = pd.DataFrame([input_data])

    required_columns = ['location', 'rest_type', 'cuisines', 'listed_in(type)']
    for column in required_columns:
        if column not in input_df.columns:
            input_df[column] = ''

    input_encoded = encoder.transform(input_df[['location', 'rest_type', 'cuisines', 'listed_in(type)']])
    input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out())
    input_encoded_df = input_encoded_df.reindex(columns=df_encoded.columns, fill_value=0)

    processed_input_df = pd.concat([input_encoded_df, input_df[['cost']].reset_index(drop=True)], axis=1)

    filtered_df = df[
        (df['location'] == user_location)
    ]

    # Filter out rows with 0 votes
    filtered_df = filtered_df[filtered_df['votes'] > 0]

    try:
        df_final_reindexed = filtered_df.reindex(columns=processed_input_df.columns, fill_value=0)
        similarities = cosine_similarity(processed_input_df, df_final_reindexed)
        top_indices = similarities[0].argsort()[::-1]

        start_index = (page - 1) * results_per_page
        end_index = start_index + results_per_page
        top_indices_page = top_indices[start_index:end_index]

        recommended_restaurants = filtered_df.iloc[top_indices_page].drop_duplicates(subset='name')[
            ['name', 'location', 'cost', 'rate', 'cuisines', 'votes', 'book_table', 'online_order', 'listed_in(type)']]

        # Sort by 'votes' in descending order
        recommended_restaurants = recommended_restaurants.sort_values(by='votes', ascending=False)

        response = jsonify(recommended_restaurants.to_dict(orient='records'))
        response.headers.add('Access-Control-Allow-Origin', '*')
        print(input_data)
        return response

    except ValueError:
        return jsonify({'error': "This ain't working dude"}), 400


if __name__ == '__main__':
    app.run(debug=True)
