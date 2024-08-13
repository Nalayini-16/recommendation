import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
ratings = pd.read_csv('ratings.csv')

# Create the user-item matrix
user_movie_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating')
user_movie_matrix.fillna(0, inplace=True)

# Calculate user similarity matrix
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

def recommend_movies(user_id, num_recommendations):
    # Get the user's ratings
    user_ratings = user_movie_matrix.loc[user_id]
    
    # Calculate weighted sum of ratings from similar users
    similar_users = user_similarity_df[user_id]
    weighted_ratings = user_movie_matrix.T.dot(similar_users)
    
    # Create a DataFrame for weighted ratings
    weighted_ratings_df = pd.DataFrame(weighted_ratings, columns=['weighted_rating'])
    
    # Filter out movies already rated by the user
    rated_movies = user_ratings[user_ratings > 0].index
    recommendations = weighted_ratings_df.drop(index=rated_movies)
    
    # Sort by weighted rating and get the top recommendations
    recommendations = recommendations.sort_values(by='weighted_rating', ascending=False).head(num_recommendations)
    
    return recommendations

# Get recommendations for a specific user
user_id = 1
num_recommendations = 2
recommended_movies = recommend_movies(user_id, num_recommendations)
print(f"Top {num_recommendations} movie recommendations for user {user_id}:")
print(recommended_movies)

