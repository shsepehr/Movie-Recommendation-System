import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample data
data = {
    "user": ["Alice", "Alice", "Bob", "Bob", "Carol"],
    "movie": ["Titanic", "Avatar", "Titanic", "Inception", "Avatar"],
    "rating": [5, 4, 4, 5, 5]
}
df = pd.DataFrame(data)

# Create user-item matrix
matrix = df.pivot_table(index='user', columns='movie', values='rating').fillna(0)

# Compute similarity
similarity = cosine_similarity(matrix)
print("User similarity matrix:\n", similarity)
