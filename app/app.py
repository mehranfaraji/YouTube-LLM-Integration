from flask import Flask, render_template, request
import polars as pl
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import DistanceMetric
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Load YouTube API key
# YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Load the Sentence Transformer model
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

# Load the distance metric
dist_name = 'manhattan'
dist = DistanceMetric.get_metric(dist_name)

# Load the video data
df = pl.scan_parquet('data/video-index.parquet')


def returnSearchResults(query: str, df: pl.lazyframe.frame.LazyFrame, model, dist) -> np.ndarray:
    """
    Function to return indexes of top search results
    """
    query_embedding = model.encode(query).reshape(1, -1)
    
    # Compute distances between query and titles/transcripts
    dist_arr = dist.pairwise(df.select(df.columns[4:388]).collect(), query_embedding) + dist.pairwise(df.select(df.columns[388:]).collect(), query_embedding)
    
    # Search parameters
    threshold = 40
    top_k = 5

    idx_below_threshold = np.argwhere(dist_arr.flatten() < threshold).flatten()
    idx_sorted = np.argsort(dist_arr[idx_below_threshold], axis=0).flatten()

    results_df = df.select(['title', 'video_id']).collect()[idx_below_threshold[idx_sorted][:top_k]]
    

    # # Get titles and video_ids based on sorted indices
    # sorted_titles = results_df['title'][idx_sorted]
    # sorted_video_ids = results_df['video_id'][idx_sorted]
    

    # # Combine the sorted titles and video_ids into a list of dictionaries
    # results = [{"title": title, "video_id": video_id} for title, video_id in zip(sorted_titles, sorted_video_ids)]
    



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get query from the form
        query = request.form.get('query')
        
        # Search for videos
        search_results = returnSearchResults(query, df, model, dist)
        
        # Convert results to a list of dictionaries
        
        return render_template('index.html', results=search_results)
    
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
