import requests
import json
import polars as pl
import matplotlib.pyplot as plt
from youtube_transcript_api import YouTubeTranscriptApi
from os import getenv
import dotenv
from flask import Flask
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import DistanceMetric

dotenv.load_dotenv()


YOUTUBE_API_KEY = getenv("YOUTUBE_API_KEY")
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)
df = pl.scan_parquet('data/video-index.parquet')
dist_name = 'manhattan'
dist = DistanceMetric.get_metric(dist_name)

def returnSearchResults(query: str, df: pl.lazyframe.frame.LazyFrame, model, dist) -> np.ndarray:
    """
        Function to return indexes of top search results
    """
    
    # embed query
    query_embedding = model.encode(query).reshape(1, -1)
    
    # compute distances between query and titles/transcripts
    #384 is the dimesnsion for the embedding space
    dist_arr = dist.pairwise(df.select(df.columns[4:388]).collect(), query_embedding) + dist.pairwise(df.select(df.columns[388:]).collect(), query_embedding)

    # search paramaters
    threshold = 40 # eye balled threshold for manhatten distance
    top_k = 5

    # evaluate videos close to query based on threshold
    idx_below_threshold = np.argwhere(dist_arr.flatten()<threshold).flatten()
    # keep top k closest videos
    idx_sorted = np.argsort(dist_arr[idx_below_threshold], axis=0).flatten()

    # Get the rows based on the sorted indices
    results_df = df.select(['title', 'video_id']).collect()[idx_sorted]

    # Get titles and video_ids based on sorted indices
    sorted_titles = results_df['title'][idx_sorted]
    sorted_video_ids = results_df['video_id'][idx_sorted]

    # Combine the sorted titles and video_ids into a list of dictionaries
    results = [{"title": title, "video_id": video_id} for title, video_id in zip(sorted_titles, sorted_video_ids)]
    
    return results[:top_k]

results = returnSearchResults("virus", df, model, dist)

print(results)
    