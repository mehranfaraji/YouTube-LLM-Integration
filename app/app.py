from flask import Flask, render_template, request
import polars as pl
from sentence_transformers import SentenceTransformer
from sklearn.metrics import DistanceMetric
from Kurzegasgt import get_result

app = Flask(__name__)

model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)
df = pl.scan_parquet('data/video-index.parquet')
dist_name = 'manhattan'
dist = DistanceMetric.get_metric(dist_name)


@app.route('/', methods=['GET', 'POST'])
def search():
    results = None
    query = ""
    
    if request.method == 'POST':
        query = request.form['query']

        results = get_result(query, df, model, dist)

        return render_template('index.html', query=query, results=results)

    return render_template('index.html', query=query, results=results)

if __name__ == '__main__':
    app.run(debug=True)
