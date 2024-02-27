from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.indices.vector_store.base import VectorStoreIndex
from process import process_entries

# Create a new client and connect to the server
client = MongoClient(os.getenv("MONGODB_URI"), server_api=ServerApi('1'))

# connect to Atlas as a vector store
store = MongoDBAtlasVectorSearch(
    client,
    db_name=os.getenv('MONGODB_DATABASE'), # this is the database where you stored your embeddings
    collection_name=os.getenv('MONGODB_VECTORS'), # this is where your embeddings were stored in 2_load_and_index.py
    index_name=os.getenv('MONGODB_VECTOR_INDEX') # this is the name of the index you created after loading your data
)
index = VectorStoreIndex.from_vector_store(store)

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# This is just so you can easily tell the app is running
@app.route('/')
def hello_world():
    return jsonify({
        "status": "success",
        "message": "hello world"
    })

@app.route('/process', methods=['POST'])
@cross_origin()
def process():
    is_processed = process_entries(client = client)
    if not is_processed:
        return jsonify({"status": "failed", "error": "process failed"}), 400
    return jsonify({"status": "success", "message": "process successful"})


@app.route('/query', methods=['POST'])
@cross_origin()
def process_form():
    # get the query
    query = request.json["query"]

    if query is not None:
        # query your data!
        # here we have customized the number of documents returned per query to 20, because tweets are really short
        query_engine = index.as_query_engine(similarity_top_k=20)
        response = query_engine.query(query)
        return jsonify({"response": str(response)})
    else:
        return jsonify({"error": "query field is missing"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=9000)

