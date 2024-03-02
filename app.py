from dotenv import load_dotenv
load_dotenv()
import requests
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

@app.route('/create_index', methods=['POST'])
@cross_origin()
def create_index():
    user_id = request.json['user_id']
    database_name = "test"
    collection_name = user_id.split('@')[0] + '_invoice'
    vector_collection_name = user_id.split('@')[0] + '_invoice_vector'
    vector_index_name = (user_id.split('@')[0] + '_invoice_vector_index').replace('.', '_')
    uri = os.getenv("MONGODB_API_URI")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/vnd.atlas.2023-02-01+json',
    }
    payload = {
        "collectionName": vector_collection_name,
        "database": database_name,
        "name": vector_index_name,
        "type": "search",
        "mappings": {
            "dynamic": True,
            "fields": {
                "embedding": {
                    "dimensions": 1536,
                    "similarity": "cosine",
                    "type": "knnVector"
                }
            }
        }
    }
    try:
        response = requests.post(
            f"{uri}/groups/{os.getenv('MONGODB_ATLAS_GROUP_ID')}/clusters/{os.getenv('MONGODB_ATLAS_CLUSTER_NAME')}/fts/indexes",
            headers=headers,
            auth=requests.auth.HTTPDigestAuth(os.getenv('MONGODB_ATLAS_USERNAME'), os.getenv('MONGODB_ATLAS_PASSWORD')),
            json=payload
        )
        response.raise_for_status()
        return jsonify({"status": "success", "data": response.json()})
    except requests.exceptions.HTTPError as e:
        return jsonify({"status": "fail", "error": e.response.json()}), e.response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"status": "fail", "error": str(e)}), 500
    except Exception as e:
        return jsonify({"status": "fail", "error": str(e)}), 500

@app.route('/list_indexes/<database_name>/<collection_name>', methods=['GET'])
@cross_origin()
def list_indexes(database_name, collection_name):
    uri = os.getenv("MONGODB_API_URI")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/vnd.atlas.2023-02-01+json',
    }
    try:
        response = requests.get(
            f"{uri}/groups/{os.getenv('MONGODB_ATLAS_GROUP_ID')}/clusters/{os.getenv('MONGODB_ATLAS_CLUSTER_NAME')}/fts/indexes/{database_name}/{collection_name}",
            headers=headers,
            auth=requests.auth.HTTPDigestAuth(os.getenv('MONGODB_ATLAS_USERNAME'), os.getenv('MONGODB_ATLAS_PASSWORD')),
        )
        response.raise_for_status()
        return jsonify({"status": "success", "data": response.json()})
    except requests.exceptions.HTTPError as e:
        return jsonify({"status": "fail", "error": e.response.json()}), e.response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"status": "fail", "error": str(e)}), 500
    except Exception as e:
        return jsonify({"status": "fail", "error": str(e)}), 500




@app.route('/process', methods=['POST'])
@cross_origin()
def process():
    try:
        user_id = request.json['user_id']
        database_name = "test"
        collection_name = user_id.split('@')[0] + '_invoice'
        vector_collection_name = user_id.split('@')[0] + '_invoice_vector'
        vector_index_name = (user_id.split('@')[0] + '_invoice_vector_index').replace('.', '_')
        is_processed = process_entries(
            client=client,
            database_name=database_name,
            collection_name=collection_name,
            vector_collection_name=vector_collection_name,
            vector_index_name=vector_index_name
        )

        if not is_processed:
            return jsonify({"status": "fail", "error": "process failed"}), 400

        return jsonify({"status": "success", "message": "process successful"}), 200
    except KeyError:
        return jsonify({"status": "fail", "error": "invalid request body"}), 400
    except Exception as e:
        return jsonify({"status": "fail", "error": f"{e}"}), 500


@app.route('/query', methods=['POST'])
@cross_origin()
def process_form():
    # get the query
    query = request.json["query"]
    user_id = request.json['user_id']
    database_name = "test"
    vector_collection_name = user_id.split('@')[0] + '_invoice_vector'
    vector_index_name = (user_id.split('@')[0] + '_invoice_vector_index').replace('.', '_')
    store = MongoDBAtlasVectorSearch(
        client,
        db_name=database_name, # this is the database where you stored your embeddings
        collection_name=vector_collection_name, # this is where your embeddings were stored in 2_load_and_index.py
        index_name=vector_index_name # this is the name of the index you created after loading your data
    )
    index = VectorStoreIndex.from_vector_store(store)
    

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

