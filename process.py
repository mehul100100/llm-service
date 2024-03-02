## This script loads data from a mongo database into an index
## This will convert all the documents in the database into vectors
## which requires a call to OpenAI for each one, so it can take some time.
## Once the data is indexed, it will be stored as a new collection in mongodb
## and you can query it without having to re-index every time.
from dotenv import load_dotenv
load_dotenv()
from bson import ObjectId
# This will turn on really noisy logging if you want it, but it will slow things down
# import logging
# import sys
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import os
from llama_index.readers.mongo import SimpleMongoReader
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.storage.storage_context import StorageContext

# Create a new client and connect to the server
# client = MongoClient(os.getenv("MONGODB_URI"), server_api=ServerApi('1'))


# load objects from mongo and convert them into LlamaIndex Document objects
# llamaindex has a special class that does this for you
# it pulls every object in a given collection

def process_entries(client, database_name, collection_name, vector_collection_name, vector_index_name):
    # connect to Atlas as a vector store
    
    query_dict = {"processed": False}
    collection = client[database_name][collection_name]
    unprocessed_entries = collection.find(query_dict)
    reader = SimpleMongoReader(uri=os.getenv("MONGODB_URI"))
    documents = reader.load_data(
        database_name,
        collection_name, # this is the collection where the objects you loaded in 1_import got stored
        # field_names=["saleDate", "items", "storeLocation", "customer", "couponUsed", "purchaseMethod"], # these is a list of the top-level fields in your objects that will be indexed
        field_names=["text"],                               # make sure your objects have a field called "full_text" or that you change this value
        query_dict=query_dict # this is a mongo query dict that will filter your data if you don't want to index everything
    )

    store = MongoDBAtlasVectorSearch(
        client,
        db_name=database_name, # this is the database where you stored your embeddings
        collection_name=vector_collection_name, # this is where your embeddings will be stored
        index_name=vector_index_name # this is the name of the index you will need to create
    )
    # # create Atlas as a vector store
    # now create an index from all the Documents and store them in Atlas
    storage_context = StorageContext.from_defaults(vector_store=store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context,
        show_progress=True, # this will show you a progress bar as the embeddings are created
    )

    if unprocessed_entries is not None:
        for entry in unprocessed_entries:
            result = collection.update_one({'_id': entry['_id']}, {'$set': {'processed': True}})
            if result.modified_count > 0:
                print("Entry updated successfully")
            else:
                print("Failed to update entry")
                return False
        return True
    else:
        print("No unprocessed entries found")
        return False


