from qdrant_client import models,QdrantClient
from sentence_transformers import SentenceTransformer

encoder1 = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
encoder = SentenceTransformer('all-MiniLM-L6-v2')




def create_vector_database(collection_name,store = None):

    if store == None:
        storage = QdrantClient(":memory:")
    else:
        storage = QdrantClient(path = store)


    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
        size=encoder1.get_sentence_embedding_dimension(),  # Vector size is defined by used model
        distance=models.Distance.COSINE
    )
    )

