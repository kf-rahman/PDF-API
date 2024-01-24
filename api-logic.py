import os

import PyPDF2
import numpy as np
from tika import parser # pip install tika
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

from qdrant_client import models,QdrantClient

def walk_pdf_dir(pathpdfs):
    list_of_pdfs = {}
    for (dirpath, dirnames, filenames) in os.walk(pathpdfs):
        for filename in filenames:
            if filename.endswith('.pdf'):
                list_of_pdfs[filename] = os.sep.join([dirpath, filename])
    pdf_content_dict = read_pdf(list_of_pdfs)

    return pdf_content_dict


def read_pdf(listofpfds):
    pdf_content = {}
    for key, value in listofpfds.items():
        raw = parser.from_file(value)
        pdf_content[key] = raw['content']

    return pdf_content


def create_vector_database(collection_name,store = None):

    if store == None:
        storage = QdrantClient(":memory:")
    else:
        storage = QdrantClient(path = store)


    storage.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=model.get_sentence_embedding_dimension(),  # Vector size is defined by used model
            distance=models.Distance.COSINE
        )
    )

def generate(pdf_path,db ='no'):
    embdedded_pdf = {}
    pdf_data = walk_pdf_dir(pdf_path)
    for key,value in pdf_data.items():
        embeddings = model.encode(value)  # Generate embeddings
        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        embdedded_pdf[key] = embeddings_list

    if db == 'yes':
       create_vector_database('pdf')
       print('Files embedded and stored in a vector database')
    else:

        print('Files embedded')



    return None


