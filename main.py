import os

from fastapi import FastAPI, HTTPException, Request
from sentence_transformers import SentenceTransformer
import numpy as np
from tika import parser
from pydantic import BaseModel
app = FastAPI()
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

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

#Creat a data model

class Data(BaseModel):
    pathtopdf: str



@app.post("/")
#returns a list
def generate_embeddings(pdf: Data):
        embdedded_pdf = {}
        pdf_data = walk_pdf_dir(pdf.pathtopdf)
        for key, value in pdf_data.items():
            embeddings = model.encode(value)  # Generate embeddings
            embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
            embdedded_pdf[key] = embeddings_list


        return  embdedded_pdf