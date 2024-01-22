from fastapi import FastAPI, HTTPException, Request
from sentence_transformers import SentenceTransformer
import numpy as np
import pydantic
from pydantic import BaseModel
app = FastAPI()
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
#Creat a data model
#Since the client will be passing in some sort of text, we need to use Pydantic data model

class Data(BaseModel):
    content: str
@app.post("/")
#returns a list
def generate_embeddings(pdf: Data):
        pdf_data =pdf.content
        embeddings = model.encode(pdf_data)  # Generate embeddings

        # Convert embeddings to a list (if they are in NumPy array format)
        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

        return  embeddings_list