from fastapi import FastAPI, HTTPException, Request
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI()
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

@app.post("/get-embeddings/")
async def generate_embeddings(request: Request):
    try:
        data = await request.json()
        text = data['data']
        embeddings = model.encode([text])  # Generate embeddings

        # Convert embeddings to a list (if they are in NumPy array format)
        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

        return {"embeddings": embeddings_list}
    except Exception as e:
        print(e)  # For debugging
        raise HTTPException(status_code=500, detail=str(e))
