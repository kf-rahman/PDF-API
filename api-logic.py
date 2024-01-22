import os

import PyPDF2
import numpy as np
from tika import parser # pip install tika
from sentence_transformers import SentenceTransformer
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



def generate(pdf_path):
    embdedded_pdf = {}
    pdf_data = walk_pdf_dir(pdf_path)
    for key,value in pdf_data.items():
        embeddings = model.encode(value)  # Generate embeddings
        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        embdedded_pdf[key] = embeddings_list

    # Convert embeddings to a list (if they are in NumPy array format)


    print(embdedded_pdf.keys())
    return None


generate('/Users/kazirahman/PycharmProjects/PDFAPI/testing_data')