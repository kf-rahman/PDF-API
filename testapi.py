import requests

url = "http://127.0.0.1:8000/get-embeddings/"
data = {"data": "test"}
response = requests.post(f"http://127.0.0.1:8000/get-embeddings/{data[data]}")

if response.status_code == 200:
    print(response.json())
else:
    print(f"Request failed with status code: {response.status_code}")
    print(response.text)
