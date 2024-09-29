import json
from persona_private.mongodb.client import insert_data

# Open and read the JSON file
with open('ai/files/file-2024.08.20.13.28.json', 'r') as file:
    data = json.load(file)


status = insert_data(data=data)
