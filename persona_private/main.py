import json
from persona_private.mongodb.client import insert_data

# Open and read the JSON file
with open('ai/files/data_cleaned_05.10.24.json', 'r') as file:
    data = json.load(file)


status = insert_data(data=data)
