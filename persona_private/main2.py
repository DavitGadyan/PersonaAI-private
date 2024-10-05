from pymongo import MongoClient

# Create a connection to the MongoDB server
client = MongoClient('mongodb://localhost:27017/')

# Connect to the database
db = client['docs']

# Connect to the collection
collection = db['jsons']

# Execute the query to extract the top 3 countries with the highest 'moi' value
results = collection.aggregate([
    { "$unwind": "$moi" },
    {
        "$addFields": {
            "moi": {
                "$cond": {
                    "if": { "$eq": ["$moi.FileAttachmentString", ""] },
                    "then": "$$REMOVE",
                    "else": {
                        "$toInt": {
                            "$ifNull": [
                                {
                                    "$arrayElemAt": [
                                        { "$split": ["$moi.FileAttachmentString", " "] },
                                        # index of the numeric value in the split array
                                        0  # replace 0 with the correct index
                                    ]
                                },
                                # default value if the split array doesn't contain a numeric value at the specified index
                                0  # replace 0 with an appropriate default value
                            ]
                        }
                    }
                }
            }
        }
    },
    {
        "$group": {
            "_id": "$CountryName",
            "moi": { "$avg": "$moi" }
        }
    },
    { "$sort": { "moi": -1 } },
    { "$limit": 3 },
    {
        "$project": {
            "_id": 0,
            "CountryName": "$_id",
            "excludedKeys": { "$objectToArray": "$$ROOT" },
        }
    },
    {
        "$project": {
            "CountryName": 1,
            "excludedKeys": {
                "$filter": {
                    "input": "$excludedKeys",
                    "as": "key",
                    "cond": { "$not": { "$regexMatch": { "input": "$$key.k", "regex": "FileAttachment" } } }
                }
            }
        }
    },
    {
        "$project": {
            "CountryName": 1,
            "excludedKeys": { "$arrayToObject": "$excludedKeys" }
        }
    }
])






# Print the results
print("The top 3 countries with the highest 'moi' value are:")
for i, result in enumerate(results):
    print(f"{i+1}. {result['CountryName']} ({result['moi']})")
