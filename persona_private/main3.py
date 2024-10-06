from pymongo import MongoClient

# Create a connection to the MongoDB server
client = MongoClient('mongodb://localhost:27017/')

# Connect to the database
db = client['docs']

# Connect to the collection
collection = db['jsons']

results = collection.aggregate([
    { "$unwind": "$defense" },
    {
        "$addFields": {
            "defense.NatSecString": {
                "$toDouble": {
                    "$ifNull": [
                        {
                            "$arrayElemAt": [
                                {
                                    "$filter": {
                                        "input": { "$split": ["$defense.NatSecString", " "] },
                                        "as": "element",
                                        "cond": { "$regexMatch": { "input": "$$element", "regex": "^\\d+(\\.\\d+)?$" } }
                                    }
                                },
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
    },
    {
        "$group": {
            "_id": "$CountryName",
            "defense": { "$avg": "$defense.NatSecString" }
        }
    },
    { "$sort": { "defense": -1 } },
    { "$limit": 5 },
    { "$project": { "_id": 0, "CountryName": "$_id", "defense": 1 } }
])

# Print the results
print("The top 5 countries for defense are:")
for i, result in enumerate(results):
    print(f"{i+1}. {result['CountryName']} ({result['defense']})")
