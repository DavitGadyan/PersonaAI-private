'''FAST API to serve REST framework for analyzing real estate listings
'''
import os
import sys
import time
from typing import Optional, Literal
from pydantic import BaseModel

from fastapi import FastAPI
import uvicorn

from persona_private.ai.chroma_client import get_retriever, get_retriever2
from persona_private.ai.agent import rag, mistral7b_llm, codestral_llm
from persona_private.ai.chroma_client import load_docs, load_json, doc2chroma, save_docs2_chroma, get_retriever, load_json2, get_retriever3
from persona_private.mongodb.client import insert_data
import dotenv

dotenv.load_dotenv()


llm = mistral7b_llm()
code_llm = codestral_llm()
app = FastAPI()

class QueryParams(BaseModel):
    question: str

class JsonFile(BaseModel):
    file: dict
    filename: str



@app.get('/')
async def home():
    return {"message": "RAG AI Private Persona FAST API is running!!"}

@app.post('/analyze')
def process(query_params: QueryParams):
    # try:
    question = query_params.model_dump()['question']
    retriever = get_retriever3(question=question, persist_directory="docs_chromadb")
    docs = retriever.get_relevant_documents(question)
    print(len(docs))
    print(docs)
    time.sleep(10)
    answer = rag(retriever, llm, question=question)
    return {"answer": answer}
    # except Exception as e:
    #     print('Error>>>', e)
    #     return 500, {"message": f"Following error occured>> {e}"}


@app.post('/analyze_general')
def process_general(query_params: QueryParams):

    question = query_params.model_dump()['question']
    
    prompt = '''
    Generate python code and execute it return result using PyMongo package and json schema to answer question below. Convert all string values to double

    Question:
    {question}

    MongoDB Schema:
    {
        "docs": {
            "jsons": {
                "count": 249,
                "object": {
                    "CountryName": {
                        "count": 249,
                        "prop_in_object": 1.0,
                        "type": "string",
                        "types_count": {
                            "string": 249
                        }
                    },
                    "_id": {
                        "count": 249,
                        "prop_in_object": 1.0,
                        "type": "oid",
                        "types_count": {
                            "oid": 249
                        }
                    },
                    "country": {
                        "array_type": "OBJECT",
                        "array_types_count": {
                            "OBJECT": 249
                        },
                        "count": 249,
                        "object": {
                            "TotalString": {
                                "count": 249,
                                "prop_in_object": 1.0,
                                "type": "string",
                                "types_count": {
                                    "string": 249
                                }
                            }
                        },
                        "prop_in_object": 1.0,
                        "type": "ARRAY",
                        "types_count": {
                            "ARRAY": 249
                        }
                    },
                    "defense": {
                        "array_type": "mixed_scalar_object",
                        "array_types_count": {
                            "OBJECT": 249
                        },
                        "count": 249,
                        "object": {
                            "NatSecString": {
                                "count": 249,
                                "prop_in_object": 1.0,
                                "type": "string",
                                "types_count": {
                                    "string": 249
                                }
                            }
                        },
                        "prop_in_object": 1.0,
                        "type": "ARRAY",
                        "types_count": {
                            "ARRAY": 248,
                            "string": 1
                        }
                    },
                    "energy": {
                        "array_type": "OBJECT",
                        "array_types_count": {
                            "OBJECT": 249
                        },
                        "count": 249,
                        "object": {
                            "ExplString": {
                                "count": 249,
                                "prop_in_object": 1.0,
                                "type": "string",
                                "types_count": {
                                    "string": 249
                                }
                            },
                            "InvestString": {
                                "count": 249,
                                "prop_in_object": 1.0,
                                "type": "string",
                                "types_count": {
                                    "string": 249
                                }
                            },
                            "LNGString": {
                                "count": 249,
                                "prop_in_object": 1.0,
                                "type": "string",
                                "types_count": {
                                    "string": 249
                                }
                            },
                            "PrdtString": {
                                "count": 249,
                                "prop_in_object": 1.0,
                                "type": "string",
                                "types_count": {
                                    "string": 249
                                }
                            }
                        },
                        "prop_in_object": 1.0,
                        "type": "ARRAY",
                        "types_count": {
                            "ARRAY": 249
                        }
                    },
                    "ginfo": {
                        "array_type": "mixed_scalar_object",
                        "array_types_count": {
                            "OBJECT": 13
                        },
                        "count": 249,
                        "object": {
                            "AlertsString": {
                                "count": 13,
                                "prop_in_object": 0.0522,
                                "type": "string",
                                "types_count": {
                                    "string": 13
                                }
                            },
                            "ChiefOfStateString": {
                                "count": 13,
                                "prop_in_object": 0.0522,
                                "type": "string",
                                "types_count": {
                                    "string": 13
                                }
                            },
                            "DoingBusinessRankingString": {
                                "count": 13,
                                "prop_in_object": 0.0522,
                                "type": "string",
                                "types_count": {
                                    "string": 13
                                }
                            },
                            "FitchRatingsSovereignString": {
                                "count": 13,
                                "prop_in_object": 0.0522,
                                "type": "string",
                                "types_count": {
                                    "string": 13
                                }
                            },
                            "GDPPerCapitaString": {
                                "count": 13,
                                "prop_in_object": 0.0522,
                                "type": "string",
                                "types_count": {
                                    "string": 13
                                }
                            },
                            "GDPString": {
                                "count": 13,
                                "prop_in_object": 0.0522,
                                "type": "string",
                                "types_count": {
                                    "string": 13
                                }
                            },
                            "GovernmentElectionsString": {
                                "count": 13,
                                "prop_in_object": 0.0522,
                                "type": "string",
                                "types_count": {
                                    "string": 13
                                }
                            },
                            "HeadOfGovernmentString": {
                                "count": 13,
                                "prop_in_object": 0.0522,
                                "type": "string",
                                "types_count": {
                                    "string": 13
                                }
                            },
                            "InflationString": {
                                "count": 13,
                                "prop_in_object": 0.0522,
                                "type": "string",
                                "types_count": {
                                    "string": 13
                                }
                            },
                            "PopulationString": {
                                "count": 13,
                                "prop_in_object": 0.0522,
                                "type": "string",
                                "types_count": {
                                    "string": 13
                                }
                            },
                            "TitleString": {
                                "count": 13,
                                "prop_in_object": 0.0522,
                                "type": "string",
                                "types_count": {
                                    "string": 13
                                }
                            },
                            "Top3ImportsToQatarString": {
                                "count": 13,
                                "prop_in_object": 0.0522,
                                "type": "string",
                                "types_count": {
                                    "string": 13
                                }
                            },
                            "TradeSurplusOrDeficitString": {
                                "count": 13,
                                "prop_in_object": 0.0522,
                                "type": "string",
                                "types_count": {
                                    "string": 13
                                }
                            },
                            "UnemploymentString": {
                                "count": 13,
                                "prop_in_object": 0.0522,
                                "type": "string",
                                "types_count": {
                                    "string": 13
                                }
                            }
                        },
                        "prop_in_object": 1.0,
                        "type": "ARRAY",
                        "types_count": {
                            "ARRAY": 12,
                            "string": 237
                        }
                    },
                    "moci": {
                        "array_type": "OBJECT",
                        "array_types_count": {
                            "OBJECT": 249
                        },
                        "count": 249,
                        "object": {
                            "EssTrdString": {
                                "count": 249,
                                "prop_in_object": 1.0,
                                "type": "string",
                                "types_count": {
                                    "string": 249
                                }
                            },
                            "TrdFdiString": {
                                "count": 249,
                                "prop_in_object": 1.0,
                                "type": "string",
                                "types_count": {
                                    "string": 249
                                }
                            }
                        },
                        "prop_in_object": 1.0,
                        "type": "ARRAY",
                        "types_count": {
                            "ARRAY": 249
                        }
                    },
                    "mofa": {
                        "array_type": "mixed_scalar_object",
                        "array_types_count": {
                            "OBJECT": 249
                        },
                        "count": 249,
                        "object": {
                            "EsgAllyString": {
                                "count": 249,
                                "prop_in_object": 1.0,
                                "type": "string",
                                "types_count": {
                                    "string": 249
                                }
                            },
                            "MultLoyString": {
                                "count": 249,
                                "prop_in_object": 1.0,
                                "type": "string",
                                "types_count": {
                                    "string": 249
                                }
                            }
                        },
                        "prop_in_object": 1.0,
                        "type": "ARRAY",
                        "types_count": {
                            "ARRAY": 248,
                            "string": 1
                        }
                    },
                    "moi": {
                        "array_type": "OBJECT",
                        "array_types_count": {
                            "OBJECT": 249
                        },
                        "count": 249,
                        "object": {
                            "QWorkString": {
                                "count": 249,
                                "prop_in_object": 1.0,
                                "type": "string",
                                "types_count": {
                                    "string": 249
                                }
                            }
                        },
                        "prop_in_object": 1.0,
                        "type": "ARRAY",
                        "types_count": {
                            "ARRAY": 249
                        }
                    },
                    "outStandings": {
                        "array_type": "mixed_scalar_object",
                        "array_types_count": {
                            "OBJECT": 22
                        },
                        "count": 249,
                        "object": {
                            "DescriptionString": {
                                "count": 22,
                                "prop_in_object": 0.0884,
                                "type": "string",
                                "types_count": {
                                    "string": 22
                                }
                            },
                            "TitleString": {
                                "count": 22,
                                "prop_in_object": 0.0884,
                                "type": "string",
                                "types_count": {
                                    "string": 22
                                }
                            }
                        },
                        "prop_in_object": 1.0,
                        "type": "ARRAY",
                        "types_count": {
                            "ARRAY": 10,
                            "string": 239
                        }
                    },
                    "qffd": {
                        "array_type": "OBJECT",
                        "array_types_count": {
                            "OBJECT": 249
                        },
                        "count": 249,
                        "object": {
                            "AidRecipString": {
                                "count": 249,
                                "prop_in_object": 1.0,
                                "type": "string",
                                "types_count": {
                                    "string": 249
                                }
                            }
                        },
                        "prop_in_object": 1.0,
                        "type": "ARRAY",
                        "types_count": {
                            "ARRAY": 249
                        }
                    },
                    "qia": {
                        "array_type": "OBJECT",
                        "array_types_count": {
                            "OBJECT": 249
                        },
                        "count": 249,
                        "object": {
                            "QIACurString": {
                                "count": 249,
                                "prop_in_object": 1.0,
                                "type": "string",
                                "types_count": {
                                    "string": 249
                                }
                            },
                            "QIAPtosString": {
                                "count": 249,
                                "prop_in_object": 1.0,
                                "type": "string",
                                "types_count": {
                                    "string": 249
                                }
                            }
                        },
                        "prop_in_object": 1.0,
                        "type": "ARRAY",
                        "types_count": {
                            "ARRAY": 249
                        }
                    },
                    "talkingPoints": {
                        "array_type": "mixed_scalar_object",
                        "array_types_count": {
                            "OBJECT": 22
                        },
                        "count": 249,
                        "object": {
                            "DetailsString": {
                                "count": 22,
                                "prop_in_object": 0.0884,
                                "type": "string",
                                "types_count": {
                                    "string": 22
                                }
                            },
                            "SubTitleString": {
                                "count": 22,
                                "prop_in_object": 0.0884,
                                "type": "string",
                                "types_count": {
                                    "string": 22
                                }
                            },
                            "TitleString": {
                                "count": 22,
                                "prop_in_object": 0.0884,
                                "type": "string",
                                "types_count": {
                                    "string": 22
                                }
                            }
                        },
                        "prop_in_object": 1.0,
                        "type": "ARRAY",
                        "types_count": {
                            "ARRAY": 12,
                            "string": 237
                        }
                    },
                    "trips": {
                        "array_type": "mixed_scalar_object",
                        "array_types_count": {
                            "OBJECT": 92
                        },
                        "count": 249,
                        "object": {
                            "HeadOfDelegation": {
                                "count": 92,
                                "object": {
                                    "JobTitleString": {
                                        "count": 80,
                                        "prop_in_object": 0.8696,
                                        "type": "string",
                                        "types_count": {
                                            "string": 80
                                        }
                                    },
                                    "TitleString": {
                                        "count": 80,
                                        "prop_in_object": 0.8696,
                                        "type": "string",
                                        "types_count": {
                                            "string": 80
                                        }
                                    }
                                },
                                "prop_in_object": 0.3695,
                                "type": "mixed_scalar_object",
                                "types_count": {
                                    "OBJECT": 80,
                                    "string": 12
                                }
                            },
                            "Members": {
                                "array_type": "OBJECT",
                                "array_types_count": {
                                    "OBJECT": 29,
                                    "null": 67
                                },
                                "count": 92,
                                "object": {
                                    "JobTitleString": {
                                        "count": 29,
                                        "prop_in_object": 0.3152,
                                        "type": "string",
                                        "types_count": {
                                            "string": 29
                                        }
                                    },
                                    "TitleString": {
                                        "count": 29,
                                        "prop_in_object": 0.3152,
                                        "type": "string",
                                        "types_count": {
                                            "string": 29
                                        }
                                    }
                                },
                                "prop_in_object": 0.3695,
                                "type": "ARRAY",
                                "types_count": {
                                    "ARRAY": 92
                                }
                            },
                            "TitleString": {
                                "count": 92,
                                "prop_in_object": 0.3695,
                                "type": "string",
                                "types_count": {
                                    "string": 92
                                }
                            }
                        },
                        "prop_in_object": 1.0,
                        "type": "ARRAY",
                        "types_count": {
                            "ARRAY": 4,
                            "string": 245
                        }
                    }
                }
            }
        }
    }

    ## Example:

    Question:
    
    What are top 3 countries by moi?

    Answer:
    from pymongo import MongoClient

    # Create a connection to the MongoDB server
    client = MongoClient('mongodb://localhost:27017/')

    # Connect to the database
    db = client['docs']

    # Connect to the collection
    collection = db['jsons']

    results = collection.aggregate([
        { "$unwind": "$moi" },
        {
            "$addFields": {
                "moi": {
                    "$cond": {
                        "if": { "$eq": ["$moi.QWorkString", ""] },
                        "then": "$$REMOVE",
                        "else": {
                            "$toDouble": {
                                "$ifNull": [
                                    {
                                        "$arrayElemAt": [
                                            {
                                                "$filter": {
                                                    "input": { "$split": ["$moi.QWorkString", " "] },
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
        { "$project": { "_id": 0, "CountryName": "$_id", "moi": 1 } }
    ])


    # Print the results
    print("The top 3 countries with the highest 'moi' value are:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['CountryName']} ({result['moi']})")


        
    '''
    answer = llm.invoke(prompt)

    return {"answer": answer}

@app.post('/embeed')
def embeed(json_file: JsonFile):
    # print("json_file>>", json_file)
    # try:
    json_obj = json_file.model_dump()['file']
    filename = json_file.model_dump()['filename']
    # print('json_file>>', json_file)

    # insert into MongoDB
    status = insert_data(data=json_obj)
    print("status>>", status)
    print(type(json_obj))
    docs = load_json2(json_obj=json_obj, filename=filename)

    retriever = doc2chroma(docs=docs, persist_directory="docs_chromadb")
    print("retriever>>", retriever)
    return {"message": f"{filename} was save to Chroma!!!"}
    # except Exception as e:
    #     print('Error>>>', e)
    #     return 500, {"message": f"Following error occured>> {e}"}
    


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000,
    
    reload_includes=["docs_chromadb/*.sqlite3"])