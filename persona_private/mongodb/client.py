from pymongo import MongoClient


def create_client(port=27017):
    '''Create MongoDB client

    Args:
        port (int): port number
    
    Returns:
        MongoClient: A MongoDB client instance
    '''
    client = MongoClient(f"mongodb://localhost:{port}/")
    return client


def database_exists(client, db_name):
    '''Check if a MongoDB database exists

    Args:
        client (MongoClient): MongoDB client instance
        db_name (str): Name of the database

    Returns:
        bool: True if the database exists, False otherwise
    '''
    return db_name in client.list_database_names()


def table_exists(db, table_name):
    '''Check if a MongoDB collection exists within a database

    Args:
        db (Database): MongoDB database object
        table_name (str): Name of the collection (table)

    Returns:
        bool: True if the collection exists, False otherwise
    '''
    return table_name in db.list_collection_names()


def create_database_table(db_name="docs", table_name="jsons"):
    '''Create MongoDB database and table if they don't exist

    Args:
        db_name (str): Name of the database
        table_name (str): Name of the table (collection)

    Returns:
        Collection: MongoDB collection object
    '''
    client = create_client()
    
    # Check if database exists
    if not database_exists(client, db_name):
        print(f"Database '{db_name}' does not exist. It will be created.")
    
    db = client[db_name]  # Get the database
    
    # Check if table (collection) exists
    if not table_exists(db, table_name):
        print(f"Collection '{table_name}' does not exist in database '{db_name}'. It will be created.")
    
    collection = db[table_name]  # Get or create the collection
    return f"Database {db_name} and collection {table_name} are created!!"


def insert_data(data, table_name="jsons", db_name="docs"):
    '''Insert a document into a MongoDB collection

    Args:
        data (dict): The document to be inserted
        table_name (str): Name of the collection
        db_name (str): Name of the database

    Returns:
        InsertOneResult: The result of the insertion
    '''
    # Convert data to json records
    records_l = []
    for k, v in data.items():
        data[k]["CountryName"] = k
        records_l.append(data[k])


    # Get or create the collection (table)
    collection = create_database_table(db_name, table_name)
    
    # Insert the document into the collection
    result = collection.insert_many(records_l)
    print(f"{len(records_l)} records are inserted into {db_name} database {table_name} table")
    return result


