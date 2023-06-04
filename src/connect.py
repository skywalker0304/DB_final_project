from pymongo import MongoClient
import sys

def input_parsing():
    if len(sys.argv) != 3:
        print("expected 2 additional arguments")
        print(f"only get {len(sys.argv)}")
        exit(0)

    db_name = sys.argv[1]
    feature = sys.argv[2]
    connect(db_name)


def connect(db_name):
    client = MongoClient("mongodb://127.0.0.1:27017")
    print("Connection Successful")
    
    client.close()
