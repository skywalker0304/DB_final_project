from pymongo import MongoClient
import mindsdb
import sys


def start_mindsdb():
    mdb = mindsdb.MindsDB()
    mdb.start()


def connect(db_name):
    client = MongoClient("mongodb://127.0.0.1:27017")
    print("Connection Successful")
    
    client.close()
