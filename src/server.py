from pymongo import MongoClient
from preprocess import preprocess_data
from chart import *
import numpy as np
import pandas as pd
import argparse
import socket
import pickle
import threading
import time
import os


def train_model(preprocessed_data, preprocessed_test_data, predict_column):
    # extract X and Y
    X = preprocessed_data.drop([predict_column], axis=1)
    Y = preprocessed_data[predict_column].copy()

    # train model
    pinv_X = np.linalg.pinv(X.to_numpy())
    pinv_Y = np.array(Y)
    w_LIN = np.matmul(pinv_X, pinv_Y)
    print("Train complete")

    # Make predictions with new data
    Train_Y = np.matmul(X.to_numpy(), w_LIN)
    Test_Y = np.matmul(preprocessed_test_data.to_numpy(), w_LIN)
    Train_Y = list(map(round, Train_Y))
    Test_Y = list(map(round, Test_Y))
    return (Test_Y, Train_Y, Y)


def get_content(filename, substring):
    with open(filename, 'r') as file:
        content = file.read()
        substrings = content.split(substring)

    if len(substrings) >= 3:
        return substring + substring.join(substrings[2:-1])
    else:
        return ""


def handle_client(conn, addr, request):
    # Check if the database and collection exists
    database = request['database']
    if database not in db_client.list_database_names():
        response = {'response_type': 'error', 'message': 'Requested database does not exist'}
        unexpected_response(conn, response)
        return

    db = db_client[database]
    # Perform operations based on the request type
    if request['request_type'] == 'machine_learning':
        train_collection = request['train_collection']
        test_collection = request['test_collection']
        if train_collection not in db.list_collection_names():
            response = {'response_type': 'error', 'message': 'Requested train collection does not exist'}
            unexpected_response(conn, response)
            return

        if test_collection not in db.list_collection_names():
            response = {'response_type': 'error', 'message': 'Requested test collection does not exist'}
            unexpected_response(conn, response)
            return

        train_collection = db[train_collection]
        test_collection = db[test_collection]

        try:
            # User-defined request details
            preprocessing_methods = request['preprocessing_methods']
            model = request['model']
            predict_column = request['predict_column']

            # Load and preprocess the data
            train_data = pd.DataFrame(train_collection.find({}))
            preprocessed_train_data = preprocess_data(train_data, preprocessing_methods, train_data.columns)

            test_data = pd.DataFrame(test_collection.find({}))
            preprocessed_test_data = preprocess_data(test_data, preprocessing_methods, test_data.columns)

            # Access the predictions
            train_result = train_model(preprocessed_train_data, preprocessed_test_data, predict_column)
            predictions = train_result[0]
            predict_train_y = train_result[1]
            train_y = train_result[2]

            # Send back the predictions to the client
            response = {'response_type': 'predictions', 'data': np.mean(predictions)}
        except Exception as e:
            response = {'response_type': 'error', 'message': str(e)}

        # Send the response to the client
        conn.send(pickle.dumps(response))

        # Send picture
        # heatmap_fig = heatmap(predict_train_y, train_y)
        # conn.sendall(pickle.dumps(heatmap_fig))
        # print("send picture 1 successfully")

        # block the server from consecutive sending two pictures
        barchart_fig = barChart(predictions)
        conn.sendall(pickle.dumps(barchart_fig))
        print("send picture 1 successfully")

        # Close the connection
        print("Close connection")
        conn.close()

    elif request['request_type'] == 'mongodb_operation':
        user_query = request['db_operation']
        file_path = "tmp.js"
        js_code = f"use {database};\n{user_query};"
        print(f"content of file: \n{js_code}")
        with open(file_path, "w") as js_file:
            js_file.write(js_code)

        os.system("mongosh < tmp.js > tmp.out")
        os.system("rm tmp.js")
        result = get_content("tmp.out", database)
        os.system("rm tmp.out")
        print(result)
        response = {'response_type': 'mongodb_operation', 'data': result}
        # Send the response to the client
        conn.send(pickle.dumps(response))

        # Close the connection
        print("Close connection")
        conn.close()

    elif request['request_type'] == 'data_exploration':
        collection = request['collection']
        if collection not in db.list_collection_names():
            response = {'response_type': 'error', 'message': 'Requested collection does not exist'}
            unexpected_response(conn, response)
            return

        collection = db[collection]
        # Load data
        data = pd.DataFrame(collection.find({}))
        response = {'response_type': 'data_exploration', 'message': 'Sending'}

        # Send the response to the client
        conn.send(pickle.dumps(response))

        # Send picture
        if 'show_missing_values' in request['data_exploration']:
            missing_values_fig = missingMap(data)
            conn.sendall(pickle.dumps(missing_values_fig))
            print("send picture missing values successfully")

        if 'show_feature_distributions' in request['data_exploration']:
            numeric_feature_boxchart_fig = numeric_feature_boxchart(data)
            conn.sendall(pickle.dumps(numeric_feature_boxchart_fig))
            print("send picture feature distributions successfully")

        # Close the connection
        print("Close connection")
        conn.close()

    elif request['request_type'] == 'custom_operation':
        try:
            # Perform custom operations based on the request
            # You can implement your custom logic here
            # For example, you can run additional analyses, modify the dataset, etc.

            # Send back the response to the client
            response = {'response_type': 'custom_response', 'message': 'Custom operation completed successfully'}
        except Exception as e:
            response = {'response_type': 'error', 'message': str(e)}

        # Send the response to the client
        conn.send(pickle.dumps(response))

        # Close the connection
        print("Close connection")
        conn.close()


def unexpected_response(conn, response):
    conn.send(pickle.dumps(response))
    conn.close()


# Parse command line arguments
parser = argparse.ArgumentParser(description='MindsDB Preprocessing and Model')
parser.add_argument('--port', type=int, help='Specify the port number', default=8080)
args = parser.parse_args()

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to a specific port
server_socket.bind(('localhost', args.port))
print("Successful binding")

# Listen for incoming connections
server_socket.listen(1)
print(f"Server is listening on port {args.port}...")

# connect to mongodb
db_client = MongoClient('localhost', 27017)

while True:
    # Accept a new connection
    conn, addr = server_socket.accept()
    print(f"Connected to client: {addr}")

    # Receive the request from the client
    data = conn.recv(1024)
    request = pickle.loads(data)
    print(f"request from client is {request}")

    # Handle the client's request in a separate thread
    client_thread = threading.Thread(target=handle_client, args=(conn, addr, request))
    client_thread.start()
