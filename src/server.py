from pymongo import MongoClient
import numpy as np
import pandas as pd
import argparse
import socket
import pickle
import threading


def preprocess_data(data, preprocessing_methods):
    # Apply data preprocessing methods
    preprocessed_data = data.copy()

    if 'remove_duplicates' in preprocessing_methods:
        preprocessed_data.drop_duplicates(inplace=True)

    if 'handle_missing_values' in preprocessing_methods:
        preprocessed_data.fillna(method='ffill', inplace=True)

    if 'feature_scaling' in preprocessing_methods:
        preprocessed_data['feature'] = (preprocessed_data['feature'] - preprocessed_data['feature'].mean()) / preprocessed_data['feature'].std()

    return preprocessed_data


def train_model(preprocessed_data, preprocessed_test_data, predict_column):
    # extract X and Y
    X = preprocessed_data.drop([predict_column], axis=1)
    Y = preprocessed_data[predict_column].copy()

    # train model
    pinv_X = np.linalg.pinv(X.to_numpy())
    pinv_Y = np.array(Y)
    w_LIN = np.matmul(pinv_X, pinv_Y)

    # Make predictions with new data
    Test_Y = np.matmul(preprocessed_test_data.to_numpy(), w_LIN)
    Test_Y = list(map(round, Test_Y))
    return Test_Y


def handle_client(conn, addr, request):
    # Check if the database and collection exists
    database = request['database']
    train_collection = request['train_collection']
    test_collection = request['test_collection']
    if database not in db_client.list_database_names():
        response = {'response_type': 'error', 'message': 'Requested database does not exist'}
        unexpected_response(conn, response)
        return

    db = db_client['database']
    if train_collection not in db.list_collection_names():
        response = {'response_type': 'error', 'message': 'Requested train collection does not exist'}
        unexpected_response(conn, response)
        return

    if test_collection not in db.list_collection_names():
        response = {'response_type': 'error', 'message': 'Requested test collection does not exist'}
        unexpected_response(conn, response)
        return

    # Perform operations based on the request type
    if request['request_type'] == 'predict':
        try:
            # User-defined request details
            preprocessing_methods = request['preprocessing_methods']
            model = request['model']
            predict_column = request['predict_column']
            hidden_layers = request['hidden_layers']
            epochs = request['epochs']
            batch_size = request['batch_size']

            # Load and preprocess the data
            train_data = pd.DataFrame(train_collection.find())
            preprocessed_train_data = preprocess_data(train_data, preprocessing_methods)

            test_data = pd.DataFrame(test_collection.find())
            preprocessed_test_data = preprocess_data(test_data, preprocessing_methods)

            # # Specify MindsDB model details
            # model_details = {
            #     'name': 'my_model',
            #     'predict': predict_column,
            #     'data': preprocessed_train_data,
            #     'learn': {
            #         'from_data': preprocessed_train_data,
            #         'to_predict': predict_column,
            #         'model': model,
            #         'model_settings': {
            #             'hidden_layers': hidden_layers,
            #             'epochs': epochs,
            #             'batch_size': batch_size
            #         }
            #     }
            }

            # Access the predictions
            predictions = train_model(preprocessed_train_data, preprocessed_test_data, predict_column)

            # Send back the predictions to the client
            response = {'response_type': 'predictions', 'data': predictions}
        except Exception as e:
            response = {'response_type': 'error', 'message': str(e)}

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
