from pymongo import MongoClient
from preprocess import preprocess_data
import numpy as np
import pandas as pd
import argparse
import socket
import pickle
import threading

def handle_client(conn, addr, request):
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

            #connect to mongodb
            client = MongoClient('localhost', 27017)
            dbname = client['ML_final']
            train_collection = dbname["train"]
            test_collection = dbname["test"]

            # Load and preprocess the data
            data = pd.DataFrame(train_collection.find())
            preprocessed_data = preprocess_data(data, preprocessing_methods, data.columns)
            print(data[:10])
 
            # extract X and Y
            X = preprocessed_data.drop([predict_column], axis=1)
            Y = preprocessed_data[predict_column].copy()

            #train model
            pinv_X = np.linalg.pinv(X.to_numpy())
            pinv_Y = np.array(Y)
            w_LIN = np.matmul(pinv_X, pinv_Y)
            print("Train complete")

            # Make predictions with new data
            data = pd.DataFrame(test_collection.find())
            preprocessed_new_data = preprocess_data(data, preprocessing_methods, data.columns)

            print(data[:10])
            Test_Y = np.matmul(preprocessed_new_data.to_numpy(), w_LIN)
            Test_Y = list(map(round, Test_Y))

            # Access the predictions
            predictions = Test_Y

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
