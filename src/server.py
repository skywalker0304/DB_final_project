from pymongo import MongoClient
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
            collection_name = dbname["train"]

            # Load and preprocess the data
            data = pd.DataFrame(collection_name.find())
            preprocessed_data = preprocess_data(data, preprocessing_methods)

            # Specify MindsDB model details
            model_details = {
                'name': 'my_model',
                'predict': predict_column,
                'data': preprocessed_data,
                'learn': {
                    'from_data': preprocessed_data,
                    'to_predict': predict_column,
                    'model': model,
                    'model_settings': {
                        'hidden_layers': hidden_layers,
                        'epochs': epochs,
                        'batch_size': batch_size
                    }
                }
            }

            # Make predictions with new data
            
            preprocessed_new_data = data
            # Access the predictions
            predictions = preprocessed_new_data["Danceability"]

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
