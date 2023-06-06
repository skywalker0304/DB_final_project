import socket
import pickle

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

ip_addr = input("Enter the ip address you want to connect")
port = input("Enter the port you want to connect")

# Connect to the server
client_socket.connect((ip_addr, port))
print(f"Connected to server {ip_addr} on {port} port")

while True:
    # Get user's request
    db = input("Enter the database you want to use")
    collection = input("Enter the collection you want to use")
    user_input = input("Enter your request (predict, custom_operation): ")        

    if user_input == 'predict':
        preprocessing_methods_list = []
        preprocessing_methods_string = input(
            "What kind of preprocessing methods do you want to use?\n"
            "If you want to remove duplicates, please enter 1, \n"
            "If you want to handle missing values, please enter 2.\n"
            "If you want to do feature scaling, please enter 3\n"
            "It is also OK if you want to choose multiple options, "
            "e.g. 123 means do all three, and 12 means do the first two\n"
            "If you don't want to use any, please enter None"
        )

        possible_list = ["1", "2", "3", "12", "13", "21", "23", "31", "32", "123", "132", "213", "231", "312", "321", "None"]
        if preprocessing_methods_string not in possible_list:
            print("Invalid request")
            continue

        for ch in preprocessing_methods_string:
            if ch == "1":
                preprocessing_methods_list.append["remove_duplicates"]
            if ch == "2":
                preprocessing_methods_list.append["handle_missing_values"]
            if ch == "3":
                preprocessing_methods_list.append["feature_scaling"]

        predict_column = input("Enter the column you want to predict")

        # Prepare the request for prediction
        request = {
            'database': db,
            'collection': collection,
            'request_type': 'predict',
            'preprocessing_methods': preprocessing_methods_list,
            'model': 'lightwood',
            'predict_column': predict_column,
            'hidden_layers': [32, 32],
            'epochs': 1,
            'batch_size': 32
        }
    elif user_input == 'custom_operation':
        # Prepare the request for custom operation
        request = {
            'request_type': 'custom_operation'
        }
    else:
        print("Invalid request")
        continue

    # Send the request to the server
    client_socket.send(pickle.dumps(request))

    # Receive the response from the server
    data = client_socket.recv(1024)
    response = pickle.loads(data)

    # Process the response
    if response['response_type'] == 'predictions':
        print(f"Predictions: {response['data']}")
    elif response['response_type'] == 'custom_response':
        print("Custom operation completed successfully")
    elif response['response_type'] == 'error':
        print("An error occurred:", response['message'])
    else:
        print("Unknown response type")

    # Ask if the user wants to continue
    user_input = input("Do you want to continue (y/n)? ")
    if user_input.lower() != 'y':
        break

# Close the connection
client_socket.close()
