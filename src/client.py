import socket
import pickle
from PIL import Image
import matplotlib.pyplot as plt

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

ip_addr = input("Enter the ip address you want to connect: ")
port = int(input("Enter the port you want to connect: "))

# Connect to the server
client_socket.connect((ip_addr, port))
print(f"Connected to server {ip_addr} on {port} port")

while True:
    # Get user's request
    db = input("Enter the database you want to use: ")
    train_collection = input("Enter the collection you want to train on: ")
    test_collection = input("Enter the collection you want to test on: ")
    user_input = input("Enter your request (predict, custom_operation): ")        

    if user_input == 'predict':
        preprocessing_methods_list = []
        preprocessing_methods_string = input(
            "Do you want to remove duplicates? (y/n) \n"
        )
        if preprocessing_methods_string == "y":
                preprocessing_methods_list.append("remove_duplicates")

        preprocessing_methods_string = input(
            "If you want to apply standard scaling, please enter 1\n"
            "If you want to apply min_max scaling, please enter 2\n"
            "If you don't want to use any, please enter None\n"
        )

        if (preprocessing_methods_string == "1"):
            preprocessing_methods_list.append("standard_scaling")
        elif (preprocessing_methods_string == "2"):
            preprocessing_methods_list.append("min_max_scaling")

        
        preprocessing_methods_string = input(
            "If you want to impute mean, please enter 1\n"
            "If you want to impute median, please enter 2\n"
            "If you want to impute frequency, please enter 3\n"
            "If you don't want to use any, please enter None\n"
        )

        if (preprocessing_methods_string == "1"):
            preprocessing_methods_list.append("impute_mean")
        elif (preprocessing_methods_string == "2"):
            preprocessing_methods_list.append("impute_median")
        elif (preprocessing_methods_string == "3"):
            preprocessing_methods_list.append("impute_frequency")
        


        predict_column = input("Enter the column you want to predict: ")

        # Prepare the request for prediction
        request = {
            'database': db,
            'train_collection': train_collection,
            'test_collection': test_collection,
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

    # Show picture
    # image_path = input("Enter the picture file name you want to saved: ")
    for i in range(1):
        received_data = b""
        print("waiting for server")
        while True:
            data = client_socket.recv(4096)
            if not data:
                break
            received_data += data

        print("received picture, showing picture")
        received_fig = pickle.loads(received_data)
        plt.figure(received_fig.number)
        plt.show()
        print(f"show {i + 1} figure at client side")


    # with open(image_path, 'wb') as file:
    #    file.write(received_data)
    # image = Image.open(image_path)
    # plt.imshow(image)
    # plt.show()

    # Ask if the user wants to continue
    user_input = input("Do you want to continue (y/n)? ")
    if user_input.lower() != 'y':
        break

# Close the connection
client_socket.close()
