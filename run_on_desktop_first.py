# evaluate the deep model on the test dataset
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import socket
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#IP Parameters Server
TCP_IP2 = "10.0.0.5"
TCP_IP3 = "10.0.0.3"
PORT2 = 8085
PORT4 = 8086
BUFFER_SIZE = 4096



#Get the partition point from the Pi
s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP2, PORT2))
print('Waiting for partition point and input')
s.listen(2)
conn, addr = s.accept()
data=[]
print ('Device:',addr)
while 1:
    dictionary_with_partition_and_input = conn.recv(BUFFER_SIZE)
    #print("AAA")
    #print(dictionary_with_partition_and_input)
    #print(type(dictionary_with_partition_and_input))
    if not dictionary_with_partition_and_input:
        break
    #data.append(b"".join(bytes(dictionary_with_partition_and_input)))
    data.append(dictionary_with_partition_and_input)

dictionary_with_partition_and_input = pickle.loads(b"".join(data))
partition_point = int(dictionary_with_partition_and_input['partition_point'])
input_data = dictionary_with_partition_and_input['input_data']


conn.close()


def server_call(model, inputs, partition_point):
    count = 0
    for layer in model.layers[partition_point:]:
        inputs = layer(inputs)
        print("Layer ", layer)
        print(count)
        count=count+1
        print(type(inputs))
    return inputs


# run the test harness for evaluating a model
def run_test_harness():
    # load model
    model = load_model('final_model.h5')

    #model = load_model('final_model_custom_split.h5')
    model.summary()
    y_tensor_send = server_call(model, input_data, partition_point)
    y_tensor_send = y_tensor_send.numpy()
    #final_output_send = final_outputs.numpy()
    print("Sending")



    #Run the data in the partition from 0 to given value, and send the results to the Pi.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP3, PORT4))
    #s.connect((TCP_IP2, PORT4))


    data_final = pickle.dumps(y_tensor_send, protocol=pickle.HIGHEST_PROTOCOL)

    # time.sleep(1)
    s.sendall(data_final)

    s.close()





# entry point, run the test harness
run_test_harness()
