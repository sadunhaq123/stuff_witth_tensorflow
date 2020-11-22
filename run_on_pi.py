# evaluate the deep model on the test dataset
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import socket
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#IP Parameters Client
TCP_IP1 = "10.0.0.5"
TCP_IP2 = "10.0.0.3"
PORT1 = 8085
PORT4 = 8086
BUFFER_SIZE = 4096
partition_point = 0



# load train and test dataset
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY



# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm

def server_call(model, inputs, partition_point):
    count = 0
    for layer in model.layers[0: partition_point]:
        inputs = layer(inputs)
        print("Layer ", layer)
        print(count)
        count=count+1
    return inputs

def server_call_zero(model, inputs):
    count = 0
    for layer in model.layers:
        inputs = layer(inputs)
        print("Layer ", layer)
        print(count)
        count=count+1
    return inputs



# run the test harness for evaluating a model
def run_test_harness():
    global partition_point
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # load model
    model = load_model('final_model.h5')
    #model = load_model('final_model_custom_split.h5')
    model.summary()

    #We take one label to check against.
    testY = testY[0]
    testY = np.expand_dims(testY, axis=0)

    #We take one input to predict
    testX = testX[0]
    #We expand its dimension by 1, as the first layer expects it in that manner
    testX = np.expand_dims(testX, axis=0)




    y_tensor = server_call(model, testX, partition_point)

    dictionary_with_partition_and_input = {
        'partition_point': partition_point,
        'input_data': y_tensor
    }

    #Send partition point and input to Desktop
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP1, PORT1))
    data_final = pickle.dumps(dictionary_with_partition_and_input, protocol=pickle.HIGHEST_PROTOCOL)

    s.sendall(data_final)

    s.close()



    #After sending partition point and inputs, waiting for results after the partition
    s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP1, PORT4))
    print('Waiting for results from cloud')
    s.listen(2)
    conn, addr = s.accept()
    data=[]
    print ('Device:',addr)
    while 1:
        tensor = conn.recv(BUFFER_SIZE)
        if not tensor: break
        data.append(tensor)

    part_outputs=pickle.loads(b"".join(data))

    conn.close()


    print("Receiving")
    #y_tensor = server_call(model, part_outputs, partition_point)


    #Applying argmax and calculating accuracy
    print(testY)
    print(part_outputs)
    y_pred = np.argmax(part_outputs, axis=1)
    Y = np.argmax(testY, axis=1)
    print(y_pred)
    print(Y)
    accuracy = ((y_pred == Y).mean()) * 100
    #accuracy = ((y_pred == Y)) * 100
    print('accuracy:', accuracy, '%')



def run_test_harness_zero():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # load model
    model = load_model('final_model.h5')
    #model = load_model('final_model_custom_split.h5')
    model.summary()
    testY = testY[0]
    testY = np.expand_dims(testY, axis=0)



    testX = testX[0]
    testX = np.expand_dims(testX, axis=0)



    y_tensor = server_call_zero(model, testX)
    #y_tensor_send = y_tensor.numpy()
    print(y_tensor)
    print(y_tensor.shape)



    #Applying argmax and calculating accuracy
    y_pred = np.argmax(y_tensor, axis=1)
    Y = np.argmax(testY, axis=1)
    print(y_pred)
    print(Y)
    accuracy = ((y_pred == Y).mean()) * 100
    print('accuracy:', accuracy, '%')


# entry point, run the test harness

def run_test_harness_eight():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # load model
    model = load_model('final_model.h5')
    #model = load_model('final_model_custom_split.h5')
    model.summary()
    testY = testY[0]
    testY = np.expand_dims(testY, axis=0)



    testX = testX[0]
    testX = np.expand_dims(testX, axis=0)



    #y_tensor = server_call_zero(model, testX)
    #y_tensor_send = y_tensor.numpy()

    dictionary_with_partition_and_input = {
        'partition_point': partition_point,
        'input_data': testX
    }

    # Send partition point and input to Desktop
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP1, PORT1))
    data_final = pickle.dumps(dictionary_with_partition_and_input, protocol=pickle.HIGHEST_PROTOCOL)

    s.sendall(data_final)

    s.close()

    # After sending partition point and inputs, waiting for results after the partition
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP1, PORT4))
    print('Waiting for results from cloud')
    s.listen(2)
    conn, addr = s.accept()
    data = []
    print('Device:', addr)
    while 1:
        tensor = conn.recv(BUFFER_SIZE)
        if not tensor: break
        data.append(tensor)

    part_outputs = pickle.loads(b"".join(data))

    conn.close()

    print("Receiving")


    print(part_outputs)
    print(part_outputs.shape)



    #Applying argmax and calculating accuracy
    y_pred = np.argmax(part_outputs, axis=1)
    Y = np.argmax(testY, axis=1)
    print(y_pred)
    print(Y)
    accuracy = ((y_pred == Y).mean()) * 100
    print('accuracy:', accuracy, '%')

if (partition_point == 8):
    run_test_harness_zero()
elif (partition_point == 0):
    run_test_harness_eight()

else:
    run_test_harness()
