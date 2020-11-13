# evaluate the deep model on the test dataset
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import socket
import pickle


#IP Parameters Server
TCP_IP2 = "10.0.0.5" #IP of Desktop machine(self)
TCP_IP3 = "10.0.0.3" #IP of Pi (remote Pi)
PORT2 = 8085
PORT4 = 8086
BUFFER_SIZE = 4096



#Get the partition point from the Pi
s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP2, PORT2))
print('Waiting for partition point')
s.listen(2)
conn, addr = s.accept()
data=[]
print ('Device:',addr)
while 1:
    partition_point = conn.recv(BUFFER_SIZE)
    partition_point = int(partition_point.decode())
    #print(partition_point)
    #print(type(partition_point))
    break

conn.close()


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

def server_call_m(model):
    for layer in model.layers:
        print("Layer ", layer)

def model_evaluation(testX, testY):
    loss, acc = model.evaluate(testX, testY)
    return acc

# run the test harness for evaluating a model
def run_test_harness():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # load model
    model = load_model('final_model.h5')
    model.summary()
    # evaluate model on test dataset
    y_tensor_send = server_call(model, testX, partition_point)
    y_tensor_send = y_tensor_send.numpy()
    #final_output_send = final_outputs.numpy()
    print(y_tensor_send)



    #Run the data in the partition from 0 to given value, and send the results to the Pi.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP3, PORT4)) #Change TCP_IP3 to TCP_IP2, if both the files are run on the same machine!

    # filename = 'final_output.npy'
    # outfile = open(filename,'wb')
    data_final = pickle.dumps(y_tensor_send, protocol=pickle.HIGHEST_PROTOCOL)

    # time.sleep(1)
    s.sendall(data_final)

    # data=s.recv(BUFFER_SIZE)
    # data_arr=pickle.loads(data)
    s.close()


    """
    print(y_tensor)
    #print(testY.shape)
    y_pred = np.argmax(y_tensor, axis=1)
    Y = np.argmax(testY, axis=1)
    print(y_pred)
    print(Y)


    accuracy = ((y_pred == Y).mean()) * 100
    print('accuracy:', accuracy, '%')
    """


# entry point, run the test harness
run_test_harness()
