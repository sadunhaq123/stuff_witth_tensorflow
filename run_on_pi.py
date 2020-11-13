# evaluate the deep model on the test dataset
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import socket
import pickle


#IP Parameters Client
TCP_IP1 = "10.0.0.5"
#TCP_IP1 = "10.0.0.3"
PORT1 = 8085
PORT4 = 8086
BUFFER_SIZE = 4096



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
    for layer in model.layers[partition_point:]:
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
    #inputs = testX
    #inputs = numpy.array(inputs)
    #print(len(inputs))
    print(testX.shape)
    #print(testY.shape)
    #y = model(inputs)
    #print(y)
    #server_call_m(model)



    #Send partition point to Desktop
    partition_point = 7
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP1, PORT1))


    data_final = str(partition_point).encode()

    # time.sleep(1)
    s.sendall(data_final)

    s.close()

    """
    y_tensor = server_call(model, testX)
    y_tensor_send = y_tensor.numpy()
    print(y_tensor_send)
    """


    #After sending partition point, waiting for results after the partition
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



    y_tensor = server_call(model, part_outputs, partition_point)
    #y_tensor_send = y_tensor.numpy()
    print(y_tensor)


    #_, acc = model.evaluate(testX, testY, verbose=0)

    #acc=0
    #loss, acc = model.evaluate(testX, testY, verbose=1)
    #print(acc)
    #print('> %.3f' % (acc * 100.0))

    #print(testY.shape)

    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP1, PORT1))

    # filename = 'final_output.npy'
    # outfile = open(filename,'wb')
    data_final = pickle.dumps(y_tensor_send, protocol=pickle.HIGHEST_PROTOCOL)

    # time.sleep(1)
    s.sendall(data_final)

    # data=s.recv(BUFFER_SIZE)
    # data_arr=pickle.loads(data)
    s.close()
    """

    #Applying argmax and calculating accuracy
    y_pred = np.argmax(y_tensor, axis=1)
    Y = np.argmax(testY, axis=1)
    #print(y_pred)
    #print(Y)
    accuracy = ((y_pred == Y).mean()) * 100
    print('accuracy:', accuracy, '%')



# entry point, run the test harness
run_test_harness()
