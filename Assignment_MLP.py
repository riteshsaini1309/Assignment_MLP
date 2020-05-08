#MLP_Assignment
#imports
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from random import random
from random import seed
from math import exp
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix





# No. of rows(rows)
no_of_samples = 200

# No. of features {F1,F2,F3,F4}=4
no_of_features = 4

# No. of redundent features (Extra Feature)
no_of_extra = 1

#types of classes
types_of_classes = 2



X, y = make_classification(n_samples=no_of_samples, n_features=no_of_features,
                                            n_redundant=no_of_extra, n_classes=types_of_classes)
df = pd.DataFrame(X, columns=['F1', 'F2', 'F3', 'F4'])


#L1 is binary label

df['L1'] = y    
df.head()
df.to_csv("dataset1.csv")

#reading the csv file by using pandas 

df=pd.read_csv('assignment.csv',index_col=0)
df.head()


#creating the network for weights

def create_W(n_inp, n_hid, n_out):
    net=list()
    H_layer = [{'W':[random() for i in range(n_inp + 1)]} for i in range(n_hid)]
    net.append(H_layer)
    O_layer = [{'W':[random() for i in range(n_hid + 1)]} for i in range(n_out)]
    net.append(O_layer)
    return net




#activation function for MLP

def activate(W, inp):
    activation=W[-1]
    for i in range(len(W)-1):
        activation+=W[i]*inp[i]
    return activation

def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))





#forward propogation to pass it on next layer using activation function

def propagation_next_layer(net,raw):
    inp=raw
    for layer in net:
        new_inp=[]
        for neuron in layer:
            activation=activate(neuron['W'], inp)
            neuron['output']=transfer(activation)
            new_inp.append(neuron['output'])
        inp=new_inp
    return inp


def transfer_derivative(out):
    return out * (1.0 - out)



#backward propogation to learn for fine tuning the weights

def backward_learn(net, expected):
    for i in reversed(range(len(net))):
        layer = net[i]
        errors = list()
        if i != len(net)-1:
            for ind in range(len(layer)):
                err = 0.0
                for neuron in net[i + 1]:
                    err += (neuron['W'][ind] * neuron['delta'])
                errors.append(err)
        else:
            for index in range(len(layer)):
                neuron = layer[index]
                errors.append(expected[index] - neuron['output'])
        for index in range(len(layer)):
            neuron = layer[index]
            neuron['delta'] = errors[index] * transfer_derivative(neuron['output'])



#update weights W on training the algo

def update_W_train(net, row, l_rate):
    for i in range(len(net)):
        inp=row[:-1]
        if i!=0:
            inp=[neuron['output'] for neuron in net[i-1]]
        for neuron in net[i]:
            for j in range(len(inp)):
                neuron['W'][j]+=l_rate*neuron['delta']*inp[j]
            neuron['W'][-1]+=l_rate*neuron['delta']




#training the network

def train_MLP(net, train, l_rate, n_epoch, n_out):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            out = propagation_next_layer(net, row)
            expected = [0 for i in range(n_out)]
            expected[int(row[-1])] = 1
            sum_error += sum([(expected[i]-out[i])**2 for i in range(len(expected))])
            backward_learn(net, expected)
            update_W_train(net, row, l_rate)
        print('=>epoch=%d, =>lrate=%.3f, =>error=%.3f' % (epoch, l_rate, sum_error))



#predicting function

def prediction_def(net, row):
    out = propagation_next_layer(net, row)
    return out.index(max(out))


dataset=np.array(df[:])
dataset


n_inp = len(dataset[0]) - 1
n_out = len(set([row[-1] for row in dataset]))
print(n_inp,n_out)


#splitting into test and train datset
#the data is split in the ratio 3:1
train_dataset=dataset[:150]
test_dataset=dataset[150:]


#feeding the datset into the network
net=create_W(n_inp,1,n_out)
train_MLP(net, train_dataset, 0.5, 100, n_out)

#learned weights of the network
for layer in net:
    print(layer)


#applying on training dataset ie. 150 rows

y_train=[]
pred=[]
for row in train_dataset:
    prediction = prediction_def(net, row)
    y_train.append(int(row[-1]))
    pred.append(prediction)

#Final Results on training-->

print("Accuracy Result: ",accuracy_score(y_train,pred))
print("Confusion Matrix: ",confusion_matrix(y_train,pred))
print("Precision Result: ",precision_score(y_train, pred))
print("recall Result: ",recall_score(y_train, pred))

#applying on testing dataset ie. 50 rows
y_test=[]
pred=[]
for row in test_dataset:
    prediction = prediction_def(net, row)
    y_test.append(row[-1])
    pred.append(prediction)


#Final Results on testing-->


print("Accuracy Result: ",accuracy_score(y_test,pred))
print("Confusion Matrix: ",confusion_matrix(y_test,pred))
print("Precision Result: ",precision_score(y_test, pred))
print("recall Result: ",recall_score(y_test, pred))







