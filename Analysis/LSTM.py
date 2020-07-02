import sys
import shutil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from os import listdir
from os.path import isfile, join
import numpy
from bisect import bisect
from random import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
from random import random
import collections
import time
import codecs
import json
from numpy import mean, sqrt, square, arange

from load_data import load_data

 
start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"
    


class ToySequenceData(object):

    def __init__(self, raw_data, max_seq_len, lead_time):
        
        self.data = []
        self.labels = []
        self.seqlen = []
        self.y_previous=[]
        self.lead_time = lead_time 

        for i in range(max_seq_len, len(raw_data)-self.lead_time):
            for j in range(len(raw_data[i])):
                s=[]
                self.seqlen.append(max_seq_len)
                for k in range(i-max_seq_len, i):
                    s.append([])
                    for l in range(len(raw_data[k][j])):
                        s[max_seq_len-(i-k)].append(raw_data[k][j][l])
                self.data.append(s)


                symbols_out_onehot=[]
                for k in range(len(raw_data[i][j])):
                    symbols_out_onehot.append(raw_data[i+self.lead_time][j][k])
                self.labels.append(symbols_out_onehot)


                symbols_out_onehot=[]
                for k in range(len(raw_data[i-1][j])):
                    symbols_out_onehot.append(raw_data[i-1][j][k])
                self.y_previous.append(symbols_out_onehot)
                
        self.batch_id = 0


                                              
    def next(self, batch_size):                                                       
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
                                              
        return batch_data, batch_labels, batch_seqlen,self.batch_id


    def getAll(self):                                                                                        
        return self.data, self.labels, self.y_previous,self.seqlen,self.batch_id
    
    
                                              


# Parameters
learning_rate = 0.01
#training_iters = 2000
training_iters = int(sys.argv[3])
batch_size = 214000
display_step = 10

# number of units in RNN cell
#n_hidden = 32
n_hidden = int(sys.argv[2])

def dynamicRNN(x, seqlen, weights, biases):
    
    print(x)
    x = tf.unstack(x, maxSeqlength, 1)
    print(x)
    
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                sequence_length=seqlen)

    print(outputs)
    
    outputs = tf.stack(outputs)
    print(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])
    print(outputs)

    batch_size = tf.shape(outputs)[0]
    index = tf.range(0, batch_size) * maxSeqlength + (seqlen - 1)
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
    print(outputs)

    return tf.add(tf.matmul(outputs, weights['out']),biases['out'])
    # outputs is hidden layer output. This is being multiplied by weights. Then biases is being added.


maxSeqlength = int(sys.argv[4])

rawCood, NumberOfAtoms = load_data("ADK_Equilibrium_MD.xyz")

#nd = normalizedData(rawCood)
#nd.scaleIt()
#Coordinates = nd.scaled_dat
Coordinates = rawCood

print(len(Coordinates), len(Coordinates[0]))

lead_time = int(sys.argv[1])
training_data = ToySequenceData(Coordinates, maxSeqlength, lead_time)

print(len(training_data.data), len(training_data.data[0]))


# tf Graph input
x = tf.placeholder("float", [None, maxSeqlength, 3],name='x')
# maxSeqlength * 3 array (maxSeqlength is our "window size"). 
# "None" is the batch size.

y = tf.placeholder("float", [None, 3],name='y')
seqlen = tf.placeholder(tf.int32, [None],name='seqlen')
# seqlen is simply an integer

tf.Variable(NumberOfAtoms,name='NumberOfAtoms')
tf.Variable(maxSeqlength,name='maxSeqlength')
tf.Variable(n_hidden,name='n_hidden')




# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, 3]),name='weights')
}
biases = {
    'out': tf.Variable(tf.random_normal([3]),name='biases')
}
# weights are the biases of the output layer.
# hidden layer dimension is 1*n_hidden. Now if you do matmul with something of dimension n_hidden*3, output will be 1*3
# Bias is just an addition at the end, so has to be 1*3.


print("before pred")
pred = dynamicRNN(x,seqlen,weights, biases)
print("after pred")

# Loss and optimizer
Errors = tf.losses.mean_squared_error(predictions=pred, labels=y)
cost = tf.reduce_mean(tf.losses.mean_squared_error(predictions=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
accuracy = tf.reduce_mean(tf.cast(cost, tf.float32),name='accuracy')

# Initializing the variables
init = tf.global_variables_initializer()


all_x, all_y, all_y_previous, all_seqlen, batch_id = training_data.getAll()
np_all_y=np.array(all_y)
np_all_y_previous=np.array(all_y_previous)
#lead_time = 15

nframes_pred = Coordinates.shape[0] - lead_time - maxSeqlength

with tf.Session() as sess:
    sess.run(init)
    # init variable is executed.

    saver = tf.train.Saver()
    # you can call it later with saver.save

    step = 1
    # Keep training until reach max iterations
    while step < training_iters:
        batch_x, batch_y, batch_seqlen,batch_id = training_data.next(batch_size)
        # Run optimization op (backprop)
        #print("Iter " + str(step) + " started training")
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       seqlen: batch_seqlen})
        if step % display_step == 0:
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y,
                                             seqlen: batch_seqlen})
            print("Iter " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss))
        

        if step % 100 == 0:
            with open("Predictions_not_scaled_" + str(lead_time) + "_" + str(n_hidden) + "_" + str(maxSeqlength) +  ".txt", 'w') as outputfile2:
                with open("output_not_scaled.txt", 'w') as outputfile:
                    my_predictions_raw = sess.run(pred, feed_dict={x: all_x, y: all_y, seqlen: all_seqlen})
                    #temp = np.reshape(my_predictions_raw, (nframes_pred, NumberOfAtoms,3))
                    #new = inversedData(nd)
                    #new.inverseIt(temp)
                    #my_predictions = np.reshape(new.inversed_dat, (nframes_pred * NumberOfAtoms,3))
                    
                    my_predictions = my_predictions_raw
                   
                    for i in range(len(Coordinates)-maxSeqlength-1):
                        err=sqrt(mean(square(my_predictions[i*NumberOfAtoms:(i+1)*NumberOfAtoms,:] - np_all_y[i*NumberOfAtoms:(i+1)*NumberOfAtoms,:])))
                        nv_err=sqrt(mean(square(np_all_y_previous[i*NumberOfAtoms:(i+1)*NumberOfAtoms,:] - np_all_y[i*NumberOfAtoms:(i+1)*NumberOfAtoms,:])))
                        outputfile.write(str(i+maxSeqlength) +'\t'+ str(err) + '\t' + str(nv_err) +'\n')
                        #print(str(i+maxSeqlength) +'\t'+ str(err) + '\t' + str(nv_err))

                    for i in range(len(all_x)):
                        outputfile2.write('frame ' + str(int(i/NumberOfAtoms)+maxSeqlength+lead_time) + ', atom '+ str(int(i%NumberOfAtoms))+', ' +str(my_predictions[i][0])+ ','+ str(my_predictions[i][1])+','+ str(my_predictions[i][2])+'\n')

                    loss = sqrt(mean(square(my_predictions - (np_all_y))))

                    Naive_loss=sqrt(mean(square(np_all_y_previous - (np_all_y))))


                    print("Overall Status in Step: " + str(step) + ",Naive Loss= " + \
                      "{:.6f}".format(Naive_loss)+", Loss= " + \
                      "{:.6f}".format(loss)+'\n\n')
                             

        step += 1
        
    print("Optimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))
