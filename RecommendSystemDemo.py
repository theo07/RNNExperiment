from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import array
from functools import reduce
import datetime
import multiprocessing
from multiprocessing import Manager
import unittest
import time
import collections

import time
import math 
import tensorflow as tf
from tensorflow.contrib import rnn

"""
preprocessing part
this part handles the data preprocessing
creating the embedding layer for the categorical data
normalising continous data
handling large data by using multicore
"""

print('TensorFlow Version (should be 1.2): ',tf.__version__)

manager = Manager()

#tracking the execution time
start_time = time.time()

#categorical data
position_size = 15 #2
industry_size = 30 #2
country_size = 254 #4
mr_info_size = 34 #2

#not using section id
#section_id_size = 1473

url_id_size = 266 #4
product_id_size = 51 #3
referral_id_size = 19 #2
title_size = 3 #2

#Male-1;Female-2;Unk-0
title_encoding ={'Ms':2, 'Mr':1, 'Miss':2, 'Mrs':2, 'Sir':1}
time_on_page_max = 3000

#31 topics
topic_size = 31 #2


#embedding the categorical data
#to update by training them in the model
#using some decent working fixed seeds
np.random.seed(700)
embed_position = np.random.uniform(low=0, high=1, size=(position_size, 2))
np.random.seed(3701)
embed_industry = np.random.uniform(low=0, high=1, size=(industry_size, 2))
np.random.seed(8700)
embed_country = np.random.uniform(low=0, high=1, size=(country_size, 4))
np.random.seed(4200)
embed_mr_info = np.random.uniform(low=0, high=1, size=(mr_info_size, 2))
#embed_section_id = np.random.uniform(low=0, high=1, size=(section_id_size, 6))
np.random.seed(1600)
embed_url_id = np.random.uniform(low=0, high=1, size=(url_id_size, 4))
np.random.seed(3800)
embed_product_id = np.random.uniform(low=0, high=1, size=(product_id_size, 3))
np.random.seed(5200)
embed_referral_id = np.random.uniform(low=0, high=1, size=(referral_id_size, 2))
np.random.seed(2800)
embed_title = np.random.uniform(low=0, high=1, size=(title_size, 2))
np.random.seed(5100)
embed_topic_id = np.random.uniform(low=0, high=1, size=(topic_size, 2))
#thread safe collection
#this will be a list of tuples
data_tuple_concurrency = manager.list([])

def normalise(value, max_value):
    return value/max_value

def one_hot_encode(x, n_classes):
        """
        One hot encode a 
        : x: category id
        : n_classes: Number of classes
        """
        x = int(x)
        verts=array.array('i',(0,)*n_classes)
        myLabel = verts.tolist()
        myLabel[x - 1] = 1
        return myLabel

def chunks_break(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def perform_embed(row):
    embed_row = []
    #appending to embed row
    #continous data first followed by categorical
    #using the feature's order from the CSV
    embed_row.append(normalise(row[4], 3000))
    embed_row.append(normalise(row[5].days, 9000))
    embed_row.append(normalise(row[6].days, 9000))
    embed_row.extend(embed_url_id[row[2]])
    embed_row.extend(embed_mr_info[row[3]])
    embed_row.extend(embed_country[row[7]])
    embed_row.extend(embed_industry[row[8]])
    embed_row.extend(embed_position[row[9]])
    embed_row.extend(embed_product_id[row[11]])
    embed_row.extend(embed_referral_id[row[12]])
    try:
        embed_row.extend(embed_title[title_encoding[row[10]]])
    except Exception as e:
        embed_row.extend(embed_title[0])
    embed_row.extend(embed_topic_id[int(row[13])])
    return embed_row

def thread_preprocess(chunk):
        chunk_data_x = []
        chunk_data_y = []  
        new_list = chunk.values
        sequence_list = []
        embed_sequence_list = []
        #use current to print thread's name
        current = multiprocessing.current_process()
        for row in new_list:
            #building the sequence
            #if empty add the entry to it
            if (len(sequence_list) == 0):
                sequence_list.append(row)
            elif (sequence_list[0][0] == row[0] and len(sequence_list) < 20):
                sequence_list.append(row)
                embed_sequence_list = [perform_embed(sequence) for sequence in sequence_list]
                #x vector represents the timesteps vectors excluding the last
                x_timesteps_vector = embed_sequence_list[:len(embed_sequence_list)-1]
                seq_len = len(x_timesteps_vector)
                y_el = embed_sequence_list[len(embed_sequence_list)-1]
                y_label = y_el[24:]
                padded_zeros = []
                #the range of one timestep feature vector
                #adding padding with zeros but tensorflow will know the exact seq length as it will be fed
                padded_zeros += [0. for i in range(26)]
                x_timesteps_vector += [padded_zeros for i in range(19 - len(x_timesteps_vector))]
                sequence_tuple = (x_timesteps_vector, y_label, seq_len)
                data_tuple_concurrency.append(sequence_tuple) 
            elif (sequence_list[0][0] != row[0]):
                sequence_list = []
                sequence_list.append(row)

class Items(object):
    #creating init of the data object
    #the object will have data(meaning the feature vectors) corresponding labels
    def __init__(self, data, labels, list_seq_len):
        self.data = data
        self.labels = labels
        self.batch_id = 0
        self.data_created = False
        #counting the number of training epochs
        self.epochsNo = 0
        self.list_seq_len = list_seq_len
        # create as many processes as there are CPUs on your machine
        self.num_processes = multiprocessing.cpu_count()
    def reset(self):
        self.data = []
        self.labels = []
        self.list_seq_len = []
        self.batch_id = 0
    def createData(self):
        #dateparsing using lambda and pandas datetime
        dateparse = lambda x: (datetime.datetime.now() - pd.datetime.strptime(str(x), '%d/%m/%y'))  if str(x)!='nan' else '0'
        while not self.data_created:
            inputPath = input("Please enter the file path of the data csv provided: ")
            try:
                df = pd.read_csv(inputPath, error_bad_lines=False, parse_dates=['last_login', 'register_date'], date_parser=dateparse)
                self.data_created = True
            except Exception as e:
                print('Cannot open file | File I|O Error | Try Again | Error as ', e)
        df = df.fillna(0)
        print('Initialised data file from the given CSV')
        #removing country;industry;position if smaller than 0 
        df.loc[(df['country_id'] < 0)] = 0
        df.loc[(df['industry_id'] < 0)] = 0
        df.loc[(df['position_id'] < 0)] = 0

        # calculate the chunk size as an integer
        chunk_size = int(df.shape[0]/self.num_processes)
        chunks = [df.iloc[df.index[i:i + chunk_size]] for i in range(0, df.shape[0], chunk_size)]

        #sending to thread process
        print('Starting multi-thread data preprocessing')
        pool = multiprocessing.Pool(processes=self.num_processes)
        pool.map(thread_preprocess, chunks)
        x_data = []
        y_data = []
        z_data = []
        #using zip to break the tuple in corresponding x;y;z; labels
        x_data, y_data, z_data = zip(*data_tuple_concurrency)
        print('Finished preprocessing')
        print('Total number of the population: ', len(data_tuple_concurrency))
        self.data = x_data
        self.labels = y_data
        self.list_seq_len = z_data
    def splitTrainTest(self):
        #splitting the data/labels/sequence with 60% for train 40% for test
        X_train, X_test, y_train, y_test, seq_len_train, seq_len_test = train_test_split(self.data, self.labels, self.list_seq_len, test_size=0.4)
        #creatin the train and test items which will be fed to the model
        train_items = Items(X_train, y_train, seq_len_train)
        test_items = Items(X_test, y_test, seq_len_test)
        print('Splitting train/test - 60/40')
        print('Training with a no of: ', len(X_train))
        return train_items, test_items
    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.epochsNo += 1
            print("Epoch no: ",  self.epochsNo)
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seq_len = (self.list_seq_len[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seq_len

"""
preprocessing part END
END of preprocessing part
NOTE: preprocessing used to be a seperate file but because of some version problems
when running on the AWS Cloud now everything belongs to the same file
"""

my_items = Items([],[],[])
my_items.reset()
my_items.createData()
#splitting the data object to train and test 60% / 40%
trainset, testset = my_items.splitTrainTest()

#tracking the execution time
start_time = time.time()

# Parameters
learning_rate = 0.009

training_iters = 200000
batch_size = 200

display_step = 10

# Network Parameters
#beta is the new parameter - controls level of regularization. Default is 0.01
beta = 0.002
n_input = 26 # data input (bigger data input because of one hot sparsity)
n_steps = 19 # data steps or time steps
n_hidden = 260 # hidden layer num of features or units or LSTM units/cells in one layer
output_embedding_size = 2 # number of classesmath
number_of_layers = 5 #number of stacked cells/layers 

# tf Graph input

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, output_embedding_size])
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_uniform([2*n_hidden, output_embedding_size], minval=-math.sqrt(6/7), maxval= math.sqrt(6/7)))
}

biases = {
    'out': tf.Variable(tf.random_normal([output_embedding_size]))
}

def BiRNN(x, weights, biases):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define lstm cells with tensorflow
    # Forward direction cell
    forwardCells = []
    backwardCells = []
    for _ in range(number_of_layers):
        #for the number of layers create a LSTMCell
        lstm_fw_cell = rnn.LSTMCell(num_units = n_hidden, state_is_tuple=True)
        # Backward direction cell
        lstm_bw_cell = rnn.LSTMCell(num_units = n_hidden, state_is_tuple=True)
        #append the lstm_cell to the cells vector
        forwardCells.append(lstm_fw_cell)
        backwardCells.append(lstm_bw_cell)

    #using the stacked forward and backward cells to create the multi rnn cell which will
    #be used as input to the bidirectional rnn 
    stacked_forwardlstm = tf.contrib.rnn.MultiRNNCell(forwardCells, state_is_tuple=True)
    stacked_backwardlstm = tf.contrib.rnn.MultiRNNCell(backwardCells, state_is_tuple=True)
     
    outputs, _, _ = rnn.static_bidirectional_rnn(stacked_forwardlstm, stacked_backwardlstm, x, sequence_length=seqlen,
                                              dtype=tf.float32)
  
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = BiRNN(x, weights, biases)

# Define loss and optimizer + l2 regularization 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
                      + beta*tf.nn.l2_loss(weights['out']))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    print(step, ' ', batch_size)
    while step * batch_size < training_iters:
        batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

        step += 1
    print("Optimization Finished!")

    test_data = testset.data
    test_label = testset.labels
    test_seqlen = testset.list_seq_len
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label, seqlen: test_seqlen}))

#tracking the execution time  
print("--- %s seconds ---" % (time.time() - start_time))
