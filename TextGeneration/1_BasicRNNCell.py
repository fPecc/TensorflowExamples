# encoding: UTF-8
# Copyright 2017 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import time
import sys
import numpy as np
import random
import io

tf.set_random_seed(0)

def print_progress(epoch, loss, acc):
    # Calculate the accuracy on the training-set.
    now = time.strftime("%c")
    print("Step "+str(epoch) + ", Accuracy: "+str(acc)+", Loss: "+str(loss)+" --- "+now);

def main():
    BATCHSIZE = 1
    ALPHASIZE = 255
    INTERNALSIZE = 512
    NLAYERS = 2
    learning_rate = float(sys.argv[1]);
    max_epochs = int(sys.argv[2]);

    data_in = "Estoy aprendiendo a predecir el siguiente caracter de una secuencia!";
    data_out = data_in[1:len(data_in)] + "\n";
    SEQLEN = len(data_in);
    encoded_data_in = np.array([[ord(c) for c in data_in]]);
    encoded_data_out = np.array([[ord(c) for c in data_out]]);

    # Building the computational graph
    tf.reset_default_graph();
    #lr = tf.placeholder(tf.float32, name='lr')  # learning rate

    # inputs
    x = tf.placeholder(tf.uint8, [None, None], name='X')    # [ BATCHSIZE, SEQLEN ]
    xo = tf.one_hot(x, ALPHASIZE, 1.0, 0.0)                 # [ BATCHSIZE, SEQLEN, ALPHASIZE ]
    batchsize = tf.shape(x)[0];
    # expected outputs = same sequence shifted by 1 since we are trying to predict the next character
    y_true = tf.placeholder(tf.uint8, [None, None], name='Y_')  # [ BATCHSIZE, SEQLEN ]
    yo_true = tf.one_hot(y_true, ALPHASIZE, 1.0, 0.0)               # [ BATCHSIZE, SEQLEN, ALPHASIZE ]
    # input state
    Hin = tf.placeholder(tf.float32, [None, INTERNALSIZE * NLAYERS], name='Hin')  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]

    cells = [tf.contrib.rnn.BasicRNNCell(INTERNALSIZE) for _ in range(NLAYERS)]
    multicell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=False)

    Yr, H = tf.nn.dynamic_rnn(multicell, xo, dtype=tf.float32, initial_state=Hin)
    # Yr: [ BATCHSIZE, SEQLEN, INTERNALSIZE ]
    # H:  [ BATCHSIZE, INTERNALSIZE*NLAYERS ] # this is the last state in the sequence

    H = tf.identity(H, name='H')  # just to give it a name

    # Softmax layer implementation:
    # Flatten the first two dimension of the output [ BATCHSIZE, SEQLEN, ALPHASIZE ] => [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    # then apply softmax readout layer. This way, the weights and biases are shared across unrolled time steps.
    # From the readout point of view, a value coming from a cell or a minibatch is the same thing

    Yflat = tf.reshape(Yr, [-1, INTERNALSIZE])    # [ BATCHSIZE x SEQLEN, INTERNALSIZE ]
    logits = tf.layers.dense(Yflat, ALPHASIZE)     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    yo_true_flat = tf.reshape(yo_true, [-1, ALPHASIZE])     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=yo_true_flat)  # [ BATCHSIZE x SEQLEN ]
    loss = tf.reshape(loss, [batchsize, -1])      # [ BATCHSIZE, SEQLEN ]
    Yo = tf.nn.softmax(logits, name='Yo')        # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    Y = tf.argmax(Yo, 1)                          # [ BATCHSIZE x SEQLEN ]
    Y = tf.reshape(Y, [batchsize, -1], name="Y")  # [ BATCHSIZE, SEQLEN ]

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # stats for display
    seqloss = tf.reduce_mean(loss, 1)
    batchloss = tf.reduce_mean(seqloss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true,tf.cast(Y,tf.uint8)), tf.float32))

    loss_summary = tf.summary.scalar("loss", batchloss)
    acc_summary = tf.summary.scalar("accuracy", accuracy)
    summaries = tf.summary.merge([loss_summary, acc_summary])
    text = tf.placeholder(tf.string, name="text");
    text_summary = tf.summary.text("output",text);

    saver = tf.train.Saver(max_to_keep=1)

    print("Graph correctly created!");

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    folder = "1_BasicRNNCell_seqlen" + str(SEQLEN) + "_batchsize" + str(BATCHSIZE) + "_intsize" + str(
        INTERNALSIZE) + "_nlayers" + str(NLAYERS)
    train_writer = tf.summary.FileWriter("logs/" + folder + "/train", sess.graph);
    val_writer = tf.summary.FileWriter("logs/" + folder + "/val", sess.graph);

    step = 0
    epoch = 0;

    f = io.open("val_samples.txt",'w+',encoding='utf-8');

    istate = np.zeros([1, INTERNALSIZE * NLAYERS])  # initial zero input state
    print("-----------------------------------------------------------");
    print("This is a text sample at step " + str(step));
    f.write(unicode("This is a text sample at step " + str(step) + ":\n"))
    text_sample = data_in[0];
    xin = np.array([[ord(data_in[0])]]);
    for k in range(SEQLEN):
        feed_dict = {x: xin, Hin: istate}
        y, ostate = sess.run([Y, H], feed_dict=feed_dict)
	c = chr(y);
        text_sample += c;
#	print y;
        xin = np.array([[ord(c)]]);
        istate = ostate;
    text_sample += "\n";
    f.write(text_sample.decode('unicode-escape'));
    s = sess.run(text_summary,feed_dict={text: text_sample.decode('unicode-escape')});
    val_writer.add_summary(s,step);
    print("-----------------------------------------------------------");

    istate = np.zeros([BATCHSIZE, INTERNALSIZE * NLAYERS])  # initial zero input state
    feed_dict_val = {x: encoded_data_in, y_true: encoded_data_out, Hin: istate};
    s = sess.run(summaries, feed_dict=feed_dict_val)
    val_writer.add_summary(s, step);

    # training loop
    print("Starting training...");
    istate = np.zeros([BATCHSIZE, INTERNALSIZE * NLAYERS])
    #outputs_examples = open("outputs.txt","w+");

    while epoch <= max_epochs:

        feed_dict = {x: encoded_data_in, y_true: encoded_data_out, Hin: istate}

        _, ostate, s, train_acc, train_loss = sess.run([optimizer, H, summaries, accuracy, batchloss], feed_dict=feed_dict)

        step += 1;

        if step % 10 == 0:
            #
            print_progress(step,train_loss,train_acc);
            train_writer.add_summary(s, step)

        if step % 10 == 0:
            #
            istate = np.zeros([BATCHSIZE, INTERNALSIZE * NLAYERS])  # initial zero input state
            feed_dict_val = {x: encoded_data_in, y_true: encoded_data_out, Hin: istate};
            s = sess.run(summaries, feed_dict=feed_dict_val)
            val_writer.add_summary(s, step);

            saver.save(sess, 'models/' + folder)
            istate = np.zeros([1, INTERNALSIZE * NLAYERS])  # initial zero input state
            print("-----------------------------------------------------------");
            print("This is a text sample at step " + str(step));
	    f.write(unicode("This is a text sample at step " + str(step) + ":\n"));
            text_sample = data_in[0];
            xin = np.array([[ord(data_in[0])]]);
            for k in range(SEQLEN):
                feed_dict = {x: xin, Hin: istate}
                y, ostate = sess.run([Y, H], feed_dict=feed_dict)
                c = chr(y);
		text_sample += c;
#		print y;
                xin = np.array([[ord(c)]]);
                istate = ostate;
	    text_sample += "\n";
	    f.write(text_sample.decode("unicode-escape"));
            s = sess.run(text_summary, feed_dict={text: text_sample.decode('unicode-escape')});
            val_writer.add_summary(s, step);
            print("-----------------------------------------------------------");

        istate = np.zeros([1, INTERNALSIZE * NLAYERS])  # initial zero input state
        epoch += 1;

    f.close();
    sess.close();

if __name__ == "__main__":
    main();
