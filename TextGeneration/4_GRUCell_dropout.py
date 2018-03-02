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
import tensorflow.contrib
import time
import sys
import numpy as np
import random
import io
import importTextData as txt

tf.set_random_seed(0)

def print_progress(epoch, loss, acc):
    # Calculate the accuracy on the training-set.
    now = time.strftime("%c")
    print("Step "+str(epoch) + ", Accuracy: "+str(acc)+", Loss: "+str(loss)+" --- "+now);

def main():
    BATCHSIZE = 100
    SEQLEN = 30
    ALPHASIZE = txt.ALPHASIZE
    INTERNALSIZE = 512
    NLAYERS = 3
    learning_rate = float(sys.argv[1]);
    max_epochs = int(sys.argv[2]);

    data = txt.Data("Text/",SEQLEN,BATCHSIZE)

    # -----------------------------------------------------------------
    # Building the computational graph

    # inputs
    x = tf.placeholder(tf.uint8, [None, None], name='X')    # [ BATCHSIZE, SEQLEN ]
    xo = tf.one_hot(x, ALPHASIZE, 1.0, 0.0)                 # [ BATCHSIZE, SEQLEN, ALPHASIZE ]
    batchsize = tf.shape(x)[0];
    p_keep = tf.placeholder(tf.float32,name="p_keep");
        # expected outputs = same sequence shifted by 1 since we are trying to predict the next character
    y_true = tf.placeholder(tf.uint8, [None, None], name='Y_')  # [ BATCHSIZE, SEQLEN ]
    yo_true = tf.one_hot(y_true, ALPHASIZE, 1.0, 0.0)               # [ BATCHSIZE, SEQLEN, ALPHASIZE ]
        # input state
    Hin = tf.placeholder(tf.float32, [None, INTERNALSIZE * NLAYERS], name='Hin')  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]

    cells = [tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.GRUCell(INTERNALSIZE),output_keep_prob=p_keep) for _ in range(NLAYERS)];

    multicell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=False)

    Yr, H = tf.nn.dynamic_rnn(multicell, xo, dtype=tf.float32, initial_state=Hin)

    H = tf.identity(H, name='H')  # just to give it a name

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
    batch_acc = tf.placeholder(tf.float32,name="batch_acc");
    batch_loss = tf.placeholder(tf.float32, name="batch_loss");
    batch_acc_summary = tf.summary.scalar("batch_acc",batch_acc);
    batch_loss_summary = tf.summary.scalar("batch_loss", batch_loss);
    batch_summaries = tf.summary.merge([batch_acc_summary,batch_loss_summary]);

    saver = tf.train.Saver(max_to_keep=1)

    print("Graph correctly created!");
    # -----------------------------------------------------------------

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
#    print sess.graph.get_operations()
    folder = "4_GOT_GRUCell__dropout_seqlen" + str(SEQLEN) + "_batchsize" + str(BATCHSIZE) + "_intsize" + str(
        INTERNALSIZE) + "_nlayers" + str(NLAYERS)
    train_writer = tf.summary.FileWriter("logs/" + folder + "/train", sess.graph);
    val_writer = tf.summary.FileWriter("logs/" + folder + "/val", sess.graph);

    step = 0
    epoch = 0;

    f = io.open("got_gru_dropout_samples.txt",'w+',encoding='utf-8');

    # Generation of a random text sample
    istate = np.zeros([1, INTERNALSIZE * NLAYERS])  # initial zero input state
    print("-----------------------------------------------------------");
    print("This is a text sample at step " + str(step));
    f.write(unicode("This is a text sample at step " + str(step) + ":\n"))
    text_sample = chr(random.randint(65,90));
    xin = np.array([[ord(text_sample)]]);
    for k in range(SEQLEN*6):
        feed_dict = {x: xin, Hin: istate, p_keep: 1.}
        y, ostate = sess.run([Y, H], feed_dict=feed_dict)
        c = chr(y);
        text_sample += c;
        xin = np.array([[ord(c)]]);
        istate = ostate;
    text_sample += "\n";
    f.write(text_sample.decode('unicode-escape'));
    f.close();
#    print text_sample.decode('unicode-escape');
    s = sess.run(text_summary,feed_dict={text: text_sample.decode('unicode-escape')});
    val_writer.add_summary(s,step);
    print("-----------------------------------------------------------");

    # Validation loop
    istate = np.zeros([BATCHSIZE, INTERNALSIZE * NLAYERS])  # initial zero input state
    val_acc = [];
    val_loss = [];
    while data.endValEpoch is False:
        x_val_batch, y_val_batch = data.getNextValBatch();
        feed_dict_val = {x: x_val_batch, y_true: y_val_batch, Hin: istate, p_keep: 1.};
        temp_loss, temp_acc = sess.run([batchloss,accuracy], feed_dict=feed_dict_val)
        val_acc.append(temp_acc);
        val_loss.append(temp_loss);
    data.endValEpoch = False;
    s = sess.run(batch_summaries,feed_dict={batch_acc: np.mean(val_acc), batch_loss: np.mean(val_loss)});
    val_writer.add_summary(s, step);
    print_progress(step,np.mean(val_loss),np.mean(val_acc))

    # Training loop
    print("Starting training...");
    istate = np.zeros([BATCHSIZE, INTERNALSIZE * NLAYERS])
    train_acc = [];
    train_loss = [];
    while epoch <= max_epochs:

        x_train_batch, y_train_batch = data.getNextTrainBatch();
        feed_dict_train = {x: x_train_batch, y_true: y_train_batch, Hin: istate, p_keep: 0.5}

        _, ostate, temp_acc, temp_loss = sess.run([optimizer, H, accuracy, batchloss], feed_dict=feed_dict_train)
        train_acc.append(temp_acc);
        train_loss.append(temp_loss);

        step += 1;

        istate = ostate;

        if step % 10 == 0:
            print_progress(step,np.mean(train_loss),np.mean(train_acc));
            s = sess.run(batch_summaries, feed_dict={batch_acc: np.mean(train_acc), batch_loss: np.mean(train_loss)});
            train_writer.add_summary(s, step)
            train_acc = [];
            train_loss = [];

        if data.endTrainEpoch is True:
            # Resetting train epoch variables and stuff
            data.endTrainEpoch = False;
            saver.save(sess, 'models/' + folder)
            train_acc = [];
            train_loss = [];
            epoch += 1;
	    istate = np.zeros([BATCHSIZE, INTERNALSIZE * NLAYERS])  # initial zero input state

	if step % 100 == 0:
            # Validation loop
            istate = np.zeros([BATCHSIZE, INTERNALSIZE * NLAYERS])  # initial zero input state
            val_acc = [];
            val_loss = [];
            while data.endValEpoch is False:
                x_val_batch, y_val_batch = data.getNextValBatch();
                feed_dict_val = {x: x_val_batch, y_true: y_val_batch, Hin: istate, p_keep: 1.};
                temp_loss, temp_acc = sess.run([batchloss, accuracy], feed_dict=feed_dict_val)
                val_acc.append(temp_acc);
                val_loss.append(temp_loss);
            data.endValEpoch = False;
            s = sess.run(batch_summaries, feed_dict={batch_acc: np.mean(val_acc), batch_loss: np.mean(val_loss)});
            val_writer.add_summary(s, step);
            istate = np.zeros([BATCHSIZE, INTERNALSIZE * NLAYERS])  # initial zero input state

        if step % 500 == 0:
	    saver.save(sess, 'models/'+folder);
            # Generation of a random text sample
            ist = np.zeros([1, INTERNALSIZE * NLAYERS])  # initial zero input state
            print("-----------------------------------------------------------");
            print("This is a text sample at step " + str(step));
	    f = io.open("got_gru_dropout_samples.txt",'a',encoding='utf-8');
            f.write(unicode("This is a text sample at step " + str(step) + ":\n"))
            text_sample = chr(random.randint(65, 90));
            xin = np.array([[ord(text_sample)]]);
            for k in range(SEQLEN*6):
                feed_dict = {x: xin, Hin: ist, p_keep: 1.}
                y, ost = sess.run([Y, H], feed_dict=feed_dict)
                c = chr(y);
                text_sample += c;
                xin = np.array([[ord(c)]]);
                ist = ost;
            text_sample += "\n";
            f.write(text_sample.decode('unicode-escape'));
#	    print text_sample.decode('unicode-escape');
            s = sess.run(text_summary, feed_dict={text: text_sample.decode('unicode-escape')});
            val_writer.add_summary(s, step);
            print("-----------------------------------------------------------");

    f.close();
    sess.close();

if __name__ == "__main__":
    main();
