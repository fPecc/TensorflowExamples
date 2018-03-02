import tensorflow as tf
import numpy as np
import random
import time


def print_progress(it, acc, loss):
    # Calculate the accuracy on the training-set.
    now = time.strftime("%c")
    print("Iteration "+str(it)+" --- Accuracy: "+str(acc)+", Loss: "+str(loss)+" --- "+now);


def main():

    learning_rate = 0.0001;
    max_iterations = 2000;

    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    tf.reset_default_graph();

    # Building the computational graph

    x = tf.placeholder(tf.float32, shape=[None, 28*28], name='x')
    #x_flat = tf.reshape(x,[-1,28*28*1]);
    y_true = tf.placeholder(tf.int32, shape=[None], name='y_true')
    y_onehot = tf.one_hot(indices=y_true, depth=10);

    output_layer = tf.layers.dense(inputs=x, units=10);

    #print tf.shape(y_onehot);
    #print tf.shape(output_layer)

    probs = tf.nn.softmax(logits=output_layer, name="softmax_tensor");

    prediction = tf.cast(tf.argmax(input=output_layer, axis=1),tf.int32);

    loss = tf.losses.softmax_cross_entropy(onehot_labels=y_onehot, logits=output_layer)
    tf.summary.scalar("loss",loss);

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    correct_prediction = tf.equal(prediction, y_true)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
    tf.summary.scalar("accuracy", accuracy);

    merged_summary = tf.summary.merge_all();

    saver = tf.train.Saver(max_to_keep=1);

    print "Graph correctly created!";

    session = tf.Session();
    session.run(tf.global_variables_initializer())
    folder = "1_mnist_fc2_dropoutno_lr" + str(learning_rate);
    train_writer = tf.summary.FileWriter("logs/" + folder + "/train", session.graph);
    val_writer = tf.summary.FileWriter("logs/" + folder + "/val", session.graph);

    i = 0;

    feed_dict_train = {x: train_data, y_true: train_labels}
    feed_dict_val = {x: eval_data, y_true: eval_labels}

    print "Starting training...";

    while i <= max_iterations:

        s,train_loss,train_acc,_ = session.run([merged_summary,loss,accuracy,optimizer], feed_dict=feed_dict_train)

        if i % 10 == 0:
            #
            print_progress(i,train_acc,train_loss);
            train_writer.add_summary(s,i);

        if i % 100 == 0:
            #
            s = session.run(merged_summary,feed_dict=feed_dict_val)
            val_writer.add_summary(s,i);

        if i % 500 == 0:
            #
            saver.save(session, 'models/' + folder)

        i += 1;

    print "Training ended, saving model...";
    saver.save(session, 'models/' + folder)

    session.close();


if __name__ == "__main__":
    main();