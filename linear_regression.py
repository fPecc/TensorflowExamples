import tensorflow as tf
import numpy as np
import random
import time
import matplotlib.pyplot as plt


def print_progress(it, acc, loss):
    # Calculate the accuracy on the training-set.
    now = time.strftime("%c")
    print("Iteration "+str(it)+" --- Accuracy: "+str(acc)+", Loss: "+str(loss)+" --- "+now);


def main():

    learning_rate = 0.001;
    max_iterations = 20000;

    # Load training and eval data
    train_x = np.linspace(0,10,50);
    train_y = 4*train_x + 1;
    train_y += np.random.uniform(low=-1,high=1,size=len(train_y));

    # Show training data
    plt.figure();
    plt.plot(train_x,train_y,'ro');
    plt.show();

    # -------------------------------------------------------------------
    # Building the computational graph
    tf.reset_default_graph();

    x = tf.placeholder(tf.float32, shape=len(train_x), name='x')
    y_true = tf.placeholder(tf.float32, shape=len(train_y), name='y_true')

    m = tf.Variable(np.random.randn(), name="m");
    tf.summary.scalar("m",m);
    b = tf.Variable(np.random.randn(), name="b");
    tf.summary.scalar("b", b);

    pred = tf.add(tf.multiply(m,x),b);

    loss = tf.losses.mean_squared_error(labels=y_true, predictions=pred)
    tf.summary.scalar("loss",loss);

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    merged_summary = tf.summary.merge_all();

    print "Graph correctly created!";
    # -------------------------------------------------------------------

    session = tf.Session();
    session.run(tf.global_variables_initializer())
    folder = "linear_regression";
    train_writer = tf.summary.FileWriter("logs/" + folder, session.graph);

    i = 0;

    feed_dict_train = {x: train_x, y_true: train_y}

    print "Starting training...";

    while i <= max_iterations:

        s,train_loss,_ = session.run([merged_summary,loss,optimizer], feed_dict=feed_dict_train)

        if i % 10 == 0:
            #
            now = time.strftime("%c")
            print("Iteration " + str(i) + " --- Loss: " + str(train_loss) + " --- " + now);
            train_writer.add_summary(s,i);

        i += 1;

    print "Training ended!";

    print "Learned parameters:";
    print "m = " + str(session.run(m));
    print "b = " + str(session.run(b));

    plt.plot(train_x, train_y, 'ro', label='Original data')
    plt.plot(train_x, session.run(pred,feed_dict={x: train_x}), label='Fitted line')
    plt.legend()
    plt.show()

    session.close();

if __name__ == "__main__":
    main();