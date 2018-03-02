import tensorflow as tf
import sys
import numpy as np
import io
import random

def main():
    INTERNALSIZE = 512
    NLAYERS = 3

    folder = sys.argv[1];
    max_chars = int(sys.argv[2]);
    reset = sys.argv[3];

    sess = tf.Session();
    new_saver = tf.train.import_meta_graph(folder + "model.meta");
    new_saver.restore(sess,tf.train.latest_checkpoint(folder));

    # Generation of a random text sample
    f = io.open("gru_generated_text.txt", 'w+', encoding='utf-8');
    istate = np.zeros([1, INTERNALSIZE * NLAYERS])  # initial zero input state
    text_sample = chr(random.randint(65, 86));
    xin = np.array([[ord(text_sample)]]);
    for k in range(max_chars):
        feed_dict = {"X:0": xin, "Hin:0": istate}
        y, ostate = sess.run(["Y:0", "H:0"], feed_dict=feed_dict)
        c = chr(y);
        if reset == 'y' and c == '.':
            ostate = np.zeros([1, INTERNALSIZE*NLAYERS]);
            text_sample += " ";
            c = chr(random.randint(65,86));
        text_sample += c;
        xin = np.array([[ord(c)]]);
        istate = ostate;
    f.write(text_sample.decode('unicode-escape'));
    f.close();
    sess.close();

if __name__ == "__main__":
    main();