import numpy as np
import os
import io
import re

# size of the alphabet that we work with
ALPHASIZE = 255


class Data:
    def __init__(self, textfolder, seqlen, batchsize):
        self.textFolder = textfolder;
        self.encodedTrainData = None;
        self.encodedValData = None;
        self.nextTrainBatch = 0;
        self.nextValBatch = 0;
        self.batchsize = batchsize;
        self.seqlen = seqlen;
        self.endTrainEpoch = False;
        self.endValEpoch = False;
        self.x_val = None;
        self.y_val = None;
        self.x_train = None;
        self.y_train = None;
        self.loadText();
        self.prepareXY();
        self.word2int = {};
        self.int2word = {};
        print("Data successfully loaded and prepared!");

    def prepareXY(self):
        # Preparation of the training data
        print("Preparing data...");
        x_temp = np.copy(self.encodedTrainData);
        while len(x_temp) % self.batchsize != 0:
            x_temp = x_temp[0:(len(x_temp) - 1)]

        y_temp = np.copy(x_temp);
        y_temp[0:len(y_temp)-1] = np.copy(x_temp[1:len(x_temp)]);
        y_temp[len(y_temp)-1] = 10;  # LF

        x_temp = np.split(x_temp, self.batchsize);
        y_temp = np.split(y_temp, self.batchsize);

        self.x_train = [self.limitSeqLength(a) for a in x_temp];
        self.y_train = [self.limitSeqLength(a) for a in y_temp];

        # Preparation of the validation data
        x_temp = np.copy(self.encodedValData);
        while len(x_temp) % self.batchsize != 0:
            x_temp = x_temp[0:(len(x_temp) - 1)]

        y_temp = np.copy(x_temp);
        y_temp[0:len(y_temp)-1] = np.copy(x_temp[1:len(x_temp)]);
        y_temp[len(y_temp)-1] = 10; # LF

        x_temp = np.split(x_temp, self.batchsize);
        y_temp = np.split(y_temp, self.batchsize);

        self.x_val = [self.limitSeqLength(a) for a in x_temp];
        self.y_val = [self.limitSeqLength(a) for a in y_temp];

    def limitSeqLength(self, a):
        while len(a) % self.seqlen != 0:
            a = a[0:len(a)-1]
        return a;

    def loadText(self):
        print("Loading text...");
        files = os.listdir(self.textFolder);
        rawDataTrain = [];
        rawDataVal = [];
        i = 1;
        for file in files:
            text = io.open(self.textFolder + file, "r", encoding="utf-8");
            print("Loading file " + file);
            rawDataTrain.extend(text.read());
            text.close();

        rawDataVal = rawDataTrain[int(len(rawDataTrain)*0.98):len(rawDataTrain)];
        rawDataTrain = rawDataTrain[0:int(len(rawDataTrain) * 0.98)];

        print("Encoding data...");
        for word in re.findall(r"[\w']+|[.,!?;]",rawDataTrain):
            
        return;

    def getNextTrainBatch(self):
        x_temp = np.zeros((self.batchsize,self.seqlen),dtype=np.uint8);
        y_temp = np.zeros((self.batchsize,self.seqlen),dtype=np.uint8);

        for i in range(len(self.x_train)):
            seqs = np.split(self.x_train[i],len(self.x_train[i])/self.seqlen);
            x_temp[i,:] = seqs[self.nextTrainBatch];
            seqs = np.split(self.y_train[i], len(self.y_train[i]) / self.seqlen);
            y_temp[i,:] = seqs[self.nextTrainBatch];

        self.nextTrainBatch += 1;
        if self.nextTrainBatch >= len(seqs):
            self.nextTrainBatch = 0;
            self.endTrainEpoch = True;

        return x_temp, y_temp;

    def getNextValBatch(self):
        x_temp = np.zeros((self.batchsize, self.seqlen), dtype=np.uint8);
        y_temp = np.zeros((self.batchsize, self.seqlen), dtype=np.uint8);

        for i in range(len(self.x_val)):
            seqs = np.split(self.x_val[i], len(self.x_val[i]) / self.seqlen);
            x_temp[i, :] = seqs[self.nextValBatch];
            seqs = np.split(self.y_val[i], len(self.y_val[i]) / self.seqlen);
            y_temp[i, :] = seqs[self.nextValBatch];

        self.nextValBatch += 1;
        if self.nextValBatch >= len(seqs):
            self.nextValBatch = 0;
            self.endValEpoch = True;

        return x_temp, y_temp;


