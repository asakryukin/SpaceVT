import pickle as pkl
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class PredictionModel():

    def __init__(self, train = False):


        self.sess = tf.Session()

        with open('data/data.pickle', 'r') as handle:
            self.mean_dst = pkl.load(handle)
            self.mean_imf = pkl.load(handle)
            self.std_dst = pkl.load(handle)
            self.std_imf = pkl.load(handle)

        if(train):
            with open('data/training_data.pickle', 'r') as handle:
                self.training_data = np.int32(pkl.load(handle))
                self.training_labels = np.int32(pkl.load(handle))
                self.train_len = len(self.training_data)

        with open('data/testing_data.pickle', 'r') as handle:
            self.testing_data = pkl.load(handle)
            self.testing_labels = pkl.load(handle)

        self.create_model()

        self.saver = tf.train.Saver()

        if(train):
            self.traing_model()
        else:
            self.saver.restore(self.sess, "model/model.ckpt")
            self.run_test()



    def create_model(self):

        self.X = tf.placeholder(tf.float32, [None, 10])
        self.Y = tf.placeholder(tf.float32, [None, 5])
        self.keep = tf.placeholder(tf.float32)

        self.W1 = tf.get_variable('W1',[10,50])
        self.b1 = tf.get_variable('b1',[50])
        self.W2 = tf.get_variable('W2', [50, 50])
        self.b2 = tf.get_variable('b2', [50])
        self.W3 = tf.get_variable('W3', [50, 5])
        self.b3 = tf.get_variable('b3', [5])

        res = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(self.X,self.W1),self.b1)),self.keep)
        res = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(res,self.W2),self.b2)),self.keep)
        self.res = tf.add(tf.matmul(res,self.W3),self.b3)

        self.loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.res,self.Y),1),0)

        self.optimizer = tf.train.AdamOptimizer(0.00001).minimize(self.loss)

    def traing_model(self):

        self.sess.run(tf.initialize_all_variables())

        losses = []
        inds = []

        best_loss = 100000
        for i in xrange(0,200001):

            indexes = np.random.randint(0,self.train_len-1,1000)

            batch_x = self.training_data[indexes]
            batch_y = self.training_labels[indexes]

            self.sess.run(self.optimizer,{self.X: batch_x, self.Y: batch_y,self.keep:0.9})

            if(i%1000==0):
                loss = self.sess.run(self.loss,{self.X: self.testing_data, self.Y: self.testing_labels,self.keep:1.0})
                print("At iteration "+str(i)+" loss: "+str(loss)+'\n')

                losses.append(loss)
                inds.append(i)

                if(loss<best_loss):
                    best_loss = loss
                    save_path = self.saver.save(self.sess, "model/model.ckpt")
                    print("Model saved in path: %s" % save_path)

        plt.plot(inds,losses)
        plt.title('Training Loss (RMSE)')
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.show()


    def run_test(self):
        loss = self.sess.run(self.loss, {self.X: self.testing_data, self.Y: self.testing_labels, self.keep: 1.0})

    def predict(self,data):

        data[0:5] = (data[0:5] - self.mean_dst)/self.std_dst
        data[5:] = (data[5:] - self.mean_imf) / self.std_imf


        res = self.sess.run(self.res, {self.X: [data], self.keep: 1.0})
        res = res[0]
        p_dst = (res*self.std_dst)+self.mean_dst

        return p_dst



