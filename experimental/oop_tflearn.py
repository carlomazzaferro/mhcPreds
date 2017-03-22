from mhcPreds.amino_acid import Data
import os
import mhcPreds.utils
import numpy
import tensorflow as tf
import tflearn
from sklearn.model_selection import train_test_split
import pandas


class TFLearnPepPred(object):
    '''
    seq2seq recurrent neural network, implemented using TFLearn.
    '''
    AVAILABLE_MODELS = ["embedding_rnn", "embedding_attention"]
    POSSIBLE_ALLELES = ['A3101', 'B1509', 'B2703', 'B1517', 'B1801', 'B1501', 'B4002', 'B3901', 'B5701', 'A6801',
                        'B5301', 'A2301', 'A2902', 'B0802', 'A3001', 'A0301', 'A0202', 'A0101', 'B4001', 'B5101',
                        'A1101', 'B4402', 'B0803', 'B5801', 'A2601', 'A0203', 'A3002', 'B4601', 'A3301', 'A6802',
                        'B3801', 'A3201', 'B3501', 'A2603', 'B0702', 'A6901', 'B0801', 'B4501', 'A0206', 'A0201',
                        'B1503', 'A2602', 'A8001', 'A2402', 'B2705', 'B4403', 'A2501', 'B5401']


    def __init__(self, allele=None, kmer_size=9, batch_size=64, verbose=None, data_dir=None):

        self.path_to_data = os.path.dirname(os.getcwd()) + '/mhcPreds/data/'
        self.xTr, self.xTe, self.yTr, self.yTe = self.generate_train_test_data(allele=allele, kmer_size=kmer_size)
        self.batch_size = batch_size
        self.verbose = verbose or 0
        self.data_dir = data_dir

    def generate_train_test_data(self, allele=None, kmer_size=9):

        train_dat = Data(self.path_to_data + 'train.txt', allele=allele,  train=True)
        test_dat = Data(self.path_to_data + 'test.txt', allele=allele, train=False)

        print('PARSED')

        kmer, aff_kmer, idx_kmer = train_dat.kmer_index_encoding(kmer_size=9)
        aff_kmer = mhcPreds.utils.ic50_to_regression_target(aff_kmer, max_ic50=50000)

        kmer_test, aff_kmer_test, idx_kmer_test = test_dat.kmer_index_encoding(kmer_size=kmer_size)
        aff_kmer_test = mhcPreds.utils.ic50_to_regression_target(aff_kmer_test, max_ic50=50000)

        xTr, xTe, yTr, yTe = kmer, kmer_test, aff_kmer, aff_kmer_test


        yTr = numpy.reshape(yTr, (yTr.shape[0], 1))
        yTe = numpy.reshape(yTe, (yTe.shape[0], 1))
        #xTr = numpy.reshape(xTr, (yTr.shape[0], 180))
        #xTe = numpy.reshape(xTe, (yTe.shape[0], 180))

        print(xTr.shape, xTe.shape, yTr.shape, yTe.shape)

        return xTr, xTe, yTr, yTe

    def optimizer(self, lr):
        opt = tflearn.RMSProp(learning_rate=lr)
        return opt

    def loss_func(self, y_pred, y_true):
        return tf.reduce_mean(tf.square(y_pred - y_true))

    def accuracy(self, y_pred, y_true):
        return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_pred, y_true))))

    def model(self, type=None, mode="train", num_layers=2, state_size=32, learning_rate=0.001, tensorboard_verbose=3):

        net = tflearn.input_data(shape=[None, 9])
        net = tflearn.embedding(net, input_dim=21, output_dim=32, weights_init='xavier')

        if type == 'bi_rnn':
            out_rnn = tflearn.bidirectional_rnn(net, tflearn.BasicLSTMCell(32), tflearn.BasicLSTMCell(32))

        elif type == 'basic_lstm':
            for i in range(4):
                net = tflearn.lstm(net, n_units=40, return_seq=True)
            #net = tflearn.lstm(net, n_units=40, return_seq=True)
            out_rnn = tflearn.lstm(net, n_units=40, return_seq=False)

        elif type == 'basic_rnn':
            out_rnn = tflearn.simple_rnn(net, 40)

        else:
            out_rnn = net

        net = tflearn.fully_connected(out_rnn, 100, activation='prelu')
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.dropout(net, 0.1)
        net = tflearn.fully_connected(net, 1, activation='sigmoid')

        """
        single_cell = getattr(tf.contrib.rnn, cell_type)(cell_size, state_is_tuple=True)

        if num_layers == 1:
            cell = single_cell
        else:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell] * num_layers)
        """

        with tf.name_scope("TargetsData"):  # placeholder for target variable (i.e. trainY input)
            targetY = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="Y")

        network = tflearn.regression(net,
                                     placeholder=targetY,
                                     optimizer=self.optimizer(learning_rate),
                                     learning_rate=learning_rate,
                                     loss=tflearn.mean_square(net, targetY),
                                     metric=self.accuracy(net, targetY),
                                     name='no rnn')

        model = tflearn.DNN(network, tensorboard_verbose=tensorboard_verbose)
        return model

    def train(self, model, num_epochs=15, validation_set=0.1):
        '''
        Train model, with specified number of epochs, and dataset size.
        Use specified model, or create one if not provided.  Load initial weights from file weights_input_fn,
        if provided. validation_set specifies what to use for the validation.
        Returns logits for prediction, as an numpy array of shape [out_seq_len, n_output_symbols].
        '''

        model.fit(self.xTr, self.yTr,
                  n_epoch=num_epochs,
                  validation_set=validation_set,
                  batch_size=self.batch_size,
                  shuffle=True
                  )
        print ("Done!")
        return model

    def predict(self, model):
        res = model.predict(self.xTe)
        return self.scoring(res)

    def scoring(self, preds):

        preds = numpy.array([mhcPreds.utils.regression_target_to_ic50(i[0]) for i in preds])
        targs = numpy.array([mhcPreds.utils.regression_target_to_ic50(i[0]) for i in self.yTe])
        scores = mhcPreds.utils.make_scores(targs, preds)
        as_df = pandas.DataFrame([preds, targs])

        return scores, as_df

if __name__ == '__main__':

    pipe = TFLearnPepPred(allele='A0101', kmer_size=9, batch_size=64, verbose=3, data_dir=None)
    model = pipe.model(type='N', mode="train", num_layers=2, state_size=10, learning_rate=0.001)
    trained = pipe.train(model)
    scores, df = pipe.predict(trained)
    print(scores, df.T)