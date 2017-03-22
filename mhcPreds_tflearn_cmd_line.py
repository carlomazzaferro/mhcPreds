from amino_acid import Data
import os
import utils
import numpy
import tensorflow as tf
import tflearn
import pandas
import random
import argparse


class TFLearnPepPred(object):

    # There are more, but I'll restrict to this for now
    POSSIBLE_ALLELES = ['A3101', 'B1509', 'B2703', 'B1517', 'B1801', 'B1501', 'B4002', 'B3901', 'B5701', 'A6801',
                        'B5301', 'A2301', 'A2902', 'B0802', 'A3001', 'A0301', 'A0202', 'A0101', 'B4001', 'B5101',
                        'A1101', 'B4402', 'B0803', 'B5801', 'A2601', 'A0203', 'A3002', 'B4601', 'A3301', 'A6802',
                        'B3801', 'A3201', 'B3501', 'A2603', 'B0702', 'A6901', 'B0801', 'B4501', 'A0206', 'A0201',
                        'B1503', 'A2602', 'A8001', 'A2402', 'B2705', 'B4403', 'A2501', 'B5401']

    TRAIN_DEFAULTS = ['A0201', 'A0301', 'A0203', 'A1101', 'A0206', 'A3101']
    AVAILABLE_MODELS = ['deep_rnn', 'embedding_rnn', 'bi_rnn']
    DATA_ENCODINGS = ['one_hot', 'kmer_embedding']

    def __init__(self, data_encoding='kmer_embedding', allele=None, kmer_size=9, verbose=None, data_dir=None):

        self.data_encoding = data_encoding
        self.path_to_data = os.path.dirname(os.getcwd()) + '/mhcPreds/data/'
        self.kmer = kmer_size
        self.xTr, self.xTe, self.yTr, self.yTe, self.idxTr, self.idxTe = self.generate_train_test_data(allele=allele, kmer_size=kmer_size)
        self.verbose = verbose or 0
        self.data_dir = data_dir

    def generate_train_test_data(self, allele=None, kmer_size=9):

        train_dat = Data(self.path_to_data + 'train.txt', allele=allele, train=True)
        test_dat = Data(self.path_to_data + 'test.txt', allele=allele, train=False)

        if self.data_encoding == 'kmer_embedding':
            kmer, aff_kmer, idx_kmer = train_dat.kmer_index_encoding(kmer_size=9)
            aff_kmer = utils.ic50_to_regression_target(aff_kmer, max_ic50=50000)

            kmer_test, aff_kmer_test, idx_kmer_test = test_dat.kmer_index_encoding(kmer_size=kmer_size)
            aff_kmer_test = utils.ic50_to_regression_target(aff_kmer_test, max_ic50=50000)

        else:
            kmer, aff_kmer, idx_kmer = train_dat.one_hot_encoding(kmer_size=9)
            aff_kmer = utils.ic50_to_regression_target(aff_kmer, max_ic50=50000)

            kmer_test, aff_kmer_test, idx_kmer_test = test_dat.one_hot_encoding(kmer_size=kmer_size)
            aff_kmer_test = utils.ic50_to_regression_target(aff_kmer_test, max_ic50=50000)

        xTr, xTe, yTr, yTe = kmer, kmer_test, aff_kmer, aff_kmer_test


        yTr = numpy.reshape(yTr, (yTr.shape[0], 1))
        yTe = numpy.reshape(yTe, (yTe.shape[0], 1))
        print(xTr.shape, xTe.shape, yTr.shape, yTe.shape)

        #if self.model_type == 'embedding_rnn':
        #    xTr = numpy.reshape(xTr, (xTr.shape[0], xTr.shape[1] * xTr.shape[2]))
        #    xTe = numpy.reshape(xTe, (xTe.shape[0], xTe.shape[1] * xTe.shape[2]))

        return xTr, xTe, yTr, yTe, idx_kmer, idx_kmer_test

    def optimizer(self, lr):
        opt = tflearn.RMSProp(learning_rate=lr) #, decay=0.9)
        return opt

    def loss_func(self, y_pred, y_true):
        return tf.reduce_mean(tf.square(y_pred - y_true))

    def accuracy(self, y_pred, y_true):
        return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_pred, y_true))))

    def add_deep_layers(self, net, model_type, out_embedding_dim, layer_size, n_layers):

        if model_type == 'embedding_rnn':
            out_rnn = tflearn.embedding(net, input_dim=21, output_dim=out_embedding_dim, weights_init='xavier')

        elif model_type == 'bi_rnn':
            out_rnn = tflearn.bidirectional_rnn(net, tflearn.BasicLSTMCell(layer_size), tflearn.BasicLSTMCell(layer_size))

        elif model_type == 'deep_rnn':
            for i in range(n_layers):
                net = tflearn.lstm(net, n_units=layer_size, return_seq=True)
            out_rnn = tflearn.lstm(net, layer_size)

        elif model_type == 'basic_rnn':
            out_rnn = tflearn.simple_rnn(net, layer_size)

        else:
            out_rnn = net

        return out_rnn

    def model(self, model_type=None, out_embedding_dim=32, layer_size=32, tensorboard_verbose=3, batch_norm=2, n_layers=2, learning_rate=0.001):

        if self.data_encoding == 'one_hot':
            input_shape = [None, self.kmer, 20]

        else:
            input_shape = [None, self.kmer]

        # Adding layers based on model type
        net = tflearn.input_data(shape=input_shape)
        deep_layers_output = self.add_deep_layers(net, model_type, out_embedding_dim, layer_size, n_layers)
        net = tflearn.fully_connected(deep_layers_output, 100, activation='prelu')

        if batch_norm > 0:
            net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.dropout(net, 0.4)
        net = tflearn.fully_connected(net, 1, activation='sigmoid')
        if batch_norm > 1:
            net = tflearn.layers.normalization.batch_normalization(net)

        with tf.name_scope("TargetsData"):  # placeholder for target variable (i.e. trainY input)
            targetY = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="Y")

        network = tflearn.regression(net,
                                     placeholder=targetY,
                                     optimizer=self.optimizer(learning_rate),
                                     learning_rate=learning_rate,
                                     loss=tflearn.mean_square(net, targetY),
                                     metric=self.accuracy(net, targetY))

        model = tflearn.DNN(network, tensorboard_verbose=tensorboard_verbose)
        return model

    def train(self, model, num_epochs=20, batch_size=64, validation_set=0.1, run_id=None):

        model.fit(self.xTr, self.yTr,
                  n_epoch=num_epochs,
                  validation_set=validation_set,
                  batch_size=batch_size,
                  shuffle=True,
                  run_id=run_id)
        print ("Done!")
        return model

    def predict(self, model):
        res = model.predict(self.xTe)
        return utils.scoring(res, self.idxTe, self.yTe)

def CommandLine(arglist=None):
    '''
    Main command line.  Accepts args, to allow for simple unit testing.
    '''
    help_text = """
    Commands:
    train - give size of training set to use, as argument
    predict - give input sequence as argument (or specify inputs via --from-file <filename>)
    """
    parser = argparse.ArgumentParser(description=help_text, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "-cmd",
        help="command")

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64)

    parser.add_argument(
        "-bn",
        "--batch-norm",
        type=int,
        help="Perform batch norm either: only after LSTM (1), after and before (2)",
        default=1)

    parser.add_argument(
        "-ls",
        "--layer-size",
        type=int,
        help="Size of inner layeres of RNN",
        default=32)

    parser.add_argument(
        "-nl",
        "--num-layers",
        type=int,
        help="Number of LSTM layers",
        default=1)

    parser.add_argument(
        '-d',
        "--embedding-size",
        type=int,
        help="Embedding layer output dimension",
        default=32)

    parser.add_argument(
        '-a',
        "--allele",
        help="Allele to use for prediction. None predicts for all alleles.",
        default='A0101')

    parser.add_argument(
        "-m",
        "--model",
        help="RNN model. Basic LSTM, Birectional LSTM or simple RNN",
        default='basic_lstm')

    parser.add_argument(
        '-c',
        "--data-encoding",
        type=str,
        help="Embedding layer output dimension",
        default='kmer_embedding')

    parser.add_argument(
        "-r",
        "--learning-rate",
        type=float,
        help="learning rate (default 0.001)",
        default=0.001)

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        help="number of trainig epochs",
        default=10)

    parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="name of model, used when generating default weights filenames",
        default=None)

    parser.add_argument(
        "-l",
        "--len",
        type=int,
        help="size of k-mer to predict on",
        default=9)

    parser.add_argument(
        "-s",
        "--save",
        type=int,
        help="Save model to --data-dir",
        default=0)

    parser.add_argument(
        "--data-dir",
        help="directory to use for saving models",
        default=None)

    parser.add_argument(
        "--cell-size",
        type=int,
        help="size of RNN cell to use (default 32)",
        default=32)

    parser.add_argument(
        "--tensorboard-verbose",
        type=int,
        help="tensorboard verbosity level (default 0)",
        default=0)

    parser.add_argument(
        "--from-file",
        type=str,
        help="name of file to take input data sequences from .tflm format",
        default=None)

    parser.add_argument(
        "--run-id",
        type=str,
        help="Name of run to be displayed in tensorboard and results folder",
        default=str(random.randint(0, 100000)))

    args = parser.parse_args(arglist)

    # def model(self, type=None,  out_embedding_dim=32, layer_size=32, tensorboard_verbose=3, batch_norm=2, n_layers=2):
    model_params = dict(type=args.model,
                        out_embedding_dim=args.embedding_size,
                        layer_size=args.layer_size,
                        batch_norm=args.batch_norm,
                        n_layers=args.num_layers,
                        tensorboard_verbose=args.tensorboard_verbose,
                        batch_size=args.batch_size)

    if args.cmd == "train_test_eval":
        pipe = TFLearnPepPred(data_encoding=args.data_encoding,
                              allele=args.allele,
                              kmer_size=args.len,
                              data_dir=args.data_dir)

        print(args.learning_rate)
        mymodel = pipe.model(model_type=args.model,
                             out_embedding_dim=args.embedding_size,
                             layer_size=args.layer_size,
                             batch_norm=args.batch_norm,
                             n_layers=args.num_layers,
                             learning_rate=args.learning_rate,
                             tensorboard_verbose=args.tensorboard_verbose)

        trained_model = pipe.train(mymodel,
                                   num_epochs=args.epochs,
                                   batch_size=args.batch_size,
                                   validation_set=0.1,
                                   run_id=args.run_id)

        print("Finished training: model -> " + args.model + ", on allele -> " + args.allele + ", of kmer size -> " + str(args.len))

        if args.save == 1:
            trained_model.save(args.data_dir + args.model + args.allele + ".tfl")

        print("Predicting ...")
        scores, df = pipe.predict(trained_model)
        print(scores, df)
        write_results(args, scores, df)
        return trained_model, scores, df


def write_results(args, scores, df):
    res_folder = os.path.dirname(os.getcwd()) + '/mhcPreds/results/'
    os.mkdir(res_folder + args.run_id)

    with open(res_folder + args.run_id + '/params_and_scores.txt', 'w') as result:

        result.write('PARAMETERS: \n')
        for arg in vars(args):
            result.write(arg + ': ' + str(getattr(args, arg)) + '\n')

        result.write('SCORES: \n \n')
        result.write('Tau: %.5f \n' % scores['tau'])
        result.write('F1: %.5f \n' % scores['f1'])
        result.write('AUC: %.5f \n' % scores['auc'])

    df.to_csv(res_folder + args.run_id + '/predictions.csv')


if __name__ == "__main__":

    CommandLine()

# python oop_tflearn_cmd_line.py -cmd 'train_test_eval' -e 1 -bn 1 -nl 3 -c 'one_hot' -a 'A0101' -m 'deep_rnn' -r 0.001
