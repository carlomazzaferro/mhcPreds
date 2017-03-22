from __future__ import division, print_function

import os
import sys
import argparse
import tflearn
import tempfile
import urllib

import numpy as np
import pandas as pd
import tensorflow as tf

# -----------------------------------------------------------------------------

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = {"workclass": 10, "education": 17, "marital_status": 8,
                       "occupation": 16, "relationship": 7, "race": 6,
                       "gender": 3, "native_country": 43, "age_binned": 14}
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]


# -----------------------------------------------------------------------------

class TFLearnWideAndDeep(object):
    '''
    Wide and deep model, implemented using TFLearn
    '''
    AVAILABLE_MODELS = ["wide", "deep", "wide+deep"]

    def __init__(self, model_type="wide", name=None, tensorboard_verbose=3,
                 deep_learning_rate=0.001, checkpoints_dir=None):
        '''
        model_type = `str`: wide or deep or wide+deep
        verbose = `bool`
        name = `str` used for run_id (defaults to model_type)
        tensorboard_verbose = `int`: logging level for tensorboard (0, 1, 2, or 3)
        wide_learning_rate = `float`: defaults to 0.001
        deep_learning_rate = `float`: defaults to 0.001
        checkpoints_dir = `str`: where checkpoint files will be stored (defaults to "CHECKPOINTS")
        '''
        self.model_type = model_type or "wide+deep"
        assert self.model_type in self.AVAILABLE_MODELS
        self.verbose = 3
        self.tensorboard_verbose = tensorboard_verbose
        self.name = name or self.model_type  # name is used for the run_id
        self.data_columns = COLUMNS
        self.continuous_columns = CONTINUOUS_COLUMNS
        self.categorical_columns = CATEGORICAL_COLUMNS  # dict with category_name: category_size
        self.label_column = LABEL_COLUMN
        self.checkpoints_dir = checkpoints_dir or "CHECKPOINTS"
        if not os.path.exists(self.checkpoints_dir):
            os.mkdir(self.checkpoints_dir)
            print("Created checkpoints directory %s" % self.checkpoints_dir)
        self.build_model(deep_learning_rate)


    def build_model(self, learning_rate=0.001,):
        '''
        Model - wide and deep - built using tflearn
        '''
        n_cc = len(self.continuous_columns)
        input_shape = [None, n_cc]

        wide_inputs = tflearn.input_data(shape=input_shape, name="wide_X")

        with tf.name_scope("Y"):  # placeholder for target variable (i.e. trainY input)
            Y_in = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="Y")

        with tf.variable_op_scope([wide_inputs], None, "cb_unit", reuse=False) as scope:
            central_bias = tflearn.variables.variable('central_bias', shape=[1],
                                                      initializer=tf.constant_initializer(np.random.randn()),
                                                      trainable=True, restore=True)
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/cb_unit', central_bias)

        deep_network = self.deep_model(wide_inputs, n_cc)
        deep_network_with_bias = tf.add(deep_network, central_bias, name="deep_with_bias")
        network = deep_network
        network = tf.add(network, central_bias, name="add_central_bias")

        # add validation monitor summaries giving confusion matrix entries
        with tf.name_scope('Monitors'):
            predictions = tf.cast(tf.greater(network, 0), tf.int64)
            print("predictions=%s" % predictions)
            Ybool = tf.cast(Y_in, tf.bool)
            print("Ybool=%s" % Ybool)
            pos = tf.boolean_mask(predictions, Ybool)
            neg = tf.boolean_mask(predictions, ~Ybool)
            psize = tf.cast(tf.shape(pos)[0], tf.int64)
            nsize = tf.cast(tf.shape(neg)[0], tf.int64)
            true_positive = tf.reduce_sum(pos, name="true_positive")
            false_negative = tf.subtract(psize, true_positive, name="false_negative")
            false_positive = tf.reduce_sum(neg, name="false_positive")
            true_negative = tf.subtract(nsize, false_positive, name="true_negative")
            overall_accuracy = tf.truediv(tf.add(true_positive, true_negative), tf.add(nsize, psize),
                                          name="overall_accuracy")
        vmset = [true_positive, true_negative, false_positive, false_negative, overall_accuracy]

        trainable_vars = tf.trainable_variables()
        tv_deep = [v for v in trainable_vars if v.name.startswith('deep_')]

        tflearn.regression(deep_network_with_bias,
                               placeholder=Y_in,
                               optimizer='adam',
                               # loss='roc_auc_score',
                               loss='binary_crossentropy',
                               metric="accuracy",
                               learning_rate=learning_rate[1],
                               validation_monitors=vmset if not 'wide' in self.model_type else None,
                               trainable_vars=tv_deep,
                               op_name="deep_regression",
                               name="Y")

        self.model = tflearn.DNN(network,
                                 tensorboard_verbose=self.tensorboard_verbose,
                                 max_checkpoints=5,
                                 checkpoint_path="%s/%s.tfl" % (self.checkpoints_dir, self.name),
                                 )


    def deep_model(self, wide_inputs, n_inputs, n_nodes=[100, 50], use_dropout=False):
        '''
        Model - deep, i.e. two-layer fully connected network model
        '''
        cc_input_var = {}
        cc_embed_var = {}
        flat_vars = []

        for cc, cc_size in self.categorical_columns.items():
            cc_input_var[cc] = tflearn.input_data(shape=[None, 1], name="%s_in" % cc, dtype=tf.int32)
            # embedding layers only work on CPU!  No GPU implementation in tensorflow, yet!
            cc_embed_var[cc] = tflearn.layers.embedding_ops.embedding(cc_input_var[cc], cc_size, 8,
                                                                      name="deep_%s_embed" % cc)
            if self.verbose:
                print("    %s_embed = %s" % (cc, cc_embed_var[cc]))
            flat_vars.append(tf.squeeze(cc_embed_var[cc], squeeze_dims=[1], name="%s_squeeze" % cc))

        network = tf.concat(1, [wide_inputs] + flat_vars, name="deep_concat")
        for k in range(len(n_nodes)):
            network = tflearn.fully_connected(network, n_nodes[k], activation="relu", name="deep_fc%d" % (k + 1))
            if use_dropout:
                network = tflearn.dropout(network, 0.5, name="deep_dropout%d" % (k + 1))
        if self.verbose:
            print("Deep model network before output %s" % network)
        network = tflearn.fully_connected(network, 1, activation="linear", name="deep_fc_output", bias=False)
        network = tf.reshape(network, [-1, 1])  # so that accuracy is binary_accuracy
        if self.verbose:
            print("Deep model network %s" % network)
        return network

    def prepare_input_data(self, input_data, name="", category_map=None):
        '''
        Prepare input data dicts
        '''
        print("-" * 40 + " Preparing %s" % name)
        X = input_data[self.continuous_columns].values.astype(np.float32)
        Y = input_data[self.label_column].values.astype(np.float32)
        Y = Y.reshape([-1, 1])
        if self.verbose:
            print("  Y shape=%s, X shape=%s" % (Y.shape, X.shape))

        X_dict = {"wide_X": X}

        # map categorical value strings to integers
        td = input_data
        if category_map is None:
            category_map = {}
            for cc in self.categorical_columns:
                if not cc in td.columns:
                    continue
                cc_values = sorted(td[cc].unique())
                cc_max = 1 + len(cc_values)
                cc_map = dict(
                    zip(cc_values, range(1, cc_max)))  # start from 1 to avoid 0:0 mapping (save 0 for missing)
                if self.verbose:
                    print("  category %s max=%s,  map=%s" % (cc, cc_max, cc_map))
                category_map[cc] = cc_map

        td = td.replace(category_map)

        # bin ages (cuts off extreme values)
        age_bins = [0, 12, 18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 80, 65535]
        td['age_binned'] = pd.cut(td['age'], age_bins, labels=False)
        td = td.replace({'age_binned': {np.nan: 0}})
        print("  %d age bins: age bins = %s" % (len(age_bins), age_bins))

        X_dict.update(
            {("%s_in" % cc): td[cc].values.astype(np.int32).reshape([-1, 1]) for cc in self.categorical_columns})

        Y_dict = {"Y": Y}

        return X_dict, Y_dict, category_map

    def train(self, n_epoch=1000, snapshot_step=10, batch_size=None):

        self.X_dict, self.Y_dict, category_map = self.prepare_input_data(self.train_data, "train data")
        self.testX_dict, self.testY_dict, _ = self.prepare_input_data(self.test_data, "test data", category_map)
        validation_batch_size = batch_size or self.testY_dict['Y'].shape[0]
        batch_size = batch_size or self.Y_dict['Y'].shape[0]

        print("Input data shape = %s; output data shape=%s, batch_size=%s" % (str(self.X_dict['wide_X'].shape),
                                                                              str(self.Y_dict['Y'].shape),
                                                                              batch_size))
        print("Test data shape = %s; output data shape=%s, validation_batch_size=%s" % (
        str(self.testX_dict['wide_X'].shape),
        str(self.testY_dict['Y'].shape),
        validation_batch_size))
        print("=" * 60 + "  Training")
        self.model.fit(self.X_dict,
                       self.Y_dict,
                       n_epoch=n_epoch,
                       validation_set=(self.testX_dict, self.testY_dict),
                       snapshot_step=snapshot_step,
                       batch_size=batch_size,
                       validation_batch_size=validation_batch_size,
                       show_metric=True,
                       snapshot_epoch=False,
                       shuffle=True,
                       run_id=self.name,
                       )

    def evaluate(self):
        logits = np.array(self.model.predict(self.testX_dict)).reshape([-1])
        print("=" * 60 + "  Evaluation")
        print("  logits: %s, min=%s, max=%s" % (logits.shape, logits.min(), logits.max()))
        probs = 1.0 / (1.0 + np.exp(-logits))
        y_pred = pd.Series((probs > 0.5).astype(np.int32))
        Y = pd.Series(self.testY_dict['Y'].astype(np.int32).reshape([-1]))
        self.confusion_matrix = self.output_confusion_matrix(Y, y_pred)
        print("=" * 60)

    def output_confusion_matrix(self, y, y_pred):
        assert y.size == y_pred.size
        print("Actual IDV")
        print(y.value_counts())
        print("Predicted IDV")
        print(y_pred.value_counts())
        print()
        print("Confusion matrix:")
        cmat = pd.crosstab(y_pred, y, rownames=['predictions'], colnames=['actual'])
        print(cmat)
        sys.stdout.flush()
        return cmat


# -----------------------------------------------------------------------------

def CommandLine(args=None):
    '''
    Main command line.  Accepts args, to allow for simple unit testing.
    '''
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    if args:
        FLAGS.__init__()
        FLAGS.__dict__.update(args)

    try:
        flags.DEFINE_string("model_type", "wide+deep", "Valid model types: {'wide', 'deep', 'wide+deep'}.")
        flags.DEFINE_string("run_name", None, "name for this run (defaults to model type)")
        flags.DEFINE_string("load_weights", None, "filename with initial weights to load")
        flags.DEFINE_string("checkpoints_dir", None, "name of directory where checkpoints should be saved")
        flags.DEFINE_integer("n_epoch", 200, "Number of training epoch steps")
        flags.DEFINE_integer("snapshot_step", 100, "Step number when snapshot (and validation testing) is done")
        flags.DEFINE_float("wide_learning_rate", 0.001, "learning rate for the wide part of the model")
        flags.DEFINE_float("deep_learning_rate", 0.001, "learning rate for the deep part of the model")
        flags.DEFINE_boolean("verbose", False, "Verbose output")
    except argparse.ArgumentError:
        pass  # so that CommandLine can be run more than once, for testing

    twad = TFLearnWideAndDeep(model_type=FLAGS.model_type, verbose=FLAGS.verbose,
                              name=FLAGS.run_name, wide_learning_rate=FLAGS.wide_learning_rate,
                              deep_learning_rate=FLAGS.deep_learning_rate,
                              checkpoints_dir=FLAGS.checkpoints_dir)
    twad.load_data()
    if FLAGS.load_weights:
        print("Loading initial weights from %s" % FLAGS.load_weights)
        twad.model.load(FLAGS.load_weights)
    twad.train(n_epoch=FLAGS.n_epoch, snapshot_step=FLAGS.snapshot_step)
    twad.evaluate()
    return twad