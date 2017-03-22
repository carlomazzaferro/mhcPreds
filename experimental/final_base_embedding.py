import os
import pandas
import numpy
import mhcPreds.utils
import tensorflow as tf
import tflearn
from mhcPreds.amino_acid import Data



if __name__ == '__main__':

    tensorboard_dir = '/tmp/tflearn_logs'
    path_to_data = os.path.dirname(os.getcwd()) + '/mhcPreds/data/'

    train_dat = Data(path_to_data + 'train.txt', allele='HLA-A-0101')
    test_dat = Data(path_to_data + 'test.txt', allele='HLA-A*01:01')

    """
    onehot, aff_one_hot, idx_one_hot = train_dat.one_hot_encoding(kmer_size=9)
    test_onehot, test_aff_one_hot, test_idx_one_hot = test_dat.one_hot_encoding(kmer_size=9)
    aff_one_hot = utils.ic50_to_regression_target(aff_one_hot, max_ic50=50000)
    """

    kmer_test, kmer_aff_test, test_idx_kmet = test_dat.kmer_index_encoding(kmer_size=9)
    kmer, aff_kmer, idx_kmer = train_dat.kmer_index_encoding(kmer_size=9)

    aff_kmer = mhcPreds.utils.ic50_to_regression_target(aff_kmer, max_ic50=50000)
    kmer_aff_test = mhcPreds.utils.ic50_to_regression_target(kmer_aff_test, max_ic50=50000)

    xTr, xTe, yTr, yTe = kmer, kmer_test, aff_kmer, kmer_aff_test

    yTr = numpy.reshape(yTr, (yTr.shape[0], 1))
    yTe = numpy.reshape(yTe, (yTe.shape[0], 1))

    print(xTr.shape, xTe.shape, yTr.shape, yTe.shape)

    x = tf.placeholder(shape=(None, 9), dtype=tf.float32)
    y_ = tf.placeholder(shape=(None, 1), dtype=tf.float32)
    keep_prob = tf.placeholder(tf.float32)

    batch_size = 75
    epochs = 800
    lr = 0.00001

    net = tflearn.input_data(placeholder=x)
    net = tflearn.embedding(net, input_dim=21, output_dim=32, weights_init='xavier')
    net = tflearn.fully_connected(net, 100, activation='prelu')
    net = tflearn.layers.normalization.batch_normalization(net)
    net = tflearn.dropout(net, 0.1)
    net = tflearn.fully_connected(net, 1, activation='sigmoid')

    with tf.name_scope('Loss'):
        with tf.name_scope('MSE'):
            loss = tf.reduce_mean(tf.square(net - y_))
    tf.summary.scalar('MSE', loss)

    with tf.name_scope('train'):
        train_op = tf.train.RMSPropOptimizer(lr).minimize(loss)

    with tf.name_scope('accuracy'):
        accuracy = tf.contrib.metrics.streaming_root_mean_squared_error(net, y_)
    tf.summary.scalar('accuracy', accuracy)

    merged_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        tflearn.is_training(True, session=sess)
        summary_writer = tf.summary.FileWriter(tensorboard_dir, graph=tf.get_default_graph())

        for step in range(epochs):
            total_batch = int(xTr.shape[0] / batch_size)

            for i in range(total_batch):

                batch_x, batch_y = mhcPreds.utils.get_batch2d(xTr, yTr, batch_size)
                sess.run(train_op, feed_dict={x: batch_x, y_: batch_y})

            if step % 10 == 0:
                # Calculate batch loss and accuracy
                # loss= sess.run(loss, feed_dict={x: batch_x, y_: batch_y})
                acc = sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y})

                print ("Iter " + str(step * batch_size) + ", Training RMSE " + str(acc))


        tflearn.is_training(False, session=sess)
        acc = sess.run([accuracy], feed_dict={x: xTe, y_: yTe})
        print('Testing RMSE:' + str(acc))
        preds = sess.run(net, feed_dict={x: xTe})

    preds = numpy.array([mhcPreds.utils.regression_target_to_ic50(i[0]) for i in preds])
    targs = numpy.array([mhcPreds.utils.regression_target_to_ic50(i[0]) for i in yTe])
    print(targs.shape, preds.shape)
    print(mhcPreds.utils.make_scores(targs, preds))
    print(pandas.DataFrame([preds, targs]).T.head())

