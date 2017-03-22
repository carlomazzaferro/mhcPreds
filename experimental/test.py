import os
import pandas
import mhcPreds.utils
from mhcPreds.amino_acid import Data
import pickle


def possible_human_allels():

    path_to_train_data = os.path.dirname(os.getcwd()) + '/mhcPreds/data/train.txt'
    path_to_test_data = os.path.dirname(os.getcwd()) + '/mhcPreds/data/test.txt'
    path_to_full_data = os.path.dirname(os.getcwd()) + '/mhcPreds/data/full_set.txt'

    dfTr = pandas.read_csv(path_to_train_data, sep='\t')
    dfTe = pandas.read_csv(path_to_test_data, sep='\t')
    dfF  = pandas.read_csv(path_to_full_data, sep='\t')

    transfTr = mhcPreds.utils.get_all_human_alleles_formatting_two(dfTr)
    transfTe = mhcPreds.utils.get_all_human_alleles_formatting_one(dfTe)
    transfF = mhcPreds.utils.get_all_human_alleles_formatting_one(dfF)

    possible_alleles = (set(transfF.keys()).intersection(transfTr.keys()).intersection(transfTe.keys()))

    return possible_alleles

def create_mapping():

    path_to_train_data = os.path.dirname(os.getcwd()) + '/mhcPreds/data/train.txt'
    path_to_test_data = os.path.dirname(os.getcwd()) + '/mhcPreds/data/test.txt'
    path_to_full_data = os.path.dirname(os.getcwd()) + '/mhcPreds/data/full_set.txt'

    dfTr = pandas.read_csv(path_to_train_data, sep='\t')
    dfTe = pandas.read_csv(path_to_test_data, sep='\t')
    dfF  = pandas.read_csv(path_to_full_data, sep='\t')

    transfTr = mhcPreds.utils.get_all_human_alleles_formatting_two(dfTr)
    transfTe = mhcPreds.utils.get_all_human_alleles_formatting_one(dfTe)
    transfF = mhcPreds.utils.get_all_human_alleles_formatting_one(dfF)

    maps = [transfTr, transfTe, transfF]

    super_dict = {}
    for k in set(k for d in maps for k in d):
        super_dict[k] = [d[k] for d in maps if k in d]

    pkl_dir = os.path.dirname(os.getcwd()) + '/mhcPreds/extras/allele_mapping'
    mhcPreds.utils.save_obj(super_dict, pkl_dir)


if __name__ == '__main__':

    POSSIBLE_ALLELES = ['A3101', 'B1509', 'B2703', 'B1517', 'B1801', 'B1501', 'B4002', 'B3901', 'B5701', 'A6801',
                        'B5301', 'A2301', 'A2902', 'B0802', 'A3001', 'A0301', 'A0202', 'A0101', 'B4001', 'B5101',
                        'A1101', 'B4402', 'B0803', 'B5801', 'A2601', 'A0203', 'A3002', 'B4601', 'A3301', 'A6802',
                        'B3801', 'A3201', 'B3501', 'A2603', 'B0702', 'A6901', 'B0801', 'B4501', 'A0206', 'A0201',
                        'B1503', 'A2602', 'A8001', 'A2402', 'B2705', 'B4403', 'A2501', 'B5401']

    pkl_dir = os.path.dirname(os.getcwd()) + '/mhcPreds/extras/allele_mapping'


    a = mhcPreds.utils.load_obj(pkl_dir)

    path_to_train_data = os.path.dirname(os.getcwd()) + '/mhcPreds/data/train.txt'
    path_to_test_data = os.path.dirname(os.getcwd()) + '/mhcPreds/data/test.txt'
    path_to_full_data = os.path.dirname(os.getcwd()) + '/mhcPreds/data/full_set.txt'

    dfTr = pandas.read_csv(path_to_train_data, sep='\t')
    dfTe = pandas.read_csv(path_to_test_data, sep='\t')
    dfF  = pandas.read_csv(path_to_full_data, sep='\t')

    ls1 = []
    ls2 = []
    ls3 = []

    for i in POSSIBLE_ALLELES:

        l1 = len(dfTr[dfTr.mhc == a[i][0]])
        ls1.append((i, l1))
        l2 = len(dfTe[dfTe.mhc == a[i][1]])
        ls2.append((i, l2))
        l3 = len(dfF[dfF.mhc == a[i][1]])
        ls2.append((i, l3))

    pandas.DataFrame(ls1, columns=['allele', 'train_samples']).sort_values(by='train_samples', ascending=False).to_csv(os.path.dirname(os.getcwd()) + '/mhcPreds/extras/allele_train_count.csv')
    pandas.DataFrame(ls2, columns=['allele', 'test_samples']).sort_values(by='test_samples', ascending=False).to_csv(os.path.dirname(os.getcwd()) + '/mhcPreds/extras/allele_test_count.csv')
    pandas.DataFrame(ls3, columns=['allele', 'all_samples']).sort_values(by='all_samples', ascending=False).to_csv(os.path.dirname(os.getcwd()) + '/mhcPreds/extras/allele_full_count.csv')

    #print(pandas.DataFrame(ls1, columns=['allele', 'train_samples']).sort_values(by='train_samples', ascending=False).allele.values)

    #mapp_1 = utils.get_all_human_alleles_formatting_two(df)
   #
   # print (mapp)
   # for all in my_alls:
    #    print(mapp[all])
import mhcPreds.utils
import numpy
import tensorflow as tf
import tflearn
from sklearn.model_selection import train_test_split
import pandas


class TFLearnPepPred(object):

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
            aff_kmer = mhcPreds.utils.ic50_to_regression_target(aff_kmer, max_ic50=50000)

            kmer_test, aff_kmer_test, idx_kmer_test = test_dat.kmer_index_encoding(kmer_size=kmer_size)
            aff_kmer_test = mhcPreds.utils.ic50_to_regression_target(aff_kmer_test, max_ic50=50000)

        else:
            kmer, aff_kmer, idx_kmer = train_dat.one_hot_encoding(kmer_size=9)
            aff_kmer = mhcPreds.utils.ic50_to_regression_target(aff_kmer, max_ic50=50000)

            kmer_test, aff_kmer_test, idx_kmer_test = test_dat.one_hot_encoding(kmer_size=kmer_size)
            aff_kmer_test = mhcPreds.utils.ic50_to_regression_target(aff_kmer_test, max_ic50=50000)

        xTr, xTe, yTr, yTe = kmer, kmer_test, aff_kmer, aff_kmer_test


        yTr = numpy.reshape(yTr, (yTr.shape[0], 1))
        yTe = numpy.reshape(yTe, (yTe.shape[0], 1))
        print(xTr.shape, xTe.shape, yTr.shape, yTe.shape)

        #if self.model_type == 'embedding_rnn':
        #    xTr = numpy.reshape(xTr, (xTr.shape[0], xTr.shape[1] * xTr.shape[2]))
        #    xTe = numpy.reshape(xTe, (xTe.shape[0], xTe.shape[1] * xTe.shape[2]))

        return xTr, xTe, yTr, yTe, idx_kmer, idx_kmer_test


if __name__ == '__main__':


    res = pandas.read_csv(os.path.dirname(os.getcwd()) + '/mhcPreds/results/19646/predictions.csv')
    pipe = TFLearnPepPred(allele='A0101')


    res['idx'] = pipe.idxTe
    print(res)

    avg_pred = []
    avg_lab = []
    for i in set(pipe.idxTe):
        sliced = res[res['idx'] == i]
        num_vals = len(sliced)
        avg_pred.append(sum(sliced['0'].values)/num_vals)
        avg_lab.append(sum(sliced['1'].values)/num_vals)

    print(pandas.DataFrame([list(set(pipe.idxTe)), avg_pred, avg_lab]))