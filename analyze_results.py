import glob
import os
import pandas
import utils
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


class Results(object):

    MAIN_DIR =  os.path.dirname(os.getcwd()) + '/mhcPreds/results/'
    SUMMARY_DIR = os.path.dirname(os.getcwd()) + '/mhcPreds/results_summary/'
    REGEX_EMB = '*kmer_embedding*'
    REGEX_DEEP_RNN = '*deep_rnn*'
    RESULT = '/params_and_scores.txt'
    ALLELES = ["A0201", "A0101", "A0301", "A0203", "A1101", "A0206", "A3101"]
    NUMERIC = ['epochs', 'batch_size', 'num_layers', 'learning_rate', 'AUC', 'F1', 'Tau']

    def __init__(self):

        self.KMER_SCORES_DF = self.retrieve_files(self.REGEX_EMB)
        self.DEEP_RNN_DF = self.retrieve_files(self.REGEX_DEEP_RNN)
        self.allele_specific_score_KMER = self.allele_specific(self.KMER_SCORES_DF)
        self.allele_specific_score_RNN = self.allele_specific(self.DEEP_RNN_DF)

    def retrieve_files(self, regex):

        scores_etc = []
        directories = glob.glob(self.MAIN_DIR + regex)
        for directory in directories:
            scores_etc.append(self.open_result_file(directory))

        return scores_etc

    def open_result_file(self, f):

        lines = []
        sc = []
        with open(f + self.RESULT) as inf:
            next(inf)
            for line in inf:
                if 'SCORES' in line:
                    next(inf)
                type_ = line.strip().split(':')[0]
                val = line.strip().split(':')[1][1:]
                if type_ in self.NUMERIC:
                    lines.append([type_, float(val)])
                else:
                    lines.append([type_, val])
                #else:
                sc.append(line.strip())

        df = pandas.DataFrame(lines, columns=['PARAMS', 'VALUES']).set_index('PARAMS')
        df = df.drop(['SCORES'])

        return df

    def allele_specific(self, dfs):

        all_spec = {allele : [] for allele in self.ALLELES}

        for df in dfs:
            for allele in self.ALLELES:
                if allele in df.VALUES.values:
                    all_spec[allele].append({'SCORES': {'AUC' : df.loc['AUC'].values[0],
                                                        'TAU' : df.loc['Tau'].values[0],
                                                        'F1'  : df.loc['F1'].values[0]
                                                       },
                                             'run_id' : df.loc['run_id'].values[0]
                                             }
                                            )


        return all_spec


    def return_highest_and_avg_scores(self, allele_specific_data):

        largest_score_per_allele = {}

        for allele in self.ALLELES:

            labs = []
            taus = []
            aucs = []
            f1s = []

            allele_data = allele_specific_data[allele]
            for scores in allele_data:

                labs.append(scores['run_id'])
                taus.append(scores['SCORES']['TAU'])
                aucs.append(scores['SCORES']['AUC'])
                f1s.append(scores['SCORES']['F1'])

            MAX_AUC = numpy.argmax(aucs)

            largest_score_per_allele[allele] = [('run', labs[MAX_AUC]), ('tau', taus[MAX_AUC]), ('f1', f1s[MAX_AUC]),  ('max_auc', aucs[MAX_AUC])]
            largest_score_per_allele[allele].append(('avg_auc', numpy.mean(aucs)))

        return largest_score_per_allele

    def create_summary_csvs(self):

        score_summaries_kmer = self.return_highest_and_avg_scores(self.allele_specific_score_KMER)
        score_summaries_rnn = self.return_highest_and_avg_scores(self.allele_specific_score_RNN)
        for allele in ALLELES:
            kmers = pandas.DataFrame(score_summaries_kmer[allele]).set_index(0)
            rnns = pandas.DataFrame(score_summaries_rnn[allele]).set_index(0)
            pandas.concat([kmers, rnns], axis=1).to_csv(self.MAIN_DIR + allele + '.csv')


    def plot_roc(self):

        score_summaries_kmer = self.return_highest_and_avg_scores(self.allele_specific_score_KMER)
        score_summaries_rnn = self.return_highest_and_avg_scores(self.allele_specific_score_RNN)

        plt.figure(figsize=(30, 20))

        for i, allele in enumerate(self.ALLELES[0:6]):

            pred_file_1 = self.MAIN_DIR + score_summaries_rnn[allele][0][1]
            pred_file_2 = self.MAIN_DIR + score_summaries_kmer[allele][0][1]

            results_1 = pandas.read_csv(pred_file_1 + '/predictions.csv')
            results_2 = pandas.read_csv(pred_file_2 + '/predictions.csv')

            fpr, tpr, roc_auc = self.return_auc_metric(results_1)
            fpr1, tpr1, roc_auc1 = self.return_auc_metric(results_2)


            self.make_plot(fpr, tpr, roc_auc, fpr1, tpr1, roc_auc1, allele + 'A', i)

    def return_auc_metric(self, df):

        ic50_y_pred = df['Avg_Pred'].values
        ic50_y =  df['Original_Label'].values
        y_pred = utils.ic50_to_regression_target(ic50_y_pred, 50000)
        fpr, tpr, _ = roc_curve(ic50_y <= 500, y_pred)
        roc_auc = auc(fpr, tpr)

        return fpr, tpr, roc_auc


    def make_plot(self, fpr, tpr, roc_auc, fpr2, tpr2, roc_auc2, name_, subplot_no):

        lw = 2
        plt.subplot(2, 3, subplot_no + 1)
        plt.plot(fpr, tpr, color='deeppink',linestyle=':', linewidth=4, label='RNN Regressor (area = %0.2f)' % roc_auc)

        plt.subplot(2, 3, subplot_no + 1)
        plt.plot(fpr2, tpr2, color='navy',linestyle=':', linewidth=4, label='KMER Regressor (area = %0.2f)' % roc_auc2)

        plt.subplot(2, 3, subplot_no + 1)
        plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        if subplot_no > 2:
            plt.xlabel('False Positive Rate')
        if subplot_no in [0,3]:
            plt.ylabel('True Positive Rate')
        plt.title('ROC -  Allele %s ' % name_)
        plt.legend(loc="lower right")
        if subplot_no == 5:
            plt.savefig(self.SUMMARY_DIR + 'ROC_CURVES', orientation='portrait', bbox_inches='tight')


if __name__ == '__main__':

    ALLELES = ["A0201", "A0101", "A0301", "A0203", "A1101", "A0206", "A3101"]

    c = Results()
    c.create_summary_csvs()
    c.plot_roc()
    plt.show()




