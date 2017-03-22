from sklearn import metrics
import numpy
import scipy
import pickle
import pandas

def regression_target_to_ic50(y, max_ic50=50000.0):
    return max_ic50 ** (1.0 - y)


def ic50_to_regression_target(ic50, max_ic50=50000.0):

    log_ic50 = numpy.log(ic50) / numpy.log(max_ic50)
    regression_target = 1.0 - log_ic50
    regression_target = numpy.maximum(regression_target, 0.0)
    regression_target = numpy.minimum(regression_target, 1.0)
    return regression_target


def make_scores(ic50_y, ic50_y_pred, sample_weight=None, threshold_nm=500, max_ic50=50000):

    y_pred = ic50_to_regression_target(ic50_y_pred, max_ic50)
    auc = metrics.roc_auc_score(ic50_y <= threshold_nm, y_pred, sample_weight=sample_weight)
    f1 = metrics.f1_score(ic50_y <= threshold_nm, ic50_y_pred <= threshold_nm,  sample_weight=sample_weight)
    tau = scipy.stats.kendalltau(ic50_y_pred, ic50_y)[0]

    return dict(auc=auc, f1=f1, tau=tau)

def scoring(preds, idxTe, yTe):

    preds = numpy.array([regression_target_to_ic50(i[0]) for i in preds])
    targs = numpy.array([regression_target_to_ic50(i[0]) for i in yTe])

    as_df = pandas.DataFrame([preds, targs], index=['preds', 'targs']).T
    as_df['idx'] = idxTe

    avg_pred = []
    avg_lab = []

    for i in set(idxTe):
        sliced = as_df[as_df['idx'] == i]
        num_vals = len(sliced)
        avg_pred.append(sum(sliced['preds'].values) / num_vals)
        avg_lab.append(sum(sliced['targs'].values) / num_vals)

    scoring_df = pandas.DataFrame([list(set(idxTe)), avg_pred, avg_lab], index=['Original_Index', 'Avg_Pred', 'Original_Label']).T
    scores = make_scores(scoring_df['Original_Label'].values, scoring_df['Avg_Pred'].values)

    return scores, scoring_df

def get_batch3d(X, X_, size):

    X = X.astype('float32')
    X_ = X_.astype('float32')
    idx = numpy.random.choice(len(X), size, replace=False)
    return X[idx, :, :], X_[idx, :]


def get_batch2d(X, X_, size):
    idx = numpy.random.choice(len(X), size, replace=False)
    return X[idx, :], X_[idx, :]


def get_all_human_alleles_formatting_one(df):
    """
    Ugly hack #1 to get the alleles in the same naming convention
    :param df: df from file (train, test, or full)
    :return: mapping from the shortened name to the original name of the allele
    """

    df = df[df.species == 'human']
    alleles_from_file = list(df.mhc.unique())

    letter = [i.split('-')[1][0] for i in alleles_from_file]
    loc = [i.split('-')[1].replace('*', '')[1:].split(':') for i in alleles_from_file]

    simplified_name = []
    for i, val in enumerate(loc):
        if len(val) > 1:
            simplified_name.append(letter[i] + val[0] + val[1])
        else:
            simplified_name.append(letter[i] + val[0])

    return dict(zip(simplified_name, alleles_from_file))


def get_all_human_alleles_formatting_two(df):
    """
    Ugly hack #2 to get the alleles in the same naming convention
    :param df: df from file (train, test, or full)
    :return: mapping from the shortened name to the original name of the allele
    """
    df = df[df.species == 'human']
    alleles_from_file = list(df.mhc.unique())
    loc = [i.split('-')[1:] for i in alleles_from_file]
    loc_ = [i[1] if len(i) > 1 else i[0][1:] for i in loc]
    letter = [i[0] if len(i) > 1 else i[0][0] for i in loc]

    simplified_name = []
    for i, val in enumerate(loc_):
        if len(val) > 1:
            simplified_name.append(letter[i] + val)
        else:
            simplified_name.append(letter[i] + val)

    return dict(zip(simplified_name, alleles_from_file))


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=2)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
