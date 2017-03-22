


"""
class HyperParam(object):

    ACTION = "train_test_eval"
    FILE = "mhcPreds_tflearn_cmd_line.py"

    def create_hyperparam_file(self):


class EmbeddingNet(HyperParam):

    MODEL = 'embedding_rnn'

    def __init__(self):

        self.lr_range = [0.1, 0.01, 0.001, 0.0001]
        self.alleles = ["A0201", "A0301", "A0203", "A1101", "A0206", "A3101"]
        self.embedding_size = [25, 30, 35, 40]
        self.batch_size = [40, 50, 60, 70, 80, 100]
        print(self.ACTION)
        print(self.create_hyperparam_file())


class DeepLSTM(HyperParam):

    MODEL = 'deep_rnn'

    def __init__(self):

        self.lr_range = [0.01, 0.001, 0.0001, 0.00001]
        self.alleles = ["A0201", "A0301", "A0203", "A1101", "A0206", "A3101"]
        self.batch_size = [40, 50, 60, 70, 80, 100]
        print(self.ACTION)
        print(self.create_hyperparam_file())

"""




if __name__ == '__main__':
    emb = EmbeddingNet()