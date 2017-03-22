import itertools
import pandas
import numpy
import utils
import os
import logging


class Data(object):

    DEFAULT_ALLELE = ['A0201']

    def __init__(self, filename, allele=None, train=False):

        self.is_train_file = train
        self.pkl_dir = os.path.dirname(os.getcwd()) + '/mhcPreds/extras/allele_mapping'
        self.mapping = utils.load_obj(self.pkl_dir)
        self.df = self.allele_df(filename, allele)
        self.alleles = numpy.asarray(self.df["mhc"])
        self.peptides = numpy.asarray(self.df["sequence"])
        self.affinities = numpy.asarray(self.df["meas"])
        self.all_peps = ['A', 'R', 'N', 'D', 'C', 'Q', 'E',
                         'G', 'H', 'I', 'L', 'K', 'M', 'F',
                         'P', 'S', 'T', 'W', 'Y', 'V']

    def allele_df(self, filename, allele=None):

        if not allele:
            logging.warning('Allele name not provided, using default.')
            allele = self.DEFAULT_ALLELE
        if self.is_train_file:
            allele = self.mapping[allele][0]
        else:
            allele = self.mapping[allele][1]

        df = pandas.read_csv(filename, sep='\t')
        return df[df['mhc'] == allele]

    def one_hot_encoding(self, kmer_size=9):

        X_index, _, original_peptide_indices, counts = fixed_length_one_hot_encoding(peptides=self.peptides,
                                                                                   desired_length=kmer_size)

        original_peptide_indices = numpy.asarray(original_peptide_indices)
        kmer_affinities = self.affinities[original_peptide_indices]

        return X_index, kmer_affinities, original_peptide_indices

    def index_lookup(self, aa):
        return self.all_peps.index(aa)

    def kmer_index_encoding(self, kmer_size=9):

        X_index, _, original_peptide_indices, counts = fixed_length_index_encoding(peptides=self.peptides,
                                                                                   desired_length=kmer_size)

        original_peptide_indices = numpy.asarray(original_peptide_indices)
        kmer_affinities = self.affinities[original_peptide_indices]

        return X_index, kmer_affinities, original_peptide_indices


def fixed_length_one_hot_encoding(peptides, desired_length):

    fixed_length, original_peptide_indices, counts = fixed_length_from_many_peptides(peptides=peptides,
                                                                                     desired_length=desired_length)
    one_hot_encoding = common_amino_acids.hotshot_encoding
    X = one_hot_encoding(fixed_length, desired_length)

    return X, fixed_length, original_peptide_indices, counts


def fixed_length_index_encoding(peptides, desired_length):

    fixed_length, original_peptide_indices, counts = fixed_length_from_many_peptides(peptides=peptides,
                                                                                     desired_length=desired_length)
    index_encoding = common_amino_acids.index_encoding
    X = index_encoding(fixed_length, desired_length)

    return X, fixed_length, original_peptide_indices, counts


def fixed_length_from_many_peptides(peptides, desired_length):
    """
    Create a set of fixed-length k-mer peptides from a collection of varying
    length peptides.

    """
    all_fixed_length_peptides = []
    indices = []
    counts = []
    for i, peptide in enumerate(peptides):
        n = len(peptide)
        if n == desired_length:
            fixed_length_peptides = [peptide]
        elif n < desired_length:
            try:
                fixed_length_peptides = extend_peptide(peptide=peptide, desired_length=desired_length)
            except CombinatorialExplosion:
                logging.warn(
                    "Peptide %s is too short to be extended to length %d" % (
                        peptide, desired_length))
                continue
        else:
            fixed_length_peptides = shorten_peptide(peptide=peptide, desired_length=desired_length)
        n_fixed_length = len(fixed_length_peptides)
        all_fixed_length_peptides.extend(fixed_length_peptides)
        indices.extend([i] * n_fixed_length)
        counts.extend([n_fixed_length] * n_fixed_length)
    return all_fixed_length_peptides, indices, counts


def extend_peptide(peptide, desired_length, start_offset_extend=2, end_offset_extend=1):

    n = len(peptide)
    n_missing = desired_length - n
    if n_missing > 3:
        raise CombinatorialExplosion(
            "Cannot transform %s of length %d into a %d-mer peptide" % (
                peptide, n, desired_length))
    return [
        peptide[:i] + extra + peptide[i:]
        for i in range(start_offset_extend, n - end_offset_extend + 1)
        for extra in all_kmers(n_missing)
    ]


def shorten_peptide(peptide, desired_length, start_offset_shorten=2, end_offset_shorten=0):

    n = len(peptide)
    assert n > desired_length, \
        "%s (length = %d) is too short! Must be longer than %d" % (
            peptide, n, desired_length)
    n_skip = n - desired_length
    assert n_skip > 0, \
        "Expected length of peptide %s %d to be greater than %d" % (
            peptide, n, desired_length)
    end_range = n - end_offset_shorten - n_skip + 1
    return [
        peptide[:i] + peptide[i + n_skip:]
        for i in range(start_offset_shorten, end_range)
    ]


class Alphabet(object):
    """
    Used to track the order of amino acids used for peptide encodings
    """

    def __init__(self, **kwargs):
        self.letters_to_names = {}
        for (k, v) in kwargs.items():
            self.add(k, v)

    def add(self, letter, name):
        assert letter not in self.letters_to_names
        assert len(letter) == 1
        self.letters_to_names[letter] = name

    def letters(self):
        return list(sorted(self.letters_to_names.keys()))

    def names(self):
        return [self.letters_to_names[k] for k in self.letters()]

    def index_dict(self):
        return {c: i for (i, c) in enumerate(self.letters())}

    def copy(self):
        return Alphabet(**self.letters_to_names)

    def __getitem__(self, k):
        return self.letters_to_names[k]

    def __setitem__(self, k, v):
        self.add(k, v)

    def __len__(self):
        return len(self.letters_to_names)

    def index_encoding_list(self, peptides):
        index_dict = self.index_dict()
        return [
            [index_dict[amino_acid] for amino_acid in peptide]
            for peptide in peptides
        ]

    def index_encoding(self, peptides, peptide_length):
        """
        Encode a set of equal length peptides as a matrix of their
        amino acid indices.
        """
        X = numpy.zeros((len(peptides), peptide_length), dtype=int)
        index_dict = self.index_dict()
        for i, peptide in enumerate(peptides):
            for j, amino_acid in enumerate(peptide):
                X[i, j] = index_dict[amino_acid]
        return X

    def hotshot_encoding(self, peptides, peptide_length):
        """
        Encode a set of equal length peptides as a binary matrix,
        where each letter is transformed into a length 20 vector with a single
        element that is 1 (and the others are 0).
        """
        shape = (len(peptides), peptide_length, 20)
        index_dict = self.index_dict()
        X = numpy.zeros(shape)
        for i, peptide in enumerate(peptides):
            for j, amino_acid in enumerate(peptide):
                k = index_dict[amino_acid]
                X[i, j, k] = 1
        return X


common_amino_acids = Alphabet(**{
    "A": "Alanine",
    "R": "Arginine",
    "N": "Asparagine",
    "D": "Aspartic Acid",
    "C": "Cysteine",
    "E": "Glutamic Acid",
    "Q": "Glutamine",
    "G": "Glycine",
    "H": "Histidine",
    "I": "Isoleucine",
    "L": "Leucine",
    "K": "Lysine",
    "M": "Methionine",
    "F": "Phenylalanine",
    "P": "Proline",
    "S": "Serine",
    "T": "Threonine",
    "W": "Tryptophan",
    "Y": "Tyrosine",
    "V": "Valine",
})
common_amino_acid_letters = common_amino_acids.letters()

amino_acids_with_unknown = common_amino_acids.copy()
amino_acids_with_unknown.add("X", "Unknown")
amino_acids_with_unknown_letters = amino_acids_with_unknown.letters()


def all_kmers(k, alphabet=common_amino_acid_letters):

    alphabets = [alphabet] * k
    return [
        "".join(combination)
        for combination
        in itertools.product(*alphabets)
    ]


class CombinatorialExplosion(Exception):
    pass

"""
class ProtVec(word2vec.Word2Vec):

    def __init__(self, corpus_fname=None, corpus=None, n=3, size=100, out="corpus.txt",  sg=1, window=25, min_count=2, workers=3):

        Either fname or corpus is required.
        corpus_fname: fasta file for corpus
        corpus: corpus object implemented by gensim
        n: n of n-gram
        out: corpus output file path
        min_count: least appearance count in corpus. if the n-gram appear k times which is below min_count, the model does not remember the n-gram


        self.n = n
        self.size = size
        self.corpus_fname = corpus_fname

        if corpus is None and corpus_fname is None:
            raise Exception("Either corpus_fname or corpus is needed!")


        word2vec.Word2Vec.__init__(self, corpus, size=size, sg=sg, window=window, min_count=min_count, workers=workers)

    def to_vecs(self, seq):

        convert sequence to three n-length vectors
        e.g. 'AGAMQSASM' => [ array([  ... * 100 ], array([  ... * 100 ], array([  ... * 100 ] ]

        ngram_patterns = split_ngrams(seq, self.n)

        protvecs = []
        for ngrams in ngram_patterns:
            ngram_vecs = []
            for ngram in ngrams:
                try:
                    ngram_vecs.append(self[ngram])
                except:
                    raise Exception("Model has never trained this n-gram: " + ngram)
            protvecs.append(sum(ngram_vecs))
        return protvecs
"""



