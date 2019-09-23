import numpy as np
from keras.models import Model


def make_onehot(buf, seq_length):

    fd = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0],
          'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1],
          'N': [0, 0, 0, 0]}

    one_hot = [fd[base] for seq in buf for base in seq]
    one_hot_np = np.reshape(one_hot, (-1, seq_length, 4))
    return one_hot_np


class TrainingData:

    def __init__(self, motif_a, motif_b, N, N_mult, N_neg, seq_length):
        self.motif_a = motif_a
        self.motif_b = motif_b
        self.N = N
        self.N_mult = N_mult
        self.N_neg = N_neg
        self.seq_length = seq_length

    def rc(self, motif):
        rc_dict = {'A': 'T', 'G': 'C', 'T': 'A', 'C': 'G'}
        rc_list = [rc_dict[letter] for letter in motif]
        return ''.join(rc_list)

    def embed(self, sequence):
        # choose an appropriate position
        pos = np.random.randint(5, 85)  # choose the insert position
        # embed motif randomly
        choice = np.random.randint(0, 100)
        size_a = len(self.motif_a)
        size_b = len(self.motif_b)

        if choice < 25:
            sequence[pos: pos + size_a] = self.motif_a
        elif 25 <= choice < 50:
            sequence[pos: pos + size_a] = self.rc(self.motif_a)
        elif 50 <= choice < 75:
            sequence[pos: pos + size_b] = self.motif_b
        else:
            sequence[pos: pos + size_b] = self.rc(self.motif_b)

    def simulate_data(self):
        # Define a integer --> str dictionary
        letter = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}
        # Construct N positive and N negative sequences with background frequencies A/T=0.5
        seq_list = []
        # Unbound Synthetic Data
        for idx in range(self.N_neg):
            sequence = np.random.randint(0, 4, self.seq_length)
            sequence = ''.join([letter[x] for x in sequence])
            seq_list.append((sequence, 0))  # Note: The 0 here is the sequence label
        # Bound Synthetic Data
        # N sequences with an embedded motif b/w positions [ 25, seq_length - 25] # Buffer : 25
        for idx in range(self.N):
            sequence = np.random.randint(0, 4, self.seq_length)
            sequence = [letter[x] for x in sequence]
            self.embed(sequence)
            # Adding in the sequence N_mult times.
            for idx in range(self.N_mult):
                seq_list.append((''.join(sequence), 1))  # Doing the join after the embedding for the positive set
        # Making the Sequence Data one-hot
        dat = np.array(seq_list)[:, 0]
        dat = make_onehot(dat, seq_length=self.seq_length)
        labels = np.array(seq_list)[:, 1]
        return dat, labels


class TestData:

    def __init__(self, seq_length, model):
        self.seq_length = seq_length
        self.model = model

    def embed_test_motif(self, sequence, motif):
        pos = np.random.randint(25, self.seq_length - 25)  # Buffer : 25
        size = len(motif)
        sequence[pos: pos + size] = motif

    def simulate_test_dat(self, motif):
        # Define a integer --> str dictionary
        letter = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}
        # Simulating a 1000 sequences to measure network performance
        seq_list = []

        for idx in range(1000):
            sequence = np.random.randint(0, 4, self.seq_length)
            sequence = [letter[x] for x in sequence]
            self.embed_test_motif(sequence, motif)
            seq_list.append(''.join(sequence))  # Doing the join after the embedding for the positive set
        dat = np.array(seq_list)
        dat = make_onehot(dat, seq_length=self.seq_length)
        return np.mean(self.model.predict(dat))

