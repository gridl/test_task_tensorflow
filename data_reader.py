"""Read the data from supported text file
"""
import collections

import numpy as np


class DataReader(object):

    def __init__(self, batch_size, sequence_size, data_path='data/input.txt'):
        with open(data_path, 'r') as f:
            self.raw_data = f.read()
        self.batch_size = batch_size
        self.sequence_size = sequence_size

        # Build Vocabulary and Dictionaries
        counter = collections.Counter(self.raw_data)
        # create list of pairs aka ('letter', quantity)
        count_pairs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.unique_tokens, _ = list(zip(*count_pairs))
        # get dict of pairs aka ('letter', token_id)
        self.token_to_id = dict(zip(self.unique_tokens,
                                    range(len(self.unique_tokens))))
        # get dict of pairs aka (token_id, 'letter')
        self.id_to_token = dict(zip(range(len(self.unique_tokens)),
                                    self.unique_tokens))
        self.vocabularySize = len(self.unique_tokens)

        # convert raw data text to digits representation
        self.data_as_ids = []
        for token in self.raw_data:
            if token in self.token_to_id.keys():
                self.data_as_ids.append(self.token_to_id[token])

        self.training_data = self.data_as_ids
        self.total_batches = len(
            self.training_data) // (batch_size * sequence_size)

    def print_data_info(self):
        message = (
            "----------------------------------------\n"
            "Data total tokens: {} tokens\n"
            "Data vocabulary size: {} tokens\n"
            "Training Data total tokens: {} tokens\n"
            "----------------------------------------".format(
                len(self.raw_data),
                len(self.unique_tokens),
                len(self.training_data)
            )
        )
        print(message)

    def generateXYPairs(self):
        """Generate X and Y pairs with Y shifted by one.
        batch_size mean on how many batched all data will be slitted.
        sequence_size mean how many letters will be in one batch[0] chunk

        return
        ======
            Per one iteration next() it return on batch of X and Y with
            required size with batch/sequence size examples
        """
        batch_size = self.batch_size
        sequence_size = self.sequence_size
        raw_data = np.array(self.training_data, dtype=np.int32)

        batch_len = len(raw_data) // batch_size
        data = np.zeros([batch_size, batch_len], dtype=np.int32)
        for i in range(batch_size):
            data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

        # get the X and Y data in one array. Note that Y shifted by one letter
        # TODO: made that x and y will be produced with overlapping |abc|bcd|..
        #       no them return just with shifting |abc|def|...
        for i in range((batch_len - 1) // sequence_size):
            x = data[:, i*sequence_size: (i+1)*sequence_size]
            y = data[:, i*sequence_size+1: (i+1)*sequence_size+1]
            yield (x, y)

    def seq_to_letters(self, array):
        array = [self.id_to_token[a] for a in array]
        return ''.join(array)


if __name__ == '__main__':
    reader = DataReader()
    reader.print_data_info()
