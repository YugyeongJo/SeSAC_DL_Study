import torch
from torch.utils.data import DataLoader, random_split
from collections import defaultdict

class Vocabulary:
    """
    Vocabulary class for managing token-to-index and index-to-token mappings.
    It also handles special tokens like PAD, SOS, EOS, and OOV.
    """
    PAD = '[PAD]'
    SOS = '[SOS]'
    EOS = '[EOS]'
    OOV = '[OOV]'
    SPECIAL_TOKENS = [PAD, SOS, EOS, OOV]
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    OOV_IDX = 3

    def __init__(self, word_count, coverage=0.95):
        """
        Initializes the Vocabulary object.

        Args:
            word_count (dict): Dictionary containing words and their frequencies.
            coverage (float): Proportion of the total word frequency coverage to include in the vocabulary.
        """
        word_freq_list = []
        total = 0

        # Aggregate word frequencies and compute total count
        for word, freq in word_count.items():
            word_freq_list.append((word, freq))
            total += freq

        # Sort word frequency list in descending order of frequency
        word_freq_list = sorted(word_freq_list, key=lambda x: x[1], reverse=True)

        word2idx = {}
        idx2word = {}
        s = 0

        # Add special tokens and frequently occurring words to the vocabulary
        for idx, (word, freq) in enumerate([(e, 0) for e in Vocabulary.SPECIAL_TOKENS] + word_freq_list):
            s += freq
            if s > coverage * total:
                break
            word2idx[word] = idx
            idx2word[idx] = word

        self.word2idx = word2idx
        self.idx2word = idx2word
        self.vocab_size = len(word2idx)

    def word_to_index(self, word):
        """
        Converts a word to its corresponding index.

        Args:
            word (str): The word to be converted.

        Returns:
            int: The index of the word in the vocabulary. If the word is not found, returns the OOV index.
        """
        return self.word2idx.get(word, Vocabulary.OOV_IDX)

    def __str__(self):
        res = ''
        for k, v in self.word2idx.items():
            res += f'{k} : {v}\n'
        return res 
            

def parse_file(file_path, train_valid_test_ratio=(0.8, 0.1, 0.1), batch_size=32):
    """
    Parses the input file and splits the data into training, validation, and test sets.

    Args:
        file_path (str): Path to the dataset file.
        train_valid_test_ratio (tuple): Ratios for splitting the data into train, validation, and test sets.
        batch_size (int): Number of samples per batch.

    Returns:
        tuple: A tuple containing data loaders for training, validation, and test sets, as well as the source and target vocabularies.
    """
    # Read and process the dataset file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = []

        source_word_count = defaultdict(int)
        target_word_count = defaultdict(int)

        # Process each line in the dataset file
        for line in f.readlines():
            line = line.strip()
            lst = line.split('\t')
            
            if len(lst) == 3:
                source, target, etc = lst
            elif len(lst) == 2:
                source, target = lst 

            if len(source.split()) and len(target.split()) <= 7:
                # Tokenize and count word frequencies for the source sequence
                source = source.split()
                target = target.split()
            
            for source_token in source:
                source_word_count[source_token] += 1

            # Tokenize and count word frequencies for the target sequence
            for target_token in target:
                target_word_count[target_token] += 1

            data.append((source, target))
            data = data[:50]

    # Create vocabularies for source and target sequences
    source_vocab = Vocabulary(source_word_count)
    target_vocab = Vocabulary(target_word_count)

    # Convert tokens in data to their corresponding indices
    for idx, (source, target) in enumerate(data):
        data[idx] = (
            list(map(source_vocab.word_to_index, source)),
            list(map(target_vocab.word_to_index, target))
        )

    # Split data into train, validation, and test sets based on the specified ratios
    lengths = [int(len(data) * ratio) for ratio in train_valid_test_ratio]
    lengths[-1] = len(data) - sum(lengths[:-1])
    datasets = random_split(data, lengths)

    # Create data loaders for each dataset split
    dataloaders = [
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: preprocessing(x, source_vocab, target_vocab)
        )
        for dataset in datasets
    ]

    return dataloaders, source_vocab, target_vocab

def preprocessing(batch, source_vocab, target_vocab):
    """
    Prepares batches of data by padding sequences and adding special tokens.

    Args:
        batch (list): A batch of data containing source and target sequences.
        source_vocab (Vocabulary): Vocabulary object for the source sequences.
        target_vocab (Vocabulary): Vocabulary object for the target sequences.

    Returns:
        tuple: Tensors for padded source and target sequences.
    """
    # Extract source and target sequences from the batch
    sources = [e[0] for e in batch]
    targets = [e[1] for e in batch]

    # Add EOS token to source sequences and SOS/EOS tokens to target sequences
    source_seqs = [seq + [source_vocab.EOS_IDX] for seq in sources]
    target_seqs = [[target_vocab.SOS_IDX] + seq + [target_vocab.EOS_IDX] for seq in targets]

    # Determine the maximum lengths for source and target sequences in the batch
    source_max_length = max(len(s) for s in source_seqs)
    target_max_length = max(len(s) for s in target_seqs)

    # Pad source sequences with PAD token
    for idx, seq in enumerate(source_seqs):
        seq += [source_vocab.PAD_IDX] * (source_max_length - len(seq))
        assert len(seq) == source_max_length, f'Expected to have {source_max_length}, now {len(seq)}'
        source_seqs[idx] = seq

    # Pad target sequences with PAD token
    for idx, seq in enumerate(target_seqs):
        seq += [target_vocab.PAD_IDX] * (target_max_length - len(seq))
        assert len(seq) == target_max_length, f'Expected to have {target_max_length}, now {len(seq)}'
        target_seqs[idx] = seq

    # Convert sequences to tensors
    return torch.tensor(source_seqs, dtype=torch.long), torch.tensor(target_seqs, dtype=torch.long)

# ==========================
# def preprocessing(batch, source_vocab, target_vocab):
#     """
#     Prepares batches of data by padding sequences and adding special tokens.

#     Args:
#         batch (list): A batch of data containing source and target sequences.
#         source_vocab (Vocabulary): Vocabulary object for the source sequences.
#         target_vocab (Vocabulary): Vocabulary object for the target sequences.

#     Returns:
#         tuple: Tensors for padded source and target sequences.
#     """
#     # Extract source and target sequences from the batch
#     sources = [e[0] for e in batch]
#     targets = [e[1] for e in batch]

#     # Add EOS token to source sequences and SOS/EOS tokens to target sequences
#     source_seqs = [seq + [source_vocab.EOS_IDX] for seq in sources]
#     target_seqs = [[target_vocab.SOS_IDX] + seq + [target_vocab.EOS_IDX] for seq in targets]

#     # Determine the maximum lengths for source and target sequences in the batch
#     source_max_length = max(len(s) for s in source_seqs)
#     target_max_length = max(len(s) for s in target_seqs)

#     # Pad source sequences with PAD token
#     for idx, seq in enumerate(source_seqs):
#         seq += [source_vocab.PAD_IDX] * (source_max_length - len(seq))
#         assert len(seq) == source_max_length, f'Expected to have {source_max_length}, now {len(seq)}'
#         source_seqs[idx] = seq

#     # Pad target sequences with PAD token
#     for idx, seq in enumerate(target_seqs):
#         seq += [target_vocab.PAD_IDX] * (target_max_length - len(seq))
#         assert len(seq) == target_max_length, f'Expected to have {target_max_length}, now {len(seq)}'
#         target_seqs[idx] = seq

#     # Convert sequences to tensors
#     return torch.tensor(source_seqs, dtype=torch.long), torch.tensor(target_seqs, dtype=torch.long)
# ==========================
if __name__ == '__main__':
    BATCH_SIZE = 1

    # Parse file and split data into train, validation, and test sets
    (train, valid, test), source_vocab, target_vocab = parse_file('./docs/seq2seq/dataset/kor.txt', batch_size=BATCH_SIZE)

    # Display a sample batch from the training set
    for source_batch, target_batch in train:
        print("source_batch : ", source_batch)
        print("target_batch : ", target_batch)
        break
