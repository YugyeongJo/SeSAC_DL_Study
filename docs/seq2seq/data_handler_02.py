import random
import torch
from torch.utils.data import DataLoader, random_split

from collections import defaultdict

# Vocabulary class to handle token-index mapping
class Vocabulary:
    """_summary_
    """
    PAD = '[PAD]'
    SOS = '[SOS]'
    EOS = '[EOS]'
    OOV = '[OOV]'
    SPECIAL_TOKENS = [PAD, SOS, EOS, OOV]
    pad_idx = 0 
    sos_idx = 1
    eos_idx = 2
    oov_idx = 3
    
    def __init__(self, word_count_threshold = 0):
        """_summary_

        Args:
            word_count_threshold (int, optional): _description_. Defaults to 0.
        """
        self.word2index = {}
        self.index2word = {}
        self.word_count = {}
        self.n_words = 0
        self.threshold = word_count_threshold
        
        # Special tokens
        self.pad_idx = Vocabulary.pad_idx
        self.sos_idx = Vocabulary.sos_idx
        self.eos_idx = Vocabulary.eos_idx
        self.oov_idx = Vocabulary.oov_idx
        
        # Initialize the special tokens
        self.add_word(Vocabulary.PAD)
        self.add_word(Vocabulary.SOS)
        self.add_word(Vocabulary.EOS)
        self.add_word(Vocabulary.OOV)
        
    def add_word(self, word):
        """_summary_

        Args:
            word (_type_): _description_
        """
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word_count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word_count[word] += 1
            
    def add_sentence(self, sentence):
        """_summary_

        Args:
            sentence (_type_): _description_
        """
        for word in sentence:
            self.add_word(word)
            
    def word_to_index(self, word):
        """_summary_

        Args:
            word (_type_): _description_
        """
        return self.word2index.get(word, self.oov_idx)
    
def parse_file(file_path, batch_size = 32, train_valid_test_ratio = (0.8, 0.1, 0.1)):
    source_vocab = Vocabulary()
    target_vocab = Vocabulary()
    
    text = open(file_path, 'r', encoding = 'utf-8').read()
    data = []
    num_samples = 0
    
    for line in text.split('\n'):
        line = line.strip()
        try:
            source, target, etc = line.split('\t')
        except ValueError:
            try:
                source, target = line.split('\t')
            except ValueError:
                continue
            
        source = source.strip().split()
        target = target.strip().split()
        source_vocab.add_sentence(source)
        target_vocab.add_sentence(target)
        
        data.append((source, target))
        num_samples += 1
        
    lengths = [int(num_samples * ratio) for ratio in train_valid_test_ratio]
    lengths[-1] = num_samples - sum(lengths[:-1])
    
    datasets = random_split(data, lengths)
    dataloaders = [DataLoader(dataset, batch_size = batch_size, shuffle = True, collate_fn = lambda x: collate_fn_language(x, source_vocab, target_vocab)) for dataset in datasets]
        
    return dataloaders, source_vocab, target_vocab

def collate_fn_language(batch, source_vocab, target_vocab):
    """_summary_

    Args:
        batch (_type_): _description_
        source_vocab (_type_): _description_
        target_vocab (_type_): _description_

    Returns:
        _type_: _description_
    """
    input_seqs = [item[0] for item in batch]
    target_seqs = [item[1] for item in batch]
    
    input_seqs = [seq + [Vocabulary.eos_idx] for seq in input_seqs]
    target_seqs = [[Vocabulary.sos_idx] + seq + [Vocabulary.eos_idx] for seq in target_seqs]
        
    input_max_length = max([len(s) for s in input_seqs])
    target_max_length = max([len(s) for s in target_seqs])
    
    input_padded = []
    for idx, seq in enumerate(input_seqs):
        seq = seq + [Vocabulary.pad_idx] * (input_max_length - len(seq))
        assert len(seq) == input_max_length, f'Expected to have {input_max_length}, now {len(seq)}'
        input_padded.append(seq)
        
    target_padded = []
    for idx, seq in enumerate(target_seqs):
        seq = seq + [Vocabulary.pad_idx] * (target_max_length - len(seq))
        assert len(seq) == target_max_length, f'Expected to have {target_max_length}, now {len(seq)}'
        target_padded.append(seq)
        
    input_padded = torch.tensor(input_padded, dtype = torch.long)
    target_padded = torch.tensor(target_padded, dtype = torch.long)
    
    return input_padded, target_padded