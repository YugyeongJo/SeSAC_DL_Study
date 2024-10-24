"""
Data handling scripts.

1. class Vocabulary : Vocabulary wrapper class for generating/handling special tokens/index to word, and vice versa, etc, ...
"""
    
class Vocabulary:
    """
    Handles vocabulary of a given sequential dataset.
    
    Args:
        coverage (float): Coverage for determining whether the token shall be considered as OOV or not.
    Attributes:
        word2index (dict[str, int]): Dict containing token as key, and index of the token as value.
        index2word (dict[int, str]): Dict containing index of token as key, and the token as value.
        vocab_size (int): Integer represinting the vocabulary size.
    Methods:

    Variables:
        SPECIAL_TOKENS
        EOS_IDS, EOS
        SOS_IDX, SOS
        PAD_IDX, PAD
        OOV_IDX, OOV
    """
    EOS = '[EOS]'
    SOS = '[SOS]'
    PAD = '[PAD]'
    OOV = '[OOV]'
    SPECIAL_TOKENS = [EOS, SOS, PAD, OOV]
    
    def __init__(self, ):
        self.word2index: dict = {}
        self.index2word: dict = {}
        self.vocab_size: int = 0
        
        for special_token in Vocabulary.SPECIAL_TOKENS:
            self.add_word(special_token)
            
        self.eos_idx = word2index[Vocabulary.EOS]
        self.eos_idx = word2index[Vocabulary.SOS]
        self.eos_idx = word2index[Vocabulary.PAD]
        self.eos_idx = word2index[Vocabulary.OOV]
        
    def add_word(self, token: str) -> None:
        """
        Adds a token to the vocabulary if it doesn't exists.
        If it exists, do nothing.

        Args:
            token (str): The token to be added.
        """
        if token not in self.word2index:
            self.word2index[token] = self.vocab_size
            self.index2word[self.vocab_size] = token
            self.vocab_size += 1
        