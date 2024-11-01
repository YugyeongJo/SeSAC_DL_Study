import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import DataLoader
from typing import List, Optional, Callable, Tuple, Dict

from data_handler import LanguagePair, Vocabulary
from transformer_layers import Transformer

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        save_dir: str = 'models/'
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

    def train(
        self,
        train_data: DataLoader,
        valid_data: Optional[DataLoader] = None,
        num_epochs: int = 10,
        print_every: int = 1,
        evaluate_every: int = 1,
        evaluate_metrics: List[Callable] = [],
        source_vocab: Vocabulary = Vocabulary(),
        target_vocab: Vocabulary = Vocabulary()
    ) -> Tuple[List[float], List[float], Dict[str, List[float]], Dict[str, List[float]]]:
        train_loss_history = []
        valid_loss_history = []
        train_evaluation_result = defaultdict(list)
        valid_evaluation_result = defaultdict(list)

        for epoch in range(1, 1 + num_epochs):
            self.model.train()
            epoch_train_loss = 0
            epoch_train_eval = defaultdict(list)

            for batch_idx, (src, tgt) in enumerate(train_data):
                output = self.model(src) # output logit
                pred_sent = self.get_tokens_from_logit(output, target_vocab) # list of tokens
                tgt_sent = self.get_tokens_from_indices(tgt, target_vocab)

                for metric in evaluate_metrics:
                    epoch_train_eval[metric.__name__].append(metric(tgt_sent, pred_sent))

                loss = self.criterion(tgt, output)
                epoch_train_loss += loss.item()

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            if valid_data is not None and epoch % evaluate_every == 0:
                valid_loss, valid_metric, translate_result = self.evaluate(valid_data, evaluate_metrics, source_vocab, target_vocab)
            
            for metric in evaluate_metrics:
                t = epoch_train_eval[metric.__name__]

                train_evaluation_result[metric.__name__].append(sum(t) / len(t))
                valid_evaluation_result[metric.__name__].append(valid_metric[metric.__name__])

            train_loss_history.append(epoch_train_loss)
            valid_loss_history.append(valid_loss)

        return train_loss_history, valid_loss_history, dict(train_evaluation_result), dict(valid_evaluation_result)
    
    def evaluate(
        self,
        valid_data: DataLoader,
        evaluate_metrics: List[Callable] = [],
        source_vocab: Vocabulary = Vocabulary(),
        target_vocab: Vocabulary = Vocabulary()
    ) -> Tuple[float, Dict[str, float], List[Tuple[List[str], List[str], List[str]]]]:
        self.model.eval()

        valid_loss = 0
        evaluation_result = defaultdict(list)
        translation = []

        for batch_idx, (src, tgt) in enumerate(valid_data):
            output = self.model(src) # output logit
            
            loss = self.criterion(tgt, output)
            valid_loss += loss.item()

            src_sent = self.get_tokens_from_indices(src, source_vocab)
            tgt_sent = self.get_tokens_from_indices(tgt, target_vocab)
            pred_sent = self.get_tokens_from_logit(output, target_vocab) # list of tokens

            for s, t, p in zip(src_sent, tgt_sent, pred_sent):
                translation.append((s, t, p))

            for metric in evaluate_metrics:
                evaluation_result[metric.__name__].append(metric(output, pred_sent))
        
        for metric, res in evaluation_result.items():
            evaluation_result[metric] = sum(res) / len(res)
        
        return valid_loss, evaluation_result, translation

    def get_tokens_from_logit(
        self,
        logit: torch.tensor,
        vocab: Vocabulary,
    ) -> List[List[str]]:
        # logit: batch_size, seq_length, vocab.vocab_size
        batch_size, seq_length, vocab_size = logit.size()
        assert vocab_size == vocab.vocab_size

        softmax = nn.SoftMax(dim = -1)
        prob = softmax(logit) # prob[b][i][k]

        most_likely_tokens = torch.argmax(prob, dim = -1).tolist() # batch_size, seq_length

        return self.get_tokens_from_indices(most_likely_tokens, vocab)

    def get_tokens_from_indices(
        self,
        most_likely_tokens: List[List[int]],
        vocab: Vocabulary,
    ) -> List[List[str]]:

        for sent_idx, sent in enumerate(most_likely_tokens):
            for tok_idx, token in enumerate(sent):
                i = most_likely_tokens[sent_idx][tok_idx]
                most_likely_tokens[sent_idx][tok_idx] = vocab.index2word(i)
        
        return most_likely_tokens

    def get_logits(
        self,
        src: torch.tensor,
        tgt: Optional[torch.tensor]  = None,
        target_vocab: Vocabulary = Vocabulary(),
    ) -> torch.tensor:
        batch_size, src_seq_length, embedding_dim = src.size()
        decoder_input = torch.tensor([target_vocab.sos_idx for _ in range(batch_size)])

        while decoder_output == target_vocab.eos_idx:
            output = self.model(src, )


if __name__ == '__main__':
    import config
    trainer = Trainer(..., device = config.device)