"""
This module contains all implemented models to predict Named Entity Recognition (NER).
Author: Lucas Pavanelli
"""
import torch
import torch.nn as nn
from transformers import AutoModel  # or BertModel, for BERT without pretraining heads
import torchcrf
import sklearn_crfsuite
import random


class BERTSlotFilling(nn.Module):
    """
    BERTimbau model to predict NER classes.

    Parameters
    ----------
    hidden_dim : int
        Hidden layer dimension.
    num_classes : int
        Number of NER classes.

    Attributes
    ----------
    device : torch.device
        Class device
    hidden_dim : int
        Hidden layer dimension.
    num_classes : int
        Number of NER classes.
    bert : AutoModel
        BERTimbau model
    Wb : nn.Linear
        Linear layer
    softmax : nn.Softmax
        Softmax layer
    """
    def __init__(self, hidden_dim, num_classes, device=torch.device('cpu')):
        super(BERTSlotFilling, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.bert = AutoModel.from_pretrained('neuralmind/bert-large-portuguese-cased')
        self.bert.to(self.device)

        self.Wb = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, token_ids, subword_ids):
        """
        Computes probabilities for each NER class.

        Parameters
        ----------
        token_ids : torch.Tensor
            List of tokens indexes.
        subword_ids : torch.Tensor
            List of subword indexes.

        Returns
        -------
        torch.Tensor
            Probabilities for each NER class
        """
        # with torch.no_grad():
        encoded = self.bert(token_ids.to(self.device))[0][0]
        encoded = encoded[subword_ids]
        linear = self.Wb(encoded)
        probs = self.softmax(linear)
        return probs


class SimpleLinear(nn.Module):
    """
    Simple linear model to create embeddings and return entity logits

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    num_classes : int
        Number of NER classes.
    out_w2id: dict
        Map from word to id for output vocabulary.
    emb_dim: int
        Embedding layer dimension.

    Attributes
    ----------
    emb : nn.Embedding
        Embedding layer.
    emb2tag : nn.Linear
        Linear layer.
    """
    def __init__(self, vocab_size, num_classes, out_w2id, emb_dim=10):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.emb2tag = nn.Linear(emb_dim, num_classes)
        self.emb2tag.bias.data[out_w2id['O']] = 50

    def forward(self, token_ids):
        """
        Computes entity logits

        Parameters
        ----------
        token_ids : torch.Tensor
            List of tokens indexes.

        Returns
        -------
        torch.Tensor
            Entity logits
        """
        x = self.emb(token_ids)
        x = self.emb2tag(x)
        return x


class LinearLayerCRF(nn.Module):
    """
    Linear Layer + CRF model that returns probability for each NER class

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    num_classes : int
        Number of NER classes.
    out_w2id: dict
        Map from word to id for output vocabulary.

    Attributes
    ----------
    device : torch.device
        Class device.
    linear : SimpleLinear
        Linear layer.
    crf: CRF
        PyTorch CRF
    """
    def __init__(self, num_classes, vocab_size, out_w2id):
        super(LinearLayerCRF, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.linear = SimpleLinear(vocab_size, num_classes, out_w2id)
        self.crf = torchcrf.CRF(num_tags=num_classes, batch_first=True)

    def forward(self, token_ids):
        """
        Computes probabilities for each NER class.

        Parameters
        ----------
        token_ids : torch.Tensor
            List of tokens indexes.

        Returns
        -------
        torch.Tensor
            Probabilities for each NER class
        """
        # with torch.no_grad():
        emissions = self.linear(token_ids)
        encoded = self.crf.decode(emissions)
        return torch.tensor(encoded)

    def loss(self, token_ids, tag_ids):
        """
        Computes model's loss.

        Parameters
        ----------
        token_ids : torch.Tensor
            List of tokens indexes.
        tag_ids: torch.Tensor
            List of tags indexes.
        Returns
        -------
        torch.Tensor
            Model's loss
        """
        emissions = self.linear(token_ids)
        nll = self.crf(emissions, tag_ids)
        return -nll


class CRF:
    def __init__(self, algorithm='lbfgs', c1=0.1, c2=0.1):
        self.crf = sklearn_crfsuite.CRF(
            algorithm=algorithm,
            c1=c1,
            c2=c2,
            max_iterations=100,
            all_possible_transitions=True
        )

    def _word2features(self, sentence, index):
        word = sentence[index][0]
        postag = sentence[index][1]

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],
        }
        if index > 0:
            word1 = sentence[index - 1][0]
            postag1 = sentence[index - 1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True

        if index < len(sentence) - 1:
            word1 = sentence[index + 1][0]
            postag1 = sentence[index + 1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True
        return features

    def _sentence2features(self, sentence):
        return [self._word2features(sentence, i) for i in range(len(sentence))]

    def get_train_test_data(self, data_info):
        all_data = [(self._sentence2features(list(zip(s['tokens'], s['postags']))), s['tags'], s['tokens']) for s in data_info.corpus]
        size = int(0.1 * len(all_data))
        random.shuffle(all_data)
        train_data = all_data[size:]
        test_data = all_data[:size]
        x_train = [data[0] for data in train_data]
        y_train = [data[1] for data in train_data]
        x_test = [data[0] for data in test_data]
        y_true = [data[1] for data in test_data]
        test_tokens = [data[2] for data in test_data]
        return x_train, y_train, x_test, y_true, test_tokens

    def fit(self, x_train, y_train):
        self.crf.fit(x_train, y_train)

    def predict(self, x_test):
        return self.crf.predict(x_test)

