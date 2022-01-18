"""
This module contains algorithms to train and test BERT and Linear Layer + CRF models.
Author: Lucas Pavanelli
"""
import torch
import torch.nn as nn


class Trainer:
    """
    Trains and tests BERT and Linear Layer + CRF models.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model
    batch : int
        Batch size.
    is_bert : bool
        If model is a BERT model or not.
    criterion : torch.nn
        PyTorch criterion
    device  : torch.device
        PyTorch device

    Attributes
    ----------
    model : torch.nn.Module
        PyTorch model
    batch : int
        Batch size.
    is_bert : bool
        If model is a BERT model or not.
    criterion : torch.nn
        PyTorch criterion
    device  : torch.device
        PyTorch device
    """
    def __init__(self, model, batch, is_bert=False, criterion=nn.CrossEntropyLoss(), device="cpu"):
        self.model = model
        self.is_bert = is_bert
        self.batch = batch
        self.criterion = criterion
        self.device = device

    def _train_linear_layer_crf(self, train_data, optimizer, epoch):
        self.model.train()
        loss, losses = 0, 0
        for i, (token_ids, tag_ids, _) in enumerate(train_data):
            # Init
            optimizer.zero_grad()
            # Calculate loss
            l = self.model.loss(token_ids, tag_ids)
            loss += l
            losses += l
            if (i + 1) % self.batch == 0:
                # Backpropagation
                loss = loss / self.batch
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # Display
                print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, losses / i), end='\r')
                loss = 0
        print()

    def _train_bert(self, train_data, optimizer, epoch):
        loss, losses = 0, 0
        for i, (token_ids, subwords_idx, tag_ids) in enumerate(train_data):
            # Init
            optimizer.zero_grad()
            # Predict
            output = self.model(token_ids, subwords_idx)
            # Calculate loss
            l = self.criterion(output, tag_ids.to(self.device))
            loss += l
            losses += l
            if (i + 1) % self.batch == 0:
                # Backpropagation
                loss = loss / self.batch
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # Display
                print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, losses / i), end='\r')
                loss = 0
        print()

    def train(self, train_data, optimizer, epoch):
        """
        Trains model using train data.

        Parameters
        ----------
        train_data : list
            List of tuples representing train data.
        optimizer : optim.SGD
            PyTorch optimizer.
        epoch : int
            Number of epoch
        """
        self.model.train()
        if self.is_bert:
            self._train_bert(train_data, optimizer, epoch)
        else:
            self._train_linear_layer_crf(train_data, optimizer, epoch)

    def _test_linear_layer_crf(self, test_data):
        y_pred, y_true = [], []
        for i, (token_ids, tag_ids, _) in enumerate(test_data):
            output = self.model(token_ids)
            y_pred.append([int(w) for w in list(output[0])])
            y_true.append([int(w) for w in list(tag_ids[0])])
        return y_true, y_pred

    def _test_bert(self, test_data):
        y_pred, y_true = [], []
        for i, (token_ids, subwords_idx, tag_ids) in enumerate(test_data):
            output = self.model(token_ids, subwords_idx)
            y_pred.append([int(w) for w in list(torch.argmax(output, dim=1))])
            y_true.append([int(w) for w in list(tag_ids)])
        return y_true, y_pred

    def test(self, test_data):
        """
        Tests model using test data.

        Parameters
        ----------
        test_data : list
            List of tuples representing test data.

        Returns
        -------
        list
            True and predicted values
        """
        self.model.eval()
        if self.is_bert:
            return self._test_bert(test_data)
        else:
            return self._test_linear_layer_crf(test_data)
