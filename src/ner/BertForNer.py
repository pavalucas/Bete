__author__ = 'lucaspavanelli'

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";

import pickle
from transformers import BertTokenizer, BertForTokenClassification
import torch
import torch.nn as nn
from torch import optim
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from utils import ENTITIES, tag2id, id2tag


BERT_PRETRAINED_PATH = 'neuralmind/bert-large-portuguese-cased'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BertForNer(nn.Module):
    '''
    Load the slot-filler developed as a fine-tuned BERT cased model

    params:
    ---
        num_tags: number of tags which can be predicted
        tokenizer_path: path to the tokenizer model
        model_path: path to the model
        device: device to run the architecture (cpu or cuda)
        max_length: max length of the input text

    notes:
    ---
        https://huggingface.co/transformers/model_doc/bert.html#bertfortokenclassification
    '''
    def __init__(self, num_tags=len(ENTITIES), tokenizer_path=BERT_PRETRAINED_PATH, model_path=BERT_PRETRAINED_PATH,
                 device='cuda', max_length=128):
        super(BertForNer, self).__init__()
        self.device = device
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=False)
        self.model = BertForTokenClassification.from_pretrained(model_path, num_labels=num_tags)

    def forward(self, text, labels=None):
        '''
        Forward pass of the neural architecture

        params:
        ---
            texts: input messages to be tokenized and classified
            labels: gold-standard labels

        return:
        ---
            output: a TokenClassifierOutput
        '''
        tokens = self._get_tokens(text)
        return self.model(**tokens, labels=labels)

    def predict(self, text):
        '''
        Predict tags for each token in the text

        params:
        ---
            text: message to be tokenized and classified
        return:
        ---
            tokens_text: list of text's tokens
            entity_pred_text: list of predicted entities for each token
        '''
        output = self.forward(text)
        # do softmax
        softmax = nn.LogSoftmax(2)
        entity_probs = softmax(output.logits)
        entity_pred_text = [id2tag[int(w)] for w in list(entity_probs[0].argmax(dim=1))]
        tokens = self._get_tokens(text)
        tokens_text = self.tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
        return tokens_text

    def _get_tokens(self, text):
        return self.tokenizer(text, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(
            self.device)

class Trainer:
    '''
    Module for training a slot-filler model
    '''
    def __init__(self, model, train_data, dev_data, optimizer, epochs, batch_status, device, write_path,
                 early_stop=5, verbose=True):
        '''
        params:
        ---
            model: model to be trained
            train_data: training data
            dev_data: development data
            optimizer: PyTorch optimizer to use
            epochs: number of epochs
            batch_status: update the loss after each 'batch_status' updates
            device: cpu or cuda
            write_path: folder to save best model
            early_stop: number of epochs to stop after not getting a better evaluation
            verbose
        '''
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_status = batch_status
        self.device = device
        self.early_stop = early_stop
        self.verbose = verbose
        self.train_data = train_data
        self.dev_data = dev_data
        self.write_path = write_path
        if not os.path.exists(write_path):
            os.mkdir(write_path)

    def train(self):
        '''
        Train model based on the parameters specified in __init__ function
        '''
        self.evaluate()
        max_f1 = 0
        for epoch in range(self.epochs):
            self.model.train()
            losses = []

            for batch_idx, inp in enumerate(self.train_data):
                text = inp['X']

                # Get label index
                tag_idxs = [tag2id[tag] for tag in inp['y']]
                labels = torch.tensor(tag_idxs).unsqueeze(0)

                # Predict
                output = self.model(text, labels=labels.to(self.device))

                # Calculate loss
                loss = output.loss
                losses.append(float(loss))

                # Backpropagation
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Display
                if (batch_idx + 1) % self.batch_status == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTotal Loss: {:.6f}'.format(
                        epoch, batch_idx + 1, len(self.train_data),
                               100. * batch_idx / len(self.train_data), float(loss),
                        round(sum(losses) / len(losses), 5)))
            f1, acc, class_report = self.evaluate()
            print('F1: ', f1, 'Accuracy: ', acc)
            if f1 > max_f1:
                self.model.model.save_pretrained(os.path.join(self.write_path, 'model'))
                max_f1 = f1
                df = pd.DataFrame(class_report).transpose()
                df.to_csv(f'{self.write_path}classification_report.csv')
                repeat = 0
                print('Saving best model...')

    def evaluate(self):
        '''
        Evaluate model using dev_data
        '''
        self.model.eval()
        entity_pred, entity_true = [], []
        for inp in self.dev_data:
            text = inp['X']
            tag_idxs = [tag2id[tag] for tag in inp['y']]
            output = self.model(text)
            # do softmax
            softmax = nn.LogSoftmax(2)
            entity_probs = softmax(output.logits)
            entity_pred.extend([int(w) for w in list(entity_probs[0].argmax(dim=1))])
            entity_true.extend(tag_idxs)
        entity_labels = list(range(1, len(ENTITIES)))
        entity_target_names = ENTITIES[1:]
        print(classification_report(entity_true, entity_pred, labels=entity_labels, target_names=entity_target_names))
        class_report = classification_report(entity_true, entity_pred, labels=entity_labels, target_names=entity_target_names, output_dict=True)
        f1 = f1_score(entity_true, entity_pred, average='weighted', labels=entity_labels)
        acc = accuracy_score(entity_true, entity_pred)
        return f1, acc, class_report