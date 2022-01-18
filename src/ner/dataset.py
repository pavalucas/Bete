"""
This module loads and parses input data.
Author: Lucas Pavanelli
"""
import json
import os
import random
import pandas as pd
import torch
import spacy
from transformers import AutoTokenizer


class Data:
    """
    Loads data.

    Parameters
    ----------
    file_name : str
        Dataset's file name

    Attributes
    ----------
    corpus : list
        List containing tokens, tags and postags
    vocab_in : set
        Input vocabulary
    in_w2id : dict
        Map from word to id for input vocabulary
    in_id2w : dict
        Map from id to word for input vocabulary
    vocab_out : set
        Output vocabulary
    out_w2id : dict
        Map from word to id for output vocabulary
    out_id2w : dict
        Map from id to word for output vocabulary

    Methods
    -------
    preprocess()
        Preprocesses tokens and tags
    fit()
        Gets train and test data
    """
    def __init__(self, file_name):
        corpus, vocab_in, in_w2id, in_id2w, vocab_out, out_w2id, out_id2w = Data._load_data(file_name)

        self.corpus = corpus
        # input vocabulary
        self.vocab_in = vocab_in
        self.in_w2id = in_w2id
        self.in_id2w = in_id2w
        # output vocabulary
        self.vocab_out = vocab_out
        self.out_w2id = out_w2id
        self.out_id2w = out_id2w

    @staticmethod
    def _find_sub_list(sub_list, full_list):
        results = []
        sll = len(sub_list)
        for ind in (i for i, e in enumerate(full_list) if e == sub_list[0]):
            if full_list[ind:ind + sll] == sub_list:
                results.append((ind, ind + sll - 1))
        return results

    @DeprecationWarning
    @staticmethod
    def _load_data_old(path_dir):
        nlp = spacy.load("pt_core_news_lg")
        corpus, postag, vocab_in, vocab_out = [], [], ['PAD', 'UNK'], []
        for fname in [w for w in os.listdir(path_dir) if not str(w).startswith('.')]:
            data = json.load(open(os.path.join(path_dir, fname)))
            for row in data:
                doc = nlp(row['data']['text'])
                tokens = [token.text for token in doc]
                postag = [token.pos_ for token in doc]
                tags = len(tokens) * ['O']
                ners = row['completions'][0]['result']
                for ner in ners:
                    label, text = ner['value']['labels'][0], ner['value']['text'].split()
                    indexes = Data._find_sub_list(text, tokens)
                    for idx in indexes:
                        s, e = idx
                        tags[s] = 'B-' + label
                        for j in range(s + 1, e + 1):
                            tags[j] = 'I-' + label
                corpus.append({'tokens': tokens, 'tags': tags, 'postags': postag})
                vocab_in.extend(tokens)
                vocab_out.extend(tags)

        vocab_in = set(vocab_in)
        in_w2id = {w: i for i, w in enumerate(vocab_in)}
        in_id2w = {i: w for i, w in enumerate(vocab_in)}

        vocab_out = set(vocab_out)
        out_w2id = {w: i for i, w in enumerate(vocab_out)}
        out_id2w = {i: w for i, w in enumerate(vocab_out)}
        return corpus, vocab_in, in_w2id, in_id2w, vocab_out, out_w2id, out_id2w

    @staticmethod
    def _load_data(path_dir):
        nlp = spacy.load("pt_core_news_lg")
        df = pd.read_csv(path_dir)
        file_name_list = list(df['File'].unique())
        corpus, vocab_in, vocab_out = [], ['PAD', 'UNK'], []
        for file_name in file_name_list:
            file_df = df[df['File'] == file_name]
            tokens = list(file_df['Token'])

            # TODO get postag of each token -> I should get postag of whole text instead
    #         text = ' '.join(tokens)
    #         doc = nlp(text)
            postag = [nlp(token)[0].pos_ for token in tokens]

            # put tags in IOB2 format
            file_tags = list(file_df['Entity'])
            cur_tag = ''
            tags = []
            for file_tag in file_tags:
                # remove number at the end of file tag
                tag = file_tag.split('[')[0]
                if file_tag == '_':
                    tags.append('O')
                elif file_tag != cur_tag:
                    tags.append('B-' + tag)
                    cur_tag = file_tag
                else:
                    tags.append('I-' + tag)

            assert len(tokens) == len(postag)
            corpus.append({'tokens': tokens, 'tags': tags, 'postags': postag})
            vocab_in.extend(tokens)
            vocab_out.extend(tags)

        vocab_in = set(vocab_in)
        in_w2id = {w: i for i, w in enumerate(vocab_in)}
        in_id2w = {i: w for i, w in enumerate(vocab_in)}

        vocab_out = set(vocab_out)
        out_w2id = {w: i for i, w in enumerate(vocab_out)}
        out_id2w = {i: w for i, w in enumerate(vocab_out)}
        return corpus, vocab_in, in_w2id, in_id2w, vocab_out, out_w2id, out_id2w

    def preprocess(self, tokens, tags=[]):
        """
        Converts tokens and tags to PyTorch tensors

        Parameters
        ----------
        tokens : list of string
            List of tokens.
        tags : list of string
            List of tags.

        Returns
        -------
        (Tensor, Tensor)
            Tokens and tags as PyTorch tensors
        """
        token_ids = torch.tensor([[self.in_w2id[word] for word in tokens]])
        tag_ids = torch.tensor([[self.out_w2id[word] for word in tags]])
        return token_ids, tag_ids

    def fit(self):
        """
        Separates corpus into training and test set

        Returns
        -------
        (List, List)
            Tokens and tags as PyTorch tensors
        """
        data = []
        for row in self.corpus:
            tokens, tags = row['tokens'], row['tags']

            token_ids, tag_ids = self.preprocess(tokens, tags)
            data.append((token_ids, tag_ids, tokens))

        size = int(0.1 * len(data))
        random.shuffle(data)
        train_data = data[size:]
        test_data = data[:size]

        return train_data, test_data


class DataBERT(Data):
    """
    Loads data for BERT model. Inherits from Data class.

    Parameters
    ----------
    file_name : str
        Dataset's file name

    Attributes
    ----------
    tokenizer : list
        BERT's tokenizer

    Methods
    -------
    preprocess()
        Preprocesses tokens and tags
    fit()
        Gets train and test data
    """

    def __init__(self, file_name):
        super().__init__(file_name)

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-large-portuguese-cased', do_lower_case=False)

    def preprocess(self, tokens, tags=[]):
        """
        Converts tokens and tags to PyTorch tensors

        Parameters
        ----------
        tokens : list of string
            List of tokens.
        tags : list of string
            List of tags.

        Returns
        -------
        (Tensor, Tensor, Tensor)
            Tokens, subwords and tags as PyTorch tensors
        """
        token_ids = self.tokenizer.encode(' '.join(tokens), return_tensors='pt')
        tag_ids = [self.out_w2id[w] for w in tags]

        wordpieces = self.tokenizer.convert_ids_to_tokens(token_ids[0])
        subwords_idx = []  # first subword of each word
        for i, wordpiece in enumerate(wordpieces):
            if '##' not in wordpiece and i not in [0, len(wordpieces) - 1]:
                subwords_idx.append(i)

        token_ids = torch.tensor(token_ids)
        tag_ids = torch.tensor(tag_ids)
        subwords_idx = torch.tensor(subwords_idx)
        return token_ids, subwords_idx, tag_ids

    def fit(self):
        """
        Separates corpus into training and test set

        Returns
        -------
        (List, List)
            Tokens and tags as PyTorch tensors
        """
        data = []
        for row in self.corpus:
            tokens, tags = row['tokens'], row['tags']

            token_ids, subwords_idx, tag_ids = self.preprocess(tokens, tags)
            if subwords_idx.size() == tag_ids.size():
                data.append((token_ids, subwords_idx, tag_ids))

        size = int(0.1 * len(data))
        random.shuffle(data)
        train_data = data[size:]
        test_data = data[:size]

        return train_data, test_data
