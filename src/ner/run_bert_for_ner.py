from BertForNer import BertForNer, Trainer
import utils
import pickle
from transformers import AutoTokenizer
import torch
from torch import optim
from utils import ENTITIES, tag2id, id2tag

# BERT_PRETRAINED_PATH = 'pucpr/biobertpt-bio'
BERT_PRETRAINED_PATH = 'neuralmind/bert-large-portuguese-cased'
LEARNING_RATE = 1e-5
EPOCHS = 50
BATCH_STATUS = 64
EARLY_STOP = 15
WRITE_PATH = 'bertimbau/'
VERBOSE = True
MAX_LENGTH = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def align(data):
    tokenizer = AutoTokenizer.from_pretrained(BERT_PRETRAINED_PATH, do_lower_case=False)
    procdata = []
    for data_map in data:
        X_ = ' '.join(data_map['tokens'])
        y_ = data_map['tags']
        inputs = tokenizer(X_, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        new_tags = ['O']
        pos = 0
        for token in tokens[1:-1]:
            if pos >= len(y_):
                new_tags.append('O')
            elif '##' in token:
                if y_[pos - 1].split('-')[-1] != 'O':
                    # add as an inside tag
                    new_tags.append('I-' + y_[pos - 1].split('-')[-1])
                else:
                    new_tags.append('O')
            else:
                new_tags.append(y_[pos])
                pos += 1
        new_tags.append('O')
        procdata.append({'X': X_, 'y': new_tags})
    return procdata

train_pkl = pickle.load(open('../../data/train/train.pkl', 'rb'))
dev_pkl = pickle.load(open('../../data/dev/dev.pkl', 'rb'))
test_pkl = pickle.load(open('../../data/test/test.pkl', 'rb'))

train_data = align(train_pkl)
dev_data = align(dev_pkl)
test_data = align(test_pkl)

model = BertForNer(max_length=MAX_LENGTH, tokenizer_path=BERT_PRETRAINED_PATH, model_path=BERT_PRETRAINED_PATH)
model.to(device)

# optimizer
optimizer = optim.AdamW(model.model.parameters(), lr=LEARNING_RATE)

trainer = Trainer(model, train_data, dev_data, optimizer, EPOCHS, BATCH_STATUS, device, WRITE_PATH)
trainer.train()