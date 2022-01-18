import os
import time
import torch
from torch import optim
import torch.nn as nn

from portuguese_ner_biomedical.dataset import Data, DataBERT
from portuguese_ner_biomedical.model import CRF, LinearLayerCRF, BERTSlotFilling
from portuguese_ner_biomedical.evaluation import Evaluation
from portuguese_ner_biomedical.trainer import Trainer

# General constants
NUM_EXPERIMENTS = 1
OUTPUT_PATH = 'output_files/'
DATA_PATH = 'data_example/'

# Linear Layer + CRF constants
NUM_EPOCHS = 1
BATCH = 1

# BERT constants
HIDDEN_DIM = 1024


def main():
    time_str = time.strftime("%Y_%m_%d-%H:%M:%S")

    ################################################
    #           CRF
    ################################################
    crf_output_folder = OUTPUT_PATH + ('crf_%s' % time_str) + '/'
    if not os.path.exists(crf_output_folder):
        os.makedirs(crf_output_folder)
    data_info = Data(DATA_PATH)
    crf = CRF()
    evaluation = Evaluation(crf_output_folder)

    print("Evaluating CRF:")
    micro_avg_f1 = 0.0
    y_true = y_pred = test_tokens = []
    for num_experiment in range(NUM_EXPERIMENTS):
        x_train, y_train, x_test, y_true, test_tokens = crf.get_train_test_data(data_info)
        crf.fit(x_train, y_train)
        y_pred = crf.predict(x_test)
        micro_avg_f1 += evaluation.evaluate(num_experiment, y_true, y_pred)
    micro_avg_f1 /= NUM_EXPERIMENTS
    print()
    print('\tMicro average F1: %.2f' % micro_avg_f1)

    evaluation.generate_output_csv('crf_output', y_true, y_pred, test_tokens)

    ################################################
    #           Linear Layer + CRF
    ################################################
    linear_layer_crf_output_folder = OUTPUT_PATH + ('linear_layer_crf_%s' % time_str) + '/'
    if not os.path.exists(linear_layer_crf_output_folder):
        os.makedirs(linear_layer_crf_output_folder)

    data_info = Data(DATA_PATH)
    train_data, test_data = data_info.fit()
    vocab_size = len(data_info.vocab_in)
    num_classes = len(data_info.vocab_out)

    model = LinearLayerCRF(num_classes, vocab_size, data_info.out_w2id)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    trainer = Trainer(model, BATCH, is_bert=False)
    evaluation = Evaluation(linear_layer_crf_output_folder)

    micro_avg_f1 = 0.0
    y_true_text = y_pred_text = test_tokens = []
    for num_experiment in range(NUM_EXPERIMENTS):
        y_true, y_pred = trainer.test(test_data)
        for epoch in range(1, NUM_EPOCHS + 1):
            trainer.train(train_data, optimizer, epoch)
            y_true, y_pred = trainer.test(test_data)

        # get test tokens and convert output from number to text
        test_tokens = [info[-1] for info in test_data]
        y_true_text = evaluation.convert_output_to_text(y_true, data_info.out_id2w)
        y_pred_text = evaluation.convert_output_to_text(y_pred, data_info.out_id2w)

        micro_avg_f1 += evaluation.evaluate(num_experiment, y_true_text, y_pred_text)

    micro_avg_f1 /= NUM_EXPERIMENTS
    print()
    print('Micro avg F1: %.2f' % micro_avg_f1)
    evaluation.generate_output_csv('linear_layer_output', y_true_text, y_pred_text, test_tokens)

    ################################################
    #           BERT
    ################################################
    bert_output_folder = OUTPUT_PATH + ('bert_%s' % time_str) + '/'
    if not os.path.exists(bert_output_folder):
        os.makedirs(bert_output_folder)

    data_info = DataBERT(DATA_PATH)
    train_data, test_data = data_info.fit()

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    num_classes = len(data_info.vocab_out)
    model = BERTSlotFilling(HIDDEN_DIM, num_classes)
    model.to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.Adam(optimizer_grouped_parameters, lr=1e-5)
    weights = [1.] * num_classes
    weights[data_info.out_w2id['O']] = 0.01
    weights = torch.tensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    evaluation = Evaluation(bert_output_folder)
    trainer = Trainer(model, BATCH, is_bert=True, criterion=criterion)

    micro_avg_f1 = 0.0
    y_true_text = y_pred_text = test_tokens = []
    for num_experiment in range(NUM_EXPERIMENTS):
        y_true, y_pred = trainer.test(test_data)
        for epoch in range(1, NUM_EPOCHS + 1):
            trainer.train(train_data, optimizer, epoch)
            y_true, y_pred = trainer.test(test_data)

        # get test tokens and convert output from number to text
        test_tokens = [info[-1] for info in test_data]
        y_true_text = evaluation.convert_output_to_text(y_true, data_info.out_id2w)
        y_pred_text = evaluation.convert_output_to_text(y_pred, data_info.out_id2w)

        micro_avg_f1 += evaluation.evaluate(num_experiment, y_true_text, y_pred_text)

    micro_avg_f1 /= NUM_EXPERIMENTS
    print()
    print('Micro avg F1: %.2f' % micro_avg_f1)
    evaluation.generate_output_csv('bert_output', y_true_text, y_pred_text, test_tokens)


if __name__ == '__main__':
    main()