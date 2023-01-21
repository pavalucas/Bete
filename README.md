# Bete: A Brazilian Portuguese Dataset for Named Entity Recognition and Relation Extraction in the Diabetes Domain

This repository is the official implementation of Bete: A Brazilian Portuguese Dataset for Named Entity Recognition and Relation Extraction in the Diabetes Domain. 

## Training and evaluation
### Named entity recognition (NER)

To train NER model(s) in the paper, run this command:

```train
python src/ner/run_bert_for_ner.py
```
### Relation extraction (RE)

To train RE model(s) in the paper, go to [RE notebook](src/relation_extraction/re.ipynb).

## Results

Our model achieves the following performance on Bete:

### Named entity recognition

| Model              | Precision  | Recall | F1|
| ------------------ |---------------- | -------------- | -------------- |
|CRF | **80.3** | 72.9 | 76.1 |	
|BioBERTpt-bio | 73.1 | 80.5 | 76.6 |
|BioBERTpt-clin | 77.5 | 81.8 | **79.4** |
|BioBERTpt-all | 74.5 | **83.2** | 78.5 |
|BERTimbau | 72.2 | 70.2 | 70.8 |
|mBERT | 74.3 | 81.6 | 77.6 |

### Relation extraction

| Model              | Precision  | Recall | F1|
| ------------------ |---------------- | -------------- | -------------- |
|SVM | **80.0** | 46.2 | 58.1 |
|BERT-RE (mBERT) | 73.6 | **77.8** | **75.1** |
