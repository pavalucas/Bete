import os
import json
import pandas as pd

PATH_DIR = 'Etiquetagem respostas_2021_05_06/'
EXCEL_FILE_NAME = 'respostas-label_text_sentence.xlsx'
TEXT_FILE_NAME = 'respostas-text.txt'


file_list = [w for w in os.listdir(PATH_DIR) if not str(w).startswith('.')]
file_list.sort()
csv_list = []
all_text_list = []
for fname in file_list:
    data = json.load(open(os.path.join(PATH_DIR, fname)))
    for row in data:
        sentence_text = row['data']['text']
        all_text_list.append(sentence_text)
        ners = row['completions'][0]['result']
        for ner in ners:
            label, text = ner['value']['labels'][0], ner['value']['text']
            csv_list.append([
                label,
                text,
                sentence_text
            ])
writer = pd.ExcelWriter(EXCEL_FILE_NAME, engine='xlsxwriter')
df = pd.DataFrame(data=csv_list, columns=['label', 'label_text', 'sentence_text'])
df.to_excel(writer, sheet_name='respostas', index=False, encoding='utf-8')

all_text = '\n\n'.join(all_text_list)
with open(TEXT_FILE_NAME, 'w') as f:
    f.write(all_text)