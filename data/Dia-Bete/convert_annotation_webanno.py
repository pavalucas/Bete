import os
import json
import spacy

FOLDER_PATH = 'Etiquetagem respostas_2021_05_06/'
OUTPUT_PATH = 'respostas_anotadas_webanno/'

def find_sub_list(sl, l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))
    return results

def load_data(path_dir):
    nlp = spacy.load("pt_core_news_lg")
    global_index = 1
    file_list = [w for w in os.listdir(path_dir) if not str(w).startswith('.')]
    file_list.sort()
    for fname in file_list:
        data = json.load(open(os.path.join(path_dir, fname)))
        for row in data:
            label_index = 1
            sentence_text = row['data']['text']
            doc = nlp(sentence_text)
            tokens = [(token.text, token.idx, token.idx + len(token.text)) for token in doc]
            tags = len(tokens) * ['_']
            ners = row['completions'][0]['result']
            for ner in ners:
                label, text = ner['value']['labels'][0], ner['value']['text'].split()
                indexes = find_sub_list(text, [token[0] for token in tokens])
                for idx in indexes:
                    s, e = idx
                    new_label = '{}[{}]'.format(label, label_index)
                    for j in range(s, e + 1):
                        if tags[j] == '_':
                            tags[j] = new_label
                    label_index += 1

            with open('%sresposta_%04d.tsv' % (OUTPUT_PATH, global_index), 'w') as f:
                f.write('#FORMAT=WebAnno TSV 3.2\n')
                f.write('#T_SP=webanno.custom.Relaes|value\n\n\n')
                f.write('#Text={}\n'.format(sentence_text))
                token_index = 1
                for token, label in zip(tokens, tags):
                    token_text, start, end = token
                    f.write('1-{}\t{}-{}\t{}\t{}\n'.format(token_index, start, end, token_text, label))
                    token_index += 1
            global_index += 1

def main():
    load_data(FOLDER_PATH)


if __name__ == '__main__':
    main()