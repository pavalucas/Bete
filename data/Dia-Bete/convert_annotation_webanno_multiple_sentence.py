import os
import json
import spacy

FOLDER_PATH = 'Etiquetagem respostas_2021_05_06/'
OUTPUT_PATH = 'respostas_anotadas_webanno_multiple_sentence/'

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
        sentence_text_list, tokens_list, tags_list = [], [], []
        label_index = 1
        for row in data:
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
            sentence_text_list.append(sentence_text)
            tokens_list.append(tokens)
            tags_list.append(tags)

        with open('%sresposta_sentencas_%04d.tsv' % (OUTPUT_PATH, global_index), 'w') as f:
            f.write('#FORMAT=WebAnno TSV 3.2\n')
            f.write('#T_SP=webanno.custom.Relaes|value\n\n\n')
            last_pos = 0
            for sentence_index in range(len(sentence_text_list)):
                f.write('#Text={}\n'.format(sentence_text_list[sentence_index]))
                token_index = 1
                for token, label in zip(tokens_list[sentence_index], tags_list[sentence_index]):
                    token_text, start, end = token
                    f.write('{}-{}\t{}-{}\t{}\t{}\n'.format(sentence_index+1, token_index, last_pos + start, last_pos + end,
                                                            token_text, label))
                    token_index += 1
                last_pos += len(sentence_text_list[sentence_index]) + 1
                if sentence_index < len(sentence_text_list) - 1:
                    f.write('\n')
        global_index += 1

def main():
    load_data(FOLDER_PATH)


if __name__ == '__main__':
    main()