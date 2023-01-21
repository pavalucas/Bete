import pandas as pd
import itertools
import json
import utils

def _get_relation_example(src_ent: int, trg_ent: int, df_all: pd.DataFrame, is_bert : bool = False):
    text = ''
    ent_set = set()
    for i, row in df_all.iterrows():
        cur_token = row['Token'].lower()
        if row['Entity'] != '_' and row['Entity'] not in ent_set:
            if row['Entity'] == src_ent:
                if is_bert:
                    text += f'[E1]{cur_token}[/E1] '
                else:
                    text += 'ENTITY1 '
            elif row['Entity'] == trg_ent:
                if is_bert:
                    text += f'[E2]{cur_token}[/E2] '
                else:
                    text += 'ENTITY2 '
            else:
                if is_bert:
                    text += f'{cur_token} '
                else:
                    text += 'OTHER_ENTITY '
            ent_set.add(row['Entity'])
        elif row['Entity'] == '_':
            text += cur_token + ' '
    return text

def _trim_relation(sentence: str, window_size: int = 3):
    tokens = sentence.split()
    # make sure that we don't overflow but using the min and max methods
    first_index = max(tokens.index("ENTITY1") - window_size , 0)
    second_index = min(tokens.index("ENTITY2") + window_size, len(tokens))
    
    trimmed_tokens = tokens[first_index : second_index]
    return ' '.join(trimmed_tokens)

def load_relation_data_diabete(df, file_name_list, is_bert : bool = False):
    X_text, y_text = [], []
    for file_name in file_name_list:
        print(file_name)
        file_df = df[df['File'] == file_name]
        tokens_id = list(file_df['Token ID'])
        relations_id = list(file_df['Relation ID'])
        relations = list(file_df['Relation'])
        entity_set = set(list(file_df['Entity']))
        relation_set = set()
        X_text_, y_text_ = [], []
        for cur_id, (rel, rel_id) in enumerate(zip(relations, relations_id)):
            if rel_id != '_':
                rel_id_list = rel_id.split('|')
                rel_list = rel.split('|')
                for index, text in zip(rel_id_list, rel_list):
                    if text in utils.BETE_RELATIONS:
                        src_token_id = index.split('[')[0]
                        row_id = tokens_id.index(src_token_id)
                        src_ent = file_df.iloc[row_id]['Entity']
                        trg_ent = file_df.iloc[cur_id]['Entity']
                        if src_ent != trg_ent:
                            example = _get_relation_example(src_ent, trg_ent, file_df, is_bert=is_bert)
                            # example = _trim_relation(example)
                            X_text_.append(example)
                            y_text_.append(text)
                            relation_set.add((src_ent, trg_ent))
        entity_set.remove('_')
        initial_len = len(relation_set)
        for src_ent, trg_ent in itertools.product(entity_set, entity_set):
            rel = (src_ent, trg_ent)
            rel_inv = (trg_ent, src_ent)
            if src_ent != trg_ent and len(relation_set) < 2 * initial_len and rel not in relation_set and rel_inv not in relation_set:
                example = _get_relation_example(src_ent, trg_ent, file_df, is_bert=is_bert)
                X_text_.append(example)
                y_text_.append('O')
                relation_set.add(rel)
        if len(X_text_) > 0 and len(y_text_) > 0:
            X_text.extend(X_text_)
            y_text.extend(y_text_)
    return X_text, y_text

def create_bert_re_df(X_text, y_text):
    df_list = []
    for X_, y_ in zip(X_text, y_text):
        rel = 'Other'
        if y_ != 'O':
            rel = f'{y_}(e1,e2)'
        df_list.append({'sents': X_, 'relations': rel, 'relations_id': utils.bete_relation2id[y_]})
    return pd.DataFrame(df_list)

def _get_relation_example_ehealth(text: str, arg1_spans: list, arg2_spans: list):
    ind1 = arg1_spans[0][0]
    ind2 = arg1_spans[-1][1]
    ind3 = arg2_spans[0][0]
    ind4 = arg2_spans[-1][1]
    return text[:ind1] + '[E1]' + text[ind1:ind2] + '[/E1]' + text[ind2:ind3] + '[E2]' + text[ind3:ind4] + '[/E2]' + text[ind4:]

def load_relation_data_ehealth(file_name: str):
    json_map = json.load(open(file_name))
    df_list = []
    for sent in json_map:
        text = sent['text']
        relation_set = set()
        for rel in sent['relations']:
            try:
                arg1_spans = sent['keyphrases'][str(rel['arg1'])]['spans']
                arg2_spans = sent['keyphrases'][str(rel['arg2'])]['spans']
                new_text = _get_relation_example_ehealth(text, arg1_spans, arg2_spans)
                df_list.append([new_text, rel['label'] + '(e1,e2)', utils.ehealth_relation2id[rel['label']]])
                arg1_idxs = tuple(sent['keyphrases'][str(rel['arg1'])]['idxs'])
                arg2_idxs = tuple(sent['keyphrases'][str(rel['arg2'])]['idxs'])
                relation_set.add((arg1_idxs, arg2_idxs))
            except:
                pass
        initial_len = len(relation_set)
        for src_ent, trg_ent in itertools.product(sent['keyphrases'].values(), sent['keyphrases'].values()):
            # src_ent = list(src_ent_map.values())[0]
            # trg_ent = list(trg_ent_map.values())[0]
            arg1_idxs = tuple(src_ent['idxs'])
            arg2_idxs = tuple(trg_ent['idxs'])
            rel_ = (arg1_idxs, arg2_idxs)
            rel_inv_ = (arg2_idxs, arg1_idxs)
            if src_ent != trg_ent and len(relation_set) < 2 * initial_len and rel_ not in relation_set and rel_inv_ not in relation_set:
                new_text = _get_relation_example_ehealth(text, src_ent['spans'], trg_ent['spans'])
                df_list.append([new_text, 'Other', utils.ehealth_relation2id['O']])
                relation_set.add(rel_)

    return pd.DataFrame(df_list, columns=['sents', 'relations', 'relations_id'])

                        
