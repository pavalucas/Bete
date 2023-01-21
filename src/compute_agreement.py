import pandas as pd
import glob
import os
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score

def parse_ent(ent: str):
    upd_entity = ent
    if ent.endswith(']'):
        upd_entity = ent.split('[')[0]
    return upd_entity

def compute_entity_agreement():
    main_files = glob.glob('../data/Bete_main/annotation/*')
    nutrition_files = glob.glob('../data/Bete_nutrition/annotation/*')
    main_files.sort()
    nutrition_files.sort()
    all_files = {'main': main_files, 'nutrition': nutrition_files}
    df = pd.read_csv('../data/all_annotated_dataset_2023_01_18.csv')
    ent1_full, ent2_full = [], []
    for data_name, data_files in all_files.items():
        print(f"Number of {data_name} documents: {len(data_files)}")
        for cur_file in data_files:
            document_name = cur_file.split(os.sep)[-1]
            file_df = df[df['File'] == f'{data_name}_{document_name}']
            annotator_set = set(file_df['Annotator'])
            ann_tuple = []
            for annot in annotator_set:
                annotator_df = file_df[file_df['Annotator'] == annot]
                count_entity = 0
                for ann_ent in annotator_df['Entity']:
                    if ann_ent != '_':
                        count_entity += 1
                ann_tuple.append((annot, count_entity))
            sorted_list = sorted(ann_tuple, key=lambda v: (-v[1], v[0]))
            if len(sorted_list) >= 2:
                ann1 = file_df[file_df['Annotator'] == sorted_list[0][0]]
                ann2 = file_df[file_df['Annotator'] == sorted_list[1][0]]
                ent1_list, ent2_list = [], []
                for ent1, ent2 in zip(ann1['Entity'], ann2['Entity']):
                    if parse_ent(ent1) != '_' and parse_ent(ent2) != '_' and parse_ent(ent1) != '*' and parse_ent(ent2) != '*':
                    # if (parse_ent(ent1) != '_' or parse_ent(ent2) != '_') and parse_ent(ent1) != '*' and parse_ent(ent2) != '*':
                        ent1_list.append(parse_ent(ent1))
                        ent2_list.append(parse_ent(ent2))
                ent1_full.extend(ent1_list)
                ent2_full.extend(ent2_list)
    print('Classification report:')
    print(classification_report(ent1_full, ent2_full, labels=['Complication',
    'DiabetesType',
    'Dose',
    'Duration',
    'Food',
    'GlucoseValue',
    'Insulin',
    'Medication',
    'NonMedicalTreatment',
    'Set',
    'Symptom',
    'Test',
    'Time']))
    print('Cohen kappa:')
    print(cohen_kappa_score(ent1_full, ent2_full))

def parse_rel(rel: str):
    upd_rel = rel
    if '|' in rel:
        upd_rel = rel.split('|')[0]
    return upd_rel

def compute_relation_agreement():
    main_files = glob.glob('../data/Bete_main/annotation/*')
    nutrition_files = glob.glob('../data/Bete_nutrition/annotation/*')
    main_files.sort()
    nutrition_files.sort()
    all_files = {'main': main_files, 'nutrition': nutrition_files}
    df = pd.read_csv('../data/all_annotated_dataset_2023_01_18.csv')
    rel1_full, rel2_full = [], []
    for data_name, data_files in all_files.items():
        print(f"Number of {data_name} documents: {len(data_files)}")
        for cur_file in data_files:
            document_name = cur_file.split(os.sep)[-1]
            file_df = df[df['File'] == f'{data_name}_{document_name}']
            annotator_set = set(file_df['Annotator'])
            # print(document_name)
            # print(annotator_set)
            ann_tuple = []
            for annot in annotator_set:
                annotator_df = file_df[file_df['Annotator'] == annot]
                count_relation = 0
                for ann_ent in annotator_df['Relation']:
                    if ann_ent != '_':
                        count_relation += 1
                ann_tuple.append((annot, count_relation))
            sorted_list = sorted(ann_tuple, key=lambda v: (-v[1], v[0]))
            if len(sorted_list) >= 2:
                ann1 = file_df[file_df['Annotator'] == sorted_list[0][0]]
                ann2 = file_df[file_df['Annotator'] == sorted_list[1][0]]
                ent1_list, ent2_list = [], []
                for ent1, ent2 in zip(ann1['Relation'], ann2['Relation']):
                    if parse_rel(ent1) != '_' and parse_rel(ent2) != '_' and parse_rel(ent1) != '*' and parse_rel(ent2) != '*':
                    # if (parse_rel(ent1) != '_' or parse_rel(ent2) != '_') and parse_rel(ent1) != '*' and parse_rel(ent2) != '*':
                        ent1_list.append(parse_rel(ent1))
                        ent2_list.append(parse_rel(ent2))
                rel1_full.extend(ent1_list)
                rel2_full.extend(ent2_list)
    print('Classification report:')
    print(classification_report(rel1_full, rel2_full, labels=['causes', 'prevents', 'treats', 'has', 'diagnoses', 'complicates']))
    print('Cohen kappa:')
    print(cohen_kappa_score(rel1_full, rel2_full))

if __name__ == '__main__':
    print('Entity agreement:')
    compute_entity_agreement()
    print('\nRelation agreement:')
    compute_relation_agreement()