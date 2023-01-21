import pandas as pd
import numpy as np

OUTPUT_FOLDER = 'respostas_validadas_webanno_2021_10_20/'

df = pd.read_csv('CorpusNutricao.csv', sep=',')
answer_list = list(df['Resposta Nutrição'].values)
answer_list = [answer for answer in answer_list if answer is not np.nan]
for index in range(len(answer_list)):
    answer = answer_list[index]
    with open('%sresposta_%04d.txt' % (OUTPUT_FOLDER, index + 1), 'w') as f:
        f.write(answer)
