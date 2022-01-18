import glob
import os

data_folder = 'Etiquetagem respostas_2021_05_06/'
data_path_list = glob.glob('%s*' % data_folder)
data_path_list.sort()
for index, file_name in enumerate(data_path_list):
    os.rename(file_name, '%sresult_respostas_%d.json' % (data_folder, index))