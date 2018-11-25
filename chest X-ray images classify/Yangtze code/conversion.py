"""
这个文件用来将csv文件中的标签转化为one-hot编码
"""

import pandas as pd
import numpy as np

label = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
         'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding']
label_path = './Data_Entry_2017.csv'

csv_col_name = 'Image Index'
csv_col_value = 'Finding Labels'

result_frame = pd.read_csv(label_path)
result_dist = dict(zip(result_frame[csv_col_name].values, result_frame[csv_col_value].values))

name = []
label_all = []
for img in result_dist:
    labels = str(result_dist[img])
    label_true = ''
    labels = labels.split('|')
    for label_split in label:
        # print(label_split)
        if label_split in labels:
            label_true = label_true + '1'
        else:
            label_true = label_true + '0'
    name.append(str(img))
    label_all.append(str(label_true) + '/')

a = np.column_stack((name, label_all))
resultLabel = pd.DataFrame(data=a, columns=['name', 'label'])
print("process complete.")
resultLabel.to_csv('./' + "Label.csv", index=False)
