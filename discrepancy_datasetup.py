import os
import pandas as pd


def discrepancy_datasetup(config):

    dir_base = config["dir_base"]
    dataframe_location = os.path.join(dir_base,
                                      'Zach_Analysis/discrepancy_data/first_labeled_batch.xlsx')

    df = pd.concat(pd.read_excel(dataframe_location, sheet_name=None, engine='openpyxl'), ignore_index=True)
    #df = pd.read_excel(dataframe_location, engine='openpyxl')
    df = df.dropna(axis=0, how='all')

    pd.set_option('display.max_columns', None)

    label_idx = 0
    data_with_labels = pd.DataFrame(columns=['id', 'impression1', 'impression2', 'label'])
    index = -1
    num_neg = 0
    for _, row in df.iterrows():

        index += 1
        if pd.isna(row['Discrepancy']):
            continue

        impression1 = df.iloc[index-1]
        impression2 = row

        if impression1['Accession Number'] == impression2['Accession Number']:
            accession = impression2['Accession Number']
            report1 = impression1['Impression']
            #print(report1)
            report2 = impression2['Impression']
            label = impression2['Discrepancy']
            if label == 1:
                data_with_labels.loc[label_idx] = [accession, report1, report2, label]
                label_idx += 1
            else:
                if num_neg < 100:
                    data_with_labels.loc[label_idx] = [accession, report1, report2, label]
                    num_neg += 1
                    label_idx += 1

    return data_with_labels



