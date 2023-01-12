import os
import pandas as pd
import numpy as np

from nltk import sent_tokenize
import random


def discrepancy_datasetup(config):

    dir_base = config["dir_base"]
    dataframe_location = os.path.join(dir_base,
                                      'Zach_Analysis/discrepancy_data/second_labeled_batch.xlsx')

    df = pd.concat(pd.read_excel(dataframe_location, sheet_name=None, engine='openpyxl'), ignore_index=True)
    #df = pd.read_excel(dataframe_location, engine='openpyxl')
    df = df.dropna(axis=0, how='all')

    pd.set_option('display.max_columns', None)

    label_idx = 0
    data_with_labels = pd.DataFrame(columns=['id', 'impression1', 'impression2', 'label'])
    index = -1
    num_neg = 0
    for _, row in df.iterrows():

        print(row["Report Type"])
        if row["Report Type"] == "Preliminary":
            prelim_impression = row['Impression']
            print("found prelim")
        else row["Report Type"] == "Final":
            final_impression = row['Impression']
            print("found final")
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
            label = impression2['Discrepancy score']
            if label == 1:
                data_with_labels.loc[label_idx] = [accession, report1, report2, label]
                label_idx += 1
            else:
                if num_neg < 20000:
                    data_with_labels.loc[label_idx] = [accession, report1, report2, label]
                    num_neg += 1
                    label_idx += 1

    return data_with_labels

def balance_dataset(df, config):

    # synonym replacement setup
    wordReplacementPath = os.path.join(config["dir_base"], 'Zach_Analysis/lymphoma_data/words_and_their_synonyms.xlsx')

    dfWord = pd.read_excel(wordReplacementPath, engine='openpyxl')
    dfWord.set_index("word", inplace=True)

    wordDict = dfWord.to_dict()
    for key in list(wordDict["synonyms"].keys()):
        string = wordDict["synonyms"][key][2:-2]
        wordList = string.split("', '")
        wordDict["synonyms"][key] = wordList

    balanced_df = pd.DataFrame(columns=['id', 'impression1', 'impression2', 'label'])
    pos_cases = 0
    for _, row in df.iterrows():

        pos_cases += row["label"]

    neg_cases = len(df) - pos_cases
    print(f"number of positive cases: {pos_cases}")
    print(f"number of negative cases: {neg_cases}")

    frac1 = neg_cases/pos_cases
    frac2 = pos_cases/neg_cases
    aug_factor = int(np.round(np.maximum((neg_cases/pos_cases) - 1, (pos_cases/neg_cases) - 1)))
    aug_factor = 4
    print(f"aug factor: {aug_factor}")

    balanced_idx = 0
    for _, row in df.iterrows():

        if row["label"] == 1:

            orig_text1 = row["impression1"]
            orig_text2 = row["impression2"]
            #print(f"original text1: {orig_text1}")
            #print(f"original text2: {orig_text2}")
            balanced_df.loc[balanced_idx] = row
            balanced_idx += 1
            #print(row["id"])
            # augment the example as many times as defined by the aug factor
            for i in range(aug_factor):

                #text1 = shuffledTextAugmentation(orig_text1)
                text1 = orig_text1
                #print(f"text1 before: {text1}")
                #text1 = synonymsReplacement(wordDict, text1)
                #print(f"text1 after: {text1}")

                #text2 = shuffledTextAugmentation(orig_text2)
                #text2 = synonymsReplacement(wordDict, text2)
                text2 = orig_text2

                id = row["id"] + "aug" + str(i)
                balanced_df.loc[balanced_idx] = [id, text1, text2, row["label"]]
                balanced_idx += 1
                #print(f"synth 1: {text1}")
                #print(f"synth 2: {text2}")

        else:
            #print("found negative case need to augment roughly 90% of these and put back in dataframe")
            randValue = random.uniform(0, 1)
            if randValue <= (1/aug_factor):
                balanced_df.loc[balanced_idx] = row
                balanced_idx += 1
            else:
                text1 = shuffledTextAugmentation(row["impression1"])
                text1 = synonymsReplacement(wordDict, text1)
                text2 = shuffledTextAugmentation(row["impression2"])
                text2 = synonymsReplacement(wordDict, text2)

                id = row["id"] + "aug"
                balanced_df.loc[balanced_idx] = [id, text1, text2, row["label"]]
                balanced_idx += 1

    pos_cases = 0
    for _, row in balanced_df.iterrows():
        pos_cases += row["label"]

    neg_cases = len(balanced_df) - pos_cases
    print(f"balanced number of positive cases: {pos_cases}")
    print(f"balanced number of negative cases: {neg_cases}")
    return balanced_df
def shuffledTextAugmentation(text):
    sentences = sent_tokenize(text)
    random.shuffle(sentences)
    shuffledText = sentences[0]
    for i in range(1, len(sentences)):
        shuffledText += " " + sentences[i]
    return shuffledText


def synonymsReplacement(wordDict, text):
    #wordDict = wordDict
    newText = text
    for word in list(wordDict["synonyms"].keys()):
        if word in text:
            randValue = random.uniform(0, 1)
            if randValue <= .15:
                randomSample = np.random.randint(low = 0, high = len(wordDict['synonyms'][word]))
                newText = text.replace(word, wordDict["synonyms"][word][randomSample])

    return newText