import os
import pandas as pd
import numpy as np

from nltk import sent_tokenize
import random


def discrepancy_datasetup(config):

    dir_base = config["dir_base"]
    #dataframe_location = os.path.join(dir_base,'Zach_Analysis/discrepancy_data/second_labeled_batch_hand_cleaned.xlsx')
    dataframe_location = os.path.join(dir_base, 'Zach_Analysis/discrepancy_data/second_set_unique_training_data_labeled_initial.xlsx')
    #dataframe_location = os.path.join(dir_base,'Zach_Analysis/discrepancy_data/second_labeled_batch.xlsx')

    #dataframe_location = os.path.join(dir_base,'Zach_Analysis/discrepancy_data/first_labeled_batch.xlsx')

    df = pd.concat(pd.read_excel(dataframe_location, sheet_name=None, engine='openpyxl'), ignore_index=True)
    #df = pd.read_excel(dataframe_location, engine='openpyxl')
    #df = pd.read_excel(dataframe_location, sheet_name="Head CT", engine='openpyxl')
    #print(df)
    df = df.dropna(axis=0, how='all')
    #print(df)

    pd.set_option('display.max_columns', None)

    string_dic = {}
    dups = 0

    label_idx = 0
    data_with_labels = pd.DataFrame(columns=['id', 'impression1', 'impression2', 'label'])
    index = -1
    num_neg = 0
    discrepancy_that_are_nan = 0
    non_matching = 0
    prelim_impression = "string1"
    prelim_accession = "id1"
    final_impression = "string2"
    final_accession = "id2"
    prelim_num = 0
    final_num = 0
    num_same_string = 0
    num_exclude = 0


    for _, row in df.iterrows():
        if row["Discrepancy"] == "Exclude":
            num_exclude += 1
            continue
        if pd.isna(row['Accession Number']):
            continue
        if row["Report Type"] == "Preliminary":
            prelim_impression = row['Impression']
            prelim_accession = row['Accession Number']
            prelim_num += 1
        elif row["Report Type"] == "Final":
            final_impression = row['Impression']
            final_accession = row['Accession Number']
            final_num += 1
        index += 1
        #print(row['Discrepancy score'])
        #if pd.isna(row['Accession Number']):
        #    continue
        if pd.isna(row['Discrepancy']):
            discrepancy_that_are_nan += 1
            continue
        #print(row)
        if str(prelim_accession) == str(final_accession):

            if prelim_impression == final_impression:
                num_same_string += 1
                continue

            string_key = str(prelim_impression) + str(final_impression)
            if string_key in string_dic.keys():

                string_dic[string_key].append(prelim_accession)
                # df.drop(row["id"])
                dups += 1
                #print(f"label of dup: {label}")
                continue
            else:
                string_dic[string_key] = [prelim_accession]

            #if pd.isna(row['Discrepancy score']):
            #if pd.isna(row['Discrepancy']):
            #if len():
            #    continue
            #print(f"management score: {row['Discrepancy score']}")
            #print(f"discrepant: {row['Discrepancy']}")
            #if pd.isna(row['Discrepancy score']) and row['Discrepancy'] == 0:
            #    row['Discrepancy score'] = 0
            #if int(str(row['Discrepancy score'])[-1]) <= 3 or row['Discrepancy'] == 0: #was == 0
            if row['Discrepancy'] == 0:
                label = 0
                if num_neg < 2800:
                    data_with_labels.loc[label_idx] = [prelim_accession, prelim_impression, final_impression, label]
                    num_neg += 1
            else:
                #label = str(row['Discrepancy score'])
                #label = int(label[0])
                label = 1
                data_with_labels.loc[label_idx] = [prelim_accession, prelim_impression, final_impression, label]
            label_idx += 1
        else:
            non_matching += 1

        """
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
        """
    print(f"num unmatched: {non_matching}")
    print(f"discrepancy nans: {discrepancy_that_are_nan}")
    print(f"times prelim is defined: {prelim_num}")
    print(f"times final is defined: {final_num}")
    print(f"discrepancies delcared: {label_idx}")
    print(f"number of same strings: {num_same_string}")
    print(f"duplicates: {dups}")
    print(f"number of reports exluded: {num_exclude}")

    #remove_duplicate_strings(data_with_labels)
    return data_with_labels

def remove_duplicate_strings(df):

    df.set_index("id")
    string_dic = {}
    dups = 0
    # loop through the dataframe putting each ID in a dictionary with string as key if string already in dictionary add it
    for _, row in df.iterrows():
        #print(row)
        string_imp = row["impression1"]
        string_final = row["impression2"]
        string_key = string_imp + string_final
        if string_key in string_dic.keys():

            string_dic[string_key].append(row["id"])
            #df.drop(row["id"])
            dups += 1
            print(f"label of dup: {row['label']}")
        else:
            string_dic[string_key] = [row["id"]]
        # check all other


    #print(string_dic)
    print(f"duplicates: {dups}")
    return string_dic
    #return clean_df

def balance_dataset(df, config, aug_factor):

    # synonym replacement setup
    wordReplacementPath = os.path.join(config["dir_base"], 'Zach_Analysis/discrepancy_data/full_synonym_list.xlsx')

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
    #aug_factor = int(np.round(np.maximum((neg_cases/pos_cases) - 1, (pos_cases/neg_cases) - 1)))
    #aug_factor = 1
    print(f"aug factor: {aug_factor}")

    balanced_idx = 0
    for _, row in df.iterrows():

        if int(row["label"]) >= 1:

            orig_text1 = row["impression1"]
            orig_text2 = row["impression2"]
            #print(f"original text1: {orig_text1}")
            #print(f"original text2: {orig_text2}")
            balanced_df.loc[balanced_idx] = row
            balanced_idx += 1
            #print(row["id"])
            # augment the example as many times as defined by the aug factor
            for i in range(aug_factor):

                text1 = shuffledTextAugmentation(orig_text1)
                #text1 = orig_text1
                #print(f"text1 before: {text1}")
                text1 = synonymsReplacement(wordDict, text1)
                #print(f"text1 after: {text1}")

                text2 = shuffledTextAugmentation(orig_text2)
                text2 = synonymsReplacement(wordDict, text2)
                #text2 = orig_text2

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
                #balanced_df.loc[balanced_idx] = row
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
    newText = text
    for word in list(wordDict["synonyms"].keys()):
        if word in text:
            randValue = random.uniform(0, 1)
            if randValue <= .15:
                randomSample = np.random.randint(low = 0, high = len(wordDict['synonyms'][word]))
                newText = text.replace(word, wordDict["synonyms"][word][randomSample])

    return newText

def discrepancy_datasetup_second_set(config):

    dir_base = config["dir_base"]
    #dataframe_location = os.path.join(dir_base,'Zach_Analysis/discrepancy_data/second_labeled_batch_hand_cleaned.xlsx')
    #dataframe_location = os.path.join(dir_base, 'Zach_Analysis/discrepancy_data/second_set_unique_training_data_labeled_initial.xlsx')
    #dataframe_location = os.path.join(dir_base,'Zach_Analysis/discrepancy_data/used_to_train_third_model/AI_Discrep_second_set_unique_training_updated Apr 2023_new scoring.xlsx')
    dataframe_location = os.path.join(dir_base,'Zach_Analysis/discrepancy_data/used_to_train_third_model/AI_Discrep_test_set_updated Apr 2023_new scoring.xlsx')

    #dataframe_location = os.path.join(dir_base,'Zach_Analysis/discrepancy_data/second_labeled_batch.xlsx')

    #dataframe_location = os.path.join(dir_base,'Zach_Analysis/discrepancy_data/first_labeled_batch.xlsx')

    df = pd.concat(pd.read_excel(dataframe_location, sheet_name=None, engine='openpyxl'), ignore_index=True)
    #df = pd.read_excel(dataframe_location, engine='openpyxl')
    #df = pd.read_excel(dataframe_location, sheet_name="Head CT", engine='openpyxl')
    #print(df)
    df = df.dropna(axis=0, how='all')
    #print(df)


    pd.set_option('display.max_columns', None)

    string_dic = {}
    dups = 0

    label_idx = 0
    data_with_labels = pd.DataFrame(columns=['id', 'impression1', 'impression2', 'label'])
    index = -1
    num_neg = 0
    discrepancy_that_are_nan = 0
    non_matching = 0
    prelim_impression = "string1"
    prelim_accession = "id1"
    final_impression = "string2"
    final_accession = "id2"
    prelim_num = 0
    final_num = 0
    num_same_string = 0
    num_exclude = 0
    prelim_with_values = 0
    num_addendums = 0


    for _, row in df.iterrows():
        if row["Discrepancy"] == "Exclude":
            num_exclude += 1
            continue
        if pd.isna(row['Accession Number']):
            continue
        if row["Report Type"] == "Addendum":
            num_addendums += 1
            continue
        if row["Report Type"] == "Preliminary":
            if not pd.isna(row["Discrepancy"]):
                prelim_with_values += 1
                # The prelim exam has a score so we need to get the final report and using the assession number
                #print(f"prelim accession number: {row['Accession Number']}")
                final_row = df.loc[(df['Accession Number'] == row["Accession Number"]) & (df['Report Type'] == 'Final')].iloc[0]
                #print(final_row)
                final_impression = final_row["Impression"]
                final_accession = final_row['Accession Number']
                final_row["Discrepancy"] = row["Discrepancy"]
                final_num += 1
            prelim_impression = row['Impression']
            prelim_accession = row['Accession Number']
            prelim_num += 1
        elif row["Report Type"] == "Final":
            final_impression = row['Impression']
            final_accession = row['Accession Number']
            final_num += 1
        index += 1
        #print(row['Discrepancy score'])
        #if pd.isna(row['Accession Number']):
        #    continue
        if pd.isna(row['Discrepancy']):
            discrepancy_that_are_nan += 1
            continue
        #print(row)
        if str(prelim_accession) == str(final_accession):

            if prelim_impression == final_impression:
                num_same_string += 1
                continue

            string_key = str(prelim_impression) + str(final_impression)
            if string_key in string_dic.keys():

                string_dic[string_key].append(prelim_accession)
                #df.drop(row["id"])
                dups += 1
                print(f"assession of dup: {prelim_accession}")
                print(f"string of dup: {string_key}")
                print(f"dic key: {string_dic[string_key]}")

                continue
            else:
                string_dic[string_key] = [prelim_accession]

            #if pd.isna(row['Discrepancy score']):
            #if pd.isna(row['Discrepancy']):
            #if len():
            #    continue
            #print(f"management score: {row['Discrepancy score']}")
            #print(f"discrepant: {row['Discrepancy']}")
            #if pd.isna(row['Discrepancy score']) and row['Discrepancy'] == 0:
            #    row['Discrepancy score'] = 0
            #if int(str(row['Discrepancy score'])[-1]) <= 3 or row['Discrepancy'] == 0: #was == 0
            if row['Discrepancy'] == 0:
                label = 0
                if num_neg < 2800:
                    data_with_labels.loc[label_idx] = [prelim_accession, prelim_impression, final_impression, label]
                    num_neg += 1
            else:
                #label = str(row['Discrepancy score'])
                #label = int(label[0])
                label = 1
                data_with_labels.loc[label_idx] = [prelim_accession, prelim_impression, final_impression, label]
            label_idx += 1
        else:
            non_matching += 1


    print(f"num unmatched: {non_matching}")
    print(f"discrepancy nans: {discrepancy_that_are_nan}")
    print(f"times prelim is defined: {prelim_num}")
    print(f"times final is defined: {final_num}")
    print(f"discrepancies delcared: {label_idx}")
    print(f"number of same strings: {num_same_string}")
    print(f"duplicates: {dups}")
    print(f"number of reports exluded: {num_exclude}")
    print(f"prelim with values: {prelim_with_values}")
    print(f"number of addendums: {num_addendums}")

    return data_with_labels