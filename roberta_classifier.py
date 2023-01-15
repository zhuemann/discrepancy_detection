import torch


class RobertaClassifier(torch.nn.Module):
    def __init__(self, model, n_class = 1):
        super(RobertaClassifier, self).__init__()
        self.lang_encoder = model
        self.classifier = torch.nn.Linear(1536, n_class)


    def forward(self, ids1, mask1, ids2, mask2, token_type_ids1, token_type_ids2): #, token_type_ids):

        lang_output1 = self.lang_encoder(ids1, mask1, token_type_ids1)
        word_rep1 = lang_output1[0]
        report_rep1 = lang_output1[1]
        lang_rep_avg1 = word_rep1
        lang_output2 = self.lang_encoder(ids2, mask2, token_type_ids2)
        word_rep2 = lang_output2[0]
        report_rep2 = lang_output2[1]
        lang_rep_avg2 = word_rep2
        # feed text through t5 then average across encoding dimension and then do two class classification
        #encoder_output1 = self.lang_encoder.encoder(input_ids=ids1, attention_mask=mask1, return_dict=True)
        #pooled_sentence1 = encoder_output1.last_hidden_state
        #lang_rep_avg1 = torch.mean(pooled_sentence1, 1)
        #print(lang_rep_avg1.size())
        # feed text through t5 then average across encoding dimension and then do two class classification
        #encoder_output2 = self.lang_encoder.encoder(input_ids=ids2, attention_mask=mask2, return_dict=True)
        #pooled_sentence2 = encoder_output2.last_hidden_state
        #lang_rep_avg2 = torch.mean(pooled_sentence2, 1)
        #print(lang_rep_avg2.size())
        both_lang_rep = torch.cat((lang_rep_avg1, lang_rep_avg2), dim=1)
        #print(both_lang_rep.size())
        output = self.classifier(both_lang_rep)

        return output