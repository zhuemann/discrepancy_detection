import torch


class RobertaClassifier(torch.nn.Module):
    def __init__(self, lang_model1, lang_model2, n_class = 1):
        super(RobertaClassifier, self).__init__()
        self.lang_encoder1 = lang_model1
        self.lang_encoder2 = lang_model2
        self.classifier = torch.nn.Linear(1536, n_class)
        #self.classifier = torch.nn.Linear(2048, n_class)


    def forward(self, ids1, mask1, ids2, mask2, token_type_ids1, token_type_ids2): #, token_type_ids):

        lang_output1 = self.lang_encoder1(ids1, mask1, token_type_ids1)
        word_rep1 = lang_output1[0]
        report_rep1 = lang_output1[1]
        lang_rep_avg1 = report_rep1
        lang_output2 = self.lang_encoder2(ids2, mask2, token_type_ids2)
        word_rep2 = lang_output2[0]
        report_rep2 = lang_output2[1]
        lang_rep_avg2 = report_rep2
        print(f"report 1 size: {report_rep1.size()}")
        print(f"report 2 size: {report_rep2.size()}")

        dot_prod = torch.tensordot(report_rep1, report_rep2, dims=([1],[1]))
        print(f"report 1: {report_rep1[0][10]}")
        print(f"report 2: {report_rep2[0][10]}")
        print(f"dot: {dot_prod[0][10]}")
        print(f"dot size: {dot_prod.size()}")


        both_lang_rep = torch.cat((lang_rep_avg1, lang_rep_avg2), dim=1)
        #print(both_lang_rep.size())
        #print(both_lang_rep.size())

        output = self.classifier(both_lang_rep)

        return output