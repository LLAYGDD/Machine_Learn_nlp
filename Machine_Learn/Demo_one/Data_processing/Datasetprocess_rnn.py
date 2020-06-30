from Demo_one.Data_processing import *

class RNNdataprocess(Dataset):
    def __init__(self,data,labels,transfrom=None):
        super(RNNdataprocess).__init__()

        data_2=[]
        labels_2=[]
        self.data=data
        self.labels=labels
        self.transfrom=transfrom
        self.data_2=data_2
        self.labels_2=labels_2

        for i in self.data:
            self.data_2.append(i)

        for j in self.labels:
            self.labels_2.append(j)


    def __len__(self):
        data_rnn=self.data_2
        return len(data_rnn)
    def __getitem__(self, index):
        d,tag=self.data_2[index],self.labels_2[index]
        return d,tag







