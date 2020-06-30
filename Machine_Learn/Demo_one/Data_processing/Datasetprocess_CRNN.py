from Demo_one.Data_processing import *


class CRNNDataset(Dataset):

    def __init__(self,data,labels,transfrom=None):
        super(CRNNDataset,self).__init__()

        data_crnn=[]
        labels_crnn=[]
        self.data=data
        self.labels=labels
        self.transfrom=transfrom
        self.data_rcnn=data_crnn
        self.labels_rcnn=labels_crnn


        for i in self.data:
            self.data_rcnn.append(i)

        for j in self.labels:
            self.labels_rcnn.append(j)

    def __len__(self):
        return len(self.data_rcnn)

    def __getitem__(self, item):
        d,tag=self.data_rcnn[item],self.labels_rcnn[item]
        return d,tag





