from Demo_one.Data_processing import *

class myDataset(Dataset):
    def __init__(self,data,labels,transfrom=None):
        super(myDataset,self).__init__()
        print("build Dataset load!")

        data_1=[]
        labels_1=[]
        self.data=data
        self.labels=labels
        self.transfrom=transfrom
        self.data_1 = data_1
        self.labels_1 = labels_1

        for i in self.data:
            self.data_1.append(i)

        for j in self.labels:
            self.labels_1.append(j)



    def __getitem__(self, idx):
        img, target = self.data_1[idx], self.labels_1[idx]
        return img, target
    def __len__(self):
        return len(self.data_1)



