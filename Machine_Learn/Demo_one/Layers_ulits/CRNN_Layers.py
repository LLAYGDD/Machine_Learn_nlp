from Demo_one.Data_processing import *


class CRNN(nn.Module):
    def __init__(self,batch_size=64,
                 hidden_dim=1,
                 output_dim=16,
                 num_layers=1):
        super(CRNN, self).__init__()

        self.batch_size=batch_size
        self,

