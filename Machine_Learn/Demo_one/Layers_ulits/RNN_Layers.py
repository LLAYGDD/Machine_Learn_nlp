from Demo_one.Data_processing import *

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn=nn.LSTM(
            input_size=10,
            hidden_size=64,
            num_layers=1,
            batch_first=True,)

        self.dense=nn.Sequential(
            nn.Linear(64*1*1,128),
            nn.ReLU(),
            nn.Linear(128,16) )

    def forward(self, x):
        r_out,(h_n,h_c)=self.rnn(x,None)
        out=self.dense(r_out[:,-1,:])
        return out


