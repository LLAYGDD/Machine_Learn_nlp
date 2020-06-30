from Demo_one.Data_processing import *


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(1,16,5,1,1),           # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (1, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 1),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.dense=nn.Sequential(
            nn.Linear(32*1*1,128),
            nn.ReLU(),
            nn.Linear(128,16))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.dense(x)
        return output   # return x for visualization





