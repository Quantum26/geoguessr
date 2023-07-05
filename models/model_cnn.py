import torch


class SentimentModel(torch.nn.Module):

    def __init__(self, embedding = lambda x: x):
        super(SentimentModel, self).__init__()
        self.embedding = embedding
        self.conv1 = torch.nn.Conv1d(in_channels=300, out_channels=64, kernel_size=3)
        self.conv2 = torch.nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.max1 = torch.nn.MaxPool1d(kernel_size=3)
        self.conv3 = torch.nn.Conv1d(in_channels=32, out_channels=16)
        self.conv4 = torch.nn.Conv1d(in_channels=16, out_channels=8)
        self.avg_pool = torch.nn.AvgPool1d()
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avg_pool(x)
        x = self.sig(x)
        return x