import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, batch ,vocab_size, length ,input_size): 
        super(CNN, self).__init__()

        # embedding_dim
        self.input_size = input_size

        # batch_size(50)
        self.batch = batch

        # word_embedding ()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=input_size)
        # kernerl_size(3*input_size), ouput_channel=1
        self.conv3 = nn.Conv2d(1, 100, (3, input_size),bias=True)

        # kernerl_size(4*input_size), ouput_channel=1
        self.conv4 = nn.Conv2d(1, 100, (4, input_size),bias=True)

        # kernerl_size(5*input_size), ouput_channel=1
        self.conv5 = nn.Conv2d(1, 100, (5, input_size),bias=True)

        # pooling(length-3+1)값 중 max 값
        self.Max3_pool = nn.MaxPool2d((length - 3 + 1, 1))

        # pooling(length-4+1)값 중 max 값
        self.Max4_pool = nn.MaxPool2d((length - 4 + 1, 1))

        # pooling(length-5+1)값 중 max 값
        self.Max5_pool = nn.MaxPool2d((length - 5 + 1, 1))

        # Fully_connected
        self.linear1 = nn.Linear(300, 1)

        # Dropout(Regularization)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # 1. 임베딩 층

        # output: [batch_size, sequence_length, embedding_size]
        output = self.embedding_layer(x)

        # x: [batch_size, channel_size, sequence_length, embedding_size]
        x = output.view(self.batch, 1, -1, self.input_size)

        # x1: [batch_size, output_channel, (length-3+1, 1)] feature map
        x1 = F.relu(self.conv3(x))

        # x2: [batch_size, output_channel, (length-4+1, 1)] feature map
        x2 = F.relu(self.conv4(x))

        # x3: [batch_size, output_channel, (length-5+1, 1)] feature map
        x3 = F.relu(self.conv5(x))

        # Pooling

        # x1: [batchsize, output_channel,1,1]
        x1 = F.relu(self.Max3_pool(x1))

        # x1: [batchsize, output_channel,1,1]
        x2 = F.relu(self.Max4_pool(x2))

        # x1: [batchsize, output_channel,1,1]
        x3 = F.relu(self.Max5_pool(x3))

        # capture and concatenate the features

        # x: [batchsize, output_channel, 1,3]
        x = torch.cat((x1, x2, x3), -1)

        # x: [batchsize, output_channel*3]
        x = x.view(self.batch, 300)

        # dropout
        x = self.dropout(x)

        # project the features to the labels

        # x: [batchsize, 1]
        x = F.sigmoid(self.linear1(x))

        return x
