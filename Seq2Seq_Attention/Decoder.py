import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, enc_hidden_dim, dec_hidden_dim, dropout_ratio, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        # 임베딩(embedding)은 원-핫 인코딩(one-hot encoding) 말고 특정 차원의 임베딩으로 매핑하는 레이어
        self.embedding = nn.Embedding(output_dim, embed_dim)

        # GRU 레이어
        self.rnn = nn.GRU((enc_hidden_dim * 2) + embed_dim, dec_hidden_dim)

        # FC 레이어
        self.fc_out = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim + embed_dim, output_dim)

        # 드롭아웃(dropout)
        self.dropout = nn.Dropout(dropout_ratio)

    # 디코더는 현재까지 출력된 문장에 대한 정보를 입력으로 받아 타겟 문장을 반환
    def forward(self, input, hidden, enc_outputs):
        # input: [배치 크기]: 단어의 개수는 항상 1개이도록 구현
        # hidden: [배치 크기, 히든 차원]
        # enc_outputs: [단어 개수, 배치 크기, 인코더 히든 차원 * 방향의 수]: 전체 단어의 출력 정보

        input = input.unsqueeze(0)
        # input: [단어 개수 = 1, 배치 크기]

        embedded = self.dropout(self.embedding(input))
        # embedded: [단어 개수 = 1, 배치 크기, 임베딩 차원]

        attention = self.attention(hidden, enc_outputs)
        # attention: [배치 크기, 단어 개수]: 실제 각 단어에 대한 어텐선(attention) 값들

        attention = attention.unsqueeze(1)
        # attention: [배치 크기, 1, 단어 개수]: 실제 각 단어에 대한 어텐선(attention) 값들

        enc_outputs = enc_outputs.permute(1, 0, 2)
        # enc_outputs: [배치 크기, 단어 개수, 인코더 히든 차원 * 방향의 수]: 전체 단어의 출력 정보

        weighted = torch.bmm(attention, enc_outputs)
        # weighted: [배치 크기, 1, 인코더 히든 차원 * 방향의 수]
        # bmm: batch matrix multiplication [Batch, n,m] * [Batch, m, p] = [Batch, n, p]

        weighted = weighted.permute(1, 0, 2)
        # weighted: [1, 배치 크기, 인코더 히든 차원 * 방향의 수]

        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input: [1, 배치 크기, 인코더 히든 차원 * 방향의 수 + embed_dim]: 어텐션이 적용된 현재 단어 입력 정보

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # output: [단어 개수, 배치 크기, 디코더 히든 차원 * 방향의 수]
        # hidden: [레이어 개수 * 방향의 수, 배치 크기, 디코더 히든 차원]: 현재까지의 모든 단어의 정보

        # 현재 예제에서는 단어 개수, 레이어 개수, 방향의 수 모두 1의 값을 가짐
        # 따라서 output: [1, 배치 크기, 디코더 히든 차원], hidden: [1, 배치 크기, 디코더 히든 차원]
        # 다시 말해 output과 hidden의 값 또한 동일

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        # prediction = [배치 크기, 출력 차원]

        # (현재 출력 단어, 현재까지의 모든 단어의 정보)
        return prediction, hidden.squeeze(0)