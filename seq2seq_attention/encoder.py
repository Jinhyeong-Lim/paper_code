import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, enc_hidden_dim,
                 dec_hidden_dim, dropout_ratio):
        super().__init__()

        # 임베딩(embedding)은 원-핫 인코딩(one-hot encoding)을 특정 차원의 임베딩으로 매핑하는 레이어
        self.embedding = nn.Embedding(input_dim, embed_dim)

        # 양방향(bidirectional) GRU 레이어
        self.rnn = nn.GRU(embed_dim, enc_hidden_dim, bidirectional=True)

        # FC 레이어
        self.fc = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)

        # 드롭아웃(dropout)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, src):
        # src: [단어 개수, 배치 크기]: 각 단어의 인덱스(index) 정보

        # embedded: [단어 개수, 배치 크기, 임베딩 차원]
        embedded = self.dropout(self.embedding(src))

        outputs, hidden = self.rnn(embedded)
        # outputs: [단어 개수, 배치 크기, 인코더 히든 차원 * 방향의 수]: 전체 단어의 출력 정보
        # hidden: [레이어 개수 * 방향의 수, 배치 크기, 인코더 히든 차원]: 현재까지의 모든 단어의 정보

        # hidden은 Bidirectional 이기 때문에
        # [forward_1, backward_1, forward_2, backward_2, ...] 형태로 구성
        # hidden[-2, :, :]은 forwards의 마지막 값
        # hidden[-1, :, :]은 backwards의 마지막 값
        # 디코더(decoder)의 첫 번째 hidden (context) vector는 인코더의 마지막 hidden을 이용
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :],
                                               hidden[-1, :, :]), dim=1)))

        # outputs은 Attention 목적으로 hidden은 context vector 목적으로 사용
        return outputs, hidden
