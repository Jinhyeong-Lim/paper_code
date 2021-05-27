import torch
import torch.nn as nn
import torch.nn.functional as f


class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()

        self.attn = nn.Linear(enc_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, hidden, enc_outputs):
        # hidden: [배치 크기, 히든 차원]: 현재까지의 모든 단어의 정보
        # enc_outputs: [단어 개수, 배치 크기, 인코더 히든 차원 * 방향의 수]: 전체 단어의 출력 정보
        src_len = enc_outputs.shape[0]

        # 현재 디코더의 히든 상태(hidden state)를 src_len만큼 반복
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        enc_outputs = enc_outputs.permute(1, 0, 2)
        # hidden: [배치 크기, 단어 개수, 디코더 히든 차원]: 현재까지의 모든 단어의 정보
        # enc_outputs: [배치 크기, 단어 개수, 인코더 히든 차원 * 방향의 수]: 전체 단어의 출력 정보

        # 디코더의 현재까지 모든 단어 정보와 인코더의 단어 정보 concatenation
        x = torch.cat((hidden, enc_outputs), dim=2)

        # energy: [배치 크기, 단어 개수, 디코더 히든 차원]
        energy = torch.tanh(self.attn(x))

        # attention: [배치 크기, 단어 개수]: 실제 각 단어에 대한 어텐선(attention) 값들
        attention = self.v(energy).squeeze(2)

        # 각 단어의 어텐션 값에 Softmax 함수 적용(합이 1)
        attention = f.softmax(attention, dim=1)

        return attention
