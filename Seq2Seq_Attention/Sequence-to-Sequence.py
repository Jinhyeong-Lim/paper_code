import torch
import torch.nn as nn
import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    # 학습할 때는 완전한 형태의 소스 문장, 타겟 문장, teacher_forcing_ratio를 넣기
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [단어 개수, 배치 크기]
        # trg: [단어 개수, 배치 크기]
        # 먼저 인코더를 거쳐 전체 출력과 문맥 벡터(context vector)를 추출
        enc_outputs, hidden = self.encoder(src)

        # 디코더(decoder)의 최종 결과를 담을 텐서 객체 만들기
        # 단어 개수
        trg_len = trg.shape[0]

        # 배치 크기
        batch_size = trg.shape[1]

        # 출력 차원
        trg_vocab_size = self.decoder.output_dim

        #모든 값이 0인 [trg_len, batch_size, trg_vocab_size] 크기의 tensor 생성
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)

        # 첫 번째 입력은 항상 <sos> 토큰
        input = trg[0, :]

        # 타겟 단어의 개수만큼 반복하여 디코더에 포워딩(forwarding)
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, enc_outputs)
            outputs[t] = output # FC를 거쳐서 나온 현재의 출력 단어 정보
            top1 = output.argmax(1) # 가장 확률이 높은 단어의 인덱스 추출

            # teacher_forcing_ratio: 학습할 때 실제 목표 출력(ground-truth)을 사용하는 비율
            teacher_force = random.random() < teacher_forcing_ratio

            # 현재의 출력 결과를 다음 입력에서 넣기
            input = trg[t] if teacher_force else top1

        return outputs