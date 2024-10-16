import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from data_handler import Vocabulary, parse_file, preprocessing  # 필요한 함수와 클래스 임포트
from rnns import RNNCellManual  # RNN 셀 임포트
from lstms import LSTMCellManual  # LSTM 셀 임포트
from attentions import LuongAttention, BahdanauAttention  # 어텐션 클래스 임포트
from seq2seq import Encoder, Decoder, Seq2Seq  # 인코더, 디코더, 시퀀스 투 시퀀스 모델 임포트
from trainer import Seq2SeqTrainer  # 트레이너 클래스 임포트

def main(num_epochs=50, batch_size=32, learning_rate=0.001):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 데이터 로드 및 전처리
    train_data, valid_data, vocab = parse_file('./dataset/kor.txt')  # 데이터 파일 경로
    train_data, valid_data = preprocessing(train_data), preprocessing(valid_data)
    
    # 2. 데이터 로더 설정
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    
    # 3. 모델 초기화
    embedding_dim = 256
    hidden_size = 512
    output_size = len(vocab)  # 어휘의 크기
    encoder = Encoder(vocab_size=len(vocab), embedding_dim=embedding_dim, hidden_size=hidden_size).to(DEVICE)
    decoder = Decoder(vocab_size=len(vocab), embedding_dim=embedding_dim, hidden_size=hidden_size).to(DEVICE)
    model = Seq2Seq(encoder, decoder).to(DEVICE)
    
    # 4. 옵티마이저 및 손실 함수 설정
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])  # <PAD>에 대한 손실 무시
    
    # 5. 트레이너 초기화 및 학습 시작
    trainer = Seq2SeqTrainer(model, train_loader, valid_loader, optimizer, criterion, DEVICE)
    trainer.train(num_epochs)

if __name__ == '__main__':
    NUM_EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    main(NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE)  # 하이퍼파라미터를 인자로 전달
