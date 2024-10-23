import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from data_handler import Vocabulary, parse_file  # 필요한 함수와 클래스 임포트
from rnns import RNNCellManual  # RNN 셀 임포트
from lstms import LSTMCellManual  # LSTM 셀 임포트
from attentions import LuongAttention, BahdanauAttention  # 어텐션 클래스 임포트
from seq2seq import Encoder, Decoder, Seq2Seq  # 인코더, 디코더, 시퀀스 투 시퀀스 모델 임포트
from ttrain_evaluation import Seq2SeqTrainer  # 트레이너 클래스 임포트
from metrics import bleu  # BLEU 점수 계산 함수 임포트

def main(num_epochs=50, batch_size=32, learning_rate=0.001):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 사용할 장치 설정
    print(f'Using device: {DEVICE}')

    # 1. 데이터 로드 및 전처리
    # 데이터셋 경로 설정
    file_path = '../../dataset/eng-fra.txt'  # 데이터셋 파일 경로

    # 데이터 파일 파싱
    dataloaders, source_vocab, target_vocab = parse_file(file_path, batch_size=batch_size)  # 데이터 로더 및 어휘 사전 생성
    train_loader, valid_loader, test_loader = dataloaders  # 데이터 로더 분리

    # 패딩 인덱스 가져오기 (어휘 사전에서)
    PAD_IDX = Vocabulary.PAD_IDX  # Vocabulary 클래스의 PAD_IDX 속성을 가져옵니다.
    
    # 2. 모델 초기화
    # 인코더와 디코더 초기화
    embedding_dim = 128  # 임베딩 차원 (필요에 따라 조정 가능)
    hidden_size = 256  # 숨겨진 상태의 크기

    # Attention 메커니즘 선택 (예: LuongAttention 또는 BahdanauAttention)
    attention = LuongAttention  # 또는 BahdanauAttention

    encoder = Encoder(source_vocab, embedding_dim, hidden_size, RNNCellManual, DEVICE).to(DEVICE)  # 인코더 모델 초기화
    decoder = Decoder(target_vocab, embedding_dim, hidden_size, RNNCellManual, attention, DEVICE).to(DEVICE)  # 디코더 모델 초기화
    model = Seq2Seq(encoder, decoder).to(DEVICE)  # 시퀀스 투 시퀀스 모델 초기화

    # 3. 손실 함수와 옵티마이저 설정
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)  # 손실 함수로 CrossEntropyLoss 사용
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam 옵티마이저 사용

    # 4. Trainer 인스턴스 생성
    trainer = Seq2SeqTrainer(
        model, 
        train_loader, 
        valid_loader, 
        optimizer, 
        criterion, 
        DEVICE, 
        encoder_model_name=encoder.model_type, 
        decoder_model_name=decoder.model_type, 
        attention_model_name=attention.__name__, 
        source_vocab=source_vocab,
        target_vocab=target_vocab
    )  # Trainer 인스턴스 생성

    # 5. 모델 학습
    trainer.train(num_epochs)  # 학습 시작

    # 6. 최종 테스트 평가
    # test_loss, test_accuracy = trainer.evaluate(test_loader)  # 테스트 데이터셋에 대한 평가
    # print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')  # 테스트 결과 출력

if __name__ == '__main__':
    NUM_EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    main(NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE)  # 하이퍼파라미터를 인자로 전달
