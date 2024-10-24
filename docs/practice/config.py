import os

# 현재 파일의 절대 경로
config_file_dir = os.path.abspath(__file__)

# 현재 파일의 디렉토리 경로

# 아래 코드와 같이 os.sep으로 해야 파일 디렉토리 표시를 os 환경마다 설정에 맞춰서 할 수 있음 = os.path.join
# DATA_DIR = f'{config_file_dir}{os.sep}data'
DATA_DIR = os.path.join(config_file_dir, 'data')
DATA_DIR = './data'