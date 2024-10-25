import os

config_file_dir = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-1])

DATA_DIR = os.path.join(config_file_dir, 'data')
en2fr_data = os.path.join(DATA_DIR, 'eng-fra.txt')

function_execution_log = 'function_execution_log.txt'
