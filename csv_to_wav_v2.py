import os
import torch
import argparse
import csv

parser = argparse.ArgumentParser()
    
parser.add_argument("-csv", "--filecsv", type=str, help= \
    "file csv.", default=None)
parser.add_argument("-f", "--folder", type=str, help= \
    "folder create wav.", default=None)
parser.add_argument('-pt', u'--speaker', type=str, help= \
    "file model PT.", default=None)
args = parser.parse_args()

device = torch.device('cpu')
torch.set_num_threads(4)
symbols = '_~абвгдеёжзийклмнопрстуфхцчшщъыьэюя +.,!?…:;–1234567890'
local_file = args.speaker

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file(f'https://models.silero.ai/models/tts/ru/{local_file}',
                                   local_file)  

model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)
sample_rate = 16000

with open(args.filecsv, encoding='utf-8') as r_file:
    file_reader = csv.reader(r_file, delimiter = ",")
    count = 0
    for row in file_reader:
        if count == 0:
            print(f'Файл содержит столбцы: {", ".join(row)}')
        else:
            print(f'Создание файла    {row[0]} - {row[1]}')
            audio_paths = model.save_wav(texts=row[1],
                audio_pathes=args.folder+row[0],
                sample_rate=sample_rate)
        count += 1
