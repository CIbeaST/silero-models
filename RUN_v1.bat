@echo off
color a
pip install --upgrade pip
pip install torch -f https://download.pytorch.org/whl/torch_stable.html
pip install -q torchaudio omegaconf
pip install -q logmmse
python demo_v1.py
pause