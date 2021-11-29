@echo off
color a
pip install -q torchaudio omegaconf
pip install -q logmmse
python demo_v1.py
pause