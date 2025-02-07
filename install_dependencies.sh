#!/bin/bash
python.exe -m pip install --upgrade pip

pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt