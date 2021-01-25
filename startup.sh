#! /bin/bash
sudo apt install python3 python3-dev python3-venv
sudo apt-get install wget
sudo apt-get install unzip
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
pip --version
git --version
pip install -r requirements.txt
echo -e "Now upload data files, and saved model files from local machine or cloud"

