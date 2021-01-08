#! /bin/bash
sudo apt update
sudo apt install python3 python3-dev python3-venv
sudo apt install git
sudo apt-get install wget
sudo apt-get install unzip
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
pip --version
git --version
git clone https://github.com/AniruddhS24/FeedbackAnalysisSystem.git
cd FeedbackAnalysisSystem
pip install -r requirements.txt
echo -e "Now upload data files, and saved model files from local machine or cloud"

