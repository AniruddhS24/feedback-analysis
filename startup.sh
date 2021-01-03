#! /bin/bash
sudo apt update
sudo apt install python3 python3-dev python3-venv
sudo apt-get install wget
sudo apt-get install unzip
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
pip --version
