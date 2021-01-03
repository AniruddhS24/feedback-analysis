## Feedback Analysis

Steps to train:
Install Python and Git and stuff:

sudo apt update
sudo apt install python3 python3-dev python3-venv
sudo apt-get install wget
sudo apt-get install unzip
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
pip --version

Upload project folder to VM (excluding venv), may need to zip contents first

navigate to root directory FeedbackAnalysis

SKIP THESE 2 STEPS IF VENV NOT NEEDED:
Create a new venv: python3 -m venv myenv
activate venv with: .\venv\Scripts\activate   or source myenv/bin/activate

Now, do
pip install -r requirements.txt

Now just run python trainer.py