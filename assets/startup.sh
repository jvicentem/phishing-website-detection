#!/bin/bash
yum update -y
sudo yum install python35 -y
sudo yum install python35-devel -y
curl -O https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py --user
python3 -m pip install pandas
python3 -m pip install numpy
python3 -m pip install requests
python3 -m pip install tabulate
python3 -m pip install six
python3 -m pip install future
python3 -m pip install pysparkling
python3 -m pip install h2o
python3 -m pip install colorama requests tabulate future --upgrade
python3 -m pip install jupyter
python3 -m pip install scipy
python3 -m pip install scikit-learn
python3 -m pip install matplotlib
