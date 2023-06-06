conda create -n mindsdb
conda activate mindsdb
pip3 install --upgrade pip setuptools wheel
pip3 install mindsdb
python -m mindsdb --api=http,mongodb,mysql
