touch datasets
cd datasets
wget -nc http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip -o tiny-imagenet-200.zip
cd ..
python tiny_imagenet.py