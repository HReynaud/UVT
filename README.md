# UVT
Ultrasound Video Transformers (UVT) for Cardiac Ejection Fraction Estimation. Code used for [add arXiv link]

# Before using this repo
You will need to request access to the EchoNet dataset by completing the form on this page: https://echonet.github.io/dynamic/index.html#dataset
Once you have access to the data, download it and give the location of the "EchoNet-Dynamic" folder as an argument to the train function in main.py

# Train a network
Experiments can be launched from the main.py file. Set the parameters directly in the code file and run the file to train the network. An example is ready to launch when running main.py.

# Test a network 
As for training, the test function is called from the main.py file. An example is ready to launch when running main.py.

# Results
The network predicts the position of the ES and ED frames in a video of arbitrary length as well as the Left Ventricle Ejection Fraction.
![alt results](https://github.com/HReynaud/UVT/blob/main/images/example.png)

# Disclaimer
The code in ResNetAE.py is taken from the ResNetAE repo (https://github.com/farrell236/ResNetAE) and pruned to the minimum.
The training code is inspired by the echonet-dynamic repo (https://github.com/echonet/dynamic).

