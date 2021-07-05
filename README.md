# UVT
Ultrasound Video Transformers (UVT) for Cardiac Ejection Fraction Estimation. Code used for https://arxiv.org/abs/2107.00977

# Before using this repo
You will need to request access to the EchoNet dataset by completing the form on this page: https://echonet.github.io/dynamic/index.html#dataset
Once you have access to the data, download it and write the path of the "EchoNet-Dynamic" folder in the <code>dataset_path</code> variable in main.py.

# Train a network
Experiments can be launched from the [main.py](https://github.com/HReynaud/UVT/blob/main/main.py) file. Set the parameters directly in the code file and run the file to train the network. An example is ready to launch when running [main.py](https://github.com/HReynaud/UVT/blob/main/main.py).

# Test a network 
As for training, the test function is called from the [main.py](https://github.com/HReynaud/UVT/blob/main/main.py) file. An example is ready to launch when running [main.py](https://github.com/HReynaud/UVT/blob/main/main.py). To download the weights of the networks used in the paper, use the [download_weights.sh](https://github.com/HReynaud/UVT/blob/main/download_weights.sh) script. The network parameters for these weights are:

<table>
<tr>
    <th>Parameter</th>
    <th>Value</th>
  </tr>
  <tr>
    <td><code>latent_dim</code></td>
    <td>1024</td>
  </tr>
  <tr>
    <td><code>num_hidden_layers</code></td>
    <td>16</td>
  </tr>
  <tr>
    <td><code>intermediate_size</code></td>
    <td>8192</td>
  </tr>
  <tr>
    <td><code>use_full_videos</code></td>
    <td>True</td>
  </tr>
  <tr>
    <td><code>SDmode</code><sup>1</sup></td>
    <td>'reg' or 'cla'</td>
  </tr>
  <tr>
    <td><code>model_path</code><sup>1</sup></td>
    <td>./output/UVT_[R/M]_[REG/CLA]</td>
  </tr>
</table>
<sup>1</sup>Adapt these to the weight file in use
<br/>
<br/>

# Results
The network predicts the position of the ES and ED frames in a video of arbitrary length as well as the Left Ventricle Ejection Fraction.
![alt results](https://github.com/HReynaud/UVT/blob/main/images/example.png)

# Disclaimer
The code in ResNetAE.py is taken from the ResNetAE repo (https://github.com/farrell236/ResNetAE) and pruned to the minimum.
The training code is inspired by the echonet-dynamic repo (https://github.com/echonet/dynamic).

