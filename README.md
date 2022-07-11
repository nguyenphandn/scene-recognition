# Scene Recognition

## Install requirements:

### Install Python3.6:
Deadsnakes is a PPA with newer releases than the default Ubuntu repositories. Add the PPA by entering the following:
```
sudo add-apt-repository ppa:deadsnakes/ppa
```
To update local repositories, use the command:
```
sudo apt update
```
Install Python3.6:
```
sudo apt install python3.6
```
Check if Python3.6 is installed:
```
ls -ls /usr/bin/python*
```
### Create virtual environment and activate it:
```
virtualenv -p python3.6 venv
source venv/bin/activate
```
### Install old version of PyTorch:
Use the below command or copy from [here](torch_install.txt):
```
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org torch===0.3.1 torchvision===0.2.0 -f https://download.pytorch.org/whl/torch_stable.html
```
### Install other requirements:
```
pip install -r requirements.txt
```

## Run model:
Use this command to predict scene categories, indoor/outdoor type, [scene attributes](https://cs.brown.edu/~gen/sunattributes.html), and the [class activation map](http://cnnlocalization.csail.mit.edu/) together from PlacesCNN:
```
python run_placesCNN.py --source path/to/image.jpg --save-txt --save-grad
```
The result is as below:
```
RESULT ON /photo-location/data/google_landmark_data/images/example.jpg
--TYPE OF ENVIRONMENT: outdoor
--SCENE CATEGORIES:
0.160 -> park
0.093 -> forest/broadleaf
0.090 -> lawn
0.071 -> orchard
0.061 -> picnic_area
--SCENE ATTRIBUTES:
trees, open area, foliage, natural light, vegetation, leaves, grass, no horizon, natural
Results saved to runs/detect/exp
1 labels saved to runs/detect/exp/labels
Done. (0.441s)
```
<img src="./example.jpg" height="200">

### Reference
Link: [Places2 Database](http://places2.csail.mit.edu), [Places1 Database](http://places.csail.mit.edu)

Please cite the following [IEEE Transaction on Pattern Analysis and Machine Intelligence paper](http://places2.csail.mit.edu/PAMI_places.pdf) if you use the data or pre-trained CNN models.

```
 @article{zhou2017places,
   title={Places: A 10 million Image Database for Scene Recognition},
   author={Zhou, Bolei and Lapedriza, Agata and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
   year={2017},
   publisher={IEEE}
 }
```

### Acknowledgements and License

Places dataset development has been partly supported by the National Science Foundation CISE directorate (#1016862), the McGovern Institute Neurotechnology Program (MINT), ONR MURI N000141010933, MIT Big Data Initiative at CSAIL, and Google, Xerox, Amazon and NVIDIA. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation and other funding agencies. 

The pretrained places-CNN models can be used under the Creative Common License (Attribution CC BY). Please give appropriate credit, such as providing a link to our paper or to the [Places Project Page](http://places2.csail.mit.edu). The copyright of all the images belongs to the image owners.
