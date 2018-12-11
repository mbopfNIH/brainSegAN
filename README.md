> NOTE: This code was originally written by Yuan Xue and Sharon Huang at Lehigh University. I have put this into a GitHub repository so I can maintain configuration control of the code. - Mike Bopf

# Semantic segmentation with Adversarial Training for brain tumor

## Dependencies:
	1. Pytorch available on http://pytorch.org/
	   TensorFlow available on Github: https://github.com/tensorflow/tensorflow. (We use tensorboard in TF to monitor the training process) 
	2. Python 2.7 or Python 3.5/3.6 
	3. In addition, please pip install the following packages:
		numpy
		scipy
		pillow
	If I miss anything here, you may encounter some errors and please let me know those errors.
	4. Using Virtual Environment or pip to install simpleITK

## Usage
	1. download dataset from https://sites.google.com/site/braintumorsegmentation/home/brats2015 (We trained our model on Brats 2015 training data)
  	
	2. Preprocess the dataset using prepare_data.py
	$ python prepare_data.py
	
	Note that you may need to change path in __init__ method in class Preprocess (line 47 in prepare_data.py). The path should contain HGG/ folder and LGG/ folder.
	Also, after run prepare_data.py, you may need to manually create a val/ folder under root folder and randomly move some training samples from train/ folder to val/ for 	validation.
	The number of samples in val/ is better be multiple of 5 so that you can always see 5 validation images in tensorboard.

	3. To train a model with the preprocessed dataset:
	$ CUDA_VISIBLE_DEVICES=X python train_adversarial.py --cuda --batchsize 15
	
	Note that X means the id of your GPU, for now we only support training with ONE GPU. If you have only one GPU, then X is 0.
	The number after --batchsize is the training batch size. I trained our model on a Titan X Pascal GPU with 12G memory. If you have access to a GPU with large memory such as 		12G, you can keep this number. Otherwise you may need to use a smaller batchsize. I suggest you to use GPU with at least 6G memory, then you can set the batch size to be 6 		or 7. If you encounter some errors such as "...out of memory...", then probably you need to use a GPU with larger memory or smaller batchsize.
	
	The default output folder is SegAN/, you can find tensorboard event file and trained models in that folder.
	
 

	4. The code is just for training for now, and I haven't cleaned the code, sorry for inconvenience.


## Results
	To monitor the training process, please run the command below under your Brats/ folder (the root folder for your codes and data)
	$ tensorboard --logdir='./' --port=6006  --reload_interval=5
	Then, open your browser and go to: http://0.0.0.0:6006, then you can monitor the training process via tensorboard.
	HOWEVER, I'm having some trouble using pytorch and tensorflow together, it worked pretty well but after I upgraded cuda, pytorch, tensorflow, etc. I always have "dlopen: cannot load any more object with static TLS" when I import torch and tensorflow together. It may work for you though.
	If you also encouter any problems with tensorboard, you can delete the code for tensorboard which is seperated by #============ TensorBoard logging ============#

	If you have any questions, please feel free to contact me, my email is yux715@lehigh.edu.
