American Sign Language (ASL) Alphabet Recognition
	
 	A deep learning project that recognizes the ASL alphabet from images using PyTorch.

	This model was also adapted for a project to train a drone to respond to hand signals


Structure 

ASL/ 
	
 	|-Archive-train_images-test_images
	|-train.py
	|-infer.py
	|-asl-model.pth
	|-venv/ - (optional virtual environment) 



 Model obtained very high accuracy of around 99%, 
   For best results crop image to only include hand 
   Along with keeping good lighting, and simple backgrounds 

To Use:
   
   	Using Infer.py, you can predict any images you'd like by uploading them and changing 
   	the path of the images 

Requirements: 
  
  	Python 3.10 
  	Pytorch 
  	torchVision
  	PIL

Dataset
		
  	Publicly available ASL alphabet dataset: Kaggle ASL Alphabet
	•	Images are organized by letter (A–Z).
	•	Make sure to keep training and test sets separate to avoid data leakage.
 	https://www.kaggle.com/datasets/grassknoted/asl-alphabet
