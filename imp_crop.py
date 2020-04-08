import torch
from IPython.display import clear_output

model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
model.eval()

clear_output()

import urllib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive
from PIL import Image
from torchvision import transforms
%matplotlib inline

mkdir work

foreground1 = 'https://live.staticflickr.com/1399/1118093174_8b723e1ee5_o.jpg' #@param {type:"string"}
#foreground2 = 'https://unsplash.com/photos/dGMcpbzcq1I.jpg' #@param {type:"string"}
BACKGROUND = 'https://live.staticflickr.com/7860/46618564664_be235e82e8_b.jpg' #@param {type:"string"}

url, filename = (BACKGROUND, "work/background.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

background = Image.open(filename).convert("RGB")
background.save('work/background.jpg')

url, filename = (foreground1, "work/foreground1.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

foreground1 = Image.open(filename).convert("RGB")
foreground1.save('work/foreground1.jpg')

url, filename = (foreground2, "work/foreground2.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

foreground2 = Image.open(filename).convert("RGB")
foreground2.save('work/foreground2.jpg')

fig = plt.figure(figsize=(20, 20))
ax1 = fig.add_subplot(121)
ax1.imshow(background)
ax2 = fig.add_subplot(122)
ax2.imshow(foreground1)
ax3 = fig.add_subplot(122)
ax3.imshow(foreground2)
ax1.title.set_text('Background Image')
ax2.title.set_text('foreground1 Image')
ax3.title.set_text('foreground2 Image')

###Function to craete the mask of foreground image

def maskgen(foreground,model):
	preprocess = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
	input_tensor = preprocess(Image.open(foreground).convert("RGB"))
	input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
	
	with torch.no_grad():
		output = model(input_batch)['out'][0]
	output_predictions = output.argmax(0)
	
	# create a color pallette, selecting a color for each class
	palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
	colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
	colors = (colors % 255).numpy().astype("uint8")
	
	# plot the semantic segmentation predictions of 21 classes in each color
	r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(Image.open(foreground).convert("RGB").size)
	r.putpalette(colors)
	bw = r.convert('L')
	bw.save(foreground.split('.')[0]+'_mask.png')
	
	# crop out object
	src1 = cv2.imread(foreground)
	src1_mask = cv2.imread(foreground.split('.')[0]+'_mask.png')
	ret,thresh1  = cv2.threshold(src1_mask,30,255,cv2.THRESH_BINARY)
	src1 = cv2.cvtColor(src1,cv2.COLOR_RGB2BGR)
	src1_mask = cv2.cvtColor(thresh1,cv2.COLOR_RGB2BGR)

	mask_out=cv2.subtract(src1_mask,src1)
	mask_out=cv2.subtract(src1_mask,mask_out)
	cv2.imwrite(foreground,cv2.cvtColor(mask_out, cv2.COLOR_RGB2BGR))
	
	# create mask
	src = cv2.imread(foreground.split('.')[0]+'_mask.png', 1)
	tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	thr,alpha = cv2.threshold(tmp,30,255,cv2.THRESH_BINARY)
	b, g, r = cv2.split(src)
	rgba = [b,g,r, alpha]
	dst = cv2.merge(rgba,4)
	cv2.imwrite(foreground.split('.')[0]+'_mask.png', dst)
	
### Calling the function ###
maskgen('work/foreground1.jpg',model)


### Defining the image size
#@title Image Parameters
#@markdown ### Background Image Size:
BACKGROUND_HEIGHT = 512 #@param {type:"integer"}
BACKGROUND_WIDTH = 1024 #@param {type:"integer"}
#@markdown ### Foreground Image Size:
FOREGROUND_HEIGHT = 512 #@param {type:"integer"}
FOREGROUND_WIDTH = 512 #@param {type:"integer"}

### Funaction for collaging the images

def img_clg(x,y,background,foreground):
	fig = plt.figure(figsize=(20, 20))
	ax1 = fig.add_subplot(111)
	background = Image.open(background)
	background = background.resize((BACKGROUND_WIDTH,BACKGROUND_HEIGHT))
	mouse = Image.open(foreground)
	mouse = mouse.resize((FOREGROUND_WIDTH,FOREGROUND_HEIGHT))
	mouse_mask = Image.open(foreground.split('.')[0]+'_mask.png')
	mouse_mask = mouse_mask.resize((FOREGROUND_WIDTH,FOREGROUND_HEIGHT))

	background.paste(mouse, (x,y), mouse_mask)
	background.save('work/background.jpg')
	ax1.set_title('Output Image')
	plt.imshow(background)
	
h, w = background.size
print('Use these x,y sliders to adjust Foreground Image Placement\n')

interactive_plot = interactive(img_clg, x=(-h, h, 10), y=(-w, w, 10),background='work/background.jpg',foreground='work/foreground.jpg')
output = interactive_plot.children[-1]
interactive_plot
	
####################################################################################