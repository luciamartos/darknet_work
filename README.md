## Decoding YOLO

In order to use YOLO with previously existing classifiers I simply have to add the photo I need to analyse to a give folder and run the following command: <br />
./darknet detector test cfg/coco.data cfg/yolo.cfg yolo.weights data/dog.jpg

Here is an explanation of the different components in play.
As it can be seen this is running the executable file darknet and running the detector.c class and needs three pieces of information after that:
1. coco.data - this file is different from what was used in YOLOv1 it contains the following information (basically datapaths):

classes= 80<br /> 
train  = /home/pjreddie/data/coco/trainvalno5k.txt<br /> 
#valid  = coco_testdev<br /> 
valid = data/coco_val_5k.list<br /> 
names = data/coco.names<br /> 
backup = /home/pjreddie/backup/<br /> 
eval=coco<br /> 

2. yolo.cfg - this contains the specifications of the network that we want to create, the YOLO network looks like:

[net]<br /> 
batch=1<br /> 
subdivisions=1<br /> 
width=416<br /> 
height=416<br /> 
channels=3<br /> 
momentum=0.9<br /> 
decay=0.0005<br /> 
angle=0<br /> 
saturation = 1.5<br /> 
exposure = 1.5<br /> 
hue=.1<br /> 

learning_rate=0.001<br /> 
max_batches = 120000<br /> 
policy=steps<br /> 
steps=-1,100,80000,100000<br /> 
scales=.1,10,.1,.1<br /> 

[convolutional]<br /> 
batch_normalize=1<br /> 
filters=32<br /> 
size=3<br /> 
stride=1<br /> 
pad=1<br /> 
activation=leaky<br /> 

.....<br /> 

[convolutional]<br /> 
size=1<br /> 
stride=1<br /> 
pad=1<br /> 
filters=425<br /> 
activation=linear<br /> 

[region]<br /> 
anchors = 0.738768,0.874946,  2.42204,2.65704,  4.30971,7.04493,  10.246,4.59428,  12.6868,11.8741<br /> 
bias_match=1<br /> 
classes=2<br /> 
coords=4<br /> 
num=5<br /> 
softmax=1<br /> 
jitter=.2<br /> 
rescore=1<br /> 

object_scale=5<br /> 
noobject_scale=1<br /> 
class_scale=1<br /> 
coord_scale=1<br /> 

absolute=1<br />
thresh = .6<br />
random=0<br />

In order to use this network we need to adjust the number of classes we are using (filters = classes). Example for how filters work: <br />

The last convolutional layer has 10 filters because we have 10 classes. It outputs a 7 x 7 x 10 image. We just want 10 predictions total so we use an average pooling layer to take the average across the image for each channel. This will give us our 10 predictions. We use a softmax to convert the predictions into a probability distribution and a cost layer to calculate our error.<br />

3. yolo.weights - this is a file with the initial weights of the network 
4. data/dog.jpg - the image that you are trying to compare 


## Train your own classifier steps
1. You can re-design your net or use an existing one, this is the .cfg file. I was using the yolo.cfg net and changed the number of classifiers and filters using filters=(classes + 5)*5 = 35. Note this is highly inefficient if you are just training 1 classifier.
2. Create a .data file containing the number of classes you are training and the paths to the .txt files with lists of paths of training and validation data. Example:
      1 classes= 20
		  2 train  = <path-to-voc>/train.txt
		  3 valid  = <path-to-voc>2007_test.txt
  		4 names = data/voc.names
		  5 backup = backup
3. detector.c - change train and backup path
4. Create .txt files with the paths of the images being trained and validation data in darknet/data/labels - Labels are a .txt file for each image and they should have the format: <object-class> <x> <y> <width> <height> 
5. change the .names file - this contains a list of the names of the classes that you are going to be training/executing on. Example:
		aeroplane<br />
		bicycle<br />
		bird<br />
		boat<br />
		bottle<br />
		bus<br />
6. Edit the yolo.c file: change the paths of the trainning images and back up director.
7. Put training data in the path specified, requires image.jpg and image.txt with the annotations in the format <object-class> <x> <y> <width> <height>, where everything is in ratios
8. Download some pre trained weights to use - this can be a randomised file of the correct size.
9. Re-make darknet again (run “make” on terminal)
10. RUN: ./darknet detector train cfg/voc.data cfg/yolo-voc.cfg darknet19_448.conv.23  (clearly use the correct names for the files which you created!)<br />
./darknet detector train data/obj.data cfg/yolo-voc.cfg darknet19_448.conv.23  (clearly use the correct names for the files which you created!)<br />

## Sample code
Please run the following lines:<br />
git clone XXXXX<br />
cd darknet_work<br />
make<br />
RUN: ./darknet detector train data/obj.data cfg/yolo-voc.cfg darknet19_448.conv.23 <br />
Note!!! If it does not work it is because the specified paths are linked to my computer so you might need to edit files like obj.data to update the paths (this should be very simple)<br />

## Change the training data
Now you need to specify the images which you are going to train: <br />
1. data/obj: put all the images you wish to train on
2. for each of the images create a corresponding imgX.txt file which you specify where each of the features are found (structure explained above)
3. obj.data: you will need to update the training data path, apart from this, unless you want to change the number of classes being trained you should not have to change anything
4. obj.names: each line should have the name of a different classifier you are training, in this case we are only training two classifiers so we have two lines
5. train.txt: this needs to have a list of paths of each of the individual images which you are training

After this you have changes everything you need!! Please not that if you wanna change the number of classifiers being trained you will also need to change the net and the yolo.c file as explained above. <br />

Now run: ./darknet detector train data/obj.data cfg/yolo-voc.cfg darknet19_448.conv.23 <br />


## Things to note
* The training will run forever, you need to stop it when you consider that it has been enough time (you can look at the precision numbers and if they barely change then you can assume if is sufficient)
* The neural net being trained on is way too big for training two classifiers
* Its training on less than 10 images, so the classifier which we get is essentially useless until you put a big enough data amount

## Resources
Darknet: https://pjreddie.com/darknet/<br />
YOLO: https://pjreddie.com/darknet/yolo/<br />
AlexeyAB tutorial: https://github.com/AlexeyAB/darknet#custom-object-detection<br />
Help to debug: https://groups.google.com/forum/#!forum/darknet<br />
Training CIFAR:  https://pjreddie.com/darknet/train-cifar/ <br />
Coursera course on ML:  https://www.coursera.org/learn/machine-learning<br />
TensorFlow tutorials: https://www.tensorflow.org/tutorials/pdes<br />
Google tutorials: https://github.com/nlintz/TensorFlow-Tutorials<br />
Darkflow: https://github.com/thtrieu/darkflow<br />
PASCAL dataset: http://host.robots.ox.ac.uk/pascal/VOC/<br />
Guanghan on YOLO: http://guanghan.info/blog/en/my-works/train-yolo/<br />
Flickr32Dataset: http://www.multimedia-computing.de/flickrlogos/#download<br />
Other databases: http://host.robots.ox.ac.uk/pascal/VOC/databases.html#TUD<br />
Q&A: https://groups.google.com/forum/#!searchin/darknet/make_labels|sort:relevance/darknet/6urKIJExKps/I0qECfnlAgAJ<br />

