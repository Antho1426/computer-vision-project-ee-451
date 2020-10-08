# ee-451-special-project
Special project of the course "Image Analysis and Pattern Recognition"





# Method

1. Getting Shapes Location
Use the first frame to detect each the location of each shape (digits + signs) (set box around the shapes of interest)
	- Function to set the location of the shapes

2. Tracking:
Track the location of the robot
	- Function to track the location of the robot
		A) Using the algorithm of Basics of Mobile Robotics of last semester?

3.1 Comparison with a box:
Saying if we are in a zone of interest and says which kind of classifier we have to use

3.2 Classification:
As soon as the robot lands on an shape, we classify it
	- Function to classify the operators
		A) Classify + from * from - with Fourier Descriptors and 	-, = and ÷ since 1, 2 or 3 regions
	- Function to classify the digits




# Interesting Key points

/!\ Robust to orientation of the digits and shapes
A) —> train the neural network on the rotated digits (does a data loader that rotates the digits exist?)
Or
B) Use the axes of inertia to find the orientation of the shapes + consider the confidence with the flipped version

/!\ Identifier une forme —> chercher dans une durée alentour

/!\ Digits and Operators alternate! We have to send the correct classifier accordingly (alternating between the classifier of the operators and the classifier of the digits) + the first shape always a digit and the last is always a =




## 2020-05-13 - Job repartition n°1

Antho: robot tracking
Coco: Operators classification
Sylvain: Set boxes around zone of interest + digits classification




