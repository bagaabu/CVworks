# CVworks
OCR
=====
This OCR work is for a package recognization project.  
The aim is identify the number from the package.  
The methods used are Opencv and Random Forest.  

There are two steps:  
1. Create the training set.  
2. Train the classifier.  
3. Then the classifier is ready to go.

It finds the contours (cv2.findContours()) from the image first, then transforms the counter to a 16*16.  
Then i set 3 features:  
1. the origanal image (worest)  
2. canny edge of the image (better)  
3. gradient of the image (little better than canny)  
Next transforms from 16*16 to 1*256, this is one sample.  
Then do this for all samples.  
The lable is a N*1 array.  
That's all for setting the training set.

Then i use sklearn to train the RandomForest with the samples and lables got before.
Firstly is still find the countours in the image.
Secondly, transform the contours into 16*16 then 1*256.
Then, do the canny and gradient(HOG).
After that, predict three times.
if all three results are same, that's fine, otherwise, pick the number which shows most.

That's all, this will works for images in white background and black numbers.the accurency is around 98%.(using testing set image)
The bad news is it can't identify it's a number or not, it will classify all patterns.(using cnn first, and get the probability of the patterns may solve this problems.)
