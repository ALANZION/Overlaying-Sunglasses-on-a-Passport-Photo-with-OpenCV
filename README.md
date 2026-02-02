# Overlaying Sunglasses on a Passport Photo with OpenCV
## PROGRAM
### Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
### Load the Face Image
faceImage = cv2.imread('pass-img.jpg')
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")
faceImage.shape
### resized_faceImage.shape
faceImage.shape
### Load the Sunglass image with Alpha channel
### (http://pluspng.com/sunglass-png-1104.html)
glassPNG = cv2.imread('sunglass-img.png',-1)
plt.imshow(glassPNG[:,:,::-1]);plt.title("glassPNG")
### Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG,(155,40))
print("image Dimension ={}".format(glassPNG.shape))
### Separate the Color and alpha channels
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]
### Display the images for clarity
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(glassBGR[:,:,::-1]);plt.title('Sunglass Color channels');
plt.subplot(122);plt.imshow(glassMask1,cmap='gray');plt.title('Sunglass Alpha channel');
### Make a copy
#faceWithGlassesNaive = resized_faceImage.copy()
faceWithGlassesNaive = faceImage.copy()

# Replace the eye region with the sunglass image
faceWithGlassesNaive[100:140,70:220]=glassBGR

plt.imshow(faceWithGlassesNaive[...,::-1])
### Make the dimensions of the mask same as the input image.
### Since Face Image is a 3-channel image, we create a 3 channel image for the mask
glassMask = cv2.merge((glassMask1,glassMask1,glassMask1))

### Make the values [0,1] since we are using arithmetic operations
glassMask = np.uint8(glassMask/255)

### Make a copy
faceWithGlassesArithmetic = faceImage.copy()

### Get the eye region from the face image
eyeROI= faceWithGlassesArithmetic[100:140,70:220]

### Use the mask to create the masked eye region
maskedEye = cv2.multiply(eyeROI,(1-  glassMask ))

### Use the mask to create the masked sunglass region
maskedGlass = cv2.multiply(glassBGR,glassMask)

# Combine the Sunglass in the Eye Region to get the augmented image
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)

### Display the intermediate results
plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(maskedEye[...,::-1]);plt.title("Masked Eye Region")
plt.subplot(132);plt.imshow(maskedGlass[...,::-1]);plt.title("Masked Sunglass Region")
plt.subplot(133);plt.imshow(eyeRoiFinal[...,::-1]);plt.title("Augmented Eye and Sunglass")
### Replace the eye ROI with the output from the previous section
faceWithGlassesArithmetic[100:140,70:220]=eyeRoiFinal

### Display the final result
plt.figure(figsize=[20,20]);
plt.subplot(121);plt.imshow(faceImage[:,:,::-1]); plt.title("Original Image");
plt.subplot(122);plt.imshow(faceWithGlassesArithmetic[:,:,::-1]);plt.title("With Sunglasses");
## OUTPUT
### FACE IMAGE:
<img width="326" height="396" alt="image" src="https://github.com/user-attachments/assets/9d51d9a5-8a46-41b0-8d29-576b27a97e58" />
### SUNGLASS:
<img width="540" height="261" alt="image" src="https://github.com/user-attachments/assets/99053a26-21ca-4c92-93fe-34e0b0b8ffc4" />

<img width="839" height="503" alt="image" src="https://github.com/user-attachments/assets/0d7fcc11-3e7d-4ff4-924b-94d21c674caa" />
