import math
import numpy as np
from skimage.util import view_as_blocks
import matplotlib.pyplot as plt
import cv2
from itertools import product
# from sklearn.decomposition import PCA
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
#
# pca = PCA(n_components = 2)
image5 = cv2.imread('img5.jpg')
image1 = cv2.imread('img1.jpg')
image2 = cv2.imread('img2.jpg')
image3 = cv2.imread('img3.jpg')
image4 = cv2.imread('img4.jpg')
image6 = cv2.imread('img6.jpg')
# # im=img1.jpg
# dict=[[image1],[image2],[image3],[image4],[image5],[image6]]
# # X_train = sc.fit_transform(dict)
# X_tra = pca.fit_transform(dict)
gamma = 0.32
#NORMALIZATION FUNCTION:converting cirular iris region to rectangular region
def unravel_iris(img, xp, yp, rp, xi, yi, ri, phase_width=300, iris_width=150):
    if img.ndim <= 2:
        pass
    elif img.ndim > 2:
        img = img[:, :, 0].copy()

    theta = np.linspace(0, 2 * np.pi, phase_width)# create a vector with phase values
    iris=np.full((iris_width, phase_width), 0.)  #create a 150x300 array with null elements
    
    #for each phase calculate the cartesian coordinates and the pixel value for these coordinates
    i=0
    while i<phase_width:
        #calculate the cartesian coordinates for the beginning and the end pixels
        x=int(xp + rp * math.cos(theta[i]))
        y=int(yp + rp * math.sin(theta[i]))
        x1=int(xi + ri * math.cos(theta[i]))
        y1=int(yi + ri * math.sin(theta[i]))
        # generate the cartesian coordinates of pixels between the beginning and end pixels
        xspace=np.linspace(x, x1, iris_width)
        yspace=np.linspace(y, y1, iris_width)
        #calculate the value for each pixel
        lis=[]
        for x, y in zip(xspace, yspace): #assign the cartesian coordinates
            if 0 <= int(x) < img.shape[1] and 0 <= int(y) < img.shape[0]:
                lis=lis+[255 - img[int(y), int(x)]]
            else:
                lis=lis+[0]
        iris[:, i]=lis
        i=i+1

    return iris


def iris_encode(img, dr=15, dtheta=15, alpha=0.4):


    # large the size of code to add the information from three frequency and from real and imaginary part
    code=np.full((view_as_blocks((img - img.mean()) / img.std(), (dr, dtheta)).shape[0] * 3,view_as_blocks((img - img.mean()) / img.std(), (dr, dtheta)).shape[1] * 2), 0.)
    s=5*67
    code_mask=np.full((view_as_blocks((img - img.mean()) / img.std(), (dr, dtheta)).shape[0] * 3, view_as_blocks((img - img.mean()) / img.std(), (dr, dtheta)).shape[1] * 2),0.)


    beta=1 / alpha
    patches= view_as_blocks((img - img.mean()) / img.std(), (dr, dtheta))#split  normalized image to blocks
    mask=view_as_blocks(np.logical_and(20 < img, img < 255), (dr, dtheta))#creating a mask to exclude non-iris pixels
    for i, row in enumerate(patches):
        for (j, p),(k,w) in product(enumerate(row),enumerate([8, 16, 32])):  #for k, w in enumerate([8, 16, 32]) is changing the frequency of wavelet
                #application of the 2D Gabor wavelets on the image
                # generate the parameters
                xx, yy=np.meshgrid(np.linspace(0, 1, p.shape[0]), np.linspace(-np.pi, np.pi, p.shape[1]))
                t=np.exp(-w * 1j * (0- yy)) * np.exp(-(xx - 0) ** 2 / alpha ** 2) * np.exp(-(-yy + 0) ** 2 / beta ** 2)

                wavelet= np.array([np.linspace(0, 1, p.shape[0]) for i in range(p.shape[1])]).T * p * np.real(
                   t.T),np.array([np.linspace(0, 1, p.shape[0]) for i in range(p.shape[1])]).T * p * np.imag(t.T)
                if True:
                    code[3 * i + k, 2 * j + 1]=np.sum(wavelet[1])  # calculate the imaginary part
                if True:
                    code[3 * i + k, 2 * j]=np.sum(wavelet[0])  # calculate the real part


                if mask[i, j].sum() > dr * dtheta * 3 / 4:
                    code_mask[3 * i + k, 2 * j]=code_mask[3 * i + k, 2 * j + 1]=1
                else:
                    code_mask[3 * i + k, 2 * j]=code_mask[3 * i + k, 2 * j + 1]=0




    #
    if True:
        code[code >= 0] = 1
    if True:
        code[code < 0] = 0
    return code, code_mask



def circle(image, x,y,r,p,a,b):
    cv2.circle(image, (x, y), r,p, a)
    cv2.circle(image, (x, y), b, p, b)


def show_details(image,image2):
    # implement the gamma correction function
    lookUpTable=np.empty((1, 256), np.uint8)
    for i in range(256):
        t=pow(i / 255.0, gamma)
        lookUpTable[0, i]=np.clip(t * 255.0, 0, 255)
    src=cv2.LUT(image, lookUpTable)

    img=cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 5) #convert image to  grayscale  and apply the Median Blur filter with a 5X5 kernel

    img1=cv2.medianBlur(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2GRAY), 5)  #convert image to  grayscale  and apply the Median Blur filter with a 5X5 kernel

    circles=cv2.HoughCircles(cv2.medianBlur(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2GRAY), 5), cv2.HOUGH_GRADIENT, 1, 20,
                             param1=60, param2=30, minRadius=1, maxRadius=40)

    if circles is None:
        x, y, r=0, 0, 0
    else: # verify if a circle is detected
        circles=np.uint16(np.around(circles))
        x, y, r=circles[0, 0][0], circles[0, 0][1], circles[0, 0][2]  # return x and y coordinates and the radius of the circle
    # Apply Hough transform to find potential boundaries of pupil
    circles=cv2.HoughCircles(cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 5), cv2.HOUGH_GRADIENT, 1, 20,
                             param1=60, param2=30, minRadius=20, maxRadius=100)
    if circles is None:
        x_iris, y_iris, r_iris=0, 0, 0
    else:
        circles=np.uint16(np.around(circles))
        x_iris, y_iris, r_iris=circles[0, 0][0], circles[0, 0][1], circles[0, 0][2]


    
    iris = unravel_iris(image, x, y, r, x_iris, y_iris, r_iris)
    p=(255, 0, 0)
    a=3
    b=2
    p1=(0, 255, 0)
    circle(image,x,y,r,p,a,b)
    circle(image, x_iris,y_iris, r_iris, p1, a, b)
    #showing segmented image
    plt.imshow(image,cmap=plt.cm.gray)
    plt.title('Segmentation process on image1')
    # I dont want the axis values that were displayed in the previous example.
    plt.axis("off")
    plt.show()
    
    #showing normalized image
    plt.imshow(iris, cmap=plt.cm.gray)
    plt.title('Normalization process on image1')
    # i dont want the axis values that were displayed in the previous example.
    plt.axis("off")
    plt.show()
    

    code, mask = iris_encode(iris)
    #showing iris code
    plt.imshow(code, cmap=plt.cm.gray)
    plt.title('Iris code of image1')
    plt.axis("off")
    plt.show()
    #showing mask code
    plt.imshow(mask, cmap=plt.cm.gray, interpolation='none')
    plt.title('Mask code of image1')
    plt.axis("off")
    plt.show()

    cv2.circle(image, (x, y), r, (255, 255, 0), 2)
    cv2.circle(image, (x_iris, y_iris), r_iris, (0, 255, 0), 2)
    # implement the gamma correction function
    lookUpTable=np.empty((1, 256), np.uint8)
    i=0
    while i<256:

        t=pow(i / 255.0, gamma)
        lookUpTable[0, i]=np.clip(t * 255.0, 0, 255)
        i=i+1
    src2=cv2.LUT(image2, lookUpTable)

    img2=cv2.medianBlur(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), 5)#apply the Median Blur filter with a 5X5 kernel

    img22=cv2.medianBlur(cv2.cvtColor(src2.copy(), cv2.COLOR_BGR2GRAY), 5)#apply the Median Blur filter with a 5X5 kernel

    circles=cv2.HoughCircles(cv2.medianBlur(cv2.cvtColor(src2.copy(), cv2.COLOR_BGR2GRAY), 5), cv2.HOUGH_GRADIENT, 1, 20,
                             param1=60, param2=30, minRadius=1, maxRadius=40)

    if circles is None:  # verify if a circle is detected
        x2, y2, r2=0, 0, 0
    else:
        circles=np.uint16(np.around(circles))
        x2, y2, r2=circles[0, 0][0], circles[0, 0][1], circles[0, 0][2]  # return x and y coordinates and the radius of the circle

    circles=cv2.HoughCircles(cv2.medianBlur(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), 5), cv2.HOUGH_GRADIENT, 1, 20,
                             param1=60, param2=30, minRadius=20, maxRadius=100)
    if circles is None:
        x2_iris, y2_iris, r2_iris=0, 0, 0
    else:
        circles=np.uint16(np.around(circles))
        x2_iris, y2_iris, r2_iris=circles[0, 0][0], circles[0, 0][1], circles[0, 0][2]

    
    iris2 = unravel_iris(image2, x2, y2, r2, x2_iris, y2_iris, r2_iris)
    p=(255, 0, 0)
    a=3
    b=2
    p1=(0, 255, 0)
    circle(image2,x2,y2,r,p,a,b)#draw circle on image
    circle(image2, x2_iris,y2_iris, r2_iris, p1, a, b)

    #showing segmented image2
    plt.imshow(image2, cmap=plt.cm.gray)
    plt.title('Segmentation process on image2')
    plt.axis("off")
    plt.show()

    

    code2, mask2 = iris_encode(iris2)
    #showing normalized image2
    plt.imshow(iris2, cmap=plt.cm.gray)
    plt.title('Normalization process on image2')
    plt.axis("off")
    plt.show()

    #showing iris coode of image 2
    plt.imshow(code2, cmap=plt.cm.gray, interpolation='none')
    plt.title('Iris code of image2')
    plt.axis("off")
    plt.show()
    #showing mask code of image2
    plt.imshow(mask2, cmap=plt.cm.gray, interpolation='none')
    plt.title('Mask code of image2')
    plt.axis("off")
    plt.show()

    cv2.circle(image2, (x2, y2), r2, (255, 255, 0), 2)
    cv2.circle(image2, (x2_iris, y2_iris), r2_iris, (0, 255, 0), 2)    
    
    

    
#image = cv2.imread('image_102.png')
image = cv2.imread('img5.jpg')
image2 = cv2.imread('img1.jpg')

show_details(image,image2)
#start encoding images
i=0
# implement the gamma correction function
lookUpTable=np.empty((1, 256), np.uint8)
while i<256:
    t=pow(i / 255.0, gamma)
    lookUpTable[0, i]=np.clip(t * 255.0, 0, 255)
    i=i+1

src=cv2.LUT(image, lookUpTable)

img= cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) , 5) #convert image to  grayscale and apply the Median Blur filter with a 5X5 kernel

img1= cv2.medianBlur(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2GRAY), 5)  #convert image to  grayscale and apply the Median Blur filter with a 5X5 kernel

circles=cv2.HoughCircles(cv2.medianBlur(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2GRAY), 5), cv2.HOUGH_GRADIENT, 1, 20,
                         param1=60, param2=30, minRadius=1, maxRadius=40)

if circles is None:  # verify if a circle is detected
    x, y, r=0, 0, 0
else:
    circles=np.uint16(np.around(circles))
    x, y, r=circles[0, 0][0], circles[0, 0][1], circles[0, 0][
        2]  # return x and y coordinates and the radius of the circle

circles=cv2.HoughCircles(cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 5), cv2.HOUGH_GRADIENT, 1, 20,
                             param1=60, param2=30, minRadius=20, maxRadius=100)
if circles is None:
    x_iris, y_iris, r_iris=0, 0, 0
else:
    circles=np.uint16(np.around(circles))
    x_iris, y_iris, r_iris=circles[0, 0][0], circles[0, 0][1], circles[0, 0][2]



iris=unravel_iris(image, x, y, r, x_iris, y_iris, r_iris)
code, mask=iris_encode(iris)#finally encoding iris of image1
#start encoding second image
i=0
# implement the gamma correction function
while i<256:
    p=pow(i / 255.0, gamma)
    lookUpTable[0, i]=np.clip(p * 255.0, 0, 255)
    i=i+1
src=cv2.LUT(image2, lookUpTable)

circles=cv2.HoughCircles(cv2.medianBlur(cv2.cvtColor(src.copy(), cv2.COLOR_BGR2GRAY), 5), cv2.HOUGH_GRADIENT, 1, 20,
                         param1=60, param2=30, minRadius=1, maxRadius=40)

if circles is  None:  # verify if a circle is detected
    x, y, r=0, 0, 0
else:
    circles=np.uint16(np.around(circles))
    x, y, r= circle[0, 0][0], circles[0, 0][1], circles[0, 0][2]  # return x and y coordinates and the radius of the circle


circles=cv2.HoughCircles(cv2.medianBlur(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), 5), cv2.HOUGH_GRADIENT, 1, 20,
                         param1=60, param2=30, minRadius=20, maxRadius=100)
if circles is  None:
    x_iris, y_iris, r_iris=0, 0, 0
else:
    circles=np.uint16(np.around(circles))
    x_iris, y_iris, r_iris= circles[0, 0][0], circles[0, 0][1], circles[0, 0][2]


code2, mask2=iris_encode(unravel_iris(image2, x, y, r, x_iris, y_iris, r_iris)) #finally encoding the iris of image2
#calculating hamming distance and compairing with threshold 0.38
if  (np.sum(np.remainder(code + code2, 2) * mask * mask2) / np.sum(mask * mask2)) >0.38:

    print("No match found")
    print("Difference: " + str((np.sum(np.remainder(code + code2, 2) * mask * mask2) / np.sum(mask * mask2))))
else:
    print("Iris Matched")
    print("Difference: "+str((np.sum(np.remainder(code + code2, 2) * mask * mask2) / np.sum(mask * mask2))))



