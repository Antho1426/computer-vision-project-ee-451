import skimage
import numpy as np
from skimage.filters import threshold_multiotsu
from skimage.color import rgb2gray
import skimage.measure
import skimage.morphology
import skimage.segmentation
import skimage.transform



def SymbolsExtractor(image):
    """
    Highlight the shapes on the image via binarization.
    Input: -image : First image of the video
    Output: -regions : The binarized image with shapes
    """
    ImageGS=skimage.color.rgb2gray(image)
    nb_bins=256
    Hist=np.histogram(ImageGS, nb_bins)[1]
    X,Y=ImageGS.shape
    new_image=np.full(ImageGS.shape, np.max(Hist))
    upperbound=Hist[int(np.floor(nb_bins*0.75))]  #0.75 max
    for i in range(0, X):
        for j in range(0, Y):
            if ImageGS[i,j]<upperbound:
                new_image[i,j]=ImageGS[i,j]

    thresholds = threshold_multiotsu(new_image,2 )
    regions = np.digitize(new_image, bins=thresholds)

    #fig,ax = plt.subplots(figsize=(18, 6))
    #ax.imshow(regions, cmap='gray')
    #fig.show()
    return regions


def SignID(SymbolImg):
    """
    Find the corresponding operator on the given image.
    Input: -SymbolImg : Image containing an operator to be classified
    Output: -Sign : The computed operator
    """
    ClosedSymbolImg=skimage.morphology.binary_opening(SymbolImg)
    contour = skimage.measure.find_contours(ClosedSymbolImg, 0.5)
    Sign = ' %' # unknown is "modulo"
    if len(contour) == 2:
        Sign = ' ='
    if len(contour) == 3:
        Sign = ' /'
    if len(contour) == 1:
        EigVal = skimage.measure.inertia_tensor_eigvals(ClosedSymbolImg, mu=None, T=None)
        if EigVal[0] >= (2.8 * EigVal[1]):   #########10
            Sign = ' -'
        else:
            complexContour = np.matmul([1, 1j], np.asarray(contour[0]).T)
            FFTpath = np.fft.fft(complexContour)
            Amplitude = np.abs(FFTpath)
            AmplitudeRel = Amplitude / Amplitude[1]

            if AmplitudeRel[5]>=AmplitudeRel[7]:
                Sign = ' +'
            else:
                Sign = ' *'
    return (Sign)


def Symbol_finding(boxes, frame1):
    """
    Find the corresponding operator for all the shapes in boxes
    Input: -boxes : Dictionary of the shapes
          -frame1 : First image of the video
    Output: -boxes : Upgraded boxes dictionary with Operator
    """
    regions=SymbolsExtractor(frame1)
    for key in boxes.keys() :#[('Shape', 5), ('Shape', 7), ('Shape', 14), ('Shape', 19)]:  # boxes.keys() :#
        box = boxes[key]
        Area = regions[box[0][0]:box[1][0], box[0][1]:box[1][1]]#, :]
        Operator=SignID(Area)
        #print(Operator)
        boxes[key].append(Operator)
    return boxes


#SymbolFrame=SymbolsExtractor(frame1)
def SingleSymbol_finding(boxes, SymbolFrame, key):
    """
    Find the corresponding operator for the given key.
    Input: -boxes : Dictionary of the shapes
           -SymbolFrame : Image treated for the operator classification
           -key : The key corresponding to the shape to analyse
    Output: -boxes : Upgraded boxes dictionary with Operator
            -Operator : The computed operator
    """
    box = boxes[key]
    Area = SymbolFrame[box[0][0]:box[1][0], box[0][1]:box[1][1]]#, :]
    Operator=SignID(Area)
    print(Operator)
    boxes[key].append(Operator)
    return boxes, Operator

"""
#To compute the symbol for ALL boxes (giving wrong info for digits):
import Symbol_class
boxes=Symbol_class.Symbol_finding(boxes, frame1) #
#Operator is append at the end of boxes list 

##########  INTERESTING HERE: ############
#To compute the symbol for a GIVEN box:
#@beginning of code, put :
import Symbol_class
SymbolFrame=Symbol_class.SymbolsExtractor(frame1)

#When the symbol computation is required for a given key, put:
boxes, Operator=Symbol_class.SingleSymbol_finding(boxes, SymbolFrame, key) 
#Operator is the computed symbol, boxes has the symbol append at the end (for next use)

"""

