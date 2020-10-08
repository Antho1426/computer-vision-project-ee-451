from utils import region_growing

import numpy as np
import matplotlib.pyplot as plt

import cv2

import skimage
from skimage.filters import threshold_multiotsu, median


"""
The final function to use is 'get_ROI()' an gives a dictionnary with the boxes and their informations.
The keys of the dictionnary are in the form : 
"""

def get_forms(image) :
    """
    Function that highlight the regions of interest.
    
    Input : - image --> image to labelize
    
    Outputs : - shapes --> matrix with binarized image
              - forms --> dictionnary with forms info with 'key' : [#pix, form_number, form region]
    """
    
    # Dictionnary to stock information about the different shapes
    forms = dict()

    # Conversion to grayscale
    gray = skimage.color.rgb2gray(image)

    # Otsu to find the shapes
    thresholds = threshold_multiotsu(gray, classes=2)
    thresh_background = thresholds[0] * 1.38
    
                                ##### Shapes detection (cutting the background) #####
        
    shapes = np.zeros(gray.shape)
    for i in range(gray.shape[0]) :
        for j in  range(gray.shape[1]) :
            if gray[i][j] < thresh_background :
                shapes[i][j] = 255
    
    classes = np.zeros(shapes.shape)
    k = 0
    for i in range(shapes.shape[0]) :
        for j in range(shapes.shape[1]) :
            if shapes[i][j]==255 and classes[i][j] == 0 :    # When a new shape is encountered
                k += 1
                region = region_growing(shapes,[i,j],10)
                pix = 0
                color = 0
                for r in region :      # Iterations into a shape
                    pix += 1     # Counter for the number of pixel inside a shape
                    color += gray[r[0]][r[1]]
                    classes[r[0]][r[1]] = k     # Separation of the different form by labelling
                
                forms['Shape',k] =[pix, k, region]
                
                
                # Cutting of the regions with more then 500 pixels
                if forms['Shape',k][0] > 500 or forms['Shape',k][0] <= 3 :
                    w = np.where(classes==forms['Shape',k][1])
                    shapes[w[0],w[1]] = 0
                    del forms['Shape',k]
    
    return shapes, forms


def get_boxes(forms) :
    """
    Function that gives the boxes around each region of interest.
    
    Input : - forms --> dictionnary with forms info with 'key' : [#pix, form_number, form region]
    
    Outputs : - boxes --> dictionnary with bounding boxes corresponding to the forms 
                          with 'key' : [up_angle, bot_angle, center]

    """
    boxes = dict()
    for key in forms.keys() :
        region = forms[key][2]     # region where the form is localized
        
        up_angle = np.min(region, axis=0)     # left up angle of the bbox (inverse up and bot for plots)
        bot_angle = np.max(region, axis=0)     # right bottom angle of the bbox
        
        center = np.array([int((up_angle[0]+bot_angle[0])/2), int((up_angle[1]+bot_angle[1])/2)])     # Center of the bbox
        
        boxes[key] = np.array([up_angle, bot_angle, center])

    return boxes


def merge_boxes(box1, box2) :
    """
    Function that merge 2 boxes in one.
    
    Input : - box1 --> array with the infos of the 1st box : [up_angle, bot_angle, center]
            - box2 --> array with the infos of the 2nd box : [up_angle, bot_angle, center]
    
    Outputs : - new_boxes --> new box merging the 2 infos : [up_angle, bot_angle, center]

    """
    
    item_up = np.array([box1[0],box2[0]])
    item_bot = np.array([box1[1],box2[1]])
    
    up_angle = np.min(item_up, axis=0)
    bot_angle = np.max(item_bot, axis=0)
    center = np.array([int((up_angle[0]+bot_angle[0])/2), int((up_angle[1]+bot_angle[1])/2)])
    
    new_box = np.array([up_angle, bot_angle, center])
    
    return new_box
                    
        
def clean_boxes(boxes, dist_merge=15, thresh_ratio=6) :
    """
    Function that select the boxes to merge, delete the merged boxes from the
    dictionnary and delete vertically long boxes.
    
    Input : - boxes --> dictionnary with bounding boxes corresponding to the forms 
                        with 'key' : [up_angle, bot_angle, center]
            - dist_merge --> distance treshold to merge 2 boxes
    
    Outputs : - boxes --> new dictionnary with bounding boxes corresponding to 
                          keeping the bboxes of interest with 'key' : [up_angle, bot_angle, center]

    """
    
    i = -1; b = 0;
    while i != (len(boxes.keys())-1):
        
        for i, key1 in enumerate(boxes.keys()):
            for j, key2 in enumerate(boxes.keys()) :
                if j<=i :
                    continue
                
                # Extraction of the angle of interest of the distances computations
                tup1 = np.array([boxes[key1][0], boxes[key2][1]])
                tup2 = np.array([boxes[key1][1], boxes[key2][0]])
                tup3 = np.array([boxes[key1][0], boxes[key2][0]])
                tup4 = np.array([boxes[key1][1], boxes[key2][1]])
                
                # Distance computations for treshold comparison 
                dist1 = ((tup1[0][0]-tup1[1][0])**2+(tup1[0][1]-tup1[1][1])**2)**(1/2)     # x-axis
                dist2 = ((tup2[0][0]-tup2[1][0])**2+(tup2[0][1]-tup2[1][1])**2)**(1/2)     # y-axis
                dist3 = ((tup3[0][0]-tup3[1][0])**2+(tup3[0][1]-tup3[1][1])**2)**(1/2)     # up
                dist4 = ((tup4[0][0]-tup4[1][0])**2+(tup4[0][1]-tup4[1][1])**2)**(1/2)     # down
                
                # Distances treshold for to select the boxes to merge
                if dist1 < dist_merge or dist2 < dist_merge or dist3 < dist_merge or dist4 < dist_merge:
                    b=1
                    break
            
            if b==1 :
                #b = 0
                i = 0
                break
        
        # Update of the dictionnary by removing one of the merge boxes and replacing the other
        if b==1 :
            boxes[key1] = merge_boxes(boxes[key1], boxes[key2])
            del boxes[key2]
            b=0
        
    # Remove the long boxes
    k = list()
    for key in boxes.keys() :
        large = abs(boxes[key][1][0]-boxes[key][0][0])
        long = abs(boxes[key][1][1]-boxes[key][0][1])
        ratio = large/long     # ration x_size/y_size
        if ratio > thresh_ratio or ratio < 1/thresh_ratio or large > 100 or long > 100:
            k.append(key)
    
    for l in k :
        del boxes[l]

    return boxes


def augment_boxes(image, boxes, pix_aug=5, diag_thresh_down=5, diag_thresh_up=150) :
    """
    Function that augment the big enough boxes sizes by "pix" pixels
    
    Inputs :- image --> Initial image to analyse 
            - boxes --> dictionnary with bounding boxes corresponding to the forms 
                        with 'key' : [up_angle, bot_angle, center]
            - pix --> number of pixel for augmenting the boxes
            - diag_thresh_down --> Treshold to remove the little boxes
            - diag_thresh_up --> Treshold to remove too big boxes
    
    Output : - boxes --> new dictionnary with bounding boxes corresponding to 
                          keeping the bboxes of interest with 'key' : [up_angle, bot_angle, center]
                          
    """
    
    k = list()
    for key in boxes.keys() :
        diag = ((boxes[key][0][0]-boxes[key][1][0])**2+(boxes[key][0][1]-boxes[key][1][1])**2)**(1/2)
        if diag < diag_thresh_down or diag > diag_thresh_up:
            k.append(key)
            continue
            
        boxes[key][0] = boxes[key][0] - pix_aug
        boxes[key][1] = boxes[key][1] + pix_aug
        
        # Conditions if the augmentation put the box out of the initial image
        
        if boxes[key][0][0] < 0 :
            boxes[key][0][0] = 0
        
        if boxes[key][0][1] < 0 :
            boxes[key][0][1] = 0
        
        if boxes[key][1][0] > image.shape[0] :
            boxes[key][1][0] = image.shape[0]
        
        if boxes[key][0][1] > image.shape[1] :
            boxes[key][0][1] = image.shape[1]
    
    for l in k :
        del boxes[l]
    
    return boxes


def remove_robot_boxes(boxes, loc, thresh_robot=50) :
    """
    Function that remove the boxes around the robot position
    
    Input : - boxes --> dictionnary with bounding boxes corresponding to the forms 
                        with 'key' : [up_angle, bot_angle, center]
            - loc --> robot localization : [vertical axis, horizontal axis]
            - thresh_robot --> Treshold to remove the little boxes
    
    Outputs : - boxes --> new dictionnary with bounding boxes corresponding to 
                          keeping the bboxes of interest with 'key' : [up_angle, bot_angle, center]

    """
    
    k = list()
    for key in boxes.keys() :
        dist_center = ((boxes[key][2][0]-loc[0])**2+(boxes[key][2][1]-loc[1])**2)**(1/2)
        if dist_center < thresh_robot :
            k.append(key)
            
    for l in k :
        del boxes[l]
    
    return boxes


def binarisation(image, boxes, shape, pix_thresh=55) :
    """
    Function that binarizes the shapes inside the boxes (0 or 255)
    
    Input : - boxes --> dictionnary with bounding boxes corresponding to the forms 
                        with 'key' : [up_angle, bot_angle, center]
            - image --> initial image ( entire frame of the video)
            - shape --> Binarized image with the region of interest for the entire frame
            - pix_thresh --> threshold for the number of pixels contained inside one box (if)
                        below the threshold the box is removed
    
    Outputs : - boxes --> new dictionnary with bounding boxes corresponding to 
                          keeping the bboxes of interest with of interest

    """
    
    k = list()
    for key in boxes.keys() :
        up_angle = boxes[key][0]
        bot_angle = boxes[key][1]
        
        
        gray = skimage.color.rgb2gray(image)
        thresholds = threshold_multiotsu(gray, classes=2)
        thresh_background = thresholds[0] * 1.38
        
        
        region = np.zeros([bot_angle[0]-up_angle[0], bot_angle[1]-up_angle[1]])
        box_origin = np.zeros([bot_angle[0]-up_angle[0], bot_angle[1]-up_angle[1]])
        for i in range(up_angle[0], bot_angle[0]) :
            for j in range(up_angle[1], bot_angle[1]) :
                
                box_origin[i-up_angle[0]][j-up_angle[1]] = gray[i][j]
                
                #To redo a threshold on the original image fo the box only
                if gray[i][j] < thresh_background :
                    region[i-up_angle[0]][j-up_angle[1]] = 255
                    
                """
                # To take a crop of the binarized image (shapes)
                if shape[i][j] == 255 :
                    region[i-up_angle[0]][j-up_angle[1]] = 255
                """

        
        #region = morphology.area_opening(region, )
        med = median(region)
        pix_med = len(np.where(med==255)[0])
        
        # Pixel counting
        pix = len(np.where(region==255)[0])
        
        # Removing the boxes containing less than a certain number of shape pixel
        if pix < pix_thresh or pix_med < 5:
            k.append(key)
        else :
            boxes[key] = [boxes[key][0], boxes[key][1], boxes[key][2], region, box_origin]
    
    for l in k :
        del boxes[l]
    
    return boxes



def get_ROI(image, loc,
            dist_merge=15, thresh_ratio=6,
            pix_aug=5, diag_thresh_down=5, diag_thresh_up=50,
            thresh_robot=50,
            pix_thresh=55,
            ) :
    """
    Function that gives the region of interest in an image. The forms need to be darker than
    the background.
    
    Inputs : - image --> Initial image to analyze (RGB array)
             - loc --> Robot localization
             - dist_merge --> Distance threshold to merge 2 boxes
             -  thresh_ratio --> Ratio height width too remove a long box
             - pix_aug --> Number of pixels for the augmentation
             - diag_thresh_down --> Size of the diagonal below the one boxes are removed
             - diag_thresh_up --> Treshold to remove the little boxes
             - thresh_robot --> Threshold for the size of the region around the robot to remove the boxes

    """
    
    # Find all the forms, binarizes it in "shapes" and put the informations in the dictionnary "forms"
    shapes, forms = get_forms(image)
    
    # Find the boxes around all the regions of interest
    boxes = get_boxes(forms)
    
    # Merge the too close boxes and removes the long boxes
    boxes = clean_boxes(boxes, dist_merge, thresh_ratio)
    
    # Augment the size of the boxes and remove small one
    boxes = augment_boxes(image, boxes, pix_aug, diag_thresh_down, diag_thresh_up)
    
    # Remove the boxes near to the robot location
    boxes = remove_robot_boxes(boxes, loc, thresh_robot)
    
    # Boxes binarization and removing of boxes with few pixels
    boxes = binarisation(image, boxes, shapes, pix_thresh)
    
    return boxes


## Plotting the boxes of the operators
def plot_boxes(boxes, frame):
    color = (255, 0, 0)
    thickness = 1
    image_boxes_bin = frame.copy()
    for key in boxes.keys() :
        box = boxes[key]
        image_boxes_bin = cv2.rectangle(image_boxes_bin, (box[0][1], box[0][0]), (box[1][1], box[1][0]), color, thickness)
    plt.imshow(image_boxes_bin)
    plt.title('Boxes')
    plt.show()





