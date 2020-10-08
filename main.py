#!/usr/local/bin/python3.7



# main.py



# ShowFigures
#++++++++++++++++++++++++++++++
ShowDetails = True
ShowFigures = False
ShowRequiredInformation = True
#++++++++++++++++++++++++++++++



# Setting the current working directory automatically
import os
working_directory_path = os.getcwd()


#++++++++++++++++++++++++++++++
# Training video
#path_to_video = working_directory_path+'/robot_parcours_1.avi'
# Exam Video
#path_to_video = working_directory_path+'/Sequence_final_rotated_black.avi'
# Exam video of André (n°1, simple)
#path_to_video = working_directory_path+'/Sequence_final_simple_andre.avi'
# Exam video of André (n°2)
#path_to_video = working_directory_path+'/Sequence_final_rotated_andre.avi'
# Exam video of André (n°2) modified with Photoshop ('7' replaced by '4')
#path_to_video = working_directory_path+'/Sequence_final_rotated_andre_photoshoped_from7To4.avi'
# Exam video of André (n°2) modified with Photoshop ('3' replaced by '5' in addition to previous modification)
path_to_video = working_directory_path+'/Sequence_final_rotated_andre_photoshoped_from3To5.avi'
#++++++++++++++++++++++++++++++



# Importing modules
import cv2
import time
import skimage
import imutils
import math as m
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu

# Importing other files
import box_finding              # boxes (Sylvain)
import operators_classification # operators (Coco)
import prediction               # digits (Sylvain)





start_chrono = perf_counter()

## 1) Pre-processing

## 1.a) Finding the center of the robot in the first frame

# Considering the first frame of the video
cap = cv2.VideoCapture(path_to_video)
ret, frame1 = cap.read()


# Setting up the figure
if ShowFigures:
    plt.figure(1, figsize=(8, 7))
    plt.suptitle('Robot red arrow tracking')
    plt.subplot(3,3,1)
    plt.imshow(frame1)
    plt.title(('BGR: {} x {}'.format(frame1.shape[0], frame1.shape[1])))
    # plt.figure(); plt.imshow(frame); plt.title("frame")


# 0. Converting the frame from BGR to RGB
frame_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
if ShowFigures:
    plt.subplot(3,3,2)
    plt.imshow(frame_rgb)
    plt.title(('RGB: {} x {}'.format(frame_rgb.shape[0], frame_rgb.shape[1])))


# 1. Blurring the frame
frame_blurred = cv2.GaussianBlur(frame_rgb, (11, 11), 0)
if ShowFigures:
    plt.subplot(3,3,3)
    plt.imshow(frame_blurred)
    plt.title(('Blurred: {} x {}'.format(frame_blurred.shape[0], frame_blurred.shape[1])))


# 2. Converting the blurred frame to HSV
frame_hsv = cv2.cvtColor(frame_blurred, cv2.COLOR_RGB2HSV)
if ShowFigures:
    plt.subplot(3,3,4)
    plt.imshow(frame_hsv)
    plt.title(('HSV: {} x {}'.format(frame_hsv.shape[0], frame_hsv.shape[1])))
    # plt.figure(); plt.imshow(frame_hsv); plt.title("HSV")



# 3. Finding the range of values that preserve only the red arrow of the robot
light_red = (170,100,107)
dark_red = (188,250,168)

# 4. Extracting the arrow
extracted_arrow = cv2.inRange(frame_hsv, light_red, dark_red)
if ShowFigures:
    plt.subplot(3,3,5)
    plt.imshow(extracted_arrow)
    plt.title(('Extracted arrow: {} x {}'.format(extracted_arrow.shape[0], extracted_arrow.shape[1])))
    # plt.figure(); plt.imshow(extracted_arrow)



# 5. Morphology (performing opening: a series of dilations and erosions
# to remove any small blobs left in the mask)
extracted_arrow = cv2.erode(extracted_arrow, None, iterations=2)
extracted_arrow = cv2.dilate(extracted_arrow, None, iterations=2)
if ShowFigures:
    plt.subplot(3,3,6)
    plt.imshow(extracted_arrow)
    plt.title(('Morphology: {} x {}'.format(extracted_arrow.shape[0], extracted_arrow.shape[1])))


# 6. Finding contours of the arrow and initialize its current
# (x, y) center
cnts = cv2.findContours(extracted_arrow.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# Drawing the contours
frame_copy = frame1.copy()
cv2.drawContours(frame_copy, cnts, -1, (0,255,0), 3)
if ShowFigures:
    plt.subplot(3,3,7)
    plt.imshow(frame_copy)
    plt.title(('Contours: {} x {}'.format(frame_copy.shape[0], frame_copy.shape[1])))




# 7. Finding the center of the contour
# initialize the current center
center = None
# only proceed if at least one contour was found
if len(cnts) > 0:
    # find the largest contour in the mask, then use
    # it to compute the minimum enclosing circle and
    # centroid
    c = max(cnts, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) # hor, vert
    # only proceed if the radius meets a minimum size
    if radius > 10:
        # draw the circle and centroid on the frame,
        # then update the list of tracked points
        frame_copy_2 = frame1.copy()
        cv2.circle(frame_copy_2, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        cv2.circle(frame_copy_2, center, 5, (0, 0, 255), -1)
        if ShowFigures:
            plt.subplot(3,3,8)
            plt.imshow(frame_copy_2)
            plt.title(('Identified center: {} x {}'.format(frame_copy_2.shape[0], frame_copy_2.shape[1])))

        # drawing a rectangle around the big circle
        rectangle_size = 2*radius
        start_point = (int(x-radius), int(y-radius))
        end_point = (int(x+radius), int(y+radius))
        color = (0, 255, 0) # green color in BGR
        thickness = 2 # Line thickness in [px]
        frame_copy_3 = frame1.copy()
        frame_tracked = cv2.rectangle(frame_copy_3, start_point, end_point, color, thickness) # using cv2.rectangle() method on the current frame
        cv2.putText(frame_tracked, 'robot', (start_point[0], start_point[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)
        if ShowFigures:
            plt.subplot(3,3,9)
            plt.imshow(frame_tracked)
            plt.title(('Tracked robot: {} x {}'.format(frame_tracked.shape[0], frame_tracked.shape[1])))


if ShowFigures:
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)






## 1.b) Finding all the boxes
# Converting frame1 from BGR to RGB
frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
# plt.figure(); plt.imshow(frame1); plt.title("frame1")
robot_initial_position = np.array([center[1], center[0]]) #  VERT, HOR
start_chrono_get_ROI = perf_counter()
boxes = box_finding.get_ROI(frame1, robot_initial_position) # example to access information from the dictionary: about the "two" at the top left of the original video: boxes["Shape",2][0] == array([ 23, 213]) == top left corner of the box, boxes["Shape",2][1] == array([ 55, 238]) == bottom right corner of the box, boxes["Shape",2][2] == array([ 39, 225]) == center of the box
stop_chrono_get_ROI = perf_counter()
print("Elapsed time for 'get_ROI:' ", round(stop_chrono_get_ROI - start_chrono_get_ROI, 3)," [s]") # 5.526  [s]
# plotting all the boxes
if ShowFigures:
    plt.figure()
    box_finding.plot_boxes(boxes, frame1)


# Initializing the previously visited box
previous_key = []






## 2) Real time video playing

# loading the video
video_in = cv2.VideoCapture(path_to_video)
# initializing the number of frames
frames_counter = 0
# initializing the list of tracked points (the positions of the center of the robot)
pts = []

# initializing the counters for debugging
number_of_frames_with_too_small_radius = 0
number_of_frames_with_no_contour_detected = 0
# initializing the lists of frames presenting problems for debugging
frames_with_too_small_radius_list = []
frames_with_no_contour_detected_list = []




# Initializations for the classification
#***********
# Initializations regarding the shapes identification
identification_id = 1
dist_lim = 40 # pxl
# non-definitive debug variable
identification_type_list = []
#***********



# Initialization for the computation of the on-screen digits
#-----------
# initializing the global variable "result" (result of the equation)
result = 0
# initialization of the string gathering the real digits and operators of the
# equation
equation = ''
#-----------


# setting up the variables to write and store the processed video
# Cf.: https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
# Default resolutions of the frame
frame_width = int(video_in.get(3))
frame_height = int(video_in.get(4))
# Define the codec and create VideoWriter object.
# The output is stored in 'antho_tracked_robot.avi' file.
video_out = cv2.VideoWriter('exam.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 2, (frame_width,frame_height)) # with "5" instead of "2", the output video will have a quicker frame rate and will lasts less than 10[s] (in comparison with 21[s] for the input video)


while True:
    check, frame = video_in.read()

    print("frame:\n", frame)
    print("check:", check)

    if check:
        # update the number of frames that the video contains
        frames_counter += 1
        # print some text on the video
        # (cf.: https://www.geeksforgeeks.org/python-opencv-write-text-on-video/)
        #------------------------------------
        # describe the type of font
        # to be used.
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Use putText() method for
        # inserting text on video
        #-----
        if ShowDetails:
            frameTitle = 'Frame: '
            frameNumber = str(frames_counter)
            text = frameTitle + frameNumber
            cv2.putText(frame,
                        text,
                        (470, 25),
                        font, 1,
                        (255, 0, 125),
                        2,
                        cv2.LINE_4)
        #-----
        #------------------------------------

        # 0. Converting from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 1. Burring the frame
        frame_blurred = cv2.GaussianBlur(frame_rgb, (11, 11), 0)
        # 2. Converting from RGB to HSV
        frame_hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        # 3. Range of HSV values that preserve only the red arrow
        # light_red = (173,105,110)
        # dark_red = (185,245,165)
        light_red = (170,100,107)
        dark_red = (188,250,168)
        # 4. Extracting the arrow
        extracted_arrow = cv2.inRange(frame_hsv, light_red, dark_red)
        # 5. Morphology (performing opening: a series of dilations and erosions
        # to remove any small blobs left in the mask)
        extracted_arrow = cv2.erode(extracted_arrow, None, iterations=2)
        extracted_arrow = cv2.dilate(extracted_arrow, None, iterations=2)

        # showing the extracted_arrow
        cv2.imshow("extracted_arrow", extracted_arrow)

        # 6. Finding contours of the arrow and initialize its current
        # (x, y) center
        cnts = cv2.findContours(extracted_arrow.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # 7. Finding the center of the contour
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # saving the current center
            pts.append(center)

            # only proceed if the radius meets a minimum size
            if radius > 5:

                processed_frame = frame.copy()
                color_line = (0, 0, 255) # red color in BGR
                color_square = (0, 255, 0) # green color in BGR
                thickness_line = 1
                thickness_square = 2 # Line thickness in [px]

                # drawing a rectangle around the big circle containing the arrow
                #-----
                if ShowDetails:
                    rectangle_size = 60
                    start_point_square = (int(x-rectangle_size/2), int(y-rectangle_size/2))
                    end_point_square = (int(x+rectangle_size/2), int(y+rectangle_size/2))
                    cv2.rectangle(processed_frame, start_point_square, end_point_square, color_square, thickness_square) # using cv2.rectangle() method on the current frame
                    cv2.putText(processed_frame, 'robot', (start_point_square[0], start_point_square[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)
                #-----

                # draw the trajectory of the robot
                # using cv2.line() method to draw a diagonal green line with
                # thickness of 9 px between the tracked position of the robot
                # Cf.: https://www.geeksforgeeks.org/python-opencv-cv2-line-method/
                #-----
                if ShowRequiredInformation:
                    for i in range(len(pts)-1):
                        start_point_line = pts[i]
                        end_point_line = pts[i+1]
                        cv2.line(processed_frame, start_point_line, end_point_line, color_line, thickness_line, lineType=8)
                #-----

                # draw the centroid of the current position on the frame
                # (while replicating the centers of the previous positions)
                #-----
                if ShowRequiredInformation:
                    for i in range(len(pts)):
                        # drawing a point a the center of each position
                        cv2.circle(processed_frame, pts[i], 5, color_line, -1)
                #-----

                # write the number of the corresponding frame next to each
                # position of the robot
                #-----
                if ShowDetails:
                    for i in range(len(pts)):
                        # putting a text indicating at which frame belongs each position
                        cv2.putText(processed_frame, str(i+1), (pts[i][0], pts[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_line, 1)
                #-----

                # draw the boxes
                #-----
                if ShowDetails:
                    for key in boxes.keys():
                        box = boxes[key]
                        image_boxes_bin = cv2.rectangle(processed_frame, (box[0][1], box[0][0]), (box[1][1], box[1][0]), color_line, thickness_line)
                #-----



                # Shapes identification
                #*************************************************

                # To do at each frame of the video:
                for key in boxes:
                    robot_position = np.array([center[1], center[0]]) #  VERT, HOR
                    distance = m.sqrt( (robot_position[0] - boxes[key][2][0])**2 + (robot_position[1] - boxes[key][2][1])**2 )
                    if ( (distance < dist_lim) & (key != previous_key) ): # if we pass this condition, this means that we are in the vicinity of a shape
                        # 😈 Trick: make use of the alternation operators-digits
                        # --> identify the content of the box accordingly (uneven iteration == operator, even iteration == digit)
                        # --> either consider the detection algorithm of Coco (operators) or Sylvain (digits)
                        if (identification_id % 2 == 0): # uneven identification_id --> Coco
                            identification_type_list.append('op')
                            # [...] identifying the operator
                            SymbolFrame = operators_classification.SymbolsExtractor(frame1)
                            _, pred_operator = operators_classification.SingleSymbol_finding(boxes, SymbolFrame, key)
                            # updating the equation to solve
                            equation += pred_operator


                        else: # even identification_id --> Sylvain
                            identification_type_list.append('dgt')
                            # [...] identifying the digit

                            gray = skimage.color.rgb2gray(frame1)

                            # For the case where the digits are not rotated:
                            #pred_digit = prediction.digit_prediction(boxes[key][4], boxes[key][3], gray, 'mlp16', 33.318447, 78.567444, redress=False, redress_bis=False)
                            # For the case where the digits are rotated:
                            #pred_digit = prediction.digit_prediction(boxes[key][4], boxes[key][3], gray, 'mlp15', 31.647715, 66.54372, redress=True, redress_bis=True)
                            # Trying my version of the rotated digits recognition (CNN with keras + MLP):
                            #~~~~~~~~~~
                            pred_digit, predicted_angle = prediction.digit_prediction_with_keras(boxes[key][4], boxes[key][3], gray)
                            #print('predicted_angle: ', predicted_angle)
                            #~~~~~~~~~~

                            # updating the equation to solve
                            equation += ' ' + pred_digit

                        # saving the current key
                        previous_key = key

                        # and update the identification_id (indicating that we have met and identified an additional shape)
                        identification_id += 1


                # print text of equation accordingly on video (continuously updated)
                if ShowRequiredInformation and len(equation) > 0:

                    if equation[-1] != '=': # as long as we don't have found a "=" operator
                        cv2.putText(processed_frame,
                                    equation,
                                    (25, 25),
                                    font, 1,
                                    (0, 255, 255),
                                    2,
                                    cv2.LINE_4)
                    else: # if we enter this part of the code, it means that we
                          # encounter a "=" operator. We can hence solve the
                          # equation
                        # removing the '=' at the end of "equation" using a copy
                        equation_copy = equation
                        equation_copy = equation_copy[:-1]
                        # solving the equation
                        final_equation = 'result = ' + 'round(' + equation_copy + ', 2)'
                        exec(final_equation, globals())
                        cv2.putText(processed_frame,
                                    equation + ' ' + str(result),
                                    (25, 25),
                                    font, 1,
                                    (0, 255, 255),
                                    2,
                                    cv2.LINE_4)
                #*************************************************






                # showing the processed frame
                cv2.imshow("Processed video", processed_frame)

                # write the frame into the file output file specified above
                video_out.write(processed_frame)

                key = cv2.waitKey(1)
                time.sleep(0.1) # controls the execution speed of the video dipslayed in real time by OpenCV

            else:
                number_of_frames_with_too_small_radius += 1
                frames_with_too_small_radius_list.append(frames_counter)



        else:
            number_of_frames_with_no_contour_detected += 1
            frames_with_no_contour_detected_list.append(frames_counter)


    else:
        break # If we have no more frames to read, we "break"

print("\n======")
print("Number of frames in the video: ", frames_counter)
print("~~~~~~")
print("Frames presenting detection problems:")
print("    - Frames with no (arrow) contour detected: ", number_of_frames_with_no_contour_detected, "(these are frames: ", frames_with_no_contour_detected_list,")")
print("    - Frames with too small radius (for circle containing the arrow): ", number_of_frames_with_too_small_radius, "(these are frames: ", frames_with_too_small_radius_list,")")
print("~~~~~~")
print("Tracked positions of the robot:", pts)

# Debug
print("\nidentification_id", identification_id)
print("identification_type_list", identification_type_list)

# Measurinng elapsed time
stop_chrono = perf_counter()
print("\nTotal elapsed time: ", round(stop_chrono - start_chrono, 3)," [s]")

video_in.release()
video_out.release()
cv2.destroyAllWindows()








