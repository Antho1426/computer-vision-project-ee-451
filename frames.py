import cv2

def getFrame(sec, vidcap, count, path_to_save):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(path_to_save+'/image'+str(count)+'.jpg', image)     # save frame as JPG file
    return hasFrames, image

def frame_dict(path_to_vid, path_to_save, frames) :
    vidcap = cv2.VideoCapture(path_to_vid)
    sec = 0
    frameRate = 0.5 #//it will capture image in each 0.5 second
    count=1
    success , image = getFrame(sec, vidcap, count, path_to_save)

    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success, frames['frame %d' % count] = getFrame(sec, vidcap, count, path_to_save)
        if success :
            frames['frame %d' % count] = cv2.cvtColor(frames['frame %d' % count], cv2.COLOR_BGR2RGB)
    
    return frames