import cv2
import numpy as np 
import time

def morphology(hsv):
    b_img = cv2.inRange(hsv, (20,70,100), (40,255,255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    b_img = cv2.morphologyEx(b_img, cv2.MORPH_OPEN, kernel, iterations=5)
    b_img = cv2.dilate(b_img, kernel,iterations=4)
    return b_img

def detect(img):
    img = cv2.GaussianBlur(img, (5,5), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    b_img = morphology(hsv)
    box = []
    contours, hierachy = cv2.findContours(b_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        Area = cv2.contourArea(cnt)
        if Area > 200:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)  
            cv2.putText(frame, str(int(Area)),(x,y), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            box.append([x, y, w, h,])
    return box, frame, b_img

def main():
    track_state = False 
    prev_frame_time = 0
    new_frame_time = 0
    count_track = 0
    new_pos = 0
    prev_pos = 0 
    fps_avg = []
    show_binary_img = True
    while True:
        new_frame_time = time.time()
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (850,480))
            #tracking with resized frame (smaller 3 times) 
            frame_track = cv2.resize(frame.copy(), (283,160))
            #=========================DETECT===========================
            if not(track_state):
                #detect object
                box, img_color, b_img = detect(frame)
                #detect successfully
                if len(box)!= 0:                   
                    box =np.asarray(box[0])
                    box_track = np.asarray(box/3,dtype = int) 
                    #initialize tracker
                    track_first = tracker.init(frame_track,box_track)
                    prev_pos = box_track[1]
                    print("DETECTING!!!!!")
                    # print("box",box)
                    # print(box_track)
                    cv2.putText(frame,'DETECTING...',(10,40), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                    #checking the box size
                    if box[2] >= 60 and box[3] >= 60:
                        track_state = True
                #detect fail
                else:
                    print("DETECT FAIL! DETECT AGAIN!")
                    cv2.putText(frame,'CANNOT DETECT',(10,40), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

            #==================TRACKING======================               
            else:
                #tracking
                ret, obj = tracker.update(frame_track)
                # obj = obj/2
                if ret:
                    print("TRACKING!!!!")
                    #get box coordinate to track on the resized frame
                    p1 = (int(obj[0]),int(obj[1]))
                    p2 = (int(obj[0] + obj[2]),int(obj[1] + obj[3]))
                    new_pos = obj[1]
                    cv2.rectangle(frame_track, p1,p2, (0,255,0), 2)
                    #get box coordinate to show
                    p1_show = (int(obj[0]*3),int(obj[1]*3))
                    p2_show = (int(obj[0]*3 + obj[2]*3),int(obj[1]*3 + obj[3]*3))
                    cv2.rectangle(frame, p1_show,p2_show, (0,255,0), 3)
                    cv2.putText(frame, 'TRACKING...',(10,40), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
                    count_track +=1
                    #check the difference of w,h
                    if (obj[2]/obj[3]) >= 1.15 or (obj[2]/obj[3]) <= 0.85:
                        track_state = False
                    #tracking in the specific number of frame
                    if count_track >= 35:
                        track_state = False
                        count_track = 0
                        #check the movement of box
                        if abs(new_pos - prev_pos) >= 25 or abs(new_pos - prev_pos) <= 5:
                            track_state = False
                else:
                    track_state = False
            #show binary image
            if show_binary_img:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                b_img = morphology(hsv)
                cv2.imshow("binary", b_img) 

            #calculate FPS
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            fps_avg.append(fps)

            cv2.putText(frame,"FPS: " + str(fps) ,(10,80), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            cv2.putText(frame,"Tran Ngo Minh Tri-19146033" ,(600,30), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            cv2.putText(frame,"Nguyen Hoai Nam-19146219" ,(600,60), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            cv2.putText(frame,"Press Q to exit" ,(10,450), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)
            cv2.imshow(video, frame)
            # cv2.imshow("frame_track", frame_track)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("VIDEO ENDED!!!")
            print("AVERAGE_FPS: ",int(np.mean(np.asarray(fps_avg))))
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video = "obscured_object.mp4"
    cap = cv2.VideoCapture(video)
    tracker = cv2.TrackerCSRT_create()
    ret, frame = cap.read()
    main()



