# Kirieg Check 13 
# Test Bj8rnar

from tkinter import *
from tkinter import ttk
from turtle import update        # To use combobox
from PIL import ImageTk, Image
from click import command
import cv2
import numpy as np
import os
import sys
import glob
import socket
import time, math
import cv2.aruco as aruco



#------------------------------------------------------------------
#-------------------------Socket-----------------------------------
main_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP

ip = 'localhost'     # IP 127.0.0.1
port = 5433             # Port
print("Program: Socket Created")
#------------------------------------------------------------------


#------------------------------------------------------------------
#------------------Variables---------------------------------------
tms = 100    #Times pr milliscond

global arucoRunning
global trackRunning 
arucoRunning = False
trackRunning = False

class Colors:
    Aqua = [255, 255, 0]  
    Lime = [0, 255, 0]  
    Yellow = [0, 255, 255]
    Fuchisa = [255, 0, 255]      
    Purple = [128, 0, 128]
    Blue = [255, 0, 0] 
    Red = [0, 0, 255]   
    Green = [0, 128, 0]
#------------------------------------------------------------------


#------------------------------------------------------------------   
#------------Undistort camera matrix-------------------------------
#Get the camera calibration path  
calib_path  = "Calibration/"
mtx = np.loadtxt(calib_path+'cameraMatrix.txt', delimiter=',')
dist = np.loadtxt(calib_path+'cameraDistortion.txt', delimiter=',')
#------------------------------------------------------------------


#------------------------------------------------------------------
#----------------------Tracker-------------------------------------
class Tracker:    
    def __init__(self, tracker_type, bbox, video_capture, color):
        
        self.bbox = bbox
        self.tracker_type = tracker_type 
        self.cap = video_capture  
        self.color = color
        self.tracker_running = False
        self.error = False
        self.warning = False
        
        self.dx = 0.0
        self.dy = 0.0
        self.dz = 0.0
               
        if self.tracker_type == 'MEDIANFLOW':
            self.tracker = cv2.legacy.TrackerMedianFlow_create()
        if self.tracker_type == 'MOSSE':
            self.tracker = cv2.legacy.TrackerMOSSE_create()
        if self.tracker_type == "CSRT":
            self.tracker = cv2.legacy.TrackerCSRT_create()

        # Exit if video not opened.
        if not self.cap.isOpened():
            print ("Could not open video")
            sys.exit()
        
        # Read first frame.
        self.ok, self.frame = self.cap.read()
        if not self.ok:
            print ('Cannot read video file')
            sys.exit()
        
        ########## UNDISTORT ########### (Comment out if using test video.mp4)
        # Undistort image
        h,  w = self.frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        # undistort
        frame = cv2.undistort(self.frame, mtx, dist, None, newcameramtx)  
        # crop the image
        x, y, w, h = roi
        frame = frame[y:y+h, x:x+w]
        ##############################
        
        self.ok = self.tracker.init(self.frame, self.bbox)
        
        if delta_drop.get() == "Marked object":
            self.Create_offsett_from_ref_bbox()
        elif delta_drop.get() == "Senter screen":
            self.Create_offsett_from_center_screen()
            
        Error_timer()    
        Error_detection()
        
       
        
    def Run(self):    
        if self.tracker_running == True:
            global trackRunning
            trackRunning = True
            
            
            self.error = False
            # Read a new frame
            self.ok, self.frame = self.cap.read()
            
            ########## UNDISTORT ########### (Comment out if usen video.mp4)
            h,  w = self.frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))      
            # undistort
            frame = cv2.undistort(self.frame, mtx, dist, None, newcameramtx)      
            # crop the image
            x, y, w, h = roi
            frame = frame[y:y+h, x:x+w]
            ################################
                    
            # Start timer
            timer = cv2.getTickCount()
            # Update tracker
            self.ok, self.bbox = self.tracker.update(self.frame)
            # Calculate Frames per second (FPS)
            self.fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            # Draw bounding box
            if self.ok:
                # Tracking success
                self.p1 = (int(self.bbox[0]), int(self.bbox[1]))
                self.p2 = (int(self.bbox[0] + self.bbox[2]), int(self.bbox[1] + self.bbox[3]))
                
                # Centerpoint bbox
                self.centerXbbox = int(self.bbox[0]+(self.bbox[2] / 2))
                self.centerYbbox = int(self.bbox[1]+(self.bbox[3] / 2))
                 
                if delta_drop.get() == "Marked object":
                    self.Sett_offsett_from_ref_bbox()
                elif delta_drop.get() == "Senter screen":
                    self.Sett_offsett_from_center_screen()
            else:
                self.error= True
                        
            root.after(tms, self.Run)
            
    # Start tracker 
    def Start(self):       
        self.tracker_running = True
        self.Run()
        global trackRunning
        trackRunning = True
        
    
    def Stop_tracker(self):
        self.tracker_running = False
    

    ## Calculate offsett from the center object and where det bbox was first sett:
    def Create_offsett_from_ref_bbox(self):
         # Defining corner p1 and p2 for refferancebox
        self.refBox = self.bbox
        self.refP1 = (int(self.refBox[0]), int(self.refBox[1]))
        self.refP2 = (int(self.refBox[0] + self.refBox[2]), int(self.refBox[1] + self.refBox[3]))
        # Center of refferencebox:
        self.centerXrefBox = int(self.refBox[0]+(self.refBox[2] / 2)) 
        self.centerYrefBox = int(self.refBox[1]+(self.refBox[3] / 2))
        
    def Sett_offsett_from_ref_bbox(self):
        # Create centerpoint refferencebox 
        self.cRefP1 = (self.centerXrefBox, self.centerYrefBox)
        self.cRefP2 = (self.centerXrefBox, self.centerYrefBox)
        # Uppdate delta/offset from between center refbox and bbox
        self.dx = float(self.centerXrefBox - self.centerXbbox)
        self.dy = float(self.centerYbbox - self.centerYrefBox)
        self.dz = float(self.refBox[3] - self.bbox[3])


    ## Calculate offsett from the senter of screen
    def Create_offsett_from_center_screen(self):
        # Refference to first bbox    
        self.refBox = self.bbox   
        # Frame dimension:
        frameHeight = int(self.frame.shape[0])
        frameWidth = int(self.frame.shape[1])
        # Finds center of screen:
        self.centerFrameY = int(frameHeight/2)
        self.centerFrameX = int(frameWidth/2) 
        
    def Sett_offsett_from_center_screen(self):
        # Create refferencebox center screen
        self.refP1 = (int(self.centerFrameX - self.refBox[2]/2) , int(self.centerFrameY - self.refBox[3]/2)) 
        self.refP2 = (int(self.centerFrameX + self.refBox[2]/2) , int(self.centerFrameY + self.refBox[3]/2))
        # Create centerpoint in refferencebox center screen 
        self.cRefP1 = (self.centerFrameX, self.centerFrameY)
        self.cRefP2 = (self.centerFrameX, self.centerFrameY)
        # Update delta/offset between bbox and center screen 
        self.dx = float(self.centerXbbox - self.centerFrameX)
        self.dy = float(self.centerFrameY - self.centerYbbox)
        self.dz = float(self.refBox[3] - self.bbox[3])
              
#--------------------------Tracker end----------------------------
            
            
            
#-----------------------------------------------------------------
#------------------------------Aruco------------------------------
class Aruco:   
    def __init__(self, video_capture):    
        self.cap = video_capture
        try:
            self.id_to_find  = int(entry_aruco_id.get())
            entry_aruco_id.config({"background": "white"})
        except:
            entry_aruco_id.config({"background": "pink"})
            print("Invalid entry id")
        try:
            self.marker_size  = float(entry_aruco_size.get())
            entry_aruco_size.config({"background": "white"})
        except:
            entry_aruco_size.config({"background": "pink"})
            print("Invalid entry size aruco marker")
        
        #-- Outputs
        self.x = 0
        self.y = 0
        self.z = 0
        self.roll = 0
        self.pitch = 0
        self.yaw = 0

        #--- Get the camera calibration path
        #os.chdir('./Calibration')
        calib_path  = './Calibration/'
        self.camera_matrix   = np.loadtxt(calib_path+'cameraMatrix.txt', delimiter=',') #calib_path+
        self.camera_distortion   = np.loadtxt(calib_path+'cameraDistortion.txt', delimiter=',')

        #--- 180 deg rotation matrix around the x axis
        self.R_flip  = np.zeros((3,3), dtype=np.float32)
        self.R_flip[0,0] = 1.0
        self.R_flip[1,1] =-1.0
        self.R_flip[2,2] =-1.0

        #--- Define the aruco dictionary
        if aruco_lib_drop.get() == "4x4_100":
            self.aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
        elif aruco_lib_drop.get() == "5x5_100":
            self.aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
        elif aruco_lib_drop.get() == "Classic":
            self.aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
            
        self.parameters  = aruco.DetectorParameters_create()

        #-- Font for the text in the image
        self.font = cv2.FONT_HERSHEY_PLAIN
        
        #--- Validerer rotasjonsmatrise
    def isRotationMatrix(self,R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

        #--- kalkulerer rotasjonsmatrise til eulers
    def rotationMatrixToEulerAngles(self,R):
        assert (self.isRotationMatrix(R))

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

     
    def Aruco_run(self):        
        if not trackRunning and arucoRunning:                     
            
            ret, self.frame = self.cap.read()

            #-- Convert in gray scale
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY) #-- OpenCV stores color images in Blue, Green, Red

            #-- Find all the aruco markers in the image
            corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=self.aruco_dict, parameters=self.parameters,
                                    cameraMatrix=self.camera_matrix, distCoeff=self.camera_distortion)
            
            if ids is not None and ids[0] == self.id_to_find:
                
                #-- ret = [rvec, tvec, ?]
                #-- array of rotation and position of each marker in camera frame
                #-- rvec = [[rvec_1], [rvec_2], ...]    attitude of the marker respect to camera frame
                #-- tvec = [[tvec_1], [tvec_2], ...]    position of the marker in camera frame
                ret = aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix, self.camera_distortion)

                #-- Unpack the output, get only the first
                rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]

                #-- Draw the detected marker and put a reference frame over it
                aruco.drawDetectedMarkers(self.frame, corners)
                aruco.drawAxis(self.frame, self.camera_matrix, self.camera_distortion, rvec, tvec, 10)

                #-- Obtain the rotation matrix tag->camera
                R_ct    = np.matrix(cv2.Rodrigues(rvec)[0])
                R_tc    = R_ct.T

                #-- Now get Position and attitude f the camera respect to the marker
                pos_camera = -R_tc*np.matrix(tvec).T

                #-- Update class position variables
                self.x, self.y, self.z = pos_camera[0], pos_camera[1], pos_camera[2]

                #-- Get the attitude of the camera respect to the frame
                roll_camera, pitch_camera, yaw_camera = self.rotationMatrixToEulerAngles(self.R_flip*R_tc)
                
                #-- Update class attitude variables:
                self.roll, self.pitch, self.yaw = math.degrees(roll_camera), math.degrees(pitch_camera), math.degrees(yaw_camera)
                
            if ret:
                cv2image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image).resize((1024, 720))
                imgtk = ImageTk.PhotoImage(image = img)       
                label_vid_1.imgtk = imgtk
                label_vid_1.configure(image=imgtk)
            
            root.after(tms, self.Aruco_run)

      
#----------------------Aruco END--------------------------------       
        
        
        
#-----------------------------------------------------------------------
#--------------------------GUI Update-----------------------------------

def Aruco_Click():
    Stop_all_trackers()
    a = Aruco(cap)
    global arucoRunning
    arucoRunning = True
    aList.append(a)
    aList[0].Aruco_run()
      
    
# Function that shows bbox and refbox from trackers on screen
def Show_frames_one():    
    if not arucoRunning:
        ok, frame = cap.read() 
        if ok:
            j = 50
            for obj in tList:                
                if obj.tracker_running:
                    try:                
                        # Display refbox:
                        cv2.rectangle(frame, obj.refP1, obj.refP2, Colors.Red, 2, 1 )
                        # Display bbox:
                        cv2.rectangle(frame, obj.p1, obj.p2, obj.color, thickness=1)
                        # Display centerpoint refbox:
                        cv2.rectangle(frame, obj.cRefP1, obj.cRefP2, Colors.Red, 2, 3)
                        # Display tracker type on frame:
                        cv2.putText(frame, obj.tracker_type , (30,j), cv2.FONT_HERSHEY_SIMPLEX, 0.6, obj.color, 1)                                     
                        # Display FPS on frame:
                        cv2.putText(frame, "FPS : " + str(int(obj.fps)), (180,j), cv2.FONT_HERSHEY_SIMPLEX, 0.6, obj.color, 1) 
                    
                        j+=25   # Move text down screen if more trackers
                    except:
                        print("Error update tracker")
                        obj.error = True
                    
            cv2.putText(frame, "Refbox", (30,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, Colors.Red, 1)  
                
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image).resize((1024, 720))
            imgtk = ImageTk.PhotoImage(image = img)       
            label_vid_1.imgtk = imgtk
            label_vid_1.configure(image=imgtk)
    
    root.after(tms, Show_frames_one)

# Choose the Region of interest
def New_ROI():
    ok, cv2_image = cap.read()
    if ok:
        bbox = cv2.selectROI(cv2_image)
        cv2.destroyWindow('ROI selector')
    return bbox

# Starts multiple selected trackers
def Click_multi_start():
    global arucoRunning
    arucoRunning = False
    
    bbox = New_ROI()
    
    if tracker_drop_1.get() != "0":
        tList.append(Tracker(tracker_drop_1.get(), bbox, cap, Colors.Aqua))
    if tracker_drop_2.get() != "0":
        tList.append(Tracker(tracker_drop_2.get(), bbox ,cap, Colors.Lime))
    if tracker_drop_3.get() != "0":
        tList.append(Tracker(tracker_drop_3.get(), bbox ,cap, Colors.Yellow))
        
    for obj in tList:
        obj.Start()
    
# Retrieve the selected camera source
def Camera_Select():
    return int(camera_drop_1.get())

# Function for stopping the tracker functions
def Stop_all_trackers():
    for obj in tList:
        obj.Stop_tracker()
    tList.clear()
    aList.clear()
    global trackRunning
    trackRunning=False
    global arucoRunning
    arucoRunning=False



# Updates the statusbar with the offset        
def Update_statusbar():
    if len(tList) > 0:
        statusbar_1.config(text = "Tr.1:  " + tList[0].tracker_type +" \tx: " + str(round(tList[0].dx)) + "\ty: " + str(round(tList[0].dy)) + "\tz: " + str(round(tList[0].dz)))
    else:
        statusbar_1.config(text = "Tr.1")
    if len(tList) > 1:
        statusbar_2.config(text = "Tr.2:  " + tList[1].tracker_type +" \tx: " + str(round(tList[1].dx)) + "\ty: " + str(round(tList[1].dy)) + "\tz: " + str(round(tList[1].dz)) )
    else:
        statusbar_2.config(text = "Tr.2")
    if len(tList) > 2:
        statusbar_3.config(text = "Tr.3:  " + tList[2].tracker_type +" \tx: " + str(round(tList[2].dx)) + "\ty: " + str(round(tList[2].dy)) + "\tz: " + str(round(tList[2].dz)) )
    else:
        statusbar_3.config(text = "Tr.3")   
    root.after(200,Update_statusbar)

# Controll indicators
def Update_Indicators(): 
    # Error detection indicators
    if len(tList) > 0:
        if tList[0].error:
            Indicator_1.itemconfig(my_oval_1, fill="red")
        elif tList[0].warning & tList[0].tracker_running:
            Indicator_1.itemconfig(my_oval_1, fill="yellow")
        elif tList[0].tracker_running:
            Indicator_1.itemconfig(my_oval_1, fill="lime")
        else:
            Indicator_1.itemconfig(my_oval_1, fill="grey")
    if len(tList) > 1:
        if tList[1].error:
            Indicator_2.itemconfig(my_oval_2, fill="red")
        elif tList[1].warning & tList[1].tracker_running:
            Indicator_2.itemconfig(my_oval_2, fill="yellow")
        elif tList[1].tracker_running:
            Indicator_2.itemconfig(my_oval_2, fill="lime")
        else:
            Indicator_2.itemconfig(my_oval_2, fill="grey")
    if len(tList) > 2:
        if tList[2].error:
            Indicator_3.itemconfig(my_oval_3, fill="red")
        elif tList[2].warning & tList[2].tracker_running:
            Indicator_3.itemconfig(my_oval_3, fill="yellow")
        elif tList[2].tracker_running:
            Indicator_3.itemconfig(my_oval_3, fill="lime")
        else:
            Indicator_3.itemconfig(my_oval_3, fill="grey")
    if len(tList) == 0:
        Indicator_1.itemconfig(my_oval_1, fill="grey")
        Indicator_2.itemconfig(my_oval_2, fill="grey")
        Indicator_3.itemconfig(my_oval_3, fill="grey")
        
    # Aruco indicator:
    if arucoRunning:
        Indicator_4.itemconfig(my_oval_4, fill="lime")
    if not arucoRunning:
        Indicator_4.itemconfig(my_oval_4, fill="grey")
        
        
    root.after(400, Update_Indicators)
     
    
#-------Errordetection--------  

# Timer for measuring time of movement.
def Error_timer():
    global pre_dx, pre_dy, pre_dz
    for obj in tList:
        pre_dx = obj.dx
        pre_dy = obj.dy
        pre_dz = obj.dz      
    root.after(1000, Error_timer) 
    
# Error detection if tracker moves to far in a short ammount of time, indicates someting is wrong.   
# If tracker moves more than 50 pixles in 500 millisec, warning vil be triggered.
def Error_detection():
    for obj in tList:   
        if abs(obj.dx) >= (abs(pre_dx)+100):
            obj.warning = True
        if abs(obj.dy) >= (abs(pre_dy)+100):
            obj.warning = True
        if abs(obj.dz) >= (abs(pre_dz)+100):
            obj.warning = True
    root.after(1500, Error_detection)
    
#-------------------------  

# Themporary output controll:
def Output_control():
    i=0; j=0; x=0; y=0; z=0; tracker=""
    global tx, ty, tz
    tx=0; ty=0; tz=0
    if len(tList)>0: 
        for obj in tList:
            if obj.tracker_running and not obj.error:
                x += obj.dx
                y += obj.dy
                i += 1
                j += 1
                if obj.tracker_type == "MOSSE" and len(tList)>1:     # Not possible to use Mosse for z estimation                  
                    j -= 1
                elif obj.tracker_type == "MOSSE" and len(tList)==1:  # If only Mosse tracker are beeing used
                    obj.z = 0
                    j = 1
                else:
                    z += obj.dz
                                            
                tracker = tracker + "/" + obj.tracker_type[0:3]
                
            else: i=1; j=1;   # Handling /0

        tx = round(x/i)
        ty = round(y/i)
        tz = round(z/j)
        
        statusbar_0.config(text = "Output: Tracker:"+tracker+";"+"  x: " + str(tx) + "  y: " + str(ty) + "  z: " + str(tz))
                
    elif arucoRunning and len(aList)>0:
        statusbar_0.config(text = "Output: Aruco:  x=%2.0f y=%2.0f z=%2.0f roll=%2.0f pitch=%2.0f yaw=%2.0f"%(aList[0].x, aList[0].y, aList[0].z, aList[0].roll, aList[0].pitch, aList[0].yaw))
    
    else:
        statusbar_0.config(text="Output:")
        
    root.after(200, Output_control)

# Configure enabling buttons
def Button_controll():
    if trackRunning:
        button_stop_all.config(state=NORMAL)
        button_start_multiple.config(state=DISABLED)
        button_aruco.config(state=DISABLED)
        button_calibrate.config(state=DISABLED)
    elif arucoRunning:                                             
        button_stop_all.config(state=NORMAL)
        button_start_multiple.config(state=DISABLED)
        button_aruco.config(state=DISABLED)
        button_calibrate.config(state=DISABLED)
    else:
        button_stop_all.config(state=DISABLED)
        button_start_multiple.config(state=NORMAL)
        button_aruco.config(state=NORMAL)
        button_calibrate.config(state=NORMAL)
    
    root.after(600, Button_controll)
#--------------------------------GUI Update End----------------------------------------
 
 
     

#--------------------------------Send data-------------------------------------- 
def SendData():
    telegram = ""
    global tx,ty,tz
    # standard string format: $TX000000Y000000Z000000#
    if trackRunning and len(tList)>0:
    
        telegram = ("$TRACKX%.2fY%.2fZ%.2f#"%(tx, ty, tz))
        print(telegram)
        
    elif arucoRunning and len(aList)>0:
        #$AX000000Y000000Z000000PITCH0000000YAW000000ROLL000000#
        telegram = ("$ARUCOX%.2fY%.2fZ%.2fRO%.2fPI%.2fYA%.2f#"%(aList[0].x, aList[0].y, aList[0].z, aList[0].roll, aList[0].pitch, aList[0].yaw))
        print(telegram)
    try:
        main_socket.sendto(telegram.encode(), (ip,port))
        entry_ip.config({"background": "white"})
    except:
        entry_ip.config({"background": "pink"})
        print("Cannot send data UDP")
    
    # data, client = main_socket.recvfrom(65535)          #For testing av utdata med server.
    # data = data.decode()
    # print("Main: " + data)
    
    root.after(1000, SendData)
    
# Function click connect
def Connect_UDP_Click():
    global ip, port
    try:
        port = int(entry_port.get())
        entry_port.config({"background": "white"})
    except:
        entry_port.config({"background": "pink"})
       # port = 5433
        print("Not valid port input")
  
    try:
        ip = str(entry_ip.get())
    except:
        print("Not valid IP input")
        entry_ip.config({"background": "pink"})

    print("Program: Socket Connected")
#-------------------------------Send data end-----------------------------    
# def Update_Entry_int(variable,entry):
#     try:
#         variable = entry.get()
#         int(variable)
#         entry.config({"background": "white"})
#     except:
#         entry.config({"background": "pink"})
#         print("Not valid input: " + str(variable))

    
    
    
#------------------------------------------------------------------------ 
#------------------------Calibration-------------------------------------
def Cal_Click():
    
    #Creat new window    
    Top = Toplevel()
    Top.title('Calibrate')
    #start camera button
    global start_cam_cal_btn
    start_cam_cal_btn = Button(Top, text= "Start Camera", command= lambda: Start_cam_cal())
    start_cam_cal_btn.grid(row=0, column=0, padx= 10)
    #Exit Button 
    exit_top_btn = Button(Top,text="Exit", command=lambda:Top.destroy())
    exit_top_btn.grid(row=4, column=0)
    #Camera Frame 
    frame_Cal = LabelFrame(Top, text= "Camera", padx= 5, pady= 5 )
    frame_Cal.grid(row=0, column=2, rowspan= 8)

    def Start_cam_cal():
        #Start Camera button
        global start_cam_cal_btn
        start_cam_cal_btn = Button(Top, text= "Start Camera", state= DISABLED)
        start_cam_cal_btn.grid(row=0, column=0)
        
        #Calibrate video label
        Cal_label = Label(frame_Cal)
        Cal_label.grid(row=0, column=2, rowspan=10)
        
        #Take picture button
        Take_Pic_Button = Button(Top,text="Take Picture", command= lambda:take_pic())
        Take_Pic_Button.grid(row=1, column=0)
        #Start Calibrate button  
        Start_Cal_btn = Button(Top, text= "Start Calibrate", command= lambda:Start_Calib())
        Start_Cal_btn.grid(row= 2, column=0)
        
        #Number of pictures needed Label
        Ant_pic_lab = Label(Top, text="Need 25 or more pictures:")
        Ant_pic_lab.grid(row=9, column=0, columnspan=3)

        #Camera Frame and video capture
        Cal_label.frame_num = 0
        Cal_label.grid(row=0, column=0)
        
        #Checking if the number of pictures are enough or good
        Ant_pic_err_lab1 = Label(Top, text="")
        Ant_pic_err_lab1.grid(row=10, column=0, columnspan=3)
        Ant_pic_err_lab2 = Label(Top, text="")
        Ant_pic_err_lab2.grid(row=11, column=0, columnspan=3)
        
          #Print the distorion value to label
        label_Error = Label(Top, text = "Total error: ")
        label_Error.grid(row= 5, column= 0)
        
        #Print the distorion value status to label
        Error_value = ""
        Error_status = Label(Top, text = "Error Status: "+ Error_value)
        Error_status.grid(row= 6, column=0)

        if not cap.isOpened():
            print("Cannot open camera")
            exit()
            
        def show_frames():
            #Show video stream
            cv2image= cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
            Cal_img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image = Cal_img)
            Cal_label.imgtk = imgtk
            Cal_label.frame_num += 1.
            Cal_label.configure(image=imgtk)
            Cal_label.after(20, show_frames)

        def take_pic():
            #Take picture and save to folder
            save_path = './Cal_Images'
            file_name = f"{Cal_label.frame_num}.jpg"
            complete_name = os.path.join(save_path, file_name)

            imagetk = Cal_label.imgtk
            imgpil = ImageTk.getimage( imagetk )
            imgpil.save( complete_name, "PNG")
            imgpil.close()
            Pic_taken()

        def Pic_taken():
            #count pictures taken

            Fold_Path = './Cal_Images'
            count = 0
            for path in os.listdir(Fold_Path):
                if os.path.isfile(os.path.join(Fold_Path, path)):
                    count += 1
                        
            label_Pic_counter = Label(Top, text="Pictures: "+ str(count))
            label_Pic_counter.grid(row= 3, column= 0)
        
        #Calibrate function
        def Start_Calib():               
            nRows = 9
            nCols = 6
            dimension = 33 #- mm
            workingFolder   = './Cal_Images'
            imageType       = 'jpg'
            
            # termination criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, dimension, 0.001)

            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            objp = np.zeros((nRows*nCols,3), np.float32)
            objp[:,:2] = np.mgrid[0:nCols,0:nRows].T.reshape(-1,2)

            # Arrays to store object points and image points from all the images.
            objpoints = [] # 3d point in real world space
            imgpoints = [] # 2d points in image plane.

            if len(sys.argv) < 6:
                    print("\n Not enough inputs are provided. Using the default values.\n\n" \
                        " type -h for help")
            else:
                workingFolder   = sys.argv[1]
                imageType       = sys.argv[2]
                nRows           = int(sys.argv[3])
                nCols           = int(sys.argv[4])
                dimension       = float(sys.argv[5])

            if '-h' in sys.argv or '--h' in sys.argv:
                sys.exit()

            # Find the images files
            filename    = workingFolder + "/*." + imageType
            images      = glob.glob(filename)

            if len(images) < 9:
                sys.exit()
            else:
                nPatternFound = 0
                imgNotGood = images[1]

                for fname in images:
                    if 'calibresult' in fname: continue
                    #-- Read the file and convert in greyscale
                    img     = cv2.imread(fname)
                    gray    = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                    # Find the chess board corners
                    ret, corners = cv2.findChessboardCorners(gray, (nCols,nRows),None)

                    # If found, add object points, image points (after refining them)
                    if ret == True:
                        #--- Sometimes, Harris cornes fails with crappy pictures, so
                        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

                        # Draw and display the corners
                        cv2.drawChessboardCorners(img, (nCols,nRows), corners2,ret)
                        cv2.imshow('img',img)
  
                        nPatternFound += 1
                        objpoints.append(objp)
                        imgpoints.append(corners2)
                    else:
                        imgNotGood = fname

            cv2.destroyAllWindows()    
            
            if (nPatternFound > 25):
                
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

                # Undistort an image
                img = cv2.imread(imgNotGood)
                h,  w = img.shape[:2]
                newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

                # undistort
                mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
                dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

                # crop the image
                x,y,w,h = roi
                dst = dst[y:y+h, x:x+w] 
                cv2.imwrite("calibresult.png",dst)
                np.savetxt("./Calibration/cameraMatrix.txt", mtx, delimiter=',')
                np.savetxt("./Calibration/cameraDistortion.txt", dist, delimiter=',')
                
                #Finding the distortion value
                mean_error = 0
                for i in range(len(objpoints)):
                    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                    mean_error += error
                    
                #Print the distortion value
                Print_Error = round(mean_error/len(objpoints),4)
                label_Error.config(text = "Total error:\n" + str(Print_Error))
                
                #print the distortion Status value status
                if (Print_Error < 0,1):
                    Error_value =" OK "
                else:
                    Error_value = "Not OK"   
                Error_status.config(text = "Error Status:\n"+ Error_value)
                
                #Print the status of pictures taken to start calibrate
                ant_pic_status ="You have the number of pictures needed to calibrate"
                Ant_pic_err_lab1.config(text=ant_pic_status)  

            else:
                #Print the status of pictures taken to start calibrate
                ant_pic_status ="In order to calibrate you need at least 25 good pictures..."
                Ant_pic_err_lab1.config(text=ant_pic_status)
                
        show_frames()
        Pic_taken()
    Top.mainloop()
#--------------------------Calibration end--------------------------

        


#-----------------------------------------------------------------
#-----------------------------MAIN--------------------------------
              
if __name__ == "__main__":
    
    root = Tk()
    root.title("Computer vision position estimation") 
    root.iconbitmap('favicon.ico')     # Setts icon for app
    root.geometry("1580x760")
        
        
    #-----------DEFINING Widgets:---------------

    # Frames define:
    frame_0 = LabelFrame(root, text="", padx=10, pady=10)
    frame_1 = LabelFrame(root, text="Video", padx=3, pady=3)
    
    # Label naming boxes:
    label_vid_1 = Label(frame_1)
    label_camera = Label(frame_0, text="Camera Source")
    label_window_1 = Label(frame_0, text="Tracker 1")
    label_window_2 = Label(frame_0, text="Tracker 2")
    label_window_3 = Label(frame_0, text="Tracker 3")
    label_Delta_method = Label(frame_0, text="Delta method")
    label_port = Label(frame_0, text="Port:")
    label_ip = Label(frame_0, text="IP:")
    label_x = Label(frame_0, text="")
    label_x2 = Label(frame_0, text="")
    label_x3 = Label(frame_0, text="")
    label_UDP = Label(frame_0, text="UDP Output:")
    label_id = Label(frame_0, text="Aruco id")
    label_size = Label(frame_0, text="Aruco size")
    label_dictonary = Label(frame_0, text="Dictionary")

    # Statusbar define:
    statusbar_0 = Label(frame_0, text="Output: ", bd=2, relief=SUNKEN, anchor=W, bg='white')
    statusbar_1 = Label(frame_0, text="Tr.1: ", bd=2, relief=SUNKEN, anchor=W, bg='white')
    statusbar_2 = Label(frame_0, text="Tr.2: ", bd=2, relief=SUNKEN, anchor=W, bg='white')
    statusbar_3 = Label(frame_0, text="Tr.3: ", bd=2, relief=SUNKEN, anchor=W, bg='white')
    
    # Indicators Canvas
    Indicator_1 = Canvas(frame_0, width=20, height=20)  # Create 20x20 Canvas widget
    Indicator_2 = Canvas(frame_0, width=20, height=20)  # Create 20x20 Canvas widget
    Indicator_3 = Canvas(frame_0, width=20, height=20)  # Create 20x20 Canvas widget
    Indicator_4 = Canvas(frame_0, width=20, height=20)  # Create 20x20 Canvas widget
    
    my_oval_1 = Indicator_1.create_oval(4, 4, 18, 18)  # Create a circle on the Canvas
    my_oval_2 = Indicator_2.create_oval(4, 4, 18, 18)  # Create a circle on the Canvas
    my_oval_3 = Indicator_3.create_oval(4, 4, 18, 18)  # Create a circle on the Canvas
    my_oval_4 = Indicator_4.create_oval(4, 4, 18, 18)  # Create a circle on the Canvas

    # Buttons define:
    button_quit = Button(frame_0, text="Quit", padx=10, pady=2, command=root.quit)
    button_start_multiple = Button(frame_0, text="Start", padx=10, pady=2, command=lambda:Click_multi_start())
    button_stop_all = Button(frame_0, text="Stop All", padx=10, pady=2, command=lambda:Stop_all_trackers())
    button_calibrate = Button(frame_0, text="Calibrate", padx=10, pady=2, command=lambda:Cal_Click(),)#Stop_all_trackers()
    button_aruco = Button(frame_0, text ='Aruco Start', padx=5, pady=1, command=lambda:Aruco_Click())
    button_connect = Button(frame_0, text= "Connect", padx=10, pady=2, command=lambda:Connect_UDP_Click())
    
    # Dropdown menues
    camera_drop_1 = ttk.Combobox(frame_0, value = [0,1,2,3,5])
    camera_drop_1.current(0)
    tracker_drop_1 = ttk.Combobox(frame_0, value = ["0", "MEDIANFLOW", "CSRT", "MOSSE"])
    tracker_drop_1.current(1)
    tracker_drop_2 = ttk.Combobox(frame_0, value = ["0", "MEDIANFLOW", "CSRT", "MOSSE"])
    tracker_drop_2.current(0)
    tracker_drop_3 = ttk.Combobox(frame_0, value = ["0", "MEDIANFLOW", "CSRT", "MOSSE"])
    tracker_drop_3.current(0)
    delta_drop = ttk.Combobox(frame_0, value=["Marked object", "Senter screen"])
    delta_drop.current(0)
    aruco_lib_drop = ttk.Combobox(frame_0, width=7,value = ["4x4_100", "5x5_100", "Classic"])
    aruco_lib_drop.current(0)
    
    # Text entrys
    entry_port = Entry(frame_0, width=10 ) 
    entry_ip = Entry(frame_0, width=10 )
    entry_aruco_id = Entry(frame_0, width=10)
    entry_aruco_size = Entry(frame_0, width=10)
    #---------------------------------------------------------

    #---------------PLACING ON ROOT:--------------------------
    # Label placing:
    label_vid_1.grid(row=0,column=0)
    label_camera.grid(row=0,column=0)
    label_window_1.grid(row=1,column=0)
    label_window_2.grid(row=2,column=0)
    label_window_3.grid(row=3,column=0)
    label_Delta_method.grid(row=4,column=0)
    label_x.grid(row=12,column=0)
    label_x2.grid(row=7,column=0)
    label_x3.grid(row=8,column=0)
    label_UDP.grid(row=13,column=0)
    label_port.grid(row=14,column=0)
    label_ip.grid(row=15,column=0)
    label_id.grid(row=3, column=4)
    label_size.grid(row=4, column=4)
    label_dictonary.grid(row=5,column=4)

    # Dropdown menues:
    camera_drop_1.grid(row=0,column=1)
    tracker_drop_1.grid(row=1,column=1)
    tracker_drop_2.grid(row=2,column=1)
    tracker_drop_3.grid(row=3,column=1)
    delta_drop.grid(row=4,column=1)
    aruco_lib_drop.grid(row=5,column=5)

    # Buttons:
    button_quit.grid(row=15,column=6)
    button_start_multiple.grid(row=0,column=2)
    button_stop_all.grid(row=0,column=3)
    button_calibrate.grid(row=0, column=5)
    button_aruco.grid(row=2, column=5)
    button_connect.grid(row=15, column=2)

    # Frames:
    frame_0.grid(row=0,column=0,padx=2, pady=2)
    frame_1.grid(row=0,column=1,padx=2, pady=2)

    # Statusbar:
    statusbar_0.grid(row=11,column=0,columnspan=6, sticky=W+E)
    statusbar_1.grid(row=8,column=0,columnspan=6, sticky=W+E)
    statusbar_2.grid(row=9,column=0,columnspan=6, sticky=W+E)
    statusbar_3.grid(row=10,column=0,columnspan=6, sticky=W+E)
    
    # Canvas Indicator:
    Indicator_1.grid(row=8,column=6)
    Indicator_2.grid(row=9,column=6)
    Indicator_3.grid(row=10,column=6)
    Indicator_4.grid(row=2,column=6)
    
    # Entrys
    entry_port.grid(row=14,column=1)
    entry_ip.grid(row=15,column=1)
    entry_aruco_id.grid(row=3,column=5)
    entry_aruco_size.grid(row=4,column=5)
    
    
    #-------------------------------------------------------
 



    #-------------------------------------------------------
    #-----------------Commands------------------------------
    
    # List of trackers
    tList = []
    # Aruco list
    aList = []
    
    # Default camerasource if no tracker selected
    if len(tList) == 0:
        cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture("C:/Users/egrut/OneDrive/Dokumenter/Visual Studio 2019/pythonSaves/openCV/Video/TestRovRevVentil.mp4")
    

    Button_controll()
    
    Show_frames_one()
    Update_statusbar()
    Update_Indicators()
    Output_control()
    SendData()
  
    
    root.mainloop()
    
#---------------------------Main end----------------------------