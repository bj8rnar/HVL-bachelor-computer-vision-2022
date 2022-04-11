
# Test Kirieg

from tkinter import *
from tkinter import ttk
from tracemalloc import stop
from turtle import update        # To use combobox
from PIL import ImageTk, Image
from click import command
import cv2
import numpy as np
import os
import sys
import glob
from scipy.__config__ import show

tms = 80    #Times pr milliscond
color_list = [
        [255, 0, 0],  # blue
        [255, 255, 0],  # aqua
        [0, 255, 0],  # lime
        [128, 0, 128],  # purple
        [0, 0, 255],  # red
        [255, 0, 255],  # fuchsia
        [0, 128, 0],  # green
        [128, 128, 0],  # teal
        [0, 0, 128],  # maroon
        [0, 128, 128],  # olive
        [0, 255, 255],  # yellow
    ]

class Tracker:  ####################################
    
    def __init__(self, tracker_type, bbox, video_capture, color):
        
        self.bbox = bbox
        self.tracker_type = tracker_type 
        self.cap = video_capture
        self.tracker_running = False
        self.color = color
        
        self.dx = 0.0
        self.dy = 0.0
        self.dz = 0.0
               
        if self.tracker_type == 'BOOSTING':
            self.tracker = cv2.legacy.TrackerBoosting_create()           
        if self.tracker_type == 'MIL':
            self.tracker = cv2.TrackerMIL_create()
        if self.tracker_type == 'KCF':
            self.tracker = cv2.legacy.TrackerKCF_create()
        if self.tracker_type == 'TLD':
            self.tracker = cv2.legacy.TrackerTLD_create()
        if self.tracker_type == 'MEDIANFLOW':
            self.tracker = cv2.legacy.TrackerMedianFlow_create()
        if self.tracker_type == 'MOSSE':
            self.tracker = cv2.legacy.TrackerMOSSE_create()
        if self.tracker_type == "CSRT":
            self.tracker = cv2.legacy.TrackerCSRT_create()
        if self.tracker_type == 'DaSiamRPN':
            params = cv2.TrackerDaSiamRPN_Params()
            params.model = "model/DaSiamRPN/dasiamrpn_model.onnx"
            params.kernel_r1 = "model/DaSiamRPN/dasiamrpn_kernel_r1.onnx"
            params.kernel_cls1 = "model/DaSiamRPN/dasiamrpn_kernel_cls1.onnx"
            self.tracker = cv2.TrackerDaSiamRPN_create(params)  

        # Exit if video not opened.
        if not self.cap.isOpened():
            print ("Could not open video")
            sys.exit()
        
        # Read first frame.
        self.ok, self.frame = self.cap.read()
        if not self.ok:
            print ('Cannot read video file')
            sys.exit()

        if delta_drop.get() == "Marked object":
            self.Create_offsett_from_ref_bbox()
        elif delta_drop.get() == "Senter screen":
            self.Create_offsett_from_senter_screen()
        
    def Run(self):
    
        if self.tracker_running == True:
            self.ok = self.tracker.init(self.frame, self.bbox)
            
            # Read a new frame
            self.cap
            self.ok, self.frame = self.cap.read()
                    
            # Start timer
            timer = cv2.getTickCount()
            # Update tracker
            self.ok, self.bbox = self.tracker.update(self.frame)
            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            # Draw bounding box
            if self.ok:
                # Tracking success
                self.p1 = (int(self.bbox[0]), int(self.bbox[1]))
                self.p2 = (int(self.bbox[0] + self.bbox[2]), int(self.bbox[1] + self.bbox[3]))
                cv2.rectangle(self.frame, self.p1, self.p2, (0,255,0), 2, 1)       
                # Centerpoint bbox
                self.centerXbbox = int(self.bbox[0]+(self.bbox[2] / 2))
                self.centerYbbox = int(self.bbox[1]+(self.bbox[3] / 2))
                cv2.rectangle(self.frame, (self.centerXbbox,self.centerYbbox), (self.centerXbbox,self.centerYbbox), (0,255,0), 6, 1) #centerpoint  
                 
                if delta_drop.get() == "Marked object":
                    self.Sett_offsett_from_ref_bbox()
                elif delta_drop.get() == "Senter screen":
                    self.Sett_offsett_from_senter_screen()
                
           # else :
                # Tracking failure
                #cv2.putText(self.frame, "Tracking failure detected", (80,90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255),2)

            # Display tracker type on frame
            #cv2.putText(self.frame, self.tracker_type + " Tracker", (80,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (50,170,50),2)
            # Display FPS on frame
            #cv2.putText(self.frame, "FPS : " + str(int(fps)), (80,120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
            
            root.after(tms, self.Run)
    
     
    def Start(self):       
        #self.click_new_bbox()
        self.tracker_running = True
        self.Run()
        #show_frames_one()
        #self.show_frames()
    
    def Stop_tracker(self):
        self.tracker_running = False
        
    # Calculate offsett from the center og where det bbox was first sett:
    def Create_offsett_from_ref_bbox(self):
         # Defining corner p1 and p2 for refferance box
        self.refBox = self.bbox
        self.refP1 = (int(self.refBox[0]), int(self.refBox[1]))
        self.refP2 = (int(self.refBox[0] + self.refBox[2]), int(self.refBox[1] + self.refBox[3]))
        # Senter of refferenca Box:
        self.centerXrefBox = int(self.refBox[0]+(self.refBox[2] / 2)) 
        self.centerYrefBox = int(self.refBox[1]+(self.refBox[3] / 2))
    
    def Sett_offsett_from_ref_bbox(self):
        # Refbox + centerpoint
        cv2.rectangle(self.frame, self.refP1, self.refP2, (0,0,255), 2, 1)
        cv2.rectangle(self.frame, (self.centerXrefBox, self.centerYrefBox), (self.centerXrefBox,self.centerYrefBox), (0,0,255), 6, 1) # Centerpoint
        # Delta
        self.dx = float(self.centerXrefBox - self.centerXbbox)
        self.dy = float(self.centerYbbox - self.centerYrefBox)
        self.dz = float(self.refBox[3] - self.bbox[3])

    # Calculate offsett from the senter of the screen
    def Create_offsett_from_senter_screen(self):
        # Refference to first bbox    
        self.refBox = self.bbox   
        # Frame dimension
        frameHeight = int(self.frame.shape[0])
        frameWidth = int(self.frame.shape[1])
        # Center of screen:
        self.centerFrameY = int(frameHeight/2)
        self.centerFrameX = int(frameWidth/2)
        
    def Sett_offsett_from_senter_screen(self):
         # Center frame + point
        refP1 = (int(self.centerFrameX - self.refBox[2]/2) , int(self.centerFrameY - self.refBox[3]/2)) 
        refP2 = (int(self.centerFrameX + self.refBox[2]/2) , int(self.centerFrameY + self.refBox[3]/2))
        cv2.rectangle(self.frame, refP1, refP2, (0,0,255), 2, 1) 
        cv2.rectangle(self.frame, (self.centerFrameX, self.centerFrameY), (self.centerFrameX, self.centerFrameY), (0,0,255), 6, 1)
        
        # Delta
        self.dx = float(self.centerXbbox - self.centerFrameX)
        self.dy = float(self.centerFrameY - self.centerYbbox)
        self.dz = float(self.refBox[3] - self.bbox[3])
    
            
##############################

def show_frames_one():
    
    ok, frame = cap.read() 
    if ok:
        for obj in t:                
            if obj.tracker_running:
                    cv2.rectangle(frame, obj.p1, obj.p2, obj.color, thickness=2)
                
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   #A???
        img = Image.fromarray(cv2image).resize((800, 600))
        imgtk = ImageTk.PhotoImage(image = img)       
        label_vid_1.imgtk = imgtk
        label_vid_1.configure(image=imgtk)
    
    root.after(tms, show_frames_one)

def new_ROI():
    ok, cv2_image = cap.read()
    if ok:
        bbox = cv2.selectROI(cv2_image)
        statusbar_0.config(text = "Status:\t" + ": " + str(bbox))
        cv2.destroyWindow('ROI selector')
    return bbox

def click_multi_start():
    #cap = cv2.VideoCapture(Camera_Select())  
    bbox = new_ROI()
    
    if tracker_drop_1.get() != "0":
        t.append(Tracker(tracker_drop_1.get(), bbox, cap, color_list[0]))
    if tracker_drop_2.get() != "0":
        t.append(Tracker(tracker_drop_2.get(), bbox ,cap, color_list[1]))
    if tracker_drop_3.get() != "0":
        t.append(Tracker(tracker_drop_3.get(), bbox ,cap, color_list[2]))
        
    for obj in t:
        obj.Start()
    
def Camera_Select():
    return int(camera_drop_1.get())

def Stop_all_trackers():
    for obj in t:
        obj.Stop_tracker()
    t.clear()
        
def Update_statusbar():
    if len(t) == 1:
        statusbar_1.config(text = "Delta T1:\t" + t[0].tracker_type +"\t x: " + str(t[0].dx) + "\ty: " + str(t[0].dy) + "\tz: " + str("{:.3}".format(t[0].dz)) )
    if len(t) == 2:
        statusbar_1.config(text = "Delta T1:\t " + t[0].tracker_type +"\t x: " + str(t[0].dx) + "\ty: " + str(t[0].dy) + "\tz: " + str("{:.3}".format(t[0].dz)) )
        statusbar_2.config(text = "Delta T2:\t " + t[1].tracker_type +"\t x: " + str(t[1].dx) + "\ty: " + str(t[1].dy) + "\tz: " + str("{:.3}".format(t[1].dz)) )
    if len(t) == 3:
        statusbar_1.config(text = "Delta T1:\t " + t[0].tracker_type +"\t x: " + str(t[0].dx) + "\ty: " + str(t[0].dy) + "\tz: " + str("{:.3}".format(t[0].dz)) )
        statusbar_2.config(text = "Delta T2:\t " + t[1].tracker_type +"\t x: " + str(t[1].dx) + "\ty: " + str(t[1].dy) + "\tz: " + str("{:.3}".format(t[1].dz)) )
        statusbar_3.config(text = "Delta T3:\t " + t[2].tracker_type +"\t x: " + str(t[2].dx) + "\ty: " + str(t[2].dy) + "\tz: " + str("{:.3}".format(t[2].dz)) )
        
    root.after(100,Update_statusbar)
        
    #For å vise webcam i label_webcam
# def show_webcam():
#     frame = cap.read()[1]
#     cv2image   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
#     img   = Image.fromarray(cv2image).resize((480, 360))
#     imgtk = ImageTk.PhotoImage(image = img)
#     label_vid_1.imgtk = imgtk
#     label_vid_1.configure(image=imgtk)
#     root.after(45, show_webcam)        
   #pass


######################### - Calibrate Program - ################

def Cal_Click():
    
    workingFolder   = 'C:/Users/egrut/PyProjects/Tkinter GUI/TrackGUI/tester/Cal_Images'
    #"*/Cal_Images"
    imageType       = 'JPG'
       
    #Creat new window    
    Top = Toplevel()
    Top.title('Calibrate')
    #start camera button
    global start_cam_cal_btn
    start_cam_cal_btn = Button(Top, text= "Start Camera", command= lambda: start_cam_cal())
    start_cam_cal_btn.grid(row=0, column=0, padx= 10)
    #Exit Button 
    exit_top_btn = Button(Top,text="Exit", command=lambda:Top.destroy())
    exit_top_btn.grid(row=4, column=0)
    #Camera Frame 
    frame_Cal = LabelFrame(Top, text= "Camera", padx= 5, pady= 5 )
    frame_Cal.grid(row=0, column=2, rowspan= 8)


    def start_cam_cal():
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
            file_name = f"{Cal_label.frame_num}.jpg"
            imagetk = Cal_label.imgtk
            imgpil = ImageTk.getimage( imagetk )
            imgpil.save(workingFolder + file_name, imageType)
            imgpil.close()
            Pic_taken()
         
      
        def Pic_taken():    
            #count pictures taken
            count = 0
            for path in os.listdir(workingFolder):
                if os.path.isfile(os.path.join(workingFolder, path)):
                    count += 1              
            label_Pic_counter = Label(Top, text="Pictures: "+ str(count))
            label_Pic_counter.grid(row= 3, column= 0)
        
        #Calibrate function
        def Start_Calib():                 
            nRows = 9
            nCols = 6
            dimension = 33 #- mm
            #------------------------------------------

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
                print("\n IMAGE CALIBRATION GIVEN A SET OF IMAGES")
                print(" call: python cameracalib.py <folder> <image type> <num rows (9)> <num cols (6)> <cell dimension (25)>")
                print("\n The script will look for every image in the provided folder and will show the pattern found." \
                    " User can skip the image pressing ESC or accepting the image with RETURN. " \
                    " At the end the end the following files are created:" \
                    "  - cameraDistortion.txt" \
                    "  - cameraMatrix.txt \n\n")

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
                cv2.imwrite("/Calibration/calibresult.png",dst)
                np.savetxt("Calibration/cameraMatrixWebcam.txt", mtx, delimiter=',')
                np.savetxt("Calibration/cameraDistortionWebcam.txt", dist, delimiter=',')
                
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
#Calibrate end
###############################################################







############# - MAIN - ########################################
              
if __name__ == "__main__":
    
    root = Tk()
    root.title("Computer vision position estimation") 
    root.iconbitmap('c:/users/egrut/pyprojects/Tkinter GUI/TrackGUI/wrench.ico')     # Setts icon for app
    root.geometry("1500x700")
        
        
    #######     DEFINING Widgets:    #########

    # Frames define:
    frame_0 = LabelFrame(root, text="", padx=10, pady=10)
    frame_1 = LabelFrame(root, text="Video", padx=3, pady=3)

    # Label naming boxes:
    label_camera = Label(frame_0, text="Camera Source")
    label_window_1 = Label(frame_0, text="Tracker 1")
    label_window_2 = Label(frame_0, text="Tracekr 2")
    label_window_3 = Label(frame_0, text="Tracker 3")
    label_Delta_method = Label(frame_0, text="Delta method")

    # Statusbar define:
    statusbar_0 = Label(frame_0, text="Ref:: ", bd=2, relief=SUNKEN, anchor=W, bg='white')
    statusbar_1 = Label(frame_0, text="Tr.1: ", bd=2, relief=SUNKEN, anchor=W, bg='white')
    statusbar_2 = Label(frame_0, text="Tr.2: ", bd=2, relief=SUNKEN, anchor=W, bg='white')
    statusbar_3 = Label(frame_0, text="Tr.3: ", bd=2, relief=SUNKEN, anchor=W, bg='white')

    # Buttons define:
    button_quit = Button(frame_0, text="Quit", padx=10, pady=2, command=root.quit)
    button_start_multiple = Button(frame_0, text="Start", padx=10, pady=2, command=lambda:click_multi_start())
    button_stop_all = Button(frame_0, text="Stop All", padx=10, pady=2, command=lambda:Stop_all_trackers())
    button_calibrate = Button(frame_0, text="Calibrate", padx=10, pady=2, command=lambda:[Cal_Click(),Stop_all_trackers()])
    
    label_vid_1 = Label(frame_1)
    label_vid_1.grid(row=0,column=0)

    camera_drop_1 = ttk.Combobox(frame_0, value = [0,1,2,3,5])
    camera_drop_1.current(0)
    tracker_drop_1 = ttk.Combobox(frame_0, value = ["0","BOOSTING","MIL","KCF", "MOSSE", "MEDIANFLOW", "CSRT", "DaSiamRPN"])
    tracker_drop_1.current(4)
    tracker_drop_2 = ttk.Combobox(frame_0, value = ["0","BOOSTING","MIL","KCF", "MOSSE", "MEDIANFLOW", "CSRT", "DaSiamRPN"])
    tracker_drop_2.current(5)
    tracker_drop_3 = ttk.Combobox(frame_0, value = ["0","BOOSTING","MIL","KCF", "MOSSE", "MEDIANFLOW", "CSRT", "DaSiamRPN"])
    tracker_drop_3.current(0)
    delta_drop = ttk.Combobox(frame_0, value=["Marked object", "Senter screen"])
    delta_drop.current(0)

    ##########################################

    #########   PLACING ON ROOT:   ###########
    # Label plassering:
    label_camera.grid(row=0,column=0)
    label_window_1.grid(row=1,column=0)
    label_window_2.grid(row=2,column=0)
    label_window_3.grid(row=3,column=0)
    label_Delta_method.grid(row=4,column=0)

    # Dropdown menues:
    camera_drop_1.grid(row=0,column=1)
    tracker_drop_1.grid(row=1,column=1)
    tracker_drop_2.grid(row=2,column=1)
    tracker_drop_3.grid(row=3,column=1)
    delta_drop.grid(row=4,column=1)

    # Buttons:
    button_quit.grid(row=6,column=5)
    button_start_multiple.grid(row=0,column=3)
    button_stop_all.grid(row=0,column=4)
    button_calibrate.grid(row=0, column=5)

    # Frames:
    frame_0.grid(row=0,column=0,padx=2, pady=2)
    frame_1.grid(row=0,column=1,padx=2, pady=2)

    # Statusbar:
    statusbar_0.grid(row=7,column=0,columnspan=6, sticky=W+E)
    statusbar_1.grid(row=8,column=0,columnspan=6, sticky=W+E)
    statusbar_2.grid(row=9,column=0,columnspan=6, sticky=W+E)
    statusbar_3.grid(row=10,column=0,columnspan=6, sticky=W+E)
 
    #########################################
 
 
    ############# Commands ##################
    
    t = []
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture("C:/Users/egrut/OneDrive/Dokumenter/Visual Studio 2019/pythonSaves/openCV/Video/TestRovRevVentil.mp4")
    
    Update_statusbar()
    
    show_frames_one()
 
    
    
    root.mainloop()
