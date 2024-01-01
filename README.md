# How to Run

1. git clone
2. pip install requirements.txt


## Calibration of Camera

1. Run python3 calib.py calibration_settings.yaml
2. Now you can use the checkerboard to calibrate camera
3. With this you will get the intrinsic and extrensic parameters 

## StereoVision

1. Run python3 stereoVision.py

## depth_maps 

1. Run python3 depth_maps.py
2. To quit press 'q'

## showbodypose and BodyPose

1. Run python3 bodypose.py camera1 camera2  (camera1 -> 1st camera number camera2 -> 2nd camera number ) 
2. run ls /dev/video* to get camera number when connected
3. To view run python3 show_3d_pose.py

# What this is
Stereo camera calibration script written in python. Uses OpenCV primarily. 

# Why stereo calibrate two cameras
Allows you to obtain 3D points through triangulation from two camera views.

# Setup

Clone the repository to your PC. Then navigate to the folder in your terminal. Also print out a calibration pattern. Make sure it is as flat as you can get it. Small warps in the calibration pattern results in very poor calibration. Also, the calibration pattern should be properly sized so that both cameras can see it clearly at the same time. Checkerboards can be generated [here](https://calib.io/pages/camera-calibration-pattern-generator).

**Install required packages**

This package uses ```python3.8```. Other versions might result in issues. Only tested on Linux.

Other required packages are:
```
OpenCV
pyYAML
scipy #only if you want to triangulate. 
```
Install required packages:
```
pip3 install -r requirements.txt
```

**Calibration settings**

The camera calibration settings first need to be configured in the ```calibration_settings.yaml``` file. 

```camera0```: Put primary camera device_id here. You can check available video devices on linux with ```ls /dev/video*```. You only need to put the device number.

```camera1```: Put secondary camera device_id here. 

```frame_width``` and ```frame_height```: Camera calibration is tied with the image resolution. Once this is set, your calibration result can only be used with this resolution. Also, both cameras have to have the exact same ```width``` and ```height```. If your cameras do not support the same resolution, use cv.resize() in opencv to make them same ```width``` and ```height```. This package does not check if your camera resolutions are the same or supported by your camera, and does not raise exception. It is up to you to make sure your cameras can support this resolution.

```mono_calibration_frames```: Number of frames to use to obtain intrinsic camera parameters. Default: 10.

```stereo_calibration_frames```: Number of frames to use to obtain extrinsic camera parameters. Default: 10.

```view_resize```: If you are using a single screen and cannot see both cameras because the images are too big, then set this to 2. This will show a smaller video feed but the saved frames will still be in full resolution.

```checkerboard_box_size_scale```: This is the size of calibration pattern box in real units. For example, my calibration pattern is 3.19cm per box.

```checkerboard_rows``` and ```checkerboard_columns```: Number of crosses in your checkerboard. This is NOT the number of boxes in your checkerboard. 
![image](https://user-images.githubusercontent.com/36071915/175003788-b2477a50-6d73-45e1-a037-a317269fa9c1.png)


# Procedure

Before running the code, make sure both cameras are in their final position. Once the cameras are calibrated, their positions must remain fixed. If the cameras move, then you need to recalibrate. However, only stereo calibration is necessary in this case(Step.3 and onwards).

Run the program by invoking:
```python3 calib.py calibration_settings.yaml```. 

The calibration procedures should take less than 10 minutes.

**Check the code to see each method call corresponding to the steps below.**


**Step 1. Saving Calibration Pattern Frames**

Step1 will create ```frames``` folder and save calibration pattern frames. The number of frames saved is set by ```mono_calibration_frames```. Press SPACE when ready to save frames.

Show the calibration pattern to each camera. Don't move it too far away. When a frame is taken, move the pattern to a differnt position and try to cover different parts of the frame. Keep the pattern steady when the frame is taken.

**Step2. Obtain Intrinsic Camera Parameters**

Step2 will open the saved frames and detect calibration pattern points. Visually check that the detected points are correct. If the detected points are poor, then press "s" on keyboard to skip this frame. Otherwise press any button to use the detected points.

A good detection should look like this:

![image](https://user-images.githubusercontent.com/36071915/175025899-9e3de806-9fec-4f3c-9019-2fadf4c8365a.png)

If your code does not detect the checkerboard pattern points, ensure that your calibration patterns are well lit, and all of the pattern can be seen by the camera. Ensure that the ```checkerboard_rows``` and ```checkerboard_columns``` in the ```calibration_settings.yaml``` file is correctly set. These are NOT the number of boxes in your checkerboard pattern. 

A good calibration should result in less then 0.3 RMSE. You should aim to obtain about .15 to 0.25 RMSE.

Once the code completes, a folder named ```camera_parameters``` is created and you should see ```camera0_intrinsics.dat``` and ```camera1_intrinsics.dat``` files. These contain the intrinsic parameters of the cameras. These only need to be calibrated once for each camera. If you change position of the cameras, this does not need to be recalibrated.

**Step3. Save Calibration Frames for Both Cameras**

Show the calibration pattern to both cameras at the same time. If your calibration pattern is small or too far, you will get poor calibration. Keep the patterns very steady. Press SPACE when ready to take the frames.

The paired images will be saved in a new folder: ```frames_pair```.

**Step4. Obtain Camera0 to Camera1 Rotation and Translation**

Use the paired calibration pattern images to obtain the rotation matrix R and translation vector T that transforms points in Camera0 coordinate space to camera1 coorindate space. As before, visually ensure that detected points are correct. If the detected points are poor in any frame, press "s" to skip this pair. 

You should see something like this.

![image](https://user-images.githubusercontent.com/36071915/175031465-ddf0b965-4a4f-4983-b741-36f541bdf108.png)

Once the code completes, rotation R and translation T are returned. A good calibration should have RMSE < 0.3. Values up to 0.5 can be acceptable. If your RMSE value is too high, make sure that when taking the paired frames, you keep your hand steady. Also make sure that the calibration pattern is not too small or too far away. Keep repeating this step until a good RMSE value is obtained.

**Step5. Obtain Stereo Calibration Extrinsic Parameters**

R and T alone are not enough to triangulate a 3D point. We need to define a world space origin point and orientation. The easiest way to do this is to simply choose Camera0 position as world space origin. In general, the camera0 coordinate system is defined to be behind the camera screen:

![Camera0 coordinate system](https://docs.opencv.org/4.x/pinhole_camera_model.png)

Thus, the world origin to camera0 rotation is identity matrix and translation is a zeros vector. Then R, T obtained from previous step becomes rotation and translation from world origin to camera1. Practically what this means is that your 3D triangulated points will be with respect to the coordinate systemn sitting behind your camera0 lens, as shown above. 

Step5 code will do all of this and save ```camera0_rot_trans.dat``` and ```camera1_rot_trans.dat``` in ```camera_parameters``` folder. This completes stereo calibration. You get intrinsic and extrinsic parameters for both cameras.

As final step, Step5 shows coordinate axes shifted 60cm forward in both camera views. Since I know that the axes are shifted 60cm forward, I can check it using a tape set to 60cm. You can see that both cameras are in good alignment. This is however not a good way to check your calibration. You should try to aim for RMSE < 0.5.

If you see it in camera0 and not camera1, then change ```_zshift``` to some value that you know both cameras can see. 

 **Real time 3D body pose estimation using MediaPipe**

This is a demo on how to obtain 3D coordinates of body keypoints using MediaPipe and two calibrated cameras. Two cameras are required as there is no way to obtain global 3D coordinates from a single camera. For camera calibration, my package on github [stereo calibrate](https://github.com/TemugeB/python_stereo_camera_calibrate), my blog post on how to stereo calibrate two cameras: [link](https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html). Alternatively, follow the camera calibration at Opencv documentations: [link](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html). If you want to know some details on how this code works, take a look at my accompanying blog post here: [link](https://temugeb.github.io/python/computer_vision/2021/06/27/handpose3d.html).

![input1](media/cam0_kpts.gif "input1") ![input2](media/cam1_kpts.gif "input2") 
![output](media/pose2.gif "output")

**MediaPipe**  
Install mediapipe in your virtual environment using:
```
pip install mediapipe
```

**Requirements**  
```
Mediapipe
Python3.8
Opencv
matplotlib
```

**Usage: Getting real time 3D coordinates**  
As a demo, I've included two short video clips and corresponding camera calibration parameters. Simply run as:
```
python bodypose3d.py
```
If you want to use webcam, call the program with camera ids. For example, cameras registered to 0 and 1:
```
python bodypose3d.py 0 1
```
Make sure the corresponding camera parameters are also updated for your cameras in ```camera_parameters``` folder. My cameras were calibrated to 720px720p. The code crops the input image to this size. If your cameras are calibrated to a different resolution, make sure to change the code to your camera calibration. Also, if your cameras are different aspect ratios (i.e. 16:10, 16:9 etc), then remove the cropping calls in the code. Mediapipe crops your images and resizes them so it doesn't care if youre cameras are calibrated to 1080p, 720p or any other resolution. 

The 3D coordinate in each video frame is recorded in ```frame_p3ds``` parameter. Use this for real time application. The keypoints are indexed as below image. More keypoints can be added by including their ids at the top of the file. If keypoints are not found, then the keypoints are recorded as (-1, -1, -1). **Warning**: The code also saves keypoints for all previous frames. If you run the code for long periods, then you will run out of memory. To fix this, remove append calls to: ```kpts_3d, kpts_cam0. kpts_cam1```. When you press the ESC key, body keypoints detection will stop and three files will be saved to disk. These contain recorded 2D and 3D coordinates. 

![output](media/keypoints_ids.png "keypoint_ids")

**Usage: Viewing 3D coordinates**  
The ```bodypose3d.py``` program creates a 3D coordinates file: ```kpts_3d.dat```. To view the recorded 3D coordinates, simply call:
```
python show_3d_pose.py
```




