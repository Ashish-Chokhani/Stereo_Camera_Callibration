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

 




