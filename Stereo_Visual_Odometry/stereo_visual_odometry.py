import os
import numpy as np
import cv2
import time
from scipy.optimize import least_squares
from time import perf_counter
from matplotlib import pyplot as plt
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
import pytransform3d.rotations as pr
import pytransform3d.camera as pc
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from cycler import cycle


class VisualOdometry():
    def __init__(self, data_dir):
        self.K_l, self.P_l, self.K_r, self.P_r = self._load_calib(data_dir + './calib.txt')

        block = 11
        P1 = block * block * 8
        P2 = block * block * 32
        self.disparity = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=block, P1=P1, P2=P2)
        
        self.fastFeatures = cv2.FastFeatureDetector_create()

        self.lk_params = dict(winSize=(15, 15),
                              flags=cv2.MOTION_AFFINE,
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

    @staticmethod
    def _load_calib(filepath):

        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_l = np.reshape(params, (3, 4))
            K_l = P_l[0:3, 0:3]
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_r = np.reshape(params, (3, 4))
            K_r = P_r[0:3, 0:3]
        return K_l, P_l, K_r, P_r


    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        # Get the rotation vector
        r = dof[:3]
        # Create the rotation matrix from the rotation vector
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = dof[3:]
        # Create the transformation matrix from the rotation matrix and translation vector
        transf = self._form_transf(R, t)

        # Create the projection matrix for the i-1'th image and i'th image
        f_projection = np.matmul(self.P_l, transf)
        b_projection = np.matmul(self.P_l, np.linalg.inv(transf))

        print("Q2",Q2)
        # Make the 3D points homogenize
        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        print("ff",f_projection.T)
        # Project 3D points from i'th image to i-1'th image
        q1_pred = Q2.dot(f_projection.T)
        # Un-homogenize
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

        # Project 3D points from i-1'th image to i'th image
        q2_pred = Q1.dot(b_projection.T)
        # Un-homogenize
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

        
        # Calculate the residuals
        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
        print("res",residuals)
        return residuals

    def get_tiled_keypoints(self, img, tile_h, tile_w):
        def get_kps(x, y):
            # Get the image tile
            impatch = img[y:y + tile_h, x:x + tile_w]

            # Detect keypoints
            keypoints = self.fastFeatures.detect(impatch)
            
            # print("keypoints",keypoints)
            
            # Correct the coordinate for the point
            for pt in keypoints:
                pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

            # Get the 10 best keypoints
            if len(keypoints) > 10:
                keypoints = sorted(keypoints, key=lambda x: -x.response)
                return keypoints[:10]
            return keypoints
        # Get the image height and width
        h, w, *_ = img.shape

        # Get the keypoints for each of the tiles
        kp_list = [get_kps(x, y) for y in range(0, h, tile_h) for x in range(0, w, tile_w)]

        # Flatten the keypoint list
        kp_list_flatten = np.concatenate(kp_list)
        # print("till  ",kp_list)
        return kp_list_flatten

    def track_keypoints(self, img1, img2, kp1, max_error=4):
        # Convert the keypoints into a vector of points and expand the dims so we can select the good ones
        trackpoints1 = np.expand_dims(cv2.KeyPoint_convert(kp1), axis=1)

        # Use optical flow to find tracked counterparts
        trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None, **self.lk_params)

        # Convert the status vector to boolean so we can use it as a mask
        trackable = st.astype(bool)

        # Create a maks there selects the keypoints there was trackable and under the max error
        under_thresh = np.where(err[trackable] < max_error, True, False)

        # Use the mask to select the keypoints
        trackpoints1 = trackpoints1[trackable][under_thresh]
        trackpoints2 = np.around(trackpoints2[trackable][under_thresh])

        # Remove the keypoints there is outside the image
        h, w = img1.shape
        in_bounds = np.where(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w), True, False)
        trackpoints1 = trackpoints1[in_bounds]
        trackpoints2 = trackpoints2[in_bounds]

        return trackpoints1, trackpoints2

    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=0.0, max_disp=100.0):
        def get_idxs(q, disp):
            q_idx = q.astype(int)
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
            return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)
        
        # Get the disparity's for the feature points and mask for min_disp & max_disp
        disp1, mask1 = get_idxs(q1, disp1)
        disp2, mask2 = get_idxs(q2, disp2)
        
        # Combine the masks 
        in_bounds = np.logical_and(mask1, mask2)
        
        # Get the feature points and disparity's there was in bounds
        q1_l, q2_l, disp1, disp2 = q1[in_bounds], q2[in_bounds], disp1[in_bounds], disp2[in_bounds]
        
        # Calculate the right feature points 
        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2
        
        return q1_l, q1_r, q2_l, q2_r

    def calc_3d(self, q1_l, q1_r, q2_l, q2_r):
        print("q1_l",type(q1_l.T))
        print("q1_r",type(q1_r.T))
        # Triangulate points from i-1'th image
        Q1 = cv2.triangulatePoints(self.P_l, self.P_r, q1_l.T, q1_r.T)
        # Un-homogenize
        Q1 = np.transpose(Q1[:3] / Q1[3])

        # Triangulate points from i'th image
        Q2 = cv2.triangulatePoints(self.P_l, self.P_r, q2_l.T, q2_r.T)
        # Un-homogenize
        Q2 = np.transpose(Q2[:3] / Q2[3])
        
        
        return Q1, Q2

    def estimate_pose(self, q1, q2, Q1, Q2, max_iter=100):
        early_termination_threshold = 5

        # Initialize the min_error and early_termination counter
        min_error = float('inf')
        early_termination = 0

        for _ in range(max_iter):
            # Choose 6 random feature points
            sample_idx = np.random.choice(range(q1.shape[0]), 6)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]

            # Make the start guess
            in_guess = np.zeros(6)
            # Perform least squares optimization
            print(type(self.reprojection_residuals))
            opt_res = least_squares(self.reprojection_residuals, in_guess, method='lm', max_nfev=200,
                                    args=(sample_q1, sample_q2, sample_Q1, sample_Q2))

            # Calculate the error for the optimized transformation
            error = self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2)
            error = error.reshape((Q1.shape[0] * 2, 2))
            error = np.sum(np.linalg.norm(error, axis=1))

            # Check if the error is less the the current min error. Save the result if it is
            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1
            if early_termination == early_termination_threshold:
                # If we have not fund any better result in early_termination_threshold iterations
                break

        # Get the rotation vector
        r = out_pose[:3]
        # Make the rotation matrix
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = out_pose[3:]
        # Make the transformation matrix
        transformation_matrix = self._form_transf(R, t)
        return transformation_matrix

    def get_pose(self, old_imgL, old_imgR, new_imgL, new_imgR):
        # Get the i-1'th image and i'th image
        img1_1 = old_imgL
        img2_1 = new_imgL

        # Get teh tiled keypoints
        kp1_1 = self.get_tiled_keypoints(img1_1, 10, 20)

        # Track the keypoints
        tp1_1, tp2_1 = self.track_keypoints(img1_1, img2_1, kp1_1)

        print("dis",self.disparity)
        print("l",old_imgL.shape)
        print("r",old_imgR.shape)
        # Calculate the disparitie
        old_disp = np.divide(self.disparity.compute(old_imgL, old_imgR).astype(np.float32),16)
        new_disp = np.divide(self.disparity.compute(new_imgL, new_imgR).astype(np.float32),16)

        # Calculate the right keypoints
        tp1_1, tp1_r, tp2_1, tp2_r = self.calculate_right_qs(tp1_1, tp2_1, old_disp, new_disp)

        # Calculate the 3D points
        Q1, Q2 = self.calc_3d(tp1_1, tp1_r, tp2_1, tp2_r)
        
        print("get_Q2",Q2)
        # Estimate the transformation matrix
        transformation_matrix = self.estimate_pose(tp1_1, tp2_1, Q1, Q2)
        return transformation_matrix



skip_frames = 2
data_dir = ''
vo = VisualOdometry(data_dir)

gt_path = []
estimated_path = []
camera_pose_list = []
start_pose = np.ones((3,4))
start_translation = np.zeros((3,1))
start_rotation = np.identity(3)
start_pose = np.concatenate((start_rotation, start_translation), axis=1)


cap1 = cv2.VideoCapture(2)
cap2 = cv2.VideoCapture(4   )

cap1.set(3, 1280)
cap1.set(4, 720)
cap2.set(3, 1280)
cap2.set(4, 720)

process_frames = False
old_frame_left = None
old_frame_right = None
new_frame_left = None
new_frame_right = None
frame_counter = 0

cur_pose = start_pose

start_time = time.perf_counter()

while(cap1.isOpened() and cap2.isOpened()):
    
    ret1, new_frame_left = cap1.read()
    ret2, new_frame_right = cap2.read()
    
    print("left",new_frame_left)
    
    new_frame_left_gray = cv2.cvtColor(new_frame_left, cv2.COLOR_BGR2GRAY)
    new_frame_right_gray = cv2.cvtColor(new_frame_right, cv2.COLOR_BGR2GRAY)
    frame_counter += 1
    
    start = time.perf_counter()
    
    if process_frames and ret1 and ret2:
        
        transf = vo.get_pose(old_frame_left,old_frame_right, new_frame_left_gray,new_frame_right_gray)
        
        cur_pose = cur_pose @ transf
        
        hom_array = np.array([[0, 0, 0, 1]])
        hom_camera_pose = np.concatenate((cur_pose,hom_array), axis=0)
        camera_pose_list.append(hom_camera_pose)
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
        
    elif process_frames and ret1 is False:
        break
    
    old_frame_left = new_frame_left_gray
    old_frame_right = new_frame_right_gray
    
    if(time.perf_counter() - start_time > 5):
        process_frames = True
        
    print(process_frames)
    
    end = time.perf_counter()
    
    total_time = end - start
    fps = 1/total_time
    
    cv2.putText(new_frame_left, f'FPS: {int(fps)}',(20,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

    cv2.putText(new_frame_left, str(np.round(cur_pose[0, 0],2)),(260,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
    cv2.putText(new_frame_left, str(np.round(cur_pose[0, 1],2)),(340,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
    cv2.putText(new_frame_left, str(np.round(cur_pose[0, 2],2)),(420,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
    cv2.putText(new_frame_left, str(np.round(cur_pose[1, 0],2)),(260,90), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
    cv2.putText(new_frame_left, str(np.round(cur_pose[1, 1],2)),(340,90), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
    cv2.putText(new_frame_left, str(np.round(cur_pose[1, 2],2)),(420,90), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
    cv2.putText(new_frame_left, str(np.round(cur_pose[2, 0],2)),(260,130), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
    cv2.putText(new_frame_left, str(np.round(cur_pose[2, 1],2)),(340,130), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
    cv2.putText(new_frame_left, str(np.round(cur_pose[2, 2],2)),(420,130), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
    cv2.putText(new_frame_left, str(np.round(cur_pose[0, 3],2)),(540,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
    cv2.putText(new_frame_left, str(np.round(cur_pose[1, 3],2)),(540,90), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
    cv2.putText(new_frame_left, str(np.round(cur_pose[2, 3],2)),(540,130), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)

    cv2.imshow("img", new_frame_left)
    cv2.imshow("img2", new_frame_right)
    
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
cap1.release()
cap2.release()
    
cv2.destroyAllWindows()

print("done")


number__of_frames = 20
image_size = np.array([1280, 720])

plt.figure()
ax=plt.axes(projection='3d')

print("3d done")
print(camera_pose_list)

camera_pose_poses=np.array(camera_pose_list)

print(camera_pose_poses)

key_frames_indices=np.linspace(0,len(camera_pose_poses)-1,number__of_frames,dtype=int)
colors=cycle("rgb")


for i,c in zip(key_frames_indices,colors):
    print(vo.K_l,camera_pose_poses[i])
    pc.plot_camera(ax,vo.K_l,camera_pose_poses[i],sensor_size=image_size, c=c)
    
plt.show()

take_every_th_camera_pose=2

estimated_path = np.array(estimated_path[:: take_every_th_camera_pose])

plt.plot(estimated_path[:,0],estimated_path[:,1])
plt.show