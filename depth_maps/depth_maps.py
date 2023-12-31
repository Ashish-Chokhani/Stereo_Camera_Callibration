import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

cap_left = cv.VideoCapture(2)
cap_right = cv.VideoCapture(4)

# Check if the cameras are opened successfully
if not cap_left.isOpened() or not cap_right.isOpened():
    print("Error: Could not open cameras.")
    exit()

while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    
    if not ret_left or not ret_right:
        print("Error: Could not read frames.")
        break

    cv.imshow('Left Camera', frame_left)
    cv.imshow('Right Camera', frame_right)

    # Save the frames as left and right images
    cv.imwrite('left_image.jpg', frame_left)
    cv.imwrite('right_image.jpg', frame_right)

    # Break the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv.destroyAllWindows()



# left_image = cv.imread('../test_image/tsukuba_l.png', cv.IMREAD_GRAYSCALE)
# right_image = cv.imread('../test_image/tsukuba_r.png', cv.IMREAD_GRAYSCALE)

left_image = cv.imread('../test_image/left_image.jpg', cv.IMREAD_GRAYSCALE)
right_image = cv.imread('../test_image/right_image.jpg', cv.IMREAD_GRAYSCALE)

stereo = cv.StereoBM_create(numDisparities=0, blockSize=21)
# For each pixel algorithm will find the best disparity from 0
# Larger block size implies smoother, though less accurate disparity map
depth = stereo.compute(left_image, right_image)

plt.imshow(depth)
plt.axis('off')
plt.show()
cv.destroyAllWindows()