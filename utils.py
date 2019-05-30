import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


def calibrate_camera(nx=9, ny=6, resolution=(1280, 720), path='camera_cal/*.jpg'):
    '''
    Given a path to the calibration checker board photos, return calibration matrixes
    '''

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(ny,5,0)
    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images and image shapes
    images = glob.glob(path)
    image_shape = None

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Check for differnt image sizes
        if gray.shape != resolution[::-1]:
            gray = cv2.resize(gray, resolution)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    # Use the object points and image points to calculate camera clibrations
    _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints,
                                             resolution, None, None)

    # Return results
    return mtx, dist






