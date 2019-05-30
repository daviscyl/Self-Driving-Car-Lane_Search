import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


class Lane_Finder(object):
    '''
    Lane finding algorithm.
    '''

    def __init__(self):
        # Camera calibration params
        self.resolution = (1280, 720)
        self.camera_mtx, self.camera_dist = self.calibrate_camera()

        # Warping params
        self.src = np.float32(
            [[580, 460],
             [201, 720],
             [1100, 720],
             [702, 460]])
        self.dst = np.float32(
            [[320, 0],
             [320, 720],
             [960, 720],
             [960, 0]])
        self.warp_transform = cv2.getPerspectiveTransform(self.src, self.dst)
        self.inv_warp_transform = cv2.getPerspectiveTransform(self.dst, self.src)

        # Left & Right Lines Initiation
        self.left_line = Line()
        self.right_lane = Line()

        # Pixel select params
        self.s_thresh = (170, 255)
        self.sx_thresh = (20, 100)

        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30/720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7/640  # meters per pixel in x dimension

        # Radius Curvature calculation settings
        self.min_radius_m = 840
        self.max_radius_m = 10000
        self.max_r_diff_m = 500

        # Result Text Settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bottomLeftCornerOfText1 = (20, 50)
        self.bottomLeftCornerOfText2 = (20, 100)
        self.fontScale = 1.5
        self.fontColor = (255, 255, 255)
        self.lineType = 2

    def calibrate_camera(self, nx=9, ny=6, path='camera_cal/*.jpg'):
        '''
        Given a path to the calibration checker board photos, return calibration params
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
            if gray.shape != self.resolution[::-1]:
                gray = cv2.resize(gray, self.resolution)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        # Use the object points and image points to calculate camera clibrations
        _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, self.resolution, None, None)

        # Return results
        return mtx, dist

    def threshold_binary(self, warped):
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(warped, cv2.COLOR_RGB2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x

        # Absolute x derivative to accentuate lines away from horizontal
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= self.sx_thresh[0]) & (scaled_sobel <= self.sx_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= self.s_thresh[0]) & (s_channel <= self.s_thresh[1])] = 1

        # Return the result
        result = np.zeros_like(s_channel)
        result[(sxbinary == 1) | (s_binary == 1)] = 1
        return result

    def find_lane_box(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)

        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 70
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)

            ### Identify the nonzero pixels in x and y within the window ###
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            ### If you found > minpix pixels, recenter next window ###
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    def curvature_radius(self, polyfit_params, y):
        '''
        Calculates the bending radius given a set of second order coefficients.
        '''
        radiuses = ((1 + (2*polyfit_params[0]*y + polyfit_params[1])**2)**1.5) / (2*polyfit_params[0])
        radius = int(np.mean(radiuses))
        radius = np.sign(radius) * max(abs(radius), self.min_radius_m)

        return radius

    def pipeline(self, img):
        # Undistort
        undist = cv2.undistort(img, self.camera_mtx, self.camera_dist, None, self.camera_mtx)

        # Warp
        warped = cv2.warpPerspective(undist, self.warp_transform, self.resolution, flags=cv2.INTER_LINEAR)

        # Threshold Binary
        binary_warped = self.threshold_binary(warped)

        # Find Lane Pixels Box/Prefit
        leftx, lefty, rightx, righty, box_out_img = self.find_lane_box(binary_warped)

        # Fit Poly
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        left_fit_m = np.polyfit(lefty*self.ym_per_pix, leftx*self.xm_per_pix, 2)
        right_fit_m = np.polyfit(righty*self.ym_per_pix, rightx*self.xm_per_pix, 2)

        # Calculate Radius & Center Deviation
        y_m = np.linspace(0, self.resolution[1]*self.ym_per_pix, 10)
        left_r_m = self.curvature_radius(left_fit_m, y_m)
        right_r_m = self.curvature_radius(right_fit_m, y_m)

        bottom_m = (self.resolution[1]-1)*self.ym_per_pix
        center_m = (self.resolution[0]-1)/2*self.xm_per_pix
        lane_left_m = left_fit_m[0]*bottom_m**2 + left_fit_m[1]*bottom_m + left_fit_m[2]
        lane_right_m = right_fit_m[0]*bottom_m**2 + right_fit_m[1]*bottom_m + right_fit_m[2]
        deviation_m = round((lane_left_m + lane_right_m)/2 - center_m, 2)
        deviation_dir = 'left' if deviation_m > 0 else 'right'

        # Curvature Radius Sanity Check
        current_r_m = abs((left_r_m + right_r_m) // 2)

        # Draw Pixels & Lane on Warped View
        # Generate x and y values for plotting
        ploty = np.linspace(0, self.resolution[1]-1, self.resolution[1])
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty
        # Create an image to draw the lines on
        color_warp = np.zeros_like(warped).astype(np.uint8)
        # Colors in the left and right lane regions
        color_warp[lefty, leftx] = [0, 0, 255]
        color_warp[righty, rightx] = [255, 0, 0]
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255//2, 0))
        # Warp the blank back to original image space using inverse perspective matrix
        newwarp = cv2.warpPerspective(color_warp, self.inv_warp_transform, self.resolution)

        # Combine the result with the original image
        output = cv2.addWeighted(undist, 1, newwarp, 0.6, 0)
        cv2.putText(output, 'Curvature Radius Left = {}m, Right={}m'.format(left_r_m, right_r_m),
                    self.bottomLeftCornerOfText1,
                    self.font,
                    self.fontScale,
                    self.fontColor,
                    self.lineType)
        cv2.putText(output, 'Vehicle is {}m {} of center'.format(abs(deviation_m), deviation_dir),
                    self.bottomLeftCornerOfText2,
                    self.font,
                    self.fontScale,
                    self.fontColor,
                    self.lineType)

        # Return Final Result
        return output

    def process_video(self, video_path):
        # Open the input video file and get fps, codec, resolution info.
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # Open a VideoWriter object using above info, append 'output' to the file name.
        filename, sufix = video_path.split('.')
        out_file = cv2.VideoWriter(filename+'_output.'+sufix, fourcc, fps, resolution)

        # Main loop of of video frames
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:

                output = self.pipeline(frame)
                out_file.write(output)

                # Show the imediate result and quit on Q
                cv2.imshow('Advanced Lane-Finding Algorithm Output', output)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # Release everything if job is finished
        cap.release()
        out_file.release()
        cv2.destroyAllWindows()


class Line():
    '''
    Define a class to receive the characteristics of each line detection.
    '''

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None


finder = Lane_Finder()
finder.process_video('project_video.mp4')