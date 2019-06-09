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
        self.right_line = Line()

        # Pixel select params
        self.s_thresh = (170, 255)
        self.sx_thresh = (20, 100)

        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30/720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7/640  # meters per pixel in x dimension

        # Radius curvature calculation settings
        self.min_radius_m = 840
        self.max_radius_m = 6000

        # Lane search settings
        self.margin = 70  # margin to search for lane pixels
        self.nwindows = 9  # number of sliding windows
        self.minpix = 50  # minimum number of pixels found to recenter window

        # Result text settings
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

    def threshold_binary(self, img):
        '''
        Given a color image, use X direction gradient and saturation thresholds
        to select pixels that has a high probablity of being the lane line and
        produce a binary image where these pixel locations are 1. 
        '''
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        # Sobel x
        # Take the derivative in x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)

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

    def find_lane_sliding_window(self, binary_warped):
        '''
        Given a warped binary image, use boxes method to select pixels that are
        in the lane line region. Outputs these pixels' x and y locations.
        '''
        # Take a histogram of the bottom half of the image
        histogram = np.sum(
            binary_warped[binary_warped.shape[0]//2:, :], axis=0)

        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//self.nwindows)

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
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            # TODO: Move to visualization notebook
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            ### Identify the nonzero pixels in x and y within the window ###
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            ### If you found > minpix pixels, recenter next window ###
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
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

        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [0, 0, 255]
        out_img[righty, rightx] = [255, 0, 0]

        return leftx, lefty, rightx, righty, out_img

    def find_lane_prefit(self, binary_warped, left_fit, right_fit):

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        ploty = np.linspace(0, self.resolution[1]-1, self.resolution[1])

        ### Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        left_fitx_prev = left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2]
        left_lane_inds = ((nonzerox > (left_fitx_prev - self.margin)) & (nonzerox < (left_fitx_prev + self.margin)))
        right_fitx_prev = right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2]
        right_lane_inds = ((nonzerox > (right_fitx_prev - self.margin)) & (nonzerox < (right_fitx_prev + self.margin)))

        left_fitx = left_fit[0]*(ploty**2) + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*(ploty**2) + right_fit[1]*ploty + right_fit[2]

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Visualization ## TODO: Move this part to visualization notebook
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-self.margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+self.margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-self.margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+self.margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        return leftx, lefty, rightx, righty, result

    def curvature_radius(self, polyfit_params, y):
        '''
        Calculates the bending radius given a set of second order coefficients.
        '''
        radiuses = ((1 + (2*polyfit_params[0]*y + polyfit_params[1])**2)**1.5) / (2*polyfit_params[0])
        radius = max(int(abs(np.mean(radiuses))), self.min_radius_m)
        return radius

    def fit_poly(self, leftx, lefty, rightx, righty, yrange):
        ### Fit a second order polynomial to each with np.polyfit() ###
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, yrange-1, yrange)
        ### Calc both polynomials using ploty, left_fit and right_fit ###
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        return left_fitx, right_fitx, ploty

    def sanity_check(self, left_fitx, right_fitx, ploty):
        delta_x = right_fitx - left_fitx
        # TODO Checking that they have similar curvature

        # Checking that they are roughly parallel
        if delta_x.max() - delta_x.min() > 200:
            return False
        # Checking that they are separated by approximately the right distance horizontally
        if 540 > delta_x.mean() and delta_x.mean() > 740:
            return False
        return True

    def pipeline(self, img):
        '''
        Image processing pipeline for a single input image.

        Outputs the processed image that's been undistorted and the recognized
        lane region marked with distinct color. Curvature radius and vihicle
        relative position to the lane center is printed on the top of the output.
        '''
        # Undistort
        undist = cv2.undistort(img, self.camera_mtx, self.camera_dist, None, self.camera_mtx)

        # Threshold Binary
        binary = self.threshold_binary(undist)

        # Warp
        binary_warped = cv2.warpPerspective(binary, self.warp_transform, self.resolution, flags=cv2.INTER_LINEAR)

        # TODO: Integrate Line() object
        # Find Lane Pixels with Boxes or Prefit data
        leftx, lefty, rightx, righty, box_out_img = self.find_lane_sliding_window(binary_warped)

        # Fit Poly
        left_fitx, right_fitx, ploty = self.fit_poly(leftx, lefty, rightx, righty, self.resolution[1])

        # Sanity Check
        if self.sanity_check(left_fitx, right_fitx, ploty):
            self.right_line.detected = True
            self.left_line.detected = True

        # Calculate Radius
        midx = np.mean(np.dstack((left_fitx, right_fitx)).squeeze(), axis=1)
        mid_lane_fit = np.polyfit(ploty*self.ym_per_pix, midx*self.xm_per_pix, 2)
        y_eval = np.linspace(0, self.resolution[1]*self.ym_per_pix, 10)
        radius_m = self.curvature_radius(mid_lane_fit, y_eval)

        # Caculate Center Deviation
        deviation_m = (np.mean(midx[-10:]) - self.resolution[0]/2)*self.xm_per_pix
        deviation_m = round(deviation_m, 2)
        deviation_dir = 'left' if deviation_m > 0 else 'right'

        # Create an image to draw the lines on
        color_warp = np.zeros_like(img).astype(np.uint8)
        # Colors in the left and right lane regions
        color_warp[lefty, leftx] = [0, 0, 255]
        color_warp[righty, rightx] = [255, 0, 0]
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255//3, 0))
        # Warp the blank back to original image space using inverse perspective matrix
        newwarp = cv2.warpPerspective(
            color_warp, self.inv_warp_transform, self.resolution)

        # Combine the result with the original image
        output = cv2.addWeighted(undist, 1, newwarp, 0.9, 0)
        # Add Curvature Radius and Lane Deviation info texts to output
        cv2.putText(output, 'Curvature Radius = {}m'.format(radius_m if radius_m < self.max_radius_m else 'Inf. '),
                    self.bottomLeftCornerOfText1,
                    self.font,
                    self.fontScale,
                    self.fontColor,
                    self.lineType)
        cv2.putText(output, 'Vehicle is {}m {} of lane center'.format(abs(deviation_m), deviation_dir),
                    self.bottomLeftCornerOfText2,
                    self.font,
                    self.fontScale,
                    self.fontColor,
                    self.lineType)

        # Return Final Result
        return output

    def process_video(self, video_path, show=False):
        '''
        Wraper function for Lane_Fidner.pipeline() function.

        Opens the video file, reads one frame at a time, then process the frame
        using the image processing pipeline, and saves the processed result in
        new video file.
        '''
        # Open the input video file and get fps, codec, resolution info
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # Open a VideoWriter object using above info, append 'output' to the file name
        filename, sufix = video_path.split('.')
        out_file = cv2.VideoWriter(
            filename+'_output.'+sufix, fourcc, fps, resolution)

        # Main loop of of video frames
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                # Process single frame
                output = self.pipeline(frame)
                # Write processed frame to output file
                out_file.write(output)
                # Show the imediate result and quit on Q
                if show:
                    cv2.imshow(
                        'Advanced Lane-Finding Algorithm Output', output)
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

    # TODO: Add update function, smooth the update from frame to frame
