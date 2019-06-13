# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[calibration]: ./output_images/camera_calibration.png "Undistorted"
[undistort]: ./output_images/undistort.png "Road Transformed"
[binary]: ./output_images/binary_threshold.png "Binary Example"
[warp]: ./output_images/warp.png "Warp Example"
[window]: ./output_images/sliding_window.png "Sliding Window Example"
[prefit]: ./output_images/prev_fit.png "Previous Fit Example"
[pipeline]: ./output_images/pipeline.png "Output"
[pipelines]: ./output_images/pipeline_samples.png "8 Output"

---

## Camera Calibration

The code for this step is contained in `Lane_Finder.calibrate_camera()` which is in lines 66 through 100 of the file `utils.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

The program then loops through every image in the `camera_cal/` directory and extracts `imgpoints` of all the images.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][calibration]

As we can see that the distorted image has been mostly restored to the undistorted state.

## Pipeline (Single Images)

### 1. Undistort

In my single image processing pipeline: `Lane_Finder.pipeline()`, the undistotion step is performed as a first step before any other processing is applied (line 258 of `utils.py`). I used the `camera_mtx` and `camera_dist` transformation matrixes obtained from the **Camera Calibration** step, and used `cv2.undistort` to obtain the undistortion effect as shown bellow:

![alt text][undistort]

Although the effect can be unnoticebal but if one looks closer, he'd find that the edges of the original image has all been stretched.

### 2. Thresholded Binary Image

I used a combination of color (light channel and saturation channel) and horizonal gradient thresholds to generate a binary image (thresholding steps at `Lane_Finder.threshold_binary()` which are lines 102 through 134 in `utils.py`, this function is called in my pipeline `Lane_Finder.pipeline()` at line 260 of `utils.py`).  Here's an example of my output for this step.

![alt text][binary]

### 3. Perspective Transform

The code for my perspective transform includes a function called `cv2.warpPerspective()`, which appears in lines 263 in the file `util.py`.  The `cv2.warpPerspective()` function takes as inputs an image (`img`), as well as `warp_transform` which is calculated by `cv2.getPerspectiveTransform()` at line 30 of `utils.py` by hardcoded source (`src`) and destination (`dst`) points which I defined in line 20 through 29 in `utils.py`.

I've copied them below:

```python
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
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][warp]

### 4. Fit Polynomial

To find where the lane is, in `Lane_Fidner.pipeline()` I used two methods to further select lane pixels from the binary warped results: `Lane_Finder.find_lane_sliding_window()` (line 135 through 206 in `utils.py`), and `Lane_Finder.find_lane_prefit()` (line 208 through 230 in `utils.py`).

`Lane_Finder.find_lane_sliding_window()` uses box bounderies to find where the left line and right line pixels are, and shifts the box bounderied to the center of the pixels in the previus box. `Lane_Finder.find_lane_prefit()` simply uses the previous line fit data and looks for pixels in the new image within a margin of the previous fit line.

When the right and left line pixel selection finished, I fed those right line and left line pixels to `Line.update_pixels()` which takes those pixels and uses `np.polyfit()` to fit a 2nd order polynomial to those pixels.

The results of `Lane_Finder.find_lane_sliding_window()` + `Line.update_pixels()`

and `Lane_Finder.find_lane_prefit()` + `Line.update_pixels()`

have been visualized below:

![alt text][window]
![alt text][prefit]

#### 5. Calculated the Radius of Curvature, and Lane Center Deviation

I used the function `curvature_radius()` to calculate the line radius of curvature in lines 428 through 431 in my code in `utils.py` and this function is called whenever `Lane_Finder.pipeline()` updates the pixels of right and left line pixels via `Line.update_pixels()`, and calls `curvature_radius()` in line 415 in `Line.update_pixels()`. The final curvature of radius is calculated in line 297 in `utils.py` inside the `Lane_Finder.piepline()` by taking the average of the right line and the left line curvature radius over the last 3 frames. The curvature of raius value tend to go wild when the vehicle is traveling generally straight lines in the video, and to handle this I did a final check before putting this info to the final output image in line 304 of `utils.py` by comparing the calculated value with the `Lane_Finder`'s defined maximum value `Lane_Finder.max_radius` which is defined in line 48 of `utils.py` to be 6000 meters, and if the caculated radius exeeds the max value allowed, the pipeline would simply project `Inf.` to the image to denote the situation where the vehicle is traveling straight line.

The `Line.line_base_pos` is a propety in `Line` object, and is updated whenever `Line.update_pixels()` is called, and this happens at line 417 in `utils.py`. The final diviation is calcuated by taking the average of right and left line deviation over the past 25 frames.

The calcualted curvature radius and lane center deviation info is then projected to the pipeline output image in line 303 through 316 in `utils.py`

### Pipeline Output

The whole pipeline is implemented in `Lane_Finder.pipeline()` in lines 248 through 319 in `utils.py`.  Here are a couple of examples of my pipeline result on different test images:

![alt text][pipeline]
![alt text][pipelines]

---

## Pipeline (video)

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
