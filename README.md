
## Advanced Lane Finding Project

[//]: # (Image References)

[image1]: ./examples/camera_dist.png "Undistorted"
[image2]: ./examples/color_split.png "Road Transformed"
[image3]: ./examples/red_grads.png "Red Binary Example"
[image4]: ./examples/sat_grads.png "Saturation Binary Example"
[image5]: ./examples/comb_rs_grads.png "Combined Red-Sat Grads"
[image6]: ./examples/comb_rs_grads_region.png "Region to reduce clutter"
[image7]: ./examples/orig_wrap_test.png "Image prespective transform test"
[image8]: ./examples/bird_to_undist.png "Birds eyeview lane detection"
[video1]: ./output_videos/project_video.mp4 "Video"

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)


#### 1. Color thresholds

I have noticed that both red channel in an rgb image and saturation channel of an hls image provide best detection for yellow and white lane lines as shown below:

![alt text][image2]

#### 2. Gradients are applied for both the red channel image and the saturation channel image seperately and they are combined.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

Red channel gradients
![alt text][image3]

Saturation channel gradients
![alt text][image4]

Combining both the above images results in the following:
![alt text][image5]

Applying a region filter so as to help with lane detection:
![alt text][image6]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
# Prespective transform
bottom_left = (290, 673)
bottom_right = (1024, 673)
top_right = (757,496)
top_left = (531, 496)

src = np.float32([[bottom_left, bottom_right, top_right, top_left]])
dst = np.float32([[(460, 673), (820, 673), (820, 496), (460, 496)]])

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 290, 673      | 460, 673      | 
| 1024, 673     | 820, 673      |
| 757, 496      | 820, 496      |
| 531, 496      | 460, 496      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image7]

#### 4. Lane line identification

I ran sliding window lane detection on the thresholded images with margin of window set to 50 and minimum number of pixel detection in window set to 35, the following shows the detection on the above warped image.

![alt text][image8]

#### 5. Radius of curvature and offset from center calculation.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `get_curvature()`.  

```python
def get_curvature(leftx, lefty, rightx, righty):
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*np.max(lefty)*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*np.max(righty)*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    return left_curverad, right_curverad

def get_center_offset(left_fit, right_fit, ymax=720.0, xmax=1280.0):
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    left = left_fit[0]*ymax**2 + left_fit[1]*ymax + left_fit[2]
    right = right_fit[0]*ymax**2 + right_fit[1]*ymax + right_fit[2]
    
    return ((left + (right - left) / 2) - xmax/2)*xm_per_pix
```

The previous image contains shows the values calculated using these APIs.

---

### Pipeline (video)

Here's a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
