website: https://berkeleybear22ryan.github.io/CS180_Project4/

github: https://github.com/berkeleybear22ryan/CS180_Project4

# Image Mosaicing Project

## Overview
The goal of this project is to create an image mosaic by registering, warping, resampling, and blending multiple images together. This project involves understanding image warping using homographies, warping images to align them, and blending them smoothly to form a seamless mosaic.

In this project, I’ve performed the following tasks:
1. **Shoot and Digitize Pictures** – Acquired image sets for stitching.
2. **Recover Homographies** – Used point correspondences to compute homography matrices between pairs of images.
3. **Warp the Images** – Applied projective transformations to warp the images into a common reference frame.
4. **Blend Images into a Mosaic** – Blended the images using various techniques to produce seamless mosaics.
5. **Optional Enhancements** – Sharpened and applied Laplacian pyramid blending for smoother results.

---

## Step-by-Step Breakdown

### Part 1: Shoot and Digitize Pictures
For this project, I used pre-acquired images from several sets found online. I chose these images because they had the desired characteristics: significant overlap, varied angles, and good details to work with. The images are stored in directories like `./images/9/`, `./images/10/`, etc., and named `medium01.jpg`, `medium02.jpg`, and so on.

### Part 2: Recover Homographies
To align images, I computed the homography matrix `H` between pairs of images. A homography matrix relates two images when their transformation can be described by a projective transformation. Using point correspondences between images, I set up a system of equations to solve for the matrix `H`. The point correspondences were selected manually using an interface I built, and saved to files like `./points/9/medium01_pts.txt`, `./points/9/medium02_pts.txt`, etc.

For each correspondence, I set up the linear system:
\[
x_i h_{11} + y_i h_{12} + h_{13} - x_i' (x_i h_{31} + y_i h_{32} + 1) = 0
\]
\[
x_i h_{21} + y_i h_{22} + h_{23} - y_i' (x_i h_{31} + y_i h_{32} + 1) = 0
\]
I solved this system using least squares to ensure stability. Homographies for each pair of images were stored in the `./h_matrix/` directory for later use.

### Part 3: Warp the Images
Once I had the homographies, I used them to warp the images so they aligned with the reference image. I implemented inverse warping, where for each pixel in the warped image, I used the inverse of the homography to map it back to the original image, ensuring that no pixels were left unmapped.

Here’s the math behind the warping:
\[
x' = \frac{h_{11} x + h_{12} y + h_{13}}{h_{31} x + h_{32} y + 1}
\]
\[
y' = \frac{h_{21} x + h_{22} y + h_{23}}{h_{31} x + h_{32} y + 1}
\]
I used bilinear interpolation to compute the pixel values at non-integer locations.

### Part 4: Image Rectification
To verify my warping, I performed image rectification on individual images. In this process, I manually selected points corresponding to a known rectangular plane (like a tile) and warped them to fit a rectangle. This ensured that my homography and warping code were working correctly.

### Part 5: Blend Images into a Mosaic
Once the images were warped, I blended them to create seamless mosaics. I used alpha masks with higher weights at the center of each image and lower weights near the edges to blend the overlapping regions smoothly.

For more advanced blending, I applied Laplacian pyramid blending, where the images were first decomposed into different frequency bands using Gaussian and Laplacian pyramids. I then blended each level separately and reconstructed the mosaic from the blended pyramid. This helped reduce the visibility of seams and high-frequency artifacts between images.

### Optional: Image Sharpening
I also sharpened the final mosaic using an unsharp masking technique. After applying a Gaussian blur, I subtracted the blurred image from the original and added the result back, which accentuated the details. This was a post-processing step aimed at enhancing the visual quality of the mosaic.

---

## Directory Structure

```
code/
    finals/             # Final images
    h_matrix/           # Homography matrices
    images/             # Input images
    output/             # Output mosaics
    points/             # Selected point correspondences
    rectify/            # Rectified images
    a1.py               # Script for selecting points
    a2.py               # Script for computing homographies
    a3.py               # Script for warping images
    a4.py               # Script for rectification
    a5.py               # Script for advanced blending and sharpening
web/
    README.md
```

---

## How to Run the Code

1. **Selecting Points:**
   To select points for computing homographies, run:
   ```
   python a1.py
   ```

2. **Computing Homographies:**
   After selecting points, compute the homography matrices by running:
   ```
   python a2.py
   ```

3. **Warping Images:**
   To warp images into alignment, run:
   ```
   python a3.py
   ```

4. **Rectification:**
   To test the warping and perform image rectification, use:
   ```
   python a4.py
   ```

5. **Blending and Sharpening:**
   Finally, to create a blended mosaic and apply sharpening, use:
   ```
   python a5.py
   ```

---

## Results

- **Rectified Images:** After recovering homographies, I successfully rectified several images.
- **Mosaics:** I produced mosaics from multiple sets of images. The blending was done using alpha masks and Laplacian pyramid blending for smooth transitions.
- **Sharpened Mosaics:** The sharpened mosaics provided more detail and clarity to the final output.

---

## Parameters for Faster Execution
To speed up execution:
1. **Downsampling:** You can reduce the image resolution by a factor (e.g., 0.5) before processing. This can be done in the `a5.py` script by modifying the resizing step.
2. **Gaussian Pyramid Levels:** Decrease the number of levels in the Gaussian and Laplacian pyramids for blending. Fewer levels will reduce the computational cost.
3. **Reduce Kernel Size:** Using smaller convolution kernels for Gaussian blurs can speed up the process.

---