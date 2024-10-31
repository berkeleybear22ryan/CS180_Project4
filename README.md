website: https://berkeleybear22ryan.github.io/CS180_Project4/

github: https://github.com/berkeleybear22ryan/CS180_Project4

# Image Mosaicing Project
# Part 1 

---

---

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

---
# README for Part 2: Automatic Image Stitching

In this document, I will provide a detailed walkthrough of **Part 2** of my image stitching project, which focuses on automatically stitching images into a mosaic using feature detection, description, matching, and homography estimation. I will explain how each script (`b1.py` to `b5_1.py`) corresponds to specific steps in the pipeline and how they are linked to their respective output folders. Additionally, I will describe how I reused code from Part 1 to create the final mosaic, and I will provide insights into the overall pipeline, including the use of bash scripts for automation.

---

## Overview of Part 2

The goal of **Part 2** is to automate the process of stitching images together by implementing feature-based methods. This involves detecting keypoints, extracting feature descriptors, matching features between images, computing homographies using RANSAC, and finally stitching the images into a seamless panorama.

---

## Scripts and Their Corresponding Steps

Below is the mapping of the scripts to the steps in the project and their corresponding output folders:

1. **`b1.py`** - **Image Preprocessing**
   - **Linked with:** `step1` folder
   - **Description:** Reads and preprocesses images to prepare them for feature detection. This includes converting images to grayscale and normalizing them if necessary.

2. **`b2.py`** - **Feature Detection**
   - **Linked with:** `step2` folder
   - **Description:** Implements the Harris Corner Detector to find interest points in the images. It also applies Adaptive Non-Maximal Suppression (ANMS) to select a subset of keypoints that are well-distributed across the image.

3. **`b3.py`** - **Feature Description**
   - **Linked with:** `step3` folder
   - **Description:** Extracts feature descriptors for each keypoint detected in `b2.py`. It samples axis-aligned 8x8 patches from larger 40x40 windows around each keypoint and normalizes them.

4. **`b4.py`** - **Feature Matching**
   - **Linked with:** `step4` folder
   - **Description:** Matches feature descriptors between pairs of images using Lowe's ratio test. This step finds correspondences between images that are used for homography estimation.

5. **`b5_1.py`** - **Homography Estimation Using RANSAC**
   - **Linked with:** `step5` folder
   - **Description:** Computes the homography matrices between image pairs using RANSAC to robustly handle outliers. It saves the homographies and inlier matches for use in stitching.

---

## Reusing Code from Part 1 for Mosaic Creation

After obtaining the homographies from `b5_1.py`, I reused code from Part 1 to stitch the images together:

- **Skipping `a1.py`:** Since we have automatically generated point correspondences from `b5_1.py`, we can skip the manual point selection step (`a1.py`).
- **Using `a2.py`:** This script computes the cumulative homographies from the pairwise homographies obtained in `b5_1.py`.
- **Using `a3.py` and `a5.py`:** These scripts are responsible for warping the images using the computed homographies (`a3.py`) and blending them to create the final mosaic (`a5.py`).

---

## Automation with Bash Scripts

To streamline the pipeline, I have included three bash scripts:

1. **`runall.sh`**
   - **Main script for automation.**
   - Allows you to specify a group of images, and it runs all the necessary steps to produce the final output.
   - Usage:
     ```bash
     ./runall.sh <image_group>
     ```
     Replace `<image_group>` with the name of your image set (e.g., `9_0_P2`).

2. **`runall_shark.sh`**
   - Similar to `runall.sh` but configured for running on a different environment (e.g., a server named "shark").

3. **`runall_shark_p.sh`**
   - A variant of `runall_shark.sh` with additional parameters or configurations.

These scripts provide an easy way to execute the entire pipeline without manually running each script.

---

## Pipeline Execution

The overall pipeline for Part 2 is as follows:

1. **Preprocess Images (`b1.py`):**
   - Run `b1.py` to prepare the images.
   - Outputs are saved in the `step1` folder.

2. **Detect Features (`b2.py`):**
   - Run `b2.py` to detect keypoints using the Harris Corner Detector and ANMS.
   - Outputs are saved in the `step2` folder, including visualizations of the detected corners.

3. **Extract Feature Descriptors (`b3.py`):**
   - Run `b3.py` to extract and normalize feature descriptors.
   - Outputs are saved in the `step3` folder.

4. **Match Features (`b4.py`):**
   - Run `b4.py` to match features between image pairs using Lowe's ratio test.
   - Outputs are saved in the `step4` folder, including match indices and optional visualizations.

5. **Estimate Homographies (`b5_1.py`):**
   - Run `b5_1.py` to compute homographies using RANSAC.
   - Outputs are saved in the `step5` folder, including homography matrices and inlier matches.

6. **Create the Mosaic:**
   - Since we have the homographies and point correspondences, we can skip `a1.py`.
   - Run `a2.py` to compute cumulative homographies.
   - Run `a3.py` and/or `a5.py` to warp the images and blend them into a final mosaic.
   - The final output is saved in the appropriate output folder.

---

## Sample Walkthrough: Image Set `9_0_P2`

In the next sections, I will provide a detailed walkthrough of the pipeline using the image set `9_0_P2`. This will include:

- **Step-by-Step Execution:**
  - Commands used to run each script.
  - Descriptions of intermediate outputs and visualizations.

- **Final Mosaic:**
  - The automatically stitched panorama created from `9_0_P2`.
  - Comparison with manually stitched results if available.

---

## Step-by-Step Execution for `9_0_P2`

### 1. Preprocess Images (`b1.py`)

```bash
python b1.py --input_dir images/9_0_P2/ --output_dir part2_output/9_0_P2/step1/
```

- **Description:**
  - Reads images from `images/9_0_P2/` and preprocesses them.
  - Saves preprocessed images in `step1` folder.

### 2. Detect Features (`b2.py`)

```bash
python b2.py --input_dir part2_output/9_0_P2/step1/ --output_dir part2_output/9_0_P2/step2/
```

- **Description:**
  - Detects Harris corners and applies ANMS.
  - Saves keypoint coordinates and visualizations in `step2` folder.

### 3. Extract Feature Descriptors (`b3.py`)

```bash
python b3.py --input_dir part2_output/9_0_P2/step2/ --output_dir part2_output/9_0_P2/step3/
```

- **Description:**
  - Extracts and normalizes feature descriptors for each keypoint.
  - Saves descriptors in `step3` folder.

### 4. Match Features (`b4.py`)

```bash
python b4.py --input_dir part2_output/9_0_P2/step3/ --output_dir part2_output/9_0_P2/step4/
```

- **Description:**
  - Matches descriptors between image pairs using Lowe's ratio test.
  - Saves matched indices and visualizations in `step4` folder.

### 5. Estimate Homographies (`b5_1.py`)

```bash
python b5_1.py --input_dir part2_output/9_0_P2/step4/ --output_dir part2_output/9_0_P2/step5/
```

- **Description:**
  - Computes homographies using RANSAC.
  - Saves homography matrices and inlier matches in `step5` folder.

### 6. Create the Mosaic

Since we have the homographies, we can proceed to create the mosaic:

```bash
python a2.py --input_dir part2_output/9_0_P2/step5/ --output_dir part2_output/9_0_P2/step6/
python a3.py --input_dir part2_output/9_0_P2/step6/ --output_dir part2_output/9_0_P2/step7/
python a5.py --input_dir part2_output/9_0_P2/step7/ --output_dir part2_output/9_0_P2/final_output/
```

- **Description:**
  - `a2.py` computes cumulative homographies.
  - `a3.py` warps the images using the homographies.
  - `a5.py` blends the warped images to create the final mosaic.
  - The final mosaic is saved in `final_output` folder.

---

## Final Mosaic for `9_0_P2`

The automatically stitched mosaic for `9_0_P2` is saved in:

```
part2_output/9_0_P2/final_output/mosaic.png
```

- **Comparison with Manual Stitching:**
  - If manual stitching results are available from Part 1, include them side by side with the automatic result.
  - This comparison showcases the effectiveness of the automatic pipeline.

---

## Running the Pipeline with `runall.sh`

To simplify the process, you can use the `runall.sh` script:

```bash
./runall.sh 9_0_P2
```

- **Description:**
  - Automates all the steps from preprocessing to mosaic creation.
  - Ensure that the script has execution permissions:
    ```bash
    chmod +x runall.sh
    ```

---

## Conclusion

By following this pipeline, I successfully automated the image stitching process for multiple image sets, including `9_0_P2`. The scripts `b1.py` to `b5_1.py` correspond to the essential steps of feature-based image stitching, and by reusing code from Part 1, I efficiently created seamless mosaics.

This approach demonstrates the effectiveness of feature detection, description, matching, and robust homography estimation in automating panorama creation. The use of bash scripts like `runall.sh` further streamlines the process, making it easy to apply the pipeline to different image sets.

---

## Additional Notes

- **Parameter Tuning:**
  - Throughout the pipeline, parameters such as the number of keypoints, ratio thresholds, and RANSAC error thresholds were tuned for optimal results.
  - These parameters may need adjustment depending on the specific characteristics of the image sets.

- **Visualization:**
  - Visualizations at each step (e.g., detected corners, matched features, inlier correspondences) are saved in the respective output folders.
  - These visualizations are helpful for debugging and understanding the performance of each step.

- **Extensibility:**
  - The pipeline is designed to be extensible. By modifying the scripts or parameters, it can be adapted to different types of images or stitching scenarios.

---