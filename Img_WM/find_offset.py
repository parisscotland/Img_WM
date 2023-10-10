import cv2
import numpy as np

# Load the two images you want to match
image1 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('image3.jpg', cv2.IMREAD_GRAYSCALE)

# Create a SIFT detector
sift = cv2.SIFT_create()

# Find key points and descriptors in the images
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# Create a FLANN matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Match descriptors between the two images
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Apply Lowe's ratio test to filter good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Extract matched keypoints
matched_keypoints1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
matched_keypoints2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

# Compute the transformation (translation) matrix
transform_matrix, _ = cv2.estimateAffinePartial2D(matched_keypoints2, matched_keypoints1)

if transform_matrix is not None:
    # Extract the translation components from the transformation matrix
    tx = transform_matrix[0, 2]
    ty = transform_matrix[1, 2]
    
    # Print the offset
    print(f"Offset of the upper left corner of image2 in image1: (x={tx}, y={ty})")
    
    # Return the offset as a tuple
    offset_tuple = (tx, ty)
else:
    print("No valid transformation found.")
    offset_tuple = None

# Draw the matches on a new image
# matching_result = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Show the matching result
# cv2.imshow('SIFT Matches (FLANN)', matching_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Return the offset tuple
offset_tuple
