import cv2
import numpy as np
import random

class WMKey:
    def __init__(self, src_img):
        self.S = self._genS(src_img)
        self.src_shape = src_img.shape
        self.wm_keypoints = None # this is extracted later by getSIFT from wm_img
        self.wm_descriptors = None # this is extracted later by getSIFT from wm_img

    def getSIFT(self, wm_img):
        gray_wm_img = cv2.cvtColor(wm_img, cv2.COLOR_BGR2GRAY)  # convert rbg img to gray img for SIFT detection
        sift = cv2.SIFT_create()  # create SIFT detector
        self.wm_keypoints, self.wm_descriptors = sift.detectAndCompute(gray_wm_img, None)

    def _genPNSeq(self, _N):
        rand_idxes = list(range(_N))
        random.shuffle(rand_idxes)
        rand_bin_seq = np.ones(_N)
        half = int(_N / 2)
        rand_bin_seq[rand_idxes[0:half]] = -1
        mean_seq = np.mean(1.0 * rand_bin_seq)
        print(f'Mean of pn_seq: {mean_seq}, total entries in pn_seq: {rand_bin_seq.shape}', end='\n')

        return rand_bin_seq

    def _genPNMat(self, height, width):
        pn_seq = self._genPNSeq(height * width)
        pn_mat = pn_seq.reshape(height, width)

        mean_mat = np.mean(pn_mat)
        print(f'Mean of pn_mat: {mean_mat}, shape of pn_mat: {pn_mat.shape}', end='\n')

        return pn_mat

    def _genS(self, src_img):
        height, width, channel = src_img.shape
        list_mat = []
        for i in range(channel):
            pn_mat = self._genPNMat(height, width)
            list_mat.append(pn_mat)

        S = np.stack(list_mat, axis=-1)
        return S

class WaterMarker:
    @staticmethod
    def embedWM(src_img, wm_key, eps):
        print(f'Watermarking image', end=' ... ')
        wm_img = src_img + eps * wm_key.S
        wm_img = np.clip(wm_img, 0, 255).astype(np.uint8)
        print('Done!')

        print(f'Getting SIFT from wm_img', end=' ... ')
        wm_key.getSIFT(wm_img)
        print('Done!')
        return wm_img

    @staticmethod
    def computeSIFT(img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert rbg img to gray img for SIFT detection
        sift = cv2.SIFT_create()  # create SIFT detector
        keypoints, descriptors = sift.detectAndCompute(gray_img, None)
        return (keypoints, descriptors)

    @staticmethod
    def findOffset(src_kpts, src_descrs, tgt_kpts, tgt_descrs, verbal = False):
        keypoints1, descriptors1 = (src_kpts, src_descrs)
        keypoints2, descriptors2 = (tgt_kpts, tgt_descrs)

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
            if (verbal): print(f"Offset of the upper left corner of image2 in image1: (x={tx}, y={ty})")

            # Return the offset as a tuple
            offset_tuple = np.array([tx, ty])
        else:
            if (verbal): print("No valid transformation found.")
            offset_tuple = None # means a different image with different content

        # Return the offset tuple
        return np.round(offset_tuple).astype(int)

    @staticmethod
    def getOffSet(tgt_img, wm_key, verbal = False):
        tgt_height, tgt_width, tgt_channel = tgt_img.shape
        src_height, src_width, src_channel = wm_key.src_shape
        assert (tgt_channel == src_channel)

        if (tgt_height > src_height or tgt_width > src_width):
            if (verbal): { print('A larger height or a larger width means a different image with different content')}
            return None # a larger height or a larger width means a different image with different content
        elif (tgt_height == src_height and tgt_width == src_width):
            if (verbal): {print('the same dimension implies no crop by default')}
            return np.array([0,0]) # the same dimension implies no crop by default
        else:
            if (verbal): print('smaller height or smaller width implies potential crop')
            tgt_kpts, tgt_descrs = WaterMarker.computeSIFT(tgt_img)
            offset = WaterMarker.findOffset(wm_key.wm_keypoints, wm_key.wm_descriptors, tgt_kpts, tgt_descrs)
            return offset # smaller height or smaller width implies potential crop

    @staticmethod
    def genCandidateOffsets(S, offset, k=3):
        neighbors = []
        rows, cols = S.shape
        i, j = (offset[1], offset[0])

        for row_offset in range(-k, k + 1):
            for col_offset in range(-k, k + 1):
                neighbor_i = i + row_offset
                neighbor_j = j + col_offset

                # Check if the neighbor is within bounds and not the same as the original location
                if (
                        0 <= neighbor_i < rows
                        and 0 <= neighbor_j < cols
                        and (neighbor_i != i or neighbor_j != j)
                ):
                    neighbors.append((neighbor_i, neighbor_j))

        return neighbors

    @staticmethod
    def computeWMStrength(tgt_img, wm_key, offset=np.array([0,0])):
        height, width, _ = tgt_img.shape
        cropped_S = wm_key.S[offset[1] :offset[1]+height, offset[0]:offset[0]+width, :]
        assert (cropped_S.shape == tgt_img.shape)

        wm_strength = np.mean(tgt_img * cropped_S)
        return wm_strength

    @staticmethod
    def estimateTau(tgt_img, fpr):
        EX2 = np.mean(tgt_img * tgt_img)
        height, width, channel = tgt_img.shape
        n = height*width*channel
        tau = np.sqrt(1.0*EX2/(2*fpr*n))
        return tau

    @staticmethod
    def extractWMCrop(tgt_img, wm_key, fpr):
        print(f'Estimating tau', end=' ... ')
        tau = WaterMarker.estimateTau(tgt_img, fpr)
        print(f'tau={tau:.3f} for fpr={fpr}')

        print(f'Getting offset', end=' ... ')
        offset = WaterMarker.getOffSet(tgt_img, wm_key)
        print(f'Done! \n offset is: {offset}')

        print(f'Crop_version Extracting watermark', end=' ... ')
        if offset is None:
            print('Done! \n No watermark due to invalid offset!')
            return False
        else:
            wm_strength = WaterMarker.computeWMStrength(tgt_img, wm_key, offset)
            print(f'Done! \ncrop_version wm_strength is: {wm_strength}', end='\n')

            if wm_strength >= tau:
                return True  # means there is a watermark in the tgt_img
            else:
                return False  # means there is no watermark in the tgt_img

    @staticmethod
    def extractWMScale(tgt_img, wm_key, fpr):
        tgt_height, tgt_width, tgt_channel = tgt_img.shape
        src_height, src_width, src_channel = wm_key.src_shape
        assert (tgt_channel == src_channel)

        # rescale tgt_image to src_image dimension if dimension does not match
        if (tgt_height != src_height or tgt_width != src_width):
            tgt_img = cv2.resize(tgt_img, (src_width, src_height), interpolation=cv2.INTER_AREA)

        print(f'Estimating tau', end=' ... ')
        tau = WaterMarker.estimateTau(tgt_img, fpr)
        print(f'tau={tau:.3f} for fpr={fpr}')

        # compute watermark strength by default offset=[0,0]
        print(f'Scale_version Extracting watermark', end=' ... ')
        wm_strength = WaterMarker.computeWMStrength(tgt_img, wm_key)
        print(f'Done! \nscale_version wm_strength is: {wm_strength}', end='\n')

        if wm_strength >= tau:
            return True  # means there is a watermark in the rescaled tgt_img
        else:
            return False  # means there is no watermark in the rescaled tgt_img

    @staticmethod
    def extractWM(tgt_img, wm_key, fpr = 0.001):
        wm_result_crop = WaterMarker.extractWMCrop(tgt_img, wm_key, fpr)
        if wm_result_crop == True:
            print('Has watermark! (Crop version)')
            return True
        else:
            wm_result_scale = WaterMarker.extractWMScale(tgt_img, wm_key, fpr)
            if wm_result_scale == True:
                print('Has watermark! (Scale version)')
                return True
            else:
                print('No watermark!')
                return False

class Attacker:
    @staticmethod
    def cropImage(src_img, offset):
        # Extract the cropping parameters (x, y, width, height) from the offset tuple
        x, y, width, height = offset

        # Crop the source image based on the offset
        cropped_image = src_img[y:y + height, x:x + width]
        return cropped_image

    @staticmethod
    def scaleImage(src_img, scale_ratio=1):
        # Calculate the new dimensions based on the scale percentage
        width = int(src_img.shape[1] * scale_ratio)
        height = int(src_img.shape[0] * scale_ratio)

        # Resize the image to the specified width and height
        scaled_img = cv2.resize(src_img, (width, height), interpolation=cv2.INTER_AREA)

        return scaled_img


class Toolbox:
    def computePSNR(src_img, tgt_img):
        """
        Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.

        Parameters:
        - src_img: The source image as a NumPy ndarray.
        - tgt_img: The target (reference) image as a NumPy ndarray.

        Returns:
        - The PSNR value as a float.
        """
        # Ensure the input images have the same shape
        if src_img.shape != tgt_img.shape:
            raise ValueError("Input images must have the same dimensions.")

        # Calculate the mean squared error (MSE)
        mse = np.mean((src_img - tgt_img) ** 2)

        # Calculate the maximum possible pixel value (assuming 8-bit images)
        max_pixel_value = 255.0

        # Calculate the PSNR using the MSE and max pixel value
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))

        return psnr