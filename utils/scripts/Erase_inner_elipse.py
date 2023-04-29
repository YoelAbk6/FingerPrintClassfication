import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Directory containing fingerprint images
fingerprint_dir = "/home/uoriko/FingerPrintClassfication/data/sd04/png_txt/figs/"
fingerprint_dir_out = "/home/uoriko/FingerPrintClassfication/data/sd04/png_txt/inner/"

search_contours = False

# Iterate over all fingerprint images in the directory
for filename in os.listdir(fingerprint_dir):
    if os.path.isfile(os.path.join(fingerprint_dir, filename)):
        # Load the image
        img = cv2.imread(os.path.join(fingerprint_dir, filename))
        # Convert the image to grayscale and apply thresholding
        inverted_img = cv2.bitwise_not(img)
        gray = cv2.cvtColor(inverted_img, cv2.COLOR_BGR2GRAY)

        # plt.imshow(gray, cmap='gray')
        # plt.show()
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # plt.imshow(thresh, cmap='gray')
        # plt.show()

        erode_kernel = np.ones((3, 3), np.uint8)
        dilate_kernel = np.ones((9, 9), np.uint8)

        eroded = cv2.erode(thresh, erode_kernel, iterations=1)

        # plt.imshow(eroded, cmap='gray')
        # plt.show()

        dilated = cv2.dilate(eroded, dilate_kernel, iterations=3)

        # plt.imshow(dilated, cmap='gray')
        # plt.show()

        # Find the contours of the fingerprint
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(dilated, contours, -1, (0, 255, 0), 2)

        # Get the bounding rectangle of the largest contour (assumed to be the fingerprint)
        # idx = 0 if len(contours) == 1 else 1
        
        if search_contours:
          try:
            x, y, w, h = cv2.boundingRect(contours[-1])
          except:
            # if failed, take the image center and size
            x, y, w, h = 0, 0, img.shape[0], img.shape[1]
        else:
          x, y, w, h = 0, 0, img.shape[0], img.shape[1]

        # Calculate the center point and major/minor axes of the ellipse
        center_x = x + int(w / 2)
        center_y = y + int(h / 2)
        major_axis = int(w / 3)
        minor_axis = int(h / 3)

        # Create a white image the same size as the fingerprint
        ellipse_mask = np.full(img.shape, 255, dtype=np.uint8)

        # Draw an ellipse on the white image, we can change the angle of what's left
        cv2.ellipse(ellipse_mask, (center_x, center_y),
                    (major_axis, minor_axis), 0, 0, 360, 0, thickness=-1)

        # ----For outer----#
        # inverted_img = cv2.bitwise_not(img)

        # ----Toggle between or/and + img/inverted for inner/outer----#
        # Use the ellipse image as a mask to erase the center of the fingerprint image
        img = cv2.bitwise_or(img, ellipse_mask)

        # ----For outer----#
        # img = cv2.bitwise_not(img)

        # plt.imshow(img)
        # plt.show()

        # Save the modified image to a new file
        cv2.imwrite(os.path.join(
            fingerprint_dir_out, 'inner_' + filename), img)
