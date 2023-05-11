import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

data_dir = "/home/uoriko/FingerPrintClassfication/data/"

dirs = [
    # f"{data_dir}NIST302/auxiliary/flat/M/500/plain/png/regular",
    f"{data_dir}SOCOFing/Real",
    f"{data_dir}sd04/png_txt/figs"
]

figs = [("50", 0.25)]

for input_dir in dirs:
    for fig_string, fig_num in figs:

        output_dir = f"{os.path.dirname(input_dir)}/outer{fig_string}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Starting {input_dir} outer {fig_string}")

        # Iterate over all fingerprint images in the directory
        for filename in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, filename)):
                # Load the image
                img = cv2.imread(os.path.join(input_dir, filename))
                # Convert the image to grayscale and apply thresholding
                inverted_img = cv2.bitwise_not(img)
                gray = cv2.cvtColor(inverted_img, cv2.COLOR_BGR2GRAY)

                x, y = 0, 0
                h, w = gray.shape[:2]

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

                if len(contours) is not 0:
                    # Find the contour with the largest area
                    largest_contour = max(contours, key=cv2.contourArea)

                    cv2.drawContours(
                        dilated, [largest_contour], -1, (0, 255, 0), 2)

                    # Get the bounding rectangle of the largest contour (assumed to be the fingerprint)
                    # idx = 0 if len(contours) == 1 else 1
                    x, y, w, h = cv2.boundingRect(largest_contour)

                # Calculate the center point and major/minor axes of the ellipse
                center_x = x + int(w / 2)
                center_y = y + int(h / 2)
                major_axis = int(w * fig_num)
                minor_axis = int(h * fig_num)

                # Create a white image the same size as the fingerprint
                ellipse_mask = np.full(img.shape, 0, dtype=np.uint8)

                # Draw an ellipse on the white image, we can change the angle of what's left
                cv2.ellipse(ellipse_mask, (center_x, center_y),
                            (major_axis, minor_axis), 0, 0, 360, (255, 255, 255), thickness=-1)

                # plt.imshow(ellipse_mask)
                # plt.show()

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
                    output_dir, f'outer{fig_string}_' + filename), img)
