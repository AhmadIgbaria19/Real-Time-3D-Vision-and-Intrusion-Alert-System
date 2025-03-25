import cv2
import numpy as np
import glob
import os

class ImageUndistorter:
    def __init__(self, camera_matrix, distortion_coeffs, image_path):
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.image_path = image_path

    def undistort_images(self):
        images = glob.glob(os.path.join(self.image_path, 'calibrated_*.jpg'))  # מחפש תמונות עם פינות מסומנות

        if not images:
            print("Error: No images found for undistortion!")
            return

        for img_file in images:
            img = cv2.imread(img_file)
            h, w = img.shape[:2]

            # חישוב מטריצת המצלמה החדשה
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion_coeffs, (w, h), 1, (w, h))

            # הסרת העיוות מהתמונה
            undistorted_img = cv2.undistort(img, self.camera_matrix, self.distortion_coeffs, None, new_camera_matrix)

            # חיתוך ה-ROI כדי להסיר את המתיחות בצדדים
            x, y, w, h = roi
            undistorted_img = undistorted_img[y:y+h, x:x+w]

            # שמירת התמונה המתוקנת
            output_file = os.path.join(self.image_path, f"undistorted_{os.path.basename(img_file)}")
            cv2.imwrite(output_file, undistorted_img)

            print(f"{os.path.basename(img_file)} → תיקון עיוות נשמר כ- {os.path.basename(output_file)}")