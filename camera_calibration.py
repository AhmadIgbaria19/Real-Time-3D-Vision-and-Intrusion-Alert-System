import cv2
import numpy as np
import glob
import os

class CameraCalibration:
    def __init__(self, checkerboard_size, image_path):
        self.checkerboard_size = checkerboard_size
        self.image_path = image_path
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objpoints = []
        self.imgpoints = []
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.mse = None

    def collect_calibration_images(self):
        images = glob.glob(os.path.join(self.image_path, '*.jpg'))

        for img_file in images:
            img = cv2.imread(img_file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)

            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners2)

                cv2.drawChessboardCorners(img, self.checkerboard_size, corners2, ret)
                output_file = os.path.join(self.image_path, f"calibrated_{os.path.basename(img_file)}")
                cv2.imwrite(output_file, img)

                print(f"{os.path.basename(img_file)} successful")
            else:
                print(f"{os.path.basename(img_file)} failed")

    def calibrate_camera(self):
        if len(self.objpoints) == 0 or len(self.imgpoints) == 0:
            print("Error: No calibration data collected!")
            return

        img = cv2.imread(glob.glob(os.path.join(self.image_path, '*.jpg'))[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, self.camera_matrix, self.distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, gray.shape[::-1], None, None)

        self.compute_mse(rvecs, tvecs)

    def compute_mse(self, rvecs, tvecs):
        total_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], self.camera_matrix, self.distortion_coeffs)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error

        self.mse = total_error / len(self.objpoints)

    def print_results(self):
        print("\nCamera Calibration Results :")
        print("Camera Matrix:")
        print(self.camera_matrix)
        print("\nDistortion Coefficients:")
        print(self.distortion_coeffs)
        print(f"\nMean Reprojection Error (MSE): {self.mse:.6f}")
        print("\nCalibration completed.")