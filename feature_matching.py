import cv2
import numpy as np
import time
import os

class FeatureMatcher:
    def __init__(self, video_source, save_path, match_save_path):
        self.video_source = video_source
        self.save_path = save_path
        self.match_save_path = match_save_path
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.pair_count = 1
        self.prev_frame = None

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if not os.path.exists(self.match_save_path):
            os.makedirs(self.match_save_path)
    def match_features(self):
        cap = cv2.VideoCapture(self.video_source)

        if not cap.isOpened():
            print("Error: Could not open video source")
            return

        start_time = time.time()  

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            elapsed_time = time.time() - start_time
            if elapsed_time > 25:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.prev_frame is None:
                self.prev_frame = gray
                continue

            kp1, des1 = self.orb.detectAndCompute(self.prev_frame, None)
            kp2, des2 = self.orb.detectAndCompute(gray, None)

            if des1 is not None and des2 is not None:
                matches = self.bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)

                img_matches = cv2.drawMatches(
                    self.prev_frame, kp1, gray, kp2, matches[:25],
                    None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )

                cv2.namedWindow("Feature Matching", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Feature Matching", 1280, 720)
                cv2.imshow("Feature Matching", img_matches)

                matched_points1 = np.array([kp1[m.queryIdx].pt for m in matches])
                matched_points2 = np.array([kp2[m.trainIdx].pt for m in matches])

                np.save(os.path.join(self.match_save_path, f"matched_points1_{self.pair_count}.npy"), matched_points1)
                np.save(os.path.join(self.match_save_path, f"matched_points2_{self.pair_count}.npy"), matched_points2)

                match_img_path = os.path.join(self.match_save_path, f"match_{self.pair_count}.jpg")
                cv2.imwrite(match_img_path, img_matches)


                self.pair_count += 1
                self.prev_frame = gray

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
