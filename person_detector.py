import cv2
import numpy as np
from ultralytics import YOLO
import smtplib
import ssl
from email.message import EmailMessage
import os
from datetime import datetime

class PersonDetector:
    def __init__(self, camera_matrix, dist_coeffs):

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.model = YOLO("yolov8s.pt")
        self.email_sender = "ahmdalicr7@gmail.com"
        self.email_password = "zosj ifns gxxr dqiw"
        self.email_receiver = "ahmdalicr7@gmail.com"
        self.output_folder = "DetectedPerson"
        os.makedirs(self.output_folder, exist_ok=True)
        self.person_detected = False

    def send_email(self, image_path):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = EmailMessage()
        msg["Subject"] = "Important Mail!! Warning, Person Detected at Home!"
        msg["From"] = self.email_sender
        msg["To"] = self.email_receiver
        msg.set_content(f"Dear Mr. Igbaria Ahmad,\n\nA person has been detected entering your home at {now}.\nPlease find the attached image.\n\nStay safe.")

        with open(image_path, "rb") as f:
            img_data = f.read()
            msg.add_attachment(img_data, maintype="image", subtype="jpeg", filename=os.path.basename(image_path))

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(self.email_sender, self.email_password)
            server.send_message(msg)
            print("Email sent!")

    def run(self):
        video_source = "http://192.168.1.7:8080/video"
        cap = cv2.VideoCapture(video_source)
        flash = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1280, 720))
            frame_undistorted = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
            annotated_frame = frame_undistorted.copy()

            results = self.model(frame_undistorted, verbose=False)[0]
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            person_found = False

            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                if class_name.lower() == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(annotated_frame, "Person", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    person_found = True
                    break

            cv2.putText(annotated_frame, now, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            if person_found:
                flash = not flash
                if flash:
                    annotated_frame[:10, :] = [0, 0, 255]
                    annotated_frame[-10:, :] = [0, 0, 255]
                    annotated_frame[:, :10] = [0, 0, 255]
                    annotated_frame[:, -10:] = [0, 0, 255]

                cv2.putText(annotated_frame, "WARNING !!!", (900, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

                if not self.person_detected:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = os.path.join(self.output_folder, f"person_detected_{timestamp}.jpg")
                    cv2.putText(annotated_frame, now, (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imwrite(image_path, annotated_frame)
                    self.send_email(image_path)
                    self.person_detected = True
            else:
                flash = False
                self.person_detected = False

            cv2.imshow("Person Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
