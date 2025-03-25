from camera_calibration import CameraCalibration
from image_undistortion import ImageUndistorter
from feature_matching import FeatureMatcher
from triangulation_3d import Triangulation3D
from feature_match_visualizer import FeatureMatchVisualizer 
from person_detector import PersonDetector 

class Main:
    def __init__(self, checkerboard_size, image_path, video_source, match_save_path, output_3d_path):
        self.image_path = image_path
        self.video_source = video_source
        self.match_save_path = match_save_path
        self.output_3d_path = output_3d_path
        self.calibration = CameraCalibration(checkerboard_size, image_path)
        self.undistorter = None
        self.feature_matching = None
        self.triangulation = None

    def run(self):
        print("\nCamera Calibration : ")
        self.calibration.collect_calibration_images()
        self.calibration.calibrate_camera()
        self.calibration.print_results()

        if self.calibration.camera_matrix is not None and self.calibration.distortion_coeffs is not None:
            print("\nStep 2 : ImageUndistorter  ")
            self.undistorter = ImageUndistorter(self.calibration.camera_matrix, self.calibration.distortion_coeffs, self.image_path)
            self.undistorter.undistort_images()

            print("\nStep 3 :FeatureMatcher ")
            self.feature_matching = FeatureMatcher(self.video_source, self.image_path, self.match_save_path)
            self.feature_matching.match_features()

            print("\nPlotsForTheKeyPoints")
            ProjectPath = r"C:\Users\King\Downloads\ProjectRealTimeSystems"
            visualizer = FeatureMatchVisualizer(ProjectPath)

            visualizer.visualize_all()

            print("\n🔹Step4: Triangulation3D")
            self.triangulation = Triangulation3D(self.calibration.camera_matrix, self.calibration.distortion_coeffs, self.match_save_path, self.output_3d_path)
            self.triangulation.triangulate_points()

            #print("\nStep5 : PersonDetector  ")
            #detector = PersonDetector(self.calibration.camera_matrix, self.calibration.distortion_coeffs)
            #detector.run()

if __name__ == "__main__":
    image_path = r"C:\Users\King\Downloads\ProjectRealTimeSystems\checkboardImages"
    match_save_path = r"C:\Users\King\Downloads\ProjectRealTimeSystems\matched_points"
    output_3d_path = r"C:\Users\King\Downloads\ProjectRealTimeSystems\triangulated_3D"
    checkerboard_size = (10, 7)
    video_source = "http://192.168.1.7:8080/video"

    main_system = Main(checkerboard_size, image_path, video_source, match_save_path, output_3d_path)
    main_system.run()

