import os
import numpy as np
import matplotlib.pyplot as plt
import glob

class FeatureMatchVisualizer:
    def __init__(self, project_root_path):
        self.match_points_path = "C:\\Users\\King\\Downloads\\ProjectRealTimeSystems\\matched_points"
        self.visuals_root = os.path.join(project_root_path, "FeatureMatchPlots")
        os.makedirs(self.visuals_root, exist_ok=True)

    def visualize_all(self):
        files1 = sorted(glob.glob(os.path.join(self.match_points_path, "matched_points1_*.npy")))
        files2 = sorted(glob.glob(os.path.join(self.match_points_path, "matched_points2_*.npy")))

        for f1, f2 in zip(files1, files2):
            index = f1.split("_")[-1].split(".")[0]
            points1 = np.load(f1)
            points2 = np.load(f2)

            self.plot_lines(points1, points2, index)
            self.plot_scatter(points1, points2, index)
            self.plot_motion_vectors(points1, points2, index)
            self.plot_cumulative_histogram(points1, points2, index)
            self.plot_angle_histogram(points1, points2, index)

        print(f"✅ All visualizations saved in: {self.visuals_root}")

    def compute_delta(self, pts1, pts2):
        return np.linalg.norm(pts1 - pts2, axis=1)


    def plot_lines(self, pts1, pts2, index):
        delta = self.compute_delta(pts1, pts2)
        mean_delta = np.mean(delta)
        max_delta = np.max(delta)
        min_delta = np.min(delta)

        plt.figure(figsize=(6, 6))
        for pt1, pt2 in zip(pts1, pts2):
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='gray', alpha=0.5, linewidth=1)
        plt.scatter(pts1[:, 0], pts1[:, 1], color='blue', s=10, label='Frame t')
        plt.scatter(pts2[:, 0], pts2[:, 1], color='red', s=10, label='Frame t+1')
        plt.gca().invert_yaxis()
        plt.legend()
        plt.title("Lines Between Matched Keypoints")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.text(10, 30, f"Points: {len(delta)}", fontsize=9)
        plt.text(10, 50, f"Δ Mean: {mean_delta:.2f}", fontsize=9)
        plt.text(10, 70, f"Δ Max: {max_delta:.2f}", fontsize=9)
        plt.text(10, 90, f"Δ Min: {min_delta:.2f}", fontsize=9)
        plt.savefig(os.path.join(self.visuals_root, f"lines_points_{index}.png"))
        plt.close()

    def plot_scatter(self, pts1, pts2, index):
        plt.figure(figsize=(6, 6))
        plt.scatter(pts1[:, 0], pts1[:, 1], s=10, label='Frame t', color='blue')
        plt.scatter(pts2[:, 0], pts2[:, 1], s=10, label='Frame t+1', color='orange')
        plt.gca().invert_yaxis()
        plt.legend()
        plt.title("2D Scatter Plot of Keypoints")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig(os.path.join(self.visuals_root, f"scatter_points_{index}.png"))
        plt.close()

    def plot_motion_vectors(self, pts1, pts2, index):
        plt.figure(figsize=(6, 6))
        plt.quiver(pts1[:, 0], pts1[:, 1],
                   pts2[:, 0] - pts1[:, 0], pts2[:, 1] - pts1[:, 1],
                   angles='xy', scale_units='xy', scale=1, color='purple')
        plt.gca().invert_yaxis()
        plt.title("Motion Vectors Between Keypoints")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig(os.path.join(self.visuals_root, f"motion_vectors_{index}.png"))
        plt.close()

    def plot_cumulative_histogram(self, pts1, pts2, index):
        delta = self.compute_delta(pts1, pts2)
        sorted_d = np.sort(delta)
        cumulative = np.arange(len(delta)) / len(delta)
        plt.figure(figsize=(6, 4))
        plt.plot(sorted_d, cumulative, color='green')
        plt.title("Cumulative Histogram of Δ")
        plt.xlabel("Δ (Pixel Distance)")
        plt.ylabel("Cumulative Percentage")
        plt.grid(True)
        plt.savefig(os.path.join(self.visuals_root, f"cumulative_histogram_{index}.png"))
        plt.close()


    def plot_angle_histogram(self, pts1, pts2, index):
        vectors = pts2 - pts1
        angles = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
        plt.figure(figsize=(6, 4))
        plt.hist(angles, bins=36, color='teal', edgecolor='black')
        plt.title("Histogram of Motion Angles")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(os.path.join(self.visuals_root, f"angle_histogram_{index}.png"))
        plt.close()
