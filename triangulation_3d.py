import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

class Triangulation3D:
    def __init__(self, camera_matrix, distortion_coeffs, match_points_path, output_path):
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.match_points_path = match_points_path
        self.output_path = output_path

        self.points3d_plot_path = os.path.join(self.output_path, "3DPOINTS")
        os.makedirs(self.points3d_plot_path, exist_ok=True)

    def triangulate_points(self):
        files1 = sorted(glob.glob(os.path.join(self.match_points_path, "matched_points1_*.npy")))
        files2 = sorted(glob.glob(os.path.join(self.match_points_path, "matched_points2_*.npy")))

        if len(files1) != len(files2):
            print("Num Points dont match")
            return

        for i, (f1, f2) in enumerate(zip(files1, files2), 1):
            pts1 = np.load(f1)
            pts2 = np.load(f2)

            delta = np.linalg.norm(pts1 - pts2, axis=1)
            mask = (delta >= 4) & (delta <= 17)
            filtered_pts1 = pts1[mask]
            filtered_pts2 = pts2[mask]
            delta_filtered = delta[mask]

          
            if len(filtered_pts1) < 5:
                continue

            rvec1 = np.array([[0], [0], [0]], dtype=np.float32)
            tvec1 = np.array([[0], [0], [0]], dtype=np.float32)
            rvec2 = np.array([[0], [0], [0]], dtype=np.float32)
            tvec2 = np.array([[0.5], [0], [0]], dtype=np.float32)

            R1, _ = cv2.Rodrigues(rvec1)
            R2, _ = cv2.Rodrigues(rvec2)
            P1 = self.camera_matrix @ np.hstack((R1, tvec1))
            P2 = self.camera_matrix @ np.hstack((R2, tvec2))

            filtered_pts1 = filtered_pts1.T
            filtered_pts2 = filtered_pts2.T

            points_4d = cv2.triangulatePoints(P1, P2, filtered_pts1, filtered_pts2)
            points_3d = points_4d[:3] / points_4d[3]

            self.save_3d_points_with_stats(points_3d.T, delta_filtered, i)

        print("Triangulation3D Done!")

    def save_3d_points_with_stats(self, points_3d, delta_values, index):
        npy_path = os.path.join(self.points3d_plot_path, f"points3d_{index}.npy")
        np.save(npy_path, points_3d)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=2)
        ax.set_title(f"Triangulated 3D Points - Pair {index}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        mean_delta = np.mean(delta_values)
        max_delta = np.max(delta_values)
        min_delta = np.min(delta_values)
        num_points = len(points_3d)

        stats_text = (
            f"Δ Mean: {mean_delta:.2f}\n"
            f"Δ Max: {max_delta:.2f}\n"
            f"Δ Min: {min_delta:.2f}\n"
            f"Points: {num_points}"
        )
        ax.text2D(0.03, 0.95, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="black", alpha=0.9))

        # 🔹 גרף היסטוגרמה קטן בפינה
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_hist = inset_axes(ax, width="30%", height="30%", loc='lower right')
        ax_hist.hist(delta_values, bins=30, color='skyblue', edgecolor='black')
        ax_hist.set_title("Δ Histogram", fontsize=8)
        ax_hist.tick_params(axis='both', labelsize=6)

        plot_path = os.path.join(self.points3d_plot_path, f"points3d_{index}.png")
        plt.savefig(plot_path)
        plt.close()

