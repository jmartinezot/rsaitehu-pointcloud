import open3d as o3d
import numpy as np
from typing import Dict, Any
from .main import pointcloud_audit, get_pointcloud_after_subtracting_point_cloud

class EnhancedPointCloud:
    def __init__(self, pcd: o3d.geometry.PointCloud):
        """
        Initialize an EnhancedPointCloud object.

        :param pcd: The input point cloud.
        :type pcd: o3d.geometry.PointCloud
        """
        self.pcd = pcd
        self.audit_results = pointcloud_audit(pcd)
        self.subtraction_threshold = 0.05  # Default threshold value

    def __add__(self, other: "EnhancedPointCloud") -> "EnhancedPointCloud":
        """
        Add two EnhancedPointCloud objects.

        :param other: Another EnhancedPointCloud object.
        :type other: EnhancedPointCloud
        :return: A new EnhancedPointCloud object containing the combined point clouds.
        :rtype: EnhancedPointCloud
        """
        combined_pcd = self.pcd + other.pcd
        return EnhancedPointCloud(combined_pcd)

    def __sub__(self, other: "EnhancedPointCloud") -> "EnhancedPointCloud":
        """
        Subtract one EnhancedPointCloud from another.

        :param other: Another EnhancedPointCloud object.
        :type other: EnhancedPointCloud
        :return: A new EnhancedPointCloud object after subtraction.
        :rtype: EnhancedPointCloud
        """
        result_pcd = get_pointcloud_after_subtracting_point_cloud(self.pcd, other.pcd, self.subtraction_threshold)
        return EnhancedPointCloud(result_pcd)

    def set_subtraction_threshold(self, threshold: float):
        """
        Set the threshold for point cloud subtraction.

        :param threshold: The threshold value.
        :type threshold: float
        """
        self.subtraction_threshold = threshold

    def get_audit_results(self) -> Dict[str, Any]:
        """
        Get the audit results of the point cloud.

        :return: A dictionary containing the audit results.
        :rtype: Dict[str, Any]
        """
        return self.audit_results

    def segment_planes(self, num_planes: int, distance_threshold: float = 0.01, ransac_n: int = 3, num_iterations: int = 1000):
        """
        Segments planes from the point cloud.

        This function segments a specified number of planes from the point cloud using the RANSAC algorithm.

        :param num_planes: Number of planes to segment.
        :type num_planes: int
        :param distance_threshold: Maximum distance a point can have to an estimated plane to be considered an inlier.
        :type distance_threshold: float
        :param ransac_n: Number of points to sample for generating a plane model.
        :type ransac_n: int
        :param num_iterations: Number of iterations to run RANSAC.
        :type num_iterations: int
        :return: List of tuples containing the plane model and the inliers' indices in the original point cloud.
        :rtype: List[Tuple[np.ndarray, List[int]]]

        :Example:

        >>> import open3d as o3d
        >>> pcd = o3d.io.read_point_cloud("example.ply")
        >>> enhanced_pcd = EnhancedPointCloud(pcd)
        >>> planes = enhanced_pcd.segment_planes(num_planes=2)
        >>> for plane_model, inliers in planes:
        >>>     print("Plane model:", plane_model)
        >>>     print("Number of inliers:", len(inliers))
        """
        if num_planes <= 0:
            raise ValueError("The number of planes must be greater than zero.")
        if self.pcd.is_empty():
            raise ValueError("The point cloud is empty.")

        planes = []
        remaining_pcd = self.pcd
        original_indices = np.arange(len(self.pcd.points))

        for _ in range(num_planes):
            if remaining_pcd.is_empty():
                print("Warning: The point cloud became empty during segmentation.")
                break

            plane_model, inliers = remaining_pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations
            )
            planes.append((plane_model, original_indices[inliers].tolist()))
            remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
            original_indices = original_indices[np.setdiff1d(np.arange(len(original_indices)), inliers)]

        return planes