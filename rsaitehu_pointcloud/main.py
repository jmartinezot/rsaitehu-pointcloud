import open3d as o3d
from typing import List, Tuple, Dict, Any
import numpy as np
from copy import deepcopy

def pointcloud_audit(pcd: o3d.geometry.PointCloud) -> Dict[str, Any]:
    '''
    Audit point cloud.

    This function audits a point cloud and returns a dictionary containing various statistics and properties of the point cloud.

    :param pcd: Point cloud.
    :type pcd: o3d.geometry.PointCloud
    :return: Dictionary of audit results.
    :rtype: Dict[str, Any]

    :Example:

    .. code-block:: python

        >>> import open3d as o3d
        >>> office_dataset = o3d.data.OfficePointClouds()
        >>> office_filename = office_dataset.paths[0]
        >>> pcd = o3d.io.read_point_cloud(office_filename)
        >>> results = pointcloud_audit(pcd)
        >>> print(results)

    The returned dictionary contains the following keys:

    - **number_pcd_points**: Number of points in the point cloud.
    - **has_normals**: Boolean indicating if the point cloud has normals.
    - **has_colors**: Boolean indicating if the point cloud has colors.
    - **is_empty**: Boolean indicating if the point cloud is empty.
    - **max_x**: Maximum x-coordinate value (if not empty).
    - **min_x**: Minimum x-coordinate value (if not empty).
    - **max_y**: Maximum y-coordinate value (if not empty).
    - **min_y**: Minimum y-coordinate value (if not empty).
    - **max_z**: Maximum z-coordinate value (if not empty).
    - **min_z**: Minimum z-coordinate value (if not empty).
    - **all_points_finite**: Boolean indicating if all points are finite (if not empty).
    - **all_points_unique**: Boolean indicating if all points are unique (if not empty).
    '''
    dict_results = {}
    number_pcd_points = len(pcd.points)
    dict_results["number_pcd_points"] = number_pcd_points
    dict_results["has_normals"] = pcd.has_normals()
    dict_results["has_colors"] = pcd.has_colors()
    dict_results["is_empty"] = pcd.is_empty()
    if not pcd.is_empty():
        # get maximum and minimum values
        np_points = np.asarray(pcd.points)
        dict_results["max_x"] = np.max(np_points[:, 0])
        dict_results["min_x"] = np.min(np_points[:, 0])
        dict_results["max_y"] = np.max(np_points[:, 1])
        dict_results["min_y"] = np.min(np_points[:, 1])
        dict_results["max_z"] = np.max(np_points[:, 2])
        dict_results["min_z"] = np.min(np_points[:, 2])
        # check if all points are finite
        dict_results["all_points_finite"] = np.all(np.isfinite(np_points))
        # check if all points are unique
        dict_results["all_points_unique"] = len(np_points) == len(np.unique(np_points, axis=0))
    return dict_results

def pointcloud_sanitize(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    '''
    Sanitize point cloud.

    This function sanitizes a point cloud by removing non-finite points and duplicate points. If the point cloud has colors, the associated colors of the removed points are also removed.

    :param pcd: Point cloud.
    :type pcd: o3d.geometry.PointCloud
    :return: Sanitized point cloud.
    :rtype: o3d.geometry.PointCloud

    :Example:

    .. code-block:: python

        >>> import open3d as o3d
        >>> office_dataset = o3d.data.OfficePointClouds()
        >>> office_filename = office_dataset.paths[0]
        >>> pcd = o3d.io.read_point_cloud(office_filename)
        >>> sanitized_pcd = pointcloud_sanitize(pcd)
        >>> o3d.visualization.draw_geometries([sanitized_pcd])

    The function performs the following steps:

    - Checks if the point cloud is empty.
    - Removes non-finite points and their associated colors (if any).
    - Removes duplicate points and their associated colors (if any).
    '''
    if pcd.is_empty():
        return pcd
    # remove non-finite points and associated colors
    pcd_has_colors = pcd.has_colors()
    np_points = np.asarray(pcd.points)
    finite_mask = np.isfinite(np_points).all(axis=1)
    np_points = np_points[finite_mask]
    if pcd_has_colors:
        np_colors = np.asarray(pcd.colors)
        np_colors = np_colors[finite_mask]
    # remove duplicate points and associated colors
    _, unique_indices = np.unique(np_points, axis=0, return_index=True)
    pcd.points = o3d.utility.Vector3dVector(np_points[unique_indices])
    if pcd_has_colors:
        pcd.colors = o3d.utility.Vector3dVector(np_colors[unique_indices])
    return pcd

def get_pointcloud_after_subtracting_point_cloud(pcd: o3d.geometry.PointCloud, subtract: o3d.geometry.PointCloud,
                                                 threshold: float = 0.05) -> o3d.geometry.PointCloud:
    """
    Subtracts one point cloud from another. It removes all the points of the first point cloud that are
    closer than *threshold* to some point of the second point cloud.

    :param pcd: Point cloud to subtract from.
    :type pcd: o3d.geometry.PointCloud
    :param subtract: Point cloud to subtract.
    :type subtract: o3d.geometry.PointCloud
    :param threshold: If a point of the first point cloud is closer to some point of the second point cloud than this value, the point is removed.
    :type threshold: float
    :return: The result after subtracting the second point cloud from the first point cloud.
    :rtype: o3d.geometry.PointCloud

    :Example:

    .. code-block:: python

        >>> import open3d as o3d
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=5.0, depth=1.0)
        >>> pcd_1 = mesh_box.sample_points_uniformly(number_of_points=10000)
        >>> mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.5, height=4.0, depth=0.5)
        >>> pcd_2 = mesh_box.sample_points_uniformly(number_of_points=10000)
        >>> pcd_1.paint_uniform_color([1, 0, 0])
        >>> pcd_2.paint_uniform_color([0, 1, 0])
        >>> pcd_1_minus_pcd_2 = get_pointcloud_after_subtracting_point_cloud(pcd_1, pcd_2, threshold=0.02)
        >>> print(pcd_1_minus_pcd_2)
        PointCloud with 5861 points.
        >>> o3d.visualization.draw_geometries([pcd_1, pcd_2])
        >>> o3d.visualization.draw_geometries([pcd_1_minus_pcd_2])
        >>> pcd_2_minus_pcd_1 = get_pointcloud_after_subtracting_point_cloud(pcd_2, pcd_1, threshold=0.02)
        >>> print(pcd_2_minus_pcd_1)
        PointCloud with 4717 points.
        >>> o3d.visualization.draw_geometries([pcd_2_minus_pcd_1])

    The function performs the following steps:

    - Constructs a KDTree from the points of the second point cloud.
    - Iterates through the points of the first point cloud.
    - For each point in the first point cloud, checks if it is closer than the threshold to any point in the second point cloud.
    - If a point is closer than the threshold, it is removed from the first point cloud.
    - Returns the resulting point cloud after subtraction.
    """
    def aux_func(x, y, z):
        [_, _, d] = pcd_tree.search_knn_vector_3d([x, y, z], knn=1)
        return d[0]

    pcd_tree = o3d.geometry.KDTreeFlann(subtract)
    points = np.asarray(pcd.points)
    if len(pcd.colors) == 0:
        remaining_points = [point for point in points if
                            aux_func(point[0], point[1], point[2]) > threshold]
        pcd_result = o3d.geometry.PointCloud()
        pcd_result.points = o3d.utility.Vector3dVector(np.asarray(remaining_points))
        return pcd_result
    colors = np.asarray(pcd.colors)
    remaining_points_and_colors = [(point, color) for point, color in zip(points, colors) if
                                   aux_func(point[0], point[1], point[2]) > threshold]
    remaining_points = [item[0] for item in remaining_points_and_colors]
    remaining_colors = [item[1] for item in remaining_points_and_colors]
    pcd_result = o3d.geometry.PointCloud()
    pcd_result.points = o3d.utility.Vector3dVector(np.asarray(remaining_points))
    pcd_result.colors = o3d.utility.Vector3dVector(np.asarray(remaining_colors))
    return pcd_result

def segment_planes(pcd: o3d.geometry.PointCloud, num_planes: int, distance_threshold: float = 0.01, ransac_n: int = 3, num_iterations: int = 1000):
    """
    Segments planes from the point cloud.

    This function segments a specified number of planes from the point cloud using the RANSAC algorithm.

    :param pcd: Point cloud to segment.
    :type pcd: o3d.geometry.PointCloud
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

    .. code-block:: python

        >>> import open3d as o3d
        >>> pcd = o3d.io.read_point_cloud("example.ply")
        >>> enhanced_pcd = EnhancedPointCloud(pcd)
        >>> planes = segment_planes(num_planes=2)
        >>> for plane_model, inliers in planes:
        >>>     print("Plane model:", plane_model)
        >>>     print("Number of inliers:", len(inliers))
    """
    if num_planes <= 0:
        raise ValueError("The number of planes must be greater than zero.")
    if pcd.is_empty():
        raise ValueError("The point cloud is empty.")

    planes = []
    remaining_pcd = deepcopy(pcd)  # Clone the point cloud
    original_indices = np.arange(len(pcd.points))  # Track indices in the original point cloud

    for i in range(num_planes):
        if remaining_pcd.is_empty():
            print(f"Segmentation stopped early. Remaining point cloud is empty after {i} planes.")
            break

        # Segment a plane from the remaining point cloud
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        print(f"Plane {i+1}: {len(inliers)} inliers.")

        # Map the inliers back to the original point cloud using original_indices
        original_inliers = original_indices[inliers]

        # Store the plane model and inliers in terms of the original point cloud
        planes.append((plane_model, original_inliers.tolist()))

        # Update remaining_pcd by excluding inliers
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

        # Update original_indices by excluding the inliers
        original_indices = np.delete(original_indices, inliers)

    return planes

def get_pcd_surrounding_plane(pcd: o3d.geometry.PointCloud, plane: tuple, distance_threshold: float = 0.01) -> o3d.geometry.PointCloud:
    """
    Get the point cloud surrounding a plane.

    This function returns the point cloud surrounding a plane within a specified distance threshold.

    :param plane: A tuple (a, b, c, d) representing the plane equation ax + by + cz + d = 0.
    :type plane: tuple
    :param distance_threshold: Maximum distance a point can have to the plane to be considered surrounding.
    :type distance_threshold: float
    :return: The point cloud surrounding the plane.
    :rtype: o3d.geometry.PointCloud

    :Example:

    >>> import open3d as o3d
    >>> pcd = o3d.io.read_point_cloud("example.ply")
    >>> enhanced_pcd = EnhancedPointCloud(pcd)
    >>> surrounding_pcd = enhanced_pcd.get_pcd_surrounding_plane((1, 1, -1, -1))
    """
    if pcd.is_empty():
        raise ValueError("The point cloud is empty.")

    # Extract the plane coefficients
    a, b, c, d = plane

    # Convert point cloud to NumPy array
    points = np.asarray(self.pcd.points)

    # Compute the signed distance of each point to the plane
    distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)

    # Select points within the distance threshold
    surrounding_indices = np.where(distances <= distance_threshold)[0]

    # Extract the surrounding point cloud
    surrounding_pcd = pcd.select_by_index(surrounding_indices)

    return surrounding_pcd

def split_pointcloud_by_plane(pointcloud, plane):
    """
    Splits a point cloud into two point clouds based on a plane. Points are divided into two clouds
    depending on which side of the plane they lie. Colors and normals are preserved in the resulting point clouds.

    :param pointcloud: The input point cloud to be split.
    :type pointcloud: o3d.geometry.PointCloud
    :param plane: A tuple (a, b, c, d) representing the plane equation ax + by + cz + d = 0.
    :type plane: tuple
    :return: A tuple of two point clouds:
             - The first contains points on the positive side of the plane.
             - The second contains points on the negative or coincident side of the plane.
    :rtype: tuple(o3d.geometry.PointCloud, o3d.geometry.PointCloud)

    :Example:

    .. code-block:: python

        >>> import open3d as o3d
        >>> import numpy as np
        >>> # Load a point cloud
        >>> office_dataset = o3d.data.OfficePointClouds()
        >>> office_filename = office_dataset.paths[0]
        >>> pcd = o3d.io.read_point_cloud(office_filename)
        >>> # Define a plane (e.g., x + y - z - 1 = 0 -> a=1, b=1, c=-1, d=-1)
        >>> plane = (1, 1, -1, -1)
        >>> # Split the point cloud
        >>> positive_pcd, negative_pcd = split_pointcloud_by_plane(pcd, plane)
        >>> # Visualize the result
        >>> o3d.visualization.draw_geometries([positive_pcd], window_name="Positive Side")
        >>> o3d.visualization.draw_geometries([negative_pcd], window_name="Negative Side")

    The function performs the following steps:

    - Computes the distance of each point in the input point cloud from the plane using the plane equation.
    - Points with a positive distance are added to the first output point cloud.
    - Points with a non-positive distance are added to the second output point cloud.
    - Returns the two resulting point clouds.

    """
    # Extract plane parameters
    a, b, c, d = plane
    normal = np.array([a, b, c])
    
    # Convert point cloud to numpy arrays
    points = np.asarray(pointcloud.points)
    colors = np.asarray(pointcloud.colors) if pointcloud.has_colors() else None
    normals = np.asarray(pointcloud.normals) if pointcloud.has_normals() else None
    
    # Calculate the distance of each point from the plane
    distances = np.dot(points, normal) + d
    
    # Split points based on their distance to the plane
    positive_indices = distances > 0
    negative_indices = distances <= 0
    
    positive_points = points[positive_indices]
    negative_points = points[negative_indices]
    
    # Create point clouds for each side
    positive_pointcloud = o3d.geometry.PointCloud()
    negative_pointcloud = o3d.geometry.PointCloud()
    positive_pointcloud.points = o3d.utility.Vector3dVector(positive_points)
    negative_pointcloud.points = o3d.utility.Vector3dVector(negative_points)
    
    # Add colors if they exist
    if colors is not None:
        positive_pointcloud.colors = o3d.utility.Vector3dVector(colors[positive_indices])
        negative_pointcloud.colors = o3d.utility.Vector3dVector(colors[negative_indices])
    
    # Add normals if they exist
    if normals is not None:
        positive_pointcloud.normals = o3d.utility.Vector3dVector(normals[positive_indices])
        negative_pointcloud.normals = o3d.utility.Vector3dVector(normals[negative_indices])

    # return dictionary of positive and negative point clouds
    return {"positive": positive_pointcloud, "negative": negative_pointcloud}