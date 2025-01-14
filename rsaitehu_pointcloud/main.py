import open3d as o3d
from typing import List, Tuple, Dict, Any
import numpy as np

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