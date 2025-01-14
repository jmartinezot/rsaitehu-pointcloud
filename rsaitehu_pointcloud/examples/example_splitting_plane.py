import open3d as o3d
import numpy as np
from rsaitehu_pointcloud import EnhancedPointCloud

# Load a point cloud
office_dataset = o3d.data.OfficePointClouds()
office_filename = office_dataset.paths[0]
pcd = o3d.io.read_point_cloud(office_filename)
# Define a plane (e.g., x + y - z - 1 = 0 -> a=1, b=1, c=-1, d=-1)
plane = (1, 1, -1, -1)
# Create an EnhancedPointCloud instance
enhanced_pcd = EnhancedPointCloud(pcd)
# Split the point cloud
split_pcd = enhanced_pcd.split_by_plane(plane)
positive_pcd = split_pcd["positive"]
negative_pcd = split_pcd["negative"]
# Visualize the result
o3d.visualization.draw_geometries([positive_pcd], window_name="Positive Side")
o3d.visualization.draw_geometries([negative_pcd], window_name="Negative Side")
# Visualize the two point clouds together painted in green the positive side and in red the negative side
positive_pcd.paint_uniform_color([0, 1, 0])
negative_pcd.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([positive_pcd, negative_pcd], window_name="Positive and Negative Sides")
