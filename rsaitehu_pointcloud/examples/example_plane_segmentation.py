import open3d as o3d
import numpy as np
from rsaitehu_pointcloud import EnhancedPointCloud

if __name__ == "__main__":
    # Load the office dataset
    office_dataset = o3d.data.OfficePointClouds()
    office_filename = office_dataset.paths[0]
    pcd = o3d.io.read_point_cloud(office_filename)

    # Create an EnhancedPointCloud instance
    enhanced_pcd = EnhancedPointCloud(pcd)

    # Perform plane segmentation
    num_planes = 3
    segmented_planes = enhanced_pcd.segment_planes(num_planes=num_planes, distance_threshold=0.002, ransac_n=3, num_iterations=1000)

    # Display individual plane results
    for i, (plane_model, inliers) in enumerate(segmented_planes):
        print(f"Plane {i+1}: Model coefficients {plane_model}")
        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([np.random.rand(), np.random.rand(), np.random.rand()])
        o3d.visualization.draw_geometries([inlier_cloud], window_name=f"Plane {i+1}")

    # Visualize all planes with the remainder of the original point cloud
    remaining_pcd = pcd
    plane_clouds = []
    for i, (_, inliers) in enumerate(segmented_planes):
        inlier_cloud = remaining_pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([np.random.rand(), np.random.rand(), np.random.rand()])
        plane_clouds.append(inlier_cloud)
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

    # Add the remaining point cloud with original colors
    plane_clouds.append(remaining_pcd)

    # Visualize all together
    o3d.visualization.draw_geometries(plane_clouds, window_name="All Planes and Remaining Point Cloud")

    print("Plane segmentation completed.")