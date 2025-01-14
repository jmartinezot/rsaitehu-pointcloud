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

    # Generate random colors for each plane
    plane_colors = [np.random.rand(3) for _ in range(num_planes)]

    # Display individual plane results
    for i, (plane_model, inliers) in enumerate(segmented_planes):
        print(f"Plane {i+1}: Model coefficients {plane_model}")
        inlier_cloud = pcd.select_by_index(inliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        inlier_cloud.paint_uniform_color(plane_colors[i])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name=f"Plane {i+1}")

    # Compute the remaining point cloud as the points that are not in any inliers; take into account that
    # the indices change, so we need to join all the inliers and invert the selection
    all_inliers = np.concatenate([inliers for _, inliers in segmented_planes])
    remaining_pcd = pcd.select_by_index(all_inliers, invert=True)

    # Visualize all planes with the remainder of the original point cloud; paint the planes with random colors
    inlier_clouds = []
    for i, (plane_model, inliers) in enumerate(segmented_planes):
        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color(plane_colors[i])
        inlier_clouds.append(inlier_cloud)

    o3d.visualization.draw_geometries(inlier_clouds + [remaining_pcd], window_name="All Planes with Remaining Point Cloud")
    print("Plane segmentation completed.")