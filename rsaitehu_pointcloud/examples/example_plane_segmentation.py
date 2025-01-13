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

    print(f"Initial number of points in the point cloud: {len(np.asarray(remaining_pcd.points))}")

    for i, (_, inliers) in enumerate(segmented_planes):
        print(f"Processing Plane {i+1} with {len(inliers)} inliers.")
        
        # Select the inlier cloud
        try:
            inlier_cloud = remaining_pcd.select_by_index(inliers)
            inlier_cloud.paint_uniform_color([np.random.rand(), np.random.rand(), np.random.rand()])
            plane_clouds.append(inlier_cloud)
            print(f"Plane {i+1}: Added inlier cloud with {len(inliers)} points.")
        except Exception as e:
            print(f"Error while processing Plane {i+1}: {e}")
            raise

        # Update remaining point cloud with a deep copy
        try:
            remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True).copy()
            print(f"Remaining point cloud now has {len(np.asarray(remaining_pcd.points))} points.")

            # Check validity of remaining point cloud
            if not remaining_pcd.has_points():
                print(f"Error: Remaining point cloud is invalid after Plane {i+1}.")
                break
        except Exception as e:
            print(f"Error while updating remaining point cloud for Plane {i+1}: {e}")
            raise

    # Add the remaining point cloud with original colors
    try:
        plane_clouds.append(remaining_pcd)
        print(f"Final remaining point cloud added with {len(np.asarray(remaining_pcd.points))} points.")
    except Exception as e:
        print(f"Error while adding the remaining point cloud: {e}")
        raise

    # Visualize all together
    try:
        print(f"Visualizing {len(plane_clouds)} point clouds.")
        o3d.visualization.draw_geometries(plane_clouds, window_name="All Planes and Remaining Point Cloud")
    except Exception as e:
        print(f"Error during visualization: {e}")
        raise


    print("Plane segmentation completed.")