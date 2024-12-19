import open3d as o3d
import numpy as np
from rsaitehu_pointcloud import EnhancedPointCloud

if __name__ == "__main__":
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()

    # Populate point clouds with sample points
    pcd1.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
    pcd2.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))

    enhanced_pcd1 = EnhancedPointCloud(pcd1)
    enhanced_pcd2 = EnhancedPointCloud(pcd2)

    enhanced_pcd1.set_subtraction_threshold(0.05)
    combined_pcd = enhanced_pcd1 + enhanced_pcd2
    subtracted_pcd = enhanced_pcd1 - enhanced_pcd2

    print("Combined Point Cloud Audit Results:", combined_pcd.get_audit_results())
    print("Subtracted Point Cloud Audit Results:", subtracted_pcd.get_audit_results())

    # Visualize the point clouds
    o3d.visualization.draw_geometries([enhanced_pcd1.pcd], window_name="Point Cloud 1")
    o3d.visualization.draw_geometries([enhanced_pcd2.pcd], window_name="Point Cloud 2")
    o3d.visualization.draw_geometries([combined_pcd.pcd], window_name="Combined Point Cloud")
    o3d.visualization.draw_geometries([subtracted_pcd.pcd], window_name="Subtracted Point Cloud")
