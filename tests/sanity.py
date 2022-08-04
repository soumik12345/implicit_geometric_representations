from implicit_representation.dataloaders import PointCloud2DBatman


batman_point_cloud = PointCloud2DBatman(sample_percentage=0.1)
data = batman_point_cloud.build()
batman_point_cloud.plot_points(save_file="plot.png")
