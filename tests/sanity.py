from implicit_representation.dataloaders import PointCloud2DBatman, PointCloud2DFromFont
from implicit_representation.models import SDFModelBase


batman_point_cloud = PointCloud2DBatman(sample_percentage=0.1)
data = batman_point_cloud.build()
batman_point_cloud.plot_points(save_file="plot_points_batman.png")

font_point_cloud = PointCloud2DFromFont(
    sample_percentage=0.1,
    font_file="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    font_size=200,
)
data = font_point_cloud.build(query="Q", padding=5)
font_point_cloud.plot_points(save_file="plot_points_font.png")

model = SDFModelBase(num_points=data.shape[0])
model.visualize(save_file="plot_initial_sdf.png", distance=2)
model.visualize(save_file="plot_initial_sdf_points.png", inputs=data, distance=2)
model.summary()
