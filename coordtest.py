
import numpy as np

H = np.array([
    [2.8128700e-02, 2.0091900e-03, -4.6693600e+00],
    [8.0625700e-04, 2.5195500e-02, -5.0608800e+00],
    [3.4555400e-04, 9.2512200e-05, 4.6255300e-01]
])
# 像素坐标
pixel_coords = np.array([[559, 309], [557, 310], [558, 314], [556, 316], [554, 317], [554, 319], [550, 324], [548, 326]])

# 将像素坐标转换为齐次坐标，添加一列1
pixel_coords_homogeneous = np.column_stack((pixel_coords, np.ones((pixel_coords.shape[0], 1))))

# 使用单应矩阵对齐次坐标进行变换
world_coords_homogeneous = np.dot(H, pixel_coords_homogeneous.T).T

# 将变换后的齐次坐标转换为非齐次坐标
world_coords = world_coords_homogeneous[:, :2] / world_coords_homogeneous[:, 2:]

print(world_coords)
