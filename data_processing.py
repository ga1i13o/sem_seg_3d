import numpy as np
import random
import struct
import os
from glob import glob

# pyTorch imports
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# visualization
import open3d as o3d
import plotly.graph_objects as go

    
semantic_kitti_color_scheme = {
0 : [0, 0, 0],        # "unlabeled"
0 : [0, 0, 255],      # "outlier"
1: [245, 150, 100],  # "car"
2: [245, 230, 100],  # "bicycle"
5: [250, 80, 100],   # "bus"
3: [150, 60, 30],    # "motorcycle"
5: [255, 0, 0],      # "on-rails"
4: [180, 30, 80],    # "truck"
5: [255, 0, 0],      # "other-vehicle"
6: [30, 30, 255],    # "person"
7: [200, 40, 255],   # "bicyclist"
8: [90, 30, 150],    # "motorcyclist"
9: [255, 0, 255],    # "road"
10: [255, 150, 255],  # "parking"
11: [75, 0, 75],      # "sidewalk"
12: [75, 0, 175],     # "other-ground"
13: [0, 200, 255],    # "building"
14: [50, 120, 255],   # "fence"
0: [0, 150, 255],    # "other-structure"
9: [170, 255, 150],  # "lane-marking"
15: [0, 175, 0],      # "vegetation"
16: [0, 60, 135],     # "trunk"
17: [80, 240, 150],   # "terrain"
18: [150, 240, 255],  # "pole"
19: [0, 0, 255],      # "traffic-sign"
0: [255, 255, 50],   # "other-object"
 20: [245, 150, 100], # "moving-car"
 21: [200, 40, 255],  # "moving-bicyclist"
 22: [30, 30, 255],   # "moving-person"
 23: [90, 30, 150],   # "moving-motorcyclist"
 24: [255, 0, 0],     # "moving-on-rails"
 24: [250, 80, 100],  # "moving-bus"
 25: [180, 30, 80],   # "moving-truck"
 24: [255, 0, 0],     # "moving-other-vehicle"
}

label_remap= {
  0 : 0  ,    # "unlabeled"
  1 : 0  ,   # "outlier" mapped to "unlabeled" --------------------------mapped
  10: 1  ,   # "car"
  11: 2  ,   # "bicycle"
  13: 5  ,   # "bus" mapped to "other-vehicle" --------------------------mapped
  15: 3  ,   # "motorcycle"
  16: 5  ,   # "on-rails" mapped to "other-vehicle" ---------------------mapped
  18: 4  ,   # "truck"
  20: 5  ,   # "other-vehicle"
  30: 6  ,   # "person"
  31: 7  ,   # "bicyclist"
  32: 8  ,   # "motorcyclist"
  40: 9  ,   # "road"
  44: 10 ,   # "parking"
  48: 11 ,   # "sidewalk"
  49: 12 ,   # "other-ground"
  50: 13 ,   # "building"
  51: 14 ,   # "fence"
  52: 0  ,   # "other-structure" mapped to "unlabeled" ------------------mapped
  60: 9  ,   # "lane-marking" to "road" ---------------------------------mapped
  70: 15 ,   # "vegetation"
  71: 16 ,   # "trunk"
  72: 17 ,   # "terrain"
  80: 18 ,   # "pole"
  81: 19 ,   # "traffic-sign"
  99: 0  ,   # "other-object" to "unlabeled" ----------------------------mapped
  252: 20,    # "moving-car"
  253: 21,    # "moving-bicyclist"
  254: 22,    # "moving-person"
  255: 23,    # "moving-motorcyclist"
  256: 24,    # "moving-on-rails" mapped to "moving-other-vehicle" ------mapped
  257: 24,    # "moving-bus" mapped to "moving-other-vehicle" -----------mapped
  258: 25,    # "moving-truck"
  259: 24,
}


def sample(pointcloud, labels, numpoints_to_sample):
    """
        INPUT
            pointcloud          : list of 3D points
            labels              : list of integer labels
            numpoints_to_sample : number of points to sample
    """
    tensor = np.concatenate((pointcloud, np.reshape(labels, (labels.shape[0], 1))), axis= 1)
    tensor = np.asarray(random.choices(tensor, weights=None, cum_weights=None, k=numpoints_to_sample))
    pointcloud_ = tensor[:, 0:3]
    labels_ = tensor[:, 3]
    labels_ = np.array(labels_, dtype=np.int_)
    return pointcloud_, labels_


def readpc(pcpath, labelpath, max_dist=15**2, min_dist=4**2):
    """
    INPUT
        pcpath         : path to the point cloud ".bin" file
        labelpath      : path to the labels ".label" file
        reduced_labels : flag to select which label encoding to return
                        [True]  -> values in range [0, 1, 2]   -- default
                        [False] -> all Semantic-Kitti dataset original labels
    """
    size_float = 4
    size_small_int = 2
    pointcloud, labels = [], []

    with open(pcpath, "rb") as pc_file, open(labelpath, "rb") as label_file:
        byte = pc_file.read(size_float*4)
        label_byte = label_file.read(size_small_int)
        _ = label_file.read(size_small_int)

        while byte:
            x,y,z, _ = struct.unpack("ffff", byte)      # unpack 4 float values
            label = struct.unpack("H", label_byte)[0]   # unpach 1 Unsigned Short value
            
            d = x*x + y*y + z*z       # Euclidean norm

            if min_dist<d<max_dist:
                pointcloud.append([x, y, z])
                labels.append(label_remap[label])
            
            byte = pc_file.read(size_float*4)
            label_byte = label_file.read(size_small_int)
            _ = label_file.read(size_small_int)
    
    pointcloud  = np.array(pointcloud)
    labels      = np.array(labels)

    # return fixed_sized lists of points/labels (fixed size: numpoints)
    return pointcloud, labels


def remap_to_bgr(integer_labels, color_scheme):
  bgr_labels = []
  for n in integer_labels:
    bgr_labels.append(color_scheme[int(n)][::-1])
  np_bgr_labels = np.array(bgr_labels)
  return np_bgr_labels


def draw_geometries(geometries):
    graph_objects = []

    for geometry in geometries:
        geometry_type = geometry.get_geometry_type()

        if geometry_type == o3d.geometry.Geometry.Type.PointCloud:
            points = np.asarray(geometry.points)
            colors = None
            if geometry.has_colors():
                colors = np.asarray(geometry.colors)
            elif geometry.has_normals():
                colors = (0.5, 0.5, 0.5) + np.asarray(geometry.normals) * 0.5
            else:
                geometry.paint_uniform_color((1.0, 0.0, 0.0))
                colors = np.asarray(geometry.colors)

            scatter_3d = go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='markers', marker=dict(size=1, color=colors))
            graph_objects.append(scatter_3d)

        if geometry_type == o3d.geometry.Geometry.Type.TriangleMesh:
            triangles = np.asarray(geometry.triangles)
            vertices = np.asarray(geometry.vertices)
            colors = None
            if geometry.has_triangle_normals():
                colors = (0.5, 0.5, 0.5) + np.asarray(geometry.triangle_normals) * 0.5
                colors = tuple(map(tuple, colors))
            else:
                colors = (1.0, 0.0, 0.0)

            mesh_3d = go.Mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2], i=triangles[:,0], j=triangles[:,1], k=triangles[:,2], facecolor=colors, opacity=0.50)
            graph_objects.append(mesh_3d)

    fig = go.Figure(
        data=graph_objects,
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data'
            )
        )
    )
    fig.show()


def visualize3DPointCloud(np_pointcloud, np_labels):
  """
  INPUT
      np_pointcloud : numpy array of 3D points
      np_labels     : numpy array of integer labels
  """
  assert(len(np_pointcloud) == len(np_labels))


  pcd = o3d.geometry.PointCloud()
  v3d = o3d.utility.Vector3dVector

  # set geometry point cloud points
  pcd.points = v3d(np_pointcloud)

  # scale color values in range [0:1]
  pcd.colors = o3d.utility.Vector3dVector(np_labels / 255.0)

  # replace rendering function
  o3d.visualization.draw_geometries = draw_geometries

  # visualize the colored point cloud
  o3d.visualization.draw_geometries([pcd])


class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return  norm_pointcloud
    
class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        return torch.from_numpy(pointcloud)
    

class PointCloudData(Dataset):
    def __init__(self, dataset_path, sequences=[1], num_points=5000, max_dist=15, min_dist=4):
        # transform distances to squares (code optimization)
        max_dist *= max_dist
        min_dist *= min_dist
        self.max_dist = max_dist
        self.min_dist = min_dist

        self.dataset_path = dataset_path
        self.n_points = num_points
        self.transforms = transforms.Compose([
            Normalize(),
            ToTensor()
        ])

        all_seqs = os.listdir(os.path.join(self.dataset_path, "sequences"))
        self.pc_paths = []
        self.lb_paths = []
        for seq in all_seqs:
            if int(seq) in sequences:
                pc_path = os.path.join(self.dataset_path, "sequences", seq, "velodyne")
                lb_path = os.path.join(self.dataset_path, "sequences", seq, "labels")

                self.pc_paths += sorted(glob(pc_path + '/*'))
                self.lb_paths += sorted(glob(lb_path + '/*'))
        assert(len(self.pc_paths) == len(self.lb_paths))

    def __len__(self):
        return len(self.pc_paths)

    def __getitem__(self, idx):
        pcpath = self.pc_paths[idx]
        lbpath = self.lb_paths[idx]
        # load points and labels
        pointcloud, labels = readpc(pcpath, lbpath, max_dist=self.max_dist, min_dist=self.min_dist)
        pointcloud, labels = sample(pointcloud, labels, self.n_points)

        # transform
        torch_pointcloud = self.transforms(pointcloud)
        torch_labels      = torch.from_numpy(labels)
        return torch_pointcloud, torch_labels
