import open3d as o3d
import trimesh
import numpy as np
from copy import deepcopy


class VizContext:
    def __init__(self, non_block=False) -> None:
        self.vis = o3d.visualization.VisualizerWithKeyCallback()

        self.presenting = True
        self.confirm = False
        self.abandon = False

        def skip_callback(vis):
            self.presenting = False
            self.confirm = False

        def record_callback(vis):
            self.presenting = False
            self.confirm = True

        def abandon_callback(vis):
            self.presenting = False
            self.confirm = False
            self.abandon = True

        self.vis.register_key_callback(ord(" "), skip_callback)
        self.vis.register_key_callback(ord("X"), record_callback)
        self.vis.register_key_callback(ord("U"), abandon_callback)

        self.non_block = non_block

    def init(self):
        self.vis.create_window()

    def deinit(self):
        self.vis.destroy_window()

    def add_geometry(self, pc):
        self.vis.add_geometry(pc)

    def add_geometry_list(self, pc_list):
        for pc in pc_list:
            self.vis.add_geometry(pc)

    def update_geometry(self, pc):
        self.vis.update_geometry(pc)

    def step(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    def remove_geometry(self, pc):
        self.vis.remove_geometry(pc)

    def remove_geometry_list(self, pc_list):
        for pc in pc_list:
            self.vis.remove_geometry(pc)

    def reset(self):
        self.presenting = True
        self.confirm = False
        self.abandon = False

    def condition(self):
        return self.presenting and (not self.non_block)


def cvt_from_trimesh(mesh: trimesh.Trimesh):
    vertices = mesh.vertices
    faces = mesh.faces
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.array(
            [
                [0.8, 0.4, 0.8],
            ]
            * len(vertices)
        )
    )
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh
