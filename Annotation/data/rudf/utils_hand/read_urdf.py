import  os
import  sys
BASE_DIR=  os.path.dirname(os.path.dirname( os.path.abspath(__file__) ))                   
sys.path.append( BASE_DIR  ) 

import urdf_parser_py.urdf as URDF_PARSER
from plotly import graph_objects as go
from pytorch_kinematics.urdf_parser_py.urdf import (URDF, Box, Cylinder, Mesh, Sphere)

def read(robot_name, mesh_path, urdf_filename):
    visual = URDF.from_xml_string(open(urdf_filename).read())
    for i_link, link in enumerate(visual.links):
            print(f"Processing link #{i_link}: {link.name}")
            # load mesh
            # if len(link.visuals) == 0:
            #     continue
            # if type(link.visuals[0].geometry) == Mesh:
            #     a=a
            # elif type(link.visuals[0].geometry) == Cylinder:
            #     mesh = tm.primitives.Cylinder(
            #         radius=link.visuals[0].geometry.radius, height=link.visuals[0].geometry.length)
            # elif type(link.visuals[0].geometry) == Box:
            #     mesh = tm.primitives.Box(extents=link.visuals[0].geometry.size)
            # elif type(link.visuals[0].geometry) == Sphere:
            #     mesh = tm.primitives.Sphere(
            #         radius=link.visuals[0].geometry.radius)
            # else:
            #     print(type(link.visuals[0].geometry))
            #     raise NotImplementedError
            # try:
            #     scale = np.array(
            #         link.visuals[0].geometry.scale).reshape([1, 3])
            # except:
            #     scale = np.array([[1, 1, 1]])
            # try:
            #     rotation = transforms3d.euler.euler2mat(*link.visuals[0].origin.rpy)
            #     translation = np.reshape(link.visuals[0].origin.xyz, [1, 3])
            #     # print('---')
            #     # print(link.visuals[0].origin.rpy, rotation)
            #     # print('---')
            # except AttributeError:
            #     rotation = transforms3d.euler.euler2mat(0, 0, 0)
            #     translation = np.array([[0, 0, 0]])