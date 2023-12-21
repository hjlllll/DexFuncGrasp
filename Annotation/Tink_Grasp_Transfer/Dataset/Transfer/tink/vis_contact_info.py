import numpy as np
import open3d as o3d


def create_vertex_color(contact_info, mode="vertex_contact"):
    if mode == "vertex_contact":
        vertex_contact = contact_info["vertex_contact"]
        n_verts = vertex_contact.shape[0]
        vertex_color = np.zeros((n_verts, 3))
        vertex_color[vertex_contact == 0] = np.array([57, 57, 57]) / 255.0
        vertex_color[vertex_contact == 1] = np.array([198, 198, 198]) / 255.0
        return vertex_color
    elif mode == "contact_region":
        contact_region = contact_info["hand_region"]
        n_verts = contact_region.shape[0]
        vertex_color = np.zeros((n_verts, 3))
        # vertex_color[contact_region == 0] = np.array([138, 184, 254]) / 255.0
        vertex_color[contact_region == 0] = np.array([1, 95, 107]) / 255.0
        vertex_color[contact_region == 1] = np.array([157, 220, 92]) / 255.0
        # vertex_color[contact_region == 0] = np.array([117, 0, 0]) / 255.0
        # vertex_color[contact_region == 1] = np.array([255, 0, 0]) / 255.0
        vertex_color[contact_region == 2] = np.array([255, 138, 137]) / 255.0

        vertex_color[contact_region == 3] = np.array([117, 65, 0]) / 255.0
        vertex_color[contact_region == 4] = np.array([255, 144, 0]) / 255.0
        vertex_color[contact_region == 5] = np.array([255, 206, 134]) / 255.0

        # vertex_color[contact_region == 6] = np.array([116, 117, 0]) / 255.0
        # vertex_color[contact_region == 7] = np.array([255, 255, 0]) / 255.0
        # vertex_color[contact_region == 8] = np.array([255, 255, 131]) / 255.0
        #
        # vertex_color[contact_region == 9] = np.array([0, 117, 0]) / 255.0
        # vertex_color[contact_region == 10] = np.array([0, 255, 0]) / 255.0
        # vertex_color[contact_region == 11] = np.array([145, 255, 133]) / 255.0
        #
        # vertex_color[contact_region == 12] = np.array([0, 60, 118]) / 255.0
        # vertex_color[contact_region == 13] = np.array([0, 133, 255]) / 255.0
        # vertex_color[contact_region == 14] = np.array([136, 200, 255]) / 255.0
        #
        # vertex_color[contact_region == 15] = np.array([70, 0, 118]) / 255.0
        # vertex_color[contact_region == 16] = np.array([210, 135, 255]) / 255.0
        #
        # vertex_color[contact_region == 17] = np.array([20, 0, 150]) / 255.0
        # vertex_color[contact_region == 17] = np.array([220, 220, 220]) / 255.0
        return vertex_color
    else:
        raise ValueError(f"Unknown color mode: {mode}")


def open3d_show(
    obj_points,
    obj_verts,
    obj_faces=None,
    obj_normals=None,
    contact_info=None,
    hand_verts=None,
    hand_faces=None,
    show_hand_normals=False,
):
    import open3d as o3d

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name="Runtime HAND + OBJ",
        width=1024,
        height=768,
    )

    if hand_verts is not None:
        hand_mesh = o3d.geometry.TriangleMesh()
        hand_mesh.triangles = o3d.utility.Vector3iVector(hand_faces)
        hand_mesh.vertices = o3d.utility.Vector3dVector(hand_verts)
        hand_mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.array([[149.0 / 255.0, 163.0 / 255.0, 166.0 / 255.0]] * hand_verts.shape[0])
        )
        hand_mesh.compute_triangle_normals()
        hand_mesh.compute_vertex_normals()
        vis.add_geometry(hand_mesh)

    if obj_points is not None:
        obj_pc = o3d.geometry.PointCloud()
        obj_pc.points = o3d.utility.Vector3dVector(np.array(obj_points))
        vis.add_geometry(obj_pc)

    if show_hand_normals:
        o3d_hand_pc = o3d.geometry.PointCloud()
        o3d_hand_pc.points = hand_mesh.vertices
        o3d_hand_pc.normals = hand_mesh.vertex_normals
        # vis.add_geometry(o3d_hand_pc)

    if obj_faces is None:
        obj_verts = np.array(obj_verts) * 1000.0
        o3d_obj_pc = o3d.geometry.PointCloud()
        o3d_obj_pc.points = o3d.utility.Vector3dVector(obj_verts)
        if obj_normals is not None:
            o3d_obj_pc.normals = o3d.utility.Vector3dVector(obj_normals)
        if contact_info is not None:
            o3d_obj_pc.colors = o3d.utility.Vector3dVector(create_vertex_color(contact_info, mode="contact_region"))
        vis.add_geometry(o3d_obj_pc)
    else:
        o3d_obj_mesh = o3d.geometry.TriangleMesh()
        o3d_obj_mesh.triangles = o3d.utility.Vector3iVector(obj_faces)
        o3d_obj_mesh.vertices = o3d.utility.Vector3dVector(obj_verts)
        if contact_info is not None:
            o3d_obj_mesh.vertex_colors = o3d.utility.Vector3dVector(
                create_vertex_color(contact_info, mode="contact_region")
            )
        o3d_obj_mesh.compute_vertex_normals()
        vis.add_geometry(o3d_obj_mesh)

    t = 0
    while True:
        t+=1
        vis.update_renderer()
        vis.poll_events()
        if t == 200:
            break
