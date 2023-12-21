import numpy as np
# from manotorch.utils.anchorutils import get_rev_anchor_mapping
from scipy.spatial.distance import cdist


def get_padding_attr(anchor_mapping):
    rev_anchor_mapping = get_rev_anchor_mapping(anchor_mapping)
    anchor_id_background = len(anchor_mapping.keys())
    region_id_background = len(rev_anchor_mapping.keys())
    anchor_padding_len = max([len(v) for v in rev_anchor_mapping.values()])
    return anchor_id_background, region_id_background, anchor_padding_len


def process_contact_info(
    contact_info,
    # anchor_mapping,
    pad_vertex=False,
    pad_anchor=False,
    dist_th=1000.0,
    elasti_th=0.00,
):
    # anchor_id_background, region_id_background, anchor_padding_len = get_padding_attr(anchor_mapping)

    vertex_contact = [item["contact"] for item in contact_info]
    vertex_contact = np.array(vertex_contact, dtype=np.int)
    if pad_vertex and pad_anchor:
        # the return result will be the same length of vertex_contact
        hand_region = [item['region'] if item["contact"] == 1 else 0 for item in contact_info]
        # hand_region = [1 if item["contact"] == 1 else 0 for item in contact_info]
        hand_region = np.array(hand_region, dtype=np.int)

        # all the anchors will be padded to anchor_padding_len
        anchor_id = []
        # anchor_dist = []
        # anchor_elasti = []
        # anchor_padding_mask = []
        for item in contact_info:
            if item["contact"] == 1:
                item_anchor_id = item["anchor_id"]
        #         item_anchor_dist = item["anchor_dist"]
        #         item_anchor_elasti = item["anchor_elasti"]
        #         item_n_anchor = len(item_anchor_id)
        #         item_padding_len = anchor_padding_len - item_n_anchor
        #
        #         item_anchor_padding_mask = np.zeros((anchor_padding_len,), dtype=np.int)
        #         item_anchor_padding_mask[:item_n_anchor] = 1
        #         item_anchor_id = item_anchor_id + ([anchor_id_background] * item_padding_len)
        #         item_anchor_dist = item_anchor_dist + ([dist_th] * item_padding_len)
        #         item_anchor_elasti = item_anchor_elasti + ([elasti_th] * item_padding_len)
            else:
                item_anchor_id=0
        #         item_anchor_padding_mask = [0] * anchor_padding_len
        #         item_anchor_id = [anchor_id_background] * anchor_padding_len
        #         item_anchor_dist = [dist_th] * anchor_padding_len
        #         item_anchor_elasti = [elasti_th] * anchor_padding_len
            anchor_id.append(item_anchor_id)
        #     anchor_dist.append(item_anchor_dist)
        #     anchor_elasti.append(item_anchor_elasti)
        #     anchor_padding_mask.append(item_anchor_padding_mask)
        anchor_id = np.array(anchor_id, dtype=np.int)
        # anchor_dist = np.array(anchor_dist)
        # anchor_elasti = np.array(anchor_elasti)
        # anchor_padding_mask = np.array(anchor_padding_mask, dtype=np.int)
        # return vertex_contact, hand_region, anchor_id, anchor_dist, anchor_elasti, anchor_padding_mask
        return vertex_contact, hand_region, anchor_id

    # elif not pad_vertex and pad_anchor:
    #     # the return result will be the same length of ones in vertex_contact
    #     hand_region = [item["region"] for item in contact_info if item["contact"] == 1]
    #     hand_region = np.array(hand_region, dtype=np.int)
    #     # all the anchors will be padded to anchor_padding_len
    #     anchor_id = []
    #     anchor_dist = []
    #     anchor_elasti = []
    #     anchor_padding_mask = []
    #     for item in contact_info:
    #         if item["contact"] == 1:
    #             item_anchor_id = item["anchor_id"]
    #             item_anchor_dist = item["anchor_dist"]
    #             item_anchor_elasti = item["anchor_elasti"]
    #             item_n_anchor = len(item_anchor_id)
    #             item_padding_len = anchor_padding_len - item_n_anchor
    #
    #             item_anchor_padding_mask = np.zeros((anchor_padding_len,), dtype=np.int)
    #             item_anchor_padding_mask[:item_n_anchor] = 1
    #             item_anchor_id = item_anchor_id + ([anchor_id_background] * item_padding_len)
    #             item_anchor_dist = item_anchor_dist + ([dist_th] * item_padding_len)
    #             item_anchor_elasti = item_anchor_elasti + ([elasti_th] * item_padding_len)
    #         else:
    #             continue
    #         anchor_id.append(item_anchor_id)
    #         anchor_dist.append(item_anchor_dist)
    #         anchor_elasti.append(item_anchor_elasti)
    #         anchor_padding_mask.append(item_anchor_padding_mask)
    #     anchor_id = np.array(anchor_id, dtype=np.int)
    #     anchor_dist = np.array(anchor_dist)
    #     anchor_elasti = np.array(anchor_elasti)
    #     anchor_padding_mask = np.array(anchor_padding_mask, dtype=np.int)
    #     return vertex_contact, hand_region, anchor_id, anchor_dist, anchor_elasti, anchor_padding_mask
    # elif pad_vertex and not pad_anchor:
    #     # the return result will be the same length of vertex_contact
    #     hand_region = [item["region"] if item["contact"] == 1 else region_id_background for item in contact_info]
    #     hand_region = np.array(hand_region, dtype=np.int)
    #     # no anchors will be padded
    #     # will return list
    #     anchor_id = []
    #     anchor_dist = []
    #     anchor_elasti = []
    #     for item in contact_info:
    #         if item["contact"] == 1:
    #             item_anchor_id = item["anchor_id"]
    #             item_anchor_dist = item["anchor_dist"]
    #             item_anchor_elasti = item["anchor_elasti"]
    #         else:
    #             item_anchor_id = []
    #             item_anchor_dist = []
    #             item_anchor_elasti = []
    #         anchor_id.append(item_anchor_id)
    #         anchor_dist.append(item_anchor_dist)
    #         anchor_elasti.append(item_anchor_elasti)
    #     return vertex_contact, hand_region, anchor_id, anchor_dist, anchor_elasti, None
    # else:
    #     # the return result will be the same length of ones in vertex_contact
    #     hand_region = [item["region"] for item in contact_info if item["contact"] == 1]
    #     hand_region = np.array(hand_region, dtype=np.int)
    #     # no anchors will be padded
    #     # will return list
    #     anchor_id = []
    #     anchor_dist = []
    #     anchor_elasti = []
    #     for item in contact_info:
    #         if item["contact"] == 1:
    #             item_anchor_id = item["anchor_id"]
    #             item_anchor_dist = item["anchor_dist"]
    #             item_anchor_elasti = item["anchor_elasti"]
    #         else:
    #             continue
    #         anchor_id.append(item_anchor_id)
    #         anchor_dist.append(item_anchor_dist)
    #         anchor_elasti.append(item_anchor_elasti)
    #     return vertex_contact, hand_region, anchor_id, anchor_dist, anchor_elasti, None


def elasti_fn(x, range_th=20.0):
    x = x.copy()
    np.putmask(x, x > range_th, range_th)
    res = 0.5 * np.cos((np.pi / range_th) * x) + 0.5
    np.putmask(res, res < 1e-8, 0)
    return res


def get_mode_list(assignment, n_bins, weights=None):
    # weights: weight of each vertex
    res = np.zeros((n_bins,))
    for bin_id in range(n_bins):
        print((assignment == bin_id).astype(np.float))
        print(np.sum((assignment == bin_id).astype(np.float) * weights[0]))
        res[bin_id] = np.sum((assignment == bin_id).astype(np.float) * weights[0])
    print(res)
    maxidx = np.argmax(res)
    return maxidx


def cal_dist(
    hand_verts: np.ndarray,  # ARRAY[NHND, 3]
    hand_normals: np.ndarray,  # ARRAY[NHND, 3]
    obj_verts: np.ndarray,  # ARRAY[NOBJ, 3]
    hand_verts_region: np.ndarray,  # ARRAY[NOBJ]
    n_regions: int,
    anchor_pos: np.ndarray,
    # anchor_mapping: dict,
    n_samples: int = 32,
    # range_threshold: float = 5.0,
    range_threshold: float = 5.0,
    elasti_threshold: float = 45.0,
    elasti_cutoff: float = 0.1,
):
    vertex_blob_list = []
    obj_n_verts = obj_verts.shape[0]
    # rev_anchor_mapping = get_rev_anchor_mapping(anchor_mapping, n_region=n_regions)
    # region_size = get_region_size(hand_verts_region, n_region=n_regions)  # ARRAY[NREGION]

    # ========== STAGE 1: COMPUTE CROSS DISTANCE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print(obj_verts.shape, hand_verts.shape)
    all_dist = cdist(obj_verts, hand_verts) #* 1000  # ARRAY[NOBJ, NHND_SEL]
    h2o_vec = obj_verts[:, None, :] - hand_verts[None, :, :]  # ARRAY[NOBJ, NHND_SEL, 3]
    inner = np.einsum(
        "bij,bij->bi", h2o_vec / np.linalg.norm(h2o_vec, axis=-1, keepdims=True), hand_normals[None, :, :]
    )  # # ARRAY[NOBJ, NHND_SEL]
    # all_dist = np.where(inner > np.cos(np.pi / 6), all_dist, np.inf)
    # print('all_dist', all_dist.shape)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # ========== STAGE 2: Get 20 Closest Points>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    pts_idx = np.arange(all_dist.shape[0])[:, None]
    order_idx = np.argsort(all_dist, axis=1)  # ARRAY[NOBJ, NHND_SEL]
    # print(order_idx.shape)
    sorted_dist = all_dist[pts_idx, order_idx]
    sorted_dist_sampled = sorted_dist[:, :n_samples]  # ARRAY[NOBJ, 20]
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # ========== STAGE 3: iterate over all points >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    for point_id in range(obj_n_verts):
        dist_vec = sorted_dist_sampled[point_id, :]  # ARRAY[20]
        valid_mask = dist_vec < range_threshold
        masked_dist_vec = dist_vec[valid_mask]  # ARRAY[NSELECTED]
        if len(masked_dist_vec) == 0:
            # there is no contact
            vertex_blob_list.append({"contact": 0})
        else:
            # there is contact
            # ======= get region assignment >>>>>>
            # first we need to use np.where to collect the indexes of vertices in dist_vec
            valid_idx = np.where(valid_mask)[0]  # LIST[INT; NSELECTED]
            # then we now the samples keep the same order as in order_idx
            origin_valid_idx = order_idx[point_id, valid_idx]  # ARRAY[NSELECTED]
            # print(origin_valid_idx)
            # index them for region information
            minidx = np.argmin(masked_dist_vec)
            masked_dist_vec = masked_dist_vec[minidx]
            origin_valid_points_region = hand_verts_region[origin_valid_idx]  # ARRAY[NSELECTED]
            origin_valid_points_region = origin_valid_points_region[minidx]
            # print(origin_valid_points_region)
            # compute the mode
            # if there are ties, we will try to use the one with most mode
            # print(masked_dist_vec)
            mode_weight = range_threshold - masked_dist_vec
            # print(mode_weight)
            # print('origin_valid_points_region', origin_valid_points_region)
            # print('mode_weight', mode_weight)
            # origin_valid_points_region = np.array([0,1,2,3,4])
            # target_region = get_mode_list(origin_valid_points_region, n_regions, weights=mode_weight)
            # print(target_region)

            target_region = 0
            # print(hand_verts_region[:23])
            # print(np.isin(origin_valid_points_region, hand_verts_region[:23]))
            if np.isin(origin_valid_points_region, hand_verts_region[:23]): # Thumb
                target_region = 1
            elif np.isin(origin_valid_points_region, hand_verts_region[23:72]): # FF
                target_region = 2
            elif np.isin(origin_valid_points_region, hand_verts_region[72:121]): # MF
                target_region = 3
            elif np.isin(origin_valid_points_region, hand_verts_region[121:170]): # RF
                target_region = 4
            elif np.isin(origin_valid_points_region, hand_verts_region[170:]): # LF
                target_region = 5

            # print('target_region',target_region)
            # <<<<<<<<<<<<

            # ====== get anchor distance (by indexing dist mat) >>>>>>
            # get the anchors
            # anchor_list = rev_anchor_mapping[target_region]
            anchor_list = origin_valid_points_region
            # get the distances

            # obj_point = obj_verts[point_id : point_id + 1, :]  # ARRAY[1, 3]
            # anchor_points = anchor_pos#[anchor_list, :]  # ARRAY[NAC, 3]
            # dist_mat = cdist(obj_point, anchor_points).squeeze(0) * 1000.0
            # elasti_mat = elasti_fn(dist_mat, range_th=elasti_threshold)
            # np.putmask(elasti_mat, elasti_mat < elasti_cutoff, elasti_cutoff)
            # <<<<<<<<<<<<

            # store the result
            vertex_blob_list.append(
                {
                    "contact": 1,
                    "region": target_region,
                    "anchor_id": anchor_list,
                    # "anchor_dist": dist_mat.tolist(),
                    # "anchor_elasti": elasti_mat.tolist(),
                }
            )
    return vertex_blob_list
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
