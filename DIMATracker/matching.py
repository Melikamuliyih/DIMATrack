import numpy as np
import scipy
import torch
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from tracker import kalman_filter


def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

def hmiou(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    ious = np.zeros((len(bboxes1), len(bboxes2)), dtype=np.float)
    if ious.size == 0:
        return ious
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    yy11 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    yy12 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    yy21 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    yy22 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    o = (yy12 - yy11) / (yy22 - yy21)

    xx11 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx12 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])

    xx21 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xx22 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    overlap_width = np.maximum(0, np.minimum(xx12, xx22) - np.maximum(xx11, xx21))
    union_width = (xx12 - xx11) + (xx22 - xx21) - overlap_width
    # width = (xx12 - xx11) / (xx22 - xx21)
    width_iou = overlap_width / union_width

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    iou = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
                + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)

    iou = (iou*o + width_iou*iou)/2
    return iou

def hmiou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]
    :rtype cost_matrix np.ndarray
    """
    atlbrs = [track.tlbr for track in atracks]
    btlbrs = [track.tlbr for track in btracks]
    _ious = hmiou(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix



def bbox_overlaps_ciou(bboxes1, bboxes2):

    cious = torch.zeros((len(bboxes1), len(bboxes2)), dtype=np.float)
    if len(bboxes1) * len(bboxes2) == 0:
        return cious

    bboxes1 = np.ascontiguousarray(bboxes1, dtype=np.float)
    bboxes2 = np.ascontiguousarray(bboxes2, dtype=np.float)
    
    bboxes1 = torch.Tensor(bboxes1)
    bboxes2 = torch.Tensor(bboxes2)
    

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True

    bboxes1 = bboxes1[:,None,:]
    bboxes2 = bboxes2[None,:,:]
    w1 = bboxes1[..., 2] - bboxes1[..., 0]
    h1 = bboxes1[..., 3] - bboxes1[..., 1]
    w2 = bboxes2[..., 2] - bboxes2[..., 0]
    h2 = bboxes2[..., 3] - bboxes2[..., 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[..., 2] + bboxes1[..., 0]) / 2
    center_y1 = (bboxes1[..., 3] + bboxes1[..., 1]) / 2
    center_x2 = (bboxes2[..., 2] + bboxes2[..., 0]) / 2
    center_y2 = (bboxes2[..., 3] + bboxes2[..., 1]) / 2

    inter_max_xy = torch.min(bboxes1[..., 2:],bboxes2[..., 2:])
    inter_min_xy = torch.max(bboxes1[..., :2],bboxes2[..., :2])
    out_max_xy = torch.max(bboxes1[..., 2:],bboxes2[..., 2:])
    out_min_xy = torch.min(bboxes1[..., :2],bboxes2[..., :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:,:, 0] * inter[:,:, 1]
    inter_diag = (center_x1 - center_x2)**2 + (center_y1 - center_y2)**2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:,:, 0] ** 2) + (outer[:,:, 1] ** 2)
    union = area1+area2-inter_area
    u = (inter_diag) / outer_diag
    iou = inter_area / union
    with torch.no_grad():
        arctan = torch.atan(w2 / h2) - torch.atan(w1 / h1)
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
        S = 1 - iou
        alpha = v / (S + v)
        w_temp = 2 * w1
    ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
    cious = iou - (u + alpha * ar)
    cious = torch.clamp(cious,min=-1.0,max = 1.0)
    if exchange:
        cious = cious.T
    return cious

def bbox_overlaps_giou(bboxes1, bboxes2):
    
    giou = torch.zeros((len(bboxes1), len(bboxes2)), dtype=np.float)
    if len(bboxes1) * len(bboxes2) == 0:
        return giou

    bboxes1 = np.ascontiguousarray(bboxes1, dtype=np.float)
    bboxes2 = np.ascontiguousarray(bboxes2, dtype=np.float)
    
    bboxes1 = torch.Tensor(bboxes1)
    bboxes2 = torch.Tensor(bboxes2)
    

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    giou = torch.zeros((rows, cols))
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        giou = torch.zeros((cols, rows))
        exchange = True

    bboxes1 = bboxes1[:,None,:]
    bboxes2 = bboxes2[None,:,:]
    w1 = bboxes1[..., 2] - bboxes1[..., 0]
    h1 = bboxes1[..., 3] - bboxes1[..., 1]
    w2 = bboxes2[..., 2] - bboxes2[..., 0]
    h2 = bboxes2[..., 3] - bboxes2[..., 1]

    area1 = w1 * h1
    area2 = w2 * h2

    inter_max_xy = torch.min(bboxes1[..., 2:],bboxes2[..., 2:])
    inter_min_xy = torch.max(bboxes1[..., :2],bboxes2[..., :2])
    out_max_xy = torch.max(bboxes1[..., 2:],bboxes2[..., 2:])
    out_min_xy = torch.min(bboxes1[..., :2],bboxes2[..., :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:,:, 0] * inter[:,:, 1]
  
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_area = outer[:,:, 0] * outer[:,:, 1]
    union = area1+area2-inter_area
   
    iou = inter_area / union
  
    giou = iou - (outer_area - union)/outer_area
    giou = torch.clamp(giou,min=-1.0,max = 1.0)
    if exchange:
        giou = giou.T
    return giou

def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def tlbr_expand(tlbr, scale=1.2):
    w = tlbr[2] - tlbr[0]
    h = tlbr[3] - tlbr[1]

    half_scale = 0.5 * scale

    tlbr[0] -= half_scale * w
    tlbr[1] -= half_scale * h
    tlbr[2] += half_scale * w
    tlbr[3] += half_scale * h

    return tlbr


# def iou_distance(atracks, btracks):
#     """
#     Compute cost based on IoU
#     :type atracks: list[STrack]
#     :type btracks: list[STrack]

#     :rtype cost_matrix np.ndarray
#     """

#     if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
#         atlbrs = atracks
#         btlbrs = btracks
#     else:
#         atlbrs = [track.tlbr for track in atracks]
#         btlbrs = [track.tlbr for track in btracks]
#     _ious = ious(atlbrs, btlbrs)
#     cost_matrix = 1 - _ious

#     return cost_matrix


def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def iou_distance(atracks, btracks, type="iou"):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    if type=="ciou":
        _ious = bbox_overlaps_ciou(atlbrs, btlbrs)
    if type=="giou":
        _ious = bbox_overlaps_giou(atlbrs, btlbrs)
    else:
        _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=float)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=float)
    # print("track", track_features.shape) 
    # print("detection", det_features.shape)
    # for i, track in enumerate(tracks):
        # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1, -1), track.curr_feat.reshape(1,-1), metric))
    # print(len(tracks), len(detections))
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # / 2.0  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    # measurements = np.asarray([det.to_xyah() for det in detections])
    measurements = np.asarray([det.to_xywh() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    # measurements = np.asarray([det.to_xyah() for det in detections])
    measurements = np.asarray([det.to_xywh() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost