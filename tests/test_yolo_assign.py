import numpy as np

from src.yolo_assign import assignment_table, has_shared_point, points_inside


def test_points_inside_counts_cell_centres_per_stride():
    boxes = np.array([[1, 1, 7, 7],      # contains the stride-8 centre (4, 4) and 4 stride-4 centres
                      [9, 9, 11, 11],    # only the stride-4 centre (10, 10)
                      [0, 0, 64, 64]])   # 8x8 + 4x4 + 2x2 + 1x1 at strides 8/16/32... and 16x16 at 4
    assert points_inside(boxes, 64, 64, (8, 16, 32)).tolist() == [1, 0, 64 + 16 + 4]
    assert points_inside(boxes, 64, 64, (4, 8, 16, 32)).tolist() == [5, 1, 256 + 64 + 16 + 4]


def test_shared_point_only_for_overlaps_that_hold_a_centre():
    boxes = np.array([[0, 0, 8, 8], [2, 2, 10, 10], [20, 20, 30, 30], [29, 29, 31, 31]])
    # boxes 0/1 overlap on (2..8) which holds the stride-8 centre (4, 4); boxes 2/3 overlap on (29..30), no centre
    assert has_shared_point(boxes, 64, 64, (8,)).tolist() == [True, True, False, False]


def test_assignment_table_shares():
    frames = [(64, 64, np.array([[9, 9, 11, 11], [0, 0, 40, 40]], float), ["small", "medium"])]
    table = assignment_table(frames, [64], {"P3-P5": (8, 16, 32), "P2-P5": (4, 8, 16, 32)})
    row = lambda heads, size: table[(table.heads == heads) & (table["size"] == size)].iloc[0]
    assert row("P3-P5", "small")["no_point_%"] == 100
    assert row("P2-P5", "small")["no_point_%"] == 0
    assert row("P3-P5", "all")["objects"] == 2
