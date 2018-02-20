class Detection:
    def __init__(self):
        self.detection_id = None
        # self.snapshot_id = None
        self.type = None  # car or person etc
        self.pos = None  # 3d position of the object
        self.bbox = None  # 2d bbox of the object
        self.vehicle_class = None  # vehicle class
        self.handle = None
        self.coverage = None # how much it is visible in the game
        self.fullbox = None
        self.bbox3d_min = None
        self.bbox3d_max = None
        self.best_bbox = None  # best 2d box
        self.bbox_3d = None  # 3d box
        self.rotation = None
        self.affine = None  # affine matrix in 3d world related to camera.
        self.projected_3dpos = None
        self.projected_3dbox = None
