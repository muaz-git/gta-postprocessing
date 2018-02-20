class Snapshot:
    def __init__(self):
        self.snapshot_id = None
        self.runguid = None
        self.imagepath = None
        self.camera_pos = None # not used
        self.view_matrix = None
        self.proj_matrix = None
        self.processed = None
        self.width = None
        self.height = None
        # self.stencil_map = None
        # self.depth_map = None
        self.img_dict = {} # would contain a dictionary of images according to time and weather
        self.detections = None  # list of detections
        self.refined_stencil_coded = None  # encoded refined stencil for cars and Peds
        self.refined_stencil_colored = None  # colored refined stencil for cars and Peds
        self.debug_image = None  # debug image contains 3d and 2d bounding boxes along with overlayed segmentation mask.

