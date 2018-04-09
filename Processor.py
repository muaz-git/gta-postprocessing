# Processor.py is a helper file which provides interface to :
# 1) read directories in TIFF files,
# 2) project 3d bounding boxes on image plane,
# 3) refine stencil map,
# 4) refine bounding boxes with the help of Projection/View matrices, depth/stencil map and projected 3d bounding boxes.

import numpy as np
from shapely.geometry.point import Point
from math import *
from shapely.wkb import loads
import numpy.linalg as lin
import Snapshot
from Detection import Detection
from pathlib import Path
from libtiff import TIFF
from PIL import Image
from typing import List, Tuple
from enum import Enum
import cv2
from collections import namedtuple
from postgresql.types.geometry import Box
import colorsys
from shapely.geometry.point import Point
from shapely.geometry.multipoint import MultiPoint
from shapely.affinity import affine_transform

BBox = namedtuple('BBox', ['detection_id', 'bbox', 'coverage'])

class StenCode(Enum):
    person = 1
    car = 2


class Processor:
    def __init__(self, snapshot: Snapshot, base_data_dir, resize_debugImg=True):
        self.snapshot: Snapshot = snapshot
        self.base_data_dir = base_data_dir
        self.resize_debugImg = resize_debugImg

    # finds total number of "directories" in a tiff file
    def num_directories(self, tiffimage):
        count = 1
        TIFF.setdirectory(tiffimage, 0)
        while not TIFF.lastdirectory(tiffimage):
            TIFF.setdirectory(tiffimage, count)
            count += 1
        TIFF.setdirectory(tiffimage, 0)
        return count

    # read tiff directories
    def read_tiff(self):
        imgpath = Path(self.base_data_dir) / Path(self.snapshot.runguid) / Path(self.snapshot.imagepath, mode='r')

        if not imgpath.exists():
            raise Exception(str(imgpath), ' does not exist. And to handle this for multithreading.')

        tiffimg = TIFF.open(str(imgpath))
        img = Image.open(str(imgpath))
        w = img.width
        h = img.height
        if not w == self.snapshot.width:
            raise Exception("Width of image is not same as in database.")
        if not h == self.snapshot.height:
            raise Exception("Height of image is not same as in database.")
        del img

        image = np.empty((h, w, 4), dtype=np.uint8)
        print("\nProcessor->read_tiff() : Need to save images in to dictionary properly.")

        depth = np.empty((h, w), dtype=np.float32)
        stencil = np.empty((h, w), dtype=np.uint8)

        # reads image from particular directory of TIFF file.
        def get_img_from_directory(tiff_tmp, dir_num, dst_img):
            TIFF.setdirectory(tiff_tmp, dir_num)
            TIFF.readencodedstrip(tiff_tmp, 0, dst_img.ctypes.data, -1)

        lastdir = self.num_directories(tiffimg) - 1

        get_img_from_directory(tiffimg, dir_num=4, dst_img=image)
        get_img_from_directory(tiffimg, dir_num=lastdir-1, dst_img=depth)
        get_img_from_directory(tiffimg, dir_num=lastdir, dst_img=stencil)

        self.snapshot.img_dict["clear"] = self.swapChannels(image)
        self.snapshot.img_dict["depth"] = depth
        self.snapshot.img_dict["stencil"] = stencil

    def swapChannels(self, image):
        tmp = np.zeros_like(image)
        tmp[:, :, 0] = image[:, :, 2]
        tmp[:, :, 1] = image[:, :, 1]
        tmp[:, :, 2] = image[:, :, 0]
        tmp[:, :, 3] = image[:, :, 3]

        return tmp

    def project_3dvecs_toscreen(self, bboxes:List[MultiPoint], xforms: List[np.matrix]):
        # bboxes are relative model coordinates instead of world coordinates. Which is why we need xforms.
        boxes_matrix = [[np.matrix([[p.x], [p.y], [p.z], [1]]) for p in x.geoms] for x in bboxes]
        boxes_world = [[y @ x for x in box] for (y, box) in zip(xforms, boxes_matrix)]
        boxes_view = [[self.snapshot.view_matrix @ x for x in box] for box in boxes_world]
        boxes_proj = [[self.snapshot.proj_matrix @ x for x in box] for box in boxes_view]
        boxes_ndc = [[x / x[3] for x in box] for box in boxes_proj]
        boxes_image = [[[((vec.A1[0] + 1) / 2) * self.snapshot.width, (1 - ((vec.A1[1] + 1) / 2)) * self.snapshot.height] for vec in box] for box in
                       boxes_ndc]

        positions = [y @ np.matrix([[0], [0], [0], [1]]) for y in xforms]
        positions_view = [self.snapshot.view_matrix @ x for x in positions]
        positions_proj = [self.snapshot.proj_matrix @ x for x in positions_view]
        positions_ndc = [x / x[3] for x in positions_proj]
        positions_image = [[((vec.A1[0] + 1) / 2) * self.snapshot.width, (1 - ((vec.A1[1] + 1) / 2)) * self.snapshot.height] for vec in positions_ndc]

        return (positions_image[0], boxes_image[0])

    def refine_bbox(self):
        # detection_list = self.snapshot.detections

        # find proper way to initialize view and projection matrices.
        # view = self.snapshot.view_matrix
        # proj = self.snapshot.proj_matrix

        world_space = self.to_world_space()  # gets (1080, 1920, 4) matrix. Each (i, j, k, l) belongs to (i,j) pixel of
                                            # snapshot, which is nothing but 3d position in the world space.

        painted_output_cars = np.zeros((self.snapshot.height, self.snapshot.width), dtype=np.uint32)
        painted_output_persons = np.zeros((self.snapshot.height, self.snapshot.width), dtype=np.uint32)
        # bbox_geom = []
        # affines = []
        # worldMin= []

        for i, detection_obj in zip(range(len(self.snapshot.detections)), self.snapshot.detections):
            assert isinstance(detection_obj, Detection)

            # forms affine matrix using rotation and (x, y, z) coordinates of object.
            detection_obj.affine = self.create_affine(loads(detection_obj.rotation), loads(detection_obj.pos))

            # following code returns positions of centre and corners of the object in screen reference.
            (detection_obj.projected_3dpos, detection_obj.projected_3dbox) = self.project_3dvecs_toscreen(bboxes=[loads(detection_obj.fullbox)], xforms=[detection_obj.affine])

            # loading minimum and maximum 3d point of object. Came from model's dimensions.
            bboxes = (np.array(loads(detection_obj.bbox3d_min)), np.array(loads(detection_obj.bbox3d_max)))
            handle = detection_obj.handle
            # worldMin.append(self.paint_pixels(world_space, detection_obj.affine, bboxes, handle))
            # continue
            if detection_obj.type=='car':
                painted_output_cars += self.paint_pixels(world_space, detection_obj.affine, bboxes, handle)  # using depth map, paints pixel separating them according to depth map.

            elif detection_obj.type=='person':
                painted_output_persons += self.paint_pixels(world_space, detection_obj.affine, bboxes, handle)  # using depth map, paints pixel separating them according to depth map.

            # TODO save bbox_geom in the right form.

        # output contains painted pixels with 'handles' of bboxes according to depth.
        # stencil_cull uses evidence from stencil map and finds common pixels in "painted pixels" corresponding to
        # particular category.
        output_cars = self.stencil_cull(painted_output_cars.copy(), StenCode.car)  # contains only cars
        output_peds = self.stencil_cull(painted_output_persons.copy(), StenCode.person)  # contains only persons

        def overlap_segmask(cars, peds):
            refined_mask = np.copy(cars)

            refined_mask[peds>0]=peds[peds>0]
            return refined_mask


        refined_mask = overlap_segmask(output_cars, output_peds)
        self.set_bestbboxes(refined_mask)
        # self.set_bestbboxes(output_cars)
        # self.set_bestbboxes(output_peds)

        # output_cars = self.pseudo_color_gen(output_cars)
        # output_peds = self.pseudo_color_gen(output_peds)
        # refined = self.pseudo_color_gen(refined)
        #
        # output_cars = cv2.resize(output_cars, (0, 0), fx=0.4, fy=0.4)
        # output_peds = cv2.resize(output_peds, (0, 0), fx=0.4, fy=0.4)
        # refined = cv2.resize(refined, (0, 0), fx=0.4, fy=0.4)
        #
        # cv2.imshow("refined", refined)
        # cv2.imshow("output_cars", output_cars)
        # cv2.imshow("output_peds", output_peds)
        # cv2.waitKey()
        # cv2.waitKey()

        self.snapshot.refined_stencil_coded = refined_mask
        self.snapshot.refined_stencil_colored = self.pseudo_color_gen(refined_mask)
        self.snapshot.debug_image = self.get_debugImg()
        self.snapshot.processed = True

        # return tmp

    def set_bestbboxes(self, img):
        w = self.snapshot.width
        h = self.snapshot.height

        # best_boxes = []
        for detection_obj in self.snapshot.detections:
            assert isinstance(detection_obj, Detection)

            a = np.where(img == detection_obj.handle)

            if len(a[0]) == 0:
                continue

            bbox = Box(((np.min(a[1]) / w, np.min(a[0]) / h), (np.max(a[1]) / w, np.max(a[0]) / h)))
            count = np.count_nonzero(a)
            normw = np.min(a[1]) - np.max(a[1])
            normh = np.min(a[0]) - np.max(a[0])
            npix = normw * normh

            if npix == 0:
                cov = 0
            else:
                cov = count / npix

            # print("setting for {}".format(detection_obj.type))
            detection_obj.best_bbox = bbox
            detection_obj.coverage = cov
            # drawing best_bbox
            # myCol = self.unique_color(detection_obj.handle)
            # debug_img = self.snapshot.img_dict["clear"].copy()

            # self.draw_bestBbox(debug_img, detection_obj.best_bbox, myCol)
            # debug_img = cv2.resize(debug_img,
            #                              (int(self.snapshot.width * 0.5), int(self.snapshot.height * 0.5)))
            # cv2.imshow("debug", debug_img)
            # cv2.waitKey()


            # best_boxes.append(BBox(detection_obj.detection_id, bbox, cov))
        # return best_boxes

    # following method refines stencil map.
    # More details on stencil map: http://www.adriancourreges.com/blog/2015/11/02/gta-v-graphics-study/
    def stencil_cull(self, segmentations, cls=StenCode.car):
        stencil = self.snapshot.img_dict["stencil"]  # Do I need copy of it?
        w = self.snapshot.width
        h = self.snapshot.height


        mask = np.full((h, w), 0x7, dtype=np.uint8)
        stencil_unmask = np.bitwise_and(stencil, mask)
        thresh = cv2.compare(stencil_unmask, cls.value, cv2.CMP_EQ)
        thresh = (thresh // 255).astype(np.uint32)  # 9//2 = 4, answer is quotient.
        # person_thresh = cv2.compare(stencil_unmask, StenCode.person.value, cv2.CMP_EQ)
        # person_thresh = (person_thresh // 255).astype(np.uint32)

        return segmentations * thresh

    # forming coordinates in to world space.
    def to_world_space(self):
        w = self.snapshot.width
        h = self.snapshot.height
        view = self.snapshot.view_matrix
        proj = self.snapshot.proj_matrix
        depth = self.snapshot.img_dict["depth"]

        world_space = np.empty((h, w, 4), dtype=np.float32)
        idx = np.indices((h, w), dtype=np.float32)
        world_space[:, :, 0] = idx[1]
        world_space[:, :, 1] = idx[0]
        world_space[:, :, 2] = depth
        world_space[:, :, 3] = 1
        world_space[:, :, 0] = ((2 * world_space[:, :, 0]) / w) - 1  # scaing coordinates b/w -1 and 1
        world_space[:, :, 1] = ((2 * (h - world_space[:, :, 1])) / h) - 1  # scaing coordinates b/w -1 and 1, inverted
        world_space = lin.inv(proj) @ world_space[:, :, :, np.newaxis]
        world_space = lin.inv(view) @ world_space[:, :, :]
        world_space = np.squeeze(world_space)
        world_space[:, :] = world_space[:, :] / world_space[:, :, 3, None]
        return np.squeeze(world_space)

    def create_rot_matrix(self, euler: Point) -> np.matrix:
        x = np.radians(euler.x)
        y = np.radians(euler.y)
        z = np.radians(euler.z)
        Rx = np.array([[1, 0, 0],
                       [0, cos(x), -sin(x)],
                       [0, sin(x), cos(x)]], dtype=np.float)
        Ry = np.array([[cos(y), 0, sin(y)],
                       [0, 1, 0],
                       [-sin(y), 0, cos(y)]], dtype=np.float)
        Rz = np.array([[cos(z), -sin(z), 0],
                       [sin(z), cos(z), 0],
                       [0, 0, 1]], dtype=np.float)
        result = Rz @ Ry @ Rx
        return result

    def create_affine(self, euler: Point, translation: Point) -> np.matrix:
        rotation = self.create_rot_matrix(euler)
        x = translation.x
        y = translation.y
        z = translation.z
        translation_mtx = np.array([[1, 0, 0, x],
                                    [0, 1, 0, y],
                                    [0, 0, 1, z],
                                    [0, 0, 0, 1]], dtype=np.float64)
        rot_mtx = np.hstack((rotation, [[0], [0], [0]]))
        rot_mtx = np.vstack((rot_mtx, [0, 0, 0, 1]))
        result = translation_mtx @ rot_mtx
        return result

    def paint_pixels(self, world_space, world, box: Tuple[np.array], val):
        # depth map get used here.
        # world is affine matrix of objects, box contains two points one is minimum and the other is maximum, val is handle
        # world_space is (1080, 1920, 4)

        invv = lin.inv(world)
        model_space = lin.inv(world) @ world_space[:, :, :, np.newaxis]
        # model_space is (1080, 1920, 4, 1)
        model_space = np.squeeze(model_space)
        # model_space is (1080, 1920, 4)

        # zz= box[0]
        min = np.hstack((box[0], [0]))  # box[0]: minimum 3d box
        max = np.hstack((box[1], [2]))  # box[1]: maximum 3d box

        # in my opinion: its finding all the depth values which contains in two corners of bounding box.
        gt = model_space[:, :, :] > min
        lt = model_space[:, :, :] < max
        alllt = np.all(lt, axis=-1)
        allgt = np.all(gt, axis=-1)
        all = np.logical_and(allgt, alllt)
        # all = alllt
        return (val * all).astype(np.uint32)

    def draw_bestBbox(self, img, bbox, col, thickness):
        # drawing best_bbox
        pt1 = (int(bbox[0][0] * self.snapshot.width),
               int(bbox[0][1] * self.snapshot.height))
        pt2 = (int(bbox[1][0] * self.snapshot.width),
               int(bbox[1][1] * self.snapshot.height))
        # print("\tDrawing {0} - {1}".format(pt1, pt2))
        cv2.rectangle(img, pt1=pt1, pt2=pt2, color=col, thickness=thickness)

    # get_debugImg() draws projected corners of 3d bbox and overlway refined segmentation mask.
    def get_debugImg(self):
        debug_img = self.snapshot.img_dict["clear"].copy()
        for detectionObj in self.snapshot.detections:
            assert isinstance(detectionObj, Detection)
            if detectionObj.best_bbox is not None:
                myCol = self.unique_color(detectionObj.handle, detectionObj.type)

                # drawing 3dbox corner points.
                for x, y in detectionObj.projected_3dbox:
                    if x > self.snapshot.width or x < 0 or y > self.snapshot.height or y < 0:
                        continue
                    cv2.circle(debug_img, center=(int(x),int(y)), radius=4, color=myCol, thickness=-1)

                # drawing best_bbox
                th = 3
                if detectionObj.type == 'car':
                    th = 2

                self.draw_bestBbox(debug_img, detectionObj.best_bbox, myCol, th)
                # pt1 = (int(detectionObj.best_bbox[0][0]*self.snapshot.width), int(detectionObj.best_bbox[0][1]*self.snapshot.height))
                # pt2 = (int(detectionObj.best_bbox[1][0]*self.snapshot.width), int(detectionObj.best_bbox[1][1]*self.snapshot.height))
                # cv2.rectangle(debug_img, pt1=pt1, pt2=pt2, color=myCol)

        # overlay colored segmentation mask
        seg_mask_refined = self.pseudo_color_refined_seg_mask()
        debug_img = cv2.addWeighted(debug_img[:, :, :3], 0.80, seg_mask_refined, 0.20, 0)
        if self.resize_debugImg:
            print("Processor->get_debugImg() : resizing debug image")
            debug_img = cv2.resize(debug_img, (0, 0), fx=0.6, fy=0.6)
        return debug_img

    def unique_color(self, val, typ=None):
        v = 0.99
        if typ=='person' or typ=='property':
            s = 0.99
        elif typ == 'car':
            s = 0.3
        elif typ is None:
            s = 0.99
        else:
            print("No such class of object")
            exit()
        PHI = (1 + sqrt(5)) / 2
        n = val * PHI - floor(val * PHI)
        col = colorsys.hsv_to_rgb(n, s, v)
        return (col[0] * 256, col[1] * 256, col[2] * 256)

    def pseudo_color_gen(self, mask):
        # print(np.unique(mask))
        new_segs = np.zeros((self.snapshot.height, self.snapshot.width, 3), dtype=np.uint8)
        for val in np.unique(mask):
            if val == 0:
                new_segs[mask == val] = (0,0,0)
            else:
                new_segs[mask == val] = self.unique_color(val)
        return new_segs

    def pseudo_color_seg_mask(self):
        return self.pseudo_color_gen(self.snapshot.img_dict["stencil"])

    def pseudo_color_refined_seg_mask(self):

        new_segs = np.zeros((self.snapshot.height, self.snapshot.width, 3), dtype=np.uint8)

        for detectionObj in self.snapshot.detections:
            assert isinstance(detectionObj, Detection)

            new_segs[self.snapshot.refined_stencil_coded == detectionObj.handle] = self.unique_color(detectionObj.handle, detectionObj.type)

        return new_segs