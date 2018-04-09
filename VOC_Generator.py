# VOC_Generator.py provides an interface to :
# 1) create and save JSON annotation files similar fashion to PASCAL VOC format
# 2) save refined segmentation map in a loss-less file.
# 3) for saving space, saving images are skipped

from Snapshot import Snapshot
from Detection import Detection
from pathlib import Path
from typing import List
from os.path import join
import json
import numpy as np
import cv2

class VOC_Generator:

    # def __init__(self, snapshotList:List[Snapshot], outputFolder, save_colored_seg_mask=False):
    #     self.snapshotList = snapshotList
    #     self.output_folder = outputFolder
    #     self.runFolder = None
    #     self.save_colored_seg_mask = save_colored_seg_mask

    def __init__(self, runguid, outputFolder, save_colored_seg_mask=False ):
        self.runguid = runguid
        self.output_folder = outputFolder
        self.runFolder = None
        self.save_colored_seg_mask = save_colored_seg_mask

    # create_folders() creates directory structure similar to PASCAL_VOC dataset format with directory name "runguid" as parent.
    @staticmethod
    def create_folders(self):
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        self.runFolder = (Path(self.output_folder)/Path(self.runguid))
        self.runFolder.mkdir(exist_ok=True)
        # (self.runFolder/Path("Annotations")).mkdir(exist_ok=True)
        (self.runFolder/Path("AnnotationsJson")).mkdir(exist_ok=True)
        (self.runFolder/Path("ImageSets/Main")).mkdir(parents=True, exist_ok=True)
        (self.runFolder/Path("JPEGImages")).mkdir(exist_ok=True)
        (self.runFolder/Path("SegmentationObject")).mkdir(exist_ok=True)
        # (self.runFolder/Path("SegmentationObjectColored")).mkdir(exist_ok=True)
        (self.runFolder/Path("debugged")).mkdir(exist_ok=True)

    # save_snapshot(): saves refined segmentation map, json file and debugging image of a snapshot.
    @staticmethod
    def save_snapshot(self, snapShot:Snapshot):
        annotationJson = self.createAnnotationJSON(snapShot)
        with open(self.runFolder / "AnnotationsJson" / (str(snapShot.snapshot_id) + ".json"), 'w') as jsonFile:
            json.dump(annotationJson, jsonFile)
        if self.save_colored_seg_mask:
            print("VOC_Generator->save_snapshots() : need to save colored seg mask.")

        if snapShot.refined_stencil_coded is not None:
            np.savez_compressed(str(self.runFolder / Path("SegmentationObject") / (str(snapShot.snapshot_id) + ".npz")),
                                snapShot.refined_stencil_coded)  # saves filtered stencil map in compressed format

        if snapShot.debug_image is not None:
            cv2.imwrite(str(self.runFolder / Path("debugged") / (str(snapShot.snapshot_id) + ".jpg")),
                        snapShot.debug_image)
    @staticmethod
    def saveTrainTest(self, snapshotIdList):
        image_set_file = self.runFolder / Path("ImageSets/Main") / "trainval.txt"
        image_set_file.write_text("\n".join([str(snapshotId) for snapshotId in snapshotIdList]))

    # save_snapshots() iterates over all snapshots and save them by calling save_snapshot()
    def save_snapshots(self):
        def verifier(snapshotList:List[Snapshot]):
            # verifies if all snapshots belong to same run
            runList = []
            for s in snapshotList:
                # assert isinstance(s, Snapshot)
                runList.append(s.runguid)
            if len(set(runList))!=1:
                print("All snapshots do not belong to same run. Make sure they are. Because we want to keep files in a separate runGuid folder.")
                exit()

        def debug_npz_loader(fil_path):
            np_file = np.load(str(fil_path))
            if not 'arr_0' in np_file: return
            raw_segs = np_file['arr_0']
            print("unique elems of raw_segs = {0}".format(np.unique(raw_segs)))

        verifier(self.snapshotList)
        self.create_folders()

        for snapshot in self.snapshotList:
            # if not snapshot.processed:
            #     print("Snapshot with id = {0} is not processed".format(snapshot.snapshot_id))
            #     continue
            annotationJson = self.createAnnotationJSON(snapshot)
            with open(self.runFolder / "AnnotationsJson" / (str(snapshot.snapshot_id) + ".json"), 'w') as jsonFile:
                json.dump(annotationJson, jsonFile)

            if self.save_colored_seg_mask:
                print("VOC_Generator->save_snapshots() : need to save colored seg mask.")

            if snapshot.refined_stencil_coded is not None:
                np.savez_compressed(str(self.runFolder/Path("SegmentationObject")/(str(snapshot.snapshot_id)+".npz")), snapshot.refined_stencil_coded)  # saves filtered stencil map in compressed format


            if snapshot.debug_image is not None:
                cv2.imwrite(str(self.runFolder/Path("debugged")/(str(snapshot.snapshot_id)+".jpg")), snapshot.debug_image)

        image_set_file = self.runFolder/Path("ImageSets/Main") / "trainval.txt"
        image_set_file.write_text("\n".join([str(snapshot.snapshot_id) for snapshot in self.snapshotList]))

    # createAnnotationJSON(): parse snapshot in to a json object.
    def createAnnotationJSON(self, snapshot:Snapshot):
        root = {}
        root['folder'] = str(self.output_folder+'/'+snapshot.runguid)
        root['filename'] = join(str(snapshot.runguid), snapshot.imagepath)

        source = {}
        source['database'] = "NGV Postgres sim_annotations"
        source['annotation'] = "Grand Theft Auto V"
        source['image'] = "Grand Theft Auto V"
        root['source'] = source

        size = {}
        size['width'] = str(snapshot.width)
        size['height'] = str(snapshot.height)
        size['depth'] = "3"
        root['size'] = size

        root['segmented'] = "1"

        objects = []
        for detection in snapshot.detections:
            assert isinstance(detection, Detection)

            if detection.coverage is None: continue

            object = {}
            box = detection.best_bbox
            if box is None: continue

            xmin = box.low.x * snapshot.width
            xmax = box.high.x * snapshot.width
            ymin = box.low.y * snapshot.height
            ymax = box.high.y * snapshot.height

            if str(detection.vehicle_class) == 'Motorcycles':
                object['name'] = 'motorbike'
            else:
                object['name'] = str(detection.type)

            # TODO: get this right when we include truncated objects
            object['truncated'] = "0"
            object['occluded'] = str(1 - detection.coverage)
            object['pose'] = 'Unspecified'
            object['difficult'] = "0"

            bbox = {}
            bbox['xmin'] = str(int(xmin) + 1)
            bbox['ymin'] = str(int(ymin) + 1)
            bbox['xmax'] = str(int(xmax) + 1)
            bbox['ymax'] = str(int(ymax) + 1)
            object['bndbox'] = bbox
            object['handle'] = str(detection.handle)

            objects.append(object)
        root['objects'] = objects
        return root



