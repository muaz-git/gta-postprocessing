from configparser import ConfigParser
import postgresql as pg
from itertools import groupby
from pathlib import Path
import numpy as np

from Snapshot import Snapshot
from Detection import Detection

class db_manager:
    def __init__(self):
        self.ini_filename = "gta-postprocessing.ini.actual"
        self.db_uri = None

    def open_connection(self):
        self.parse_ini()
        self.conn = pg.open(self.db_uri)

    def close_connection(self):
        self.conn.close()

    def setFalseProcessed(self):
        self.open_connection()
        query = "update snapshots set processed=FALSE;"
        self.conn.query(query)
        self.close_connection()

    def parse_ini(self):
        CONFIG = ConfigParser()
        CONFIG.read(self.ini_filename)
        self.db_uri = CONFIG["Database"]["URI"]


    def form_snapshots_and_detections(self, db_objs):
        snapshot_tmp = Snapshot()
        snapshot_tmp.snapshot_id = db_objs[0]['snapshot_id']
        snapshot_tmp.runguid = db_objs[0]['runguid']
        snapshot_tmp.imagepath = db_objs[0]['imagepath']

        snapshot_tmp.view_matrix = np.array(db_objs[0]['view_matrix'], dtype=np.float64)
        snapshot_tmp.proj_matrix = np.array(db_objs[0]['proj_matrix'], dtype=np.float64)

        snapshot_tmp.width = db_objs[0]['width']
        snapshot_tmp.height = db_objs[0]['height']

        detectionList = []
        for obj in db_objs:
            detection_tmp = Detection()
            detection_tmp.detection_id = obj['detection_id']
            detection_tmp.type = obj['type']
            detection_tmp.pos = obj['pos']  # need to process this position
            detection_tmp.bbox = obj['bbox']  # geometry.Box
            detection_tmp.vehicle_class = obj['class']
            detection_tmp.handle = obj['handle']
            detection_tmp.best_bbox = obj['best_bbox']
            detection_tmp.fullbox = obj['fullbox']
            detection_tmp.bbox3d_min = obj['bbox3d_min']
            detection_tmp.bbox3d_max = obj['bbox3d_max']
            detection_tmp.rotation = obj['rot']
            detection_tmp.coverage = obj['coverage']
            detectionList.append(detection_tmp)

        snapshot_tmp.detections = detectionList

        return snapshot_tmp

    def getAllRuns(self, sess_id):
        query = "SELECT run_id, runguid FROM runs WHERE session_id="+str(sess_id)+"order by run_id asc"
        self.open_connection()
        prep_stmt = self.conn.query(query)
        runIds = []
        runguIds = []
        for p in prep_stmt:
            runIds.append(p['run_id'])
            runguIds.append(p['runguid'])

        self.close_connection()

        return runIds, runguIds

    def getSnapShotsFromSessionAndRun(self, sess_id, r_id):
        query = "SELECT snapshot_id, detection_id, coverage, type, class, best_bbox, runguid::text, imagepath, view_matrix," \
                "width, height, proj_matrix, handle, pos::bytea, rot::bytea, bbox, ngv_box3dpolygon(bbox3d)::bytea as fullbox," \
                "ST_MakePoint(ST_XMin(bbox3d), ST_YMin(bbox3d), ST_ZMin(bbox3d))::bytea as bbox3d_min," \
                "ST_MakePoint(ST_XMax(bbox3d), ST_YMax(bbox3d), ST_ZMax(bbox3d))::bytea as bbox3d_max FROM detections JOIN snapshots USING (snapshot_id) JOIN runs USING (run_id) JOIN sessions USING(session_id) WHERE session_id=" + str(
            sess_id) + " and run_id="+str(r_id)+" and processed=false and camera_pos <-> pos < 200 order by snapshot_id desc"
        self.open_connection()
        prep_stmt = self.conn.query(query)

        snapshotList = []
        for snapshot_id, db_objs in groupby(prep_stmt, key=lambda x: x['snapshot_id']):
            db_objs = list(db_objs)
            snapshot_tmp = self.form_snapshots_and_detections(db_objs)
            snapshotList.append(snapshot_tmp)
        self.close_connection()
        return snapshotList

    def getSnapShotsFromSession(self, sess_id):

        query = "SELECT snapshot_id, detection_id, coverage, type, class, best_bbox, runguid::text, imagepath, view_matrix," \
                "width, height, proj_matrix, handle, pos::bytea, rot::bytea, bbox, ngv_box3dpolygon(bbox3d)::bytea as fullbox," \
                "ST_MakePoint(ST_XMin(bbox3d), ST_YMin(bbox3d), ST_ZMin(bbox3d))::bytea as bbox3d_min," \
                "ST_MakePoint(ST_XMax(bbox3d), ST_YMax(bbox3d), ST_ZMax(bbox3d))::bytea as bbox3d_max FROM detections JOIN snapshots USING (snapshot_id) JOIN runs USING (run_id) JOIN sessions USING(session_id) WHERE session_id="+str(sess_id)+" and processed=false and camera_pos <-> pos < 200 order by snapshot_id desc"

        self.open_connection()
        prep_stmt = self.conn.query(query)

        snapshotList = []
        for snapshot_id, db_objs in groupby(prep_stmt, key=lambda x: x['snapshot_id']):
            db_objs = list(db_objs)
            snapshot_tmp = self.form_snapshots_and_detections(db_objs)
            snapshotList.append(snapshot_tmp)

        self.close_connection()
        return snapshotList

    # def update_bestBoxes_processed(self, snapshot:Snapshot, pixel_path: Path):
    def update_bestBoxes_processed(self, snapshot:Snapshot):
        self.open_connection()

        update_query = self.conn.prepare("UPDATE detections SET best_bbox=$1, coverage=$2 WHERE detection_id = $3")
        done_query = self.conn.prepare("UPDATE snapshots set processed=$1 where snapshot_id=$2")

        done_query(snapshot.processed, snapshot.snapshot_id)

        if snapshot.detections is not None:
            for detectionObj in snapshot.detections:
                assert isinstance(detectionObj, Detection)

                update_query(detectionObj.best_bbox, detectionObj.coverage, detectionObj.detection_id)  # just updating bestbounding boxes found earlier

        # if snapshot.refined_stencil_coded is not None:
        #     np.savez_compressed(str(pixel_path / (str(snapshot.snapshot_id) + ".npz")),
        #                         snapshot.refined_stencil_coded)  # saves filtered stencil map in compressed format

        self.close_connection()

