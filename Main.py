# Copyright (C) 2017 University Of Michigan
#
# This file is part of gta-postprocessing.
#
# gta-postprocessing is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gta-postprocessing is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gta-postprocessing.  If not, see <http://www.gnu.org/licenses/>.
#

from builtins import print

from Db_Manager import db_manager
from Processor import Processor
from Snapshot import Snapshot
from Detection import Detection
from VOC_Generator import VOC_Generator
import cv2
import numpy as np
from pathlib import Path
from threading import Semaphore
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from progressbar import ProgressBar
from multiprocessing import Lock
from typing import List
import psutil
import math


base_data_dir = '/home/muaz/archives/'
pixel_path = './out/'
output_dir = './outtt'

db = db_manager()
db.setFalseProcessed()

def process_snapshot(snapshot: Snapshot):
    processor = Processor(snapshot, base_data_dir)
    processor.read_tiff()
    processor.refine_bbox()
    usage_stat = psutil.virtual_memory()
    print("Available = ", math.ceil(usage_stat.available/1048576*100)/100, " MBs")
    return snapshot


def main_fn(snapshotList_tmp, vocObj:VOC_Generator):
    pbar = ProgressBar(max_value=len(snapshotList_tmp)).start()
    i = 0
    sem = Semaphore(2)
    pool = ProcessPoolExecutor(2)
    lck = Lock()
    results = []

    def on_done(x):
        updated_snapshot = x.result()
        assert isinstance(updated_snapshot, Snapshot)
        sem.release()
        with lck:
            nonlocal i
            nonlocal results
            pbar.update(i)
            i += 1
            db.update_bestBoxes_processed(updated_snapshot)  # This code was also saving stencil map.
            vocObj.save_snapshot(vocObj, updated_snapshot)
            results.append(updated_snapshot.snapshot_id)

    for sample_snapshot in snapshotList_tmp:
        sem.acquire()
        result = pool.submit(process_snapshot, sample_snapshot)
        result.add_done_callback(partial(on_done))

    pool.shutdown(wait=True)
    pbar.finish()

    return results


def coverage_debugger(snapshotList:List[Snapshot]):
    totalDets = 0
    totalPosit = 0
    totalNegat = 0

    for snap in snapshotList:
        for detObj in snap.detections:

            assert isinstance(detObj, Detection)
            if detObj.coverage>0:
                print("{0} has a {1}".format(snap.snapshot_id, detObj.type))
                totalPosit+=1
            else:
                totalNegat+=1
            totalDets+=1
    print("\n\n\tPos = {0}\n\tNeg = {1}\n\tTotal={2}".format(totalPosit, totalNegat, totalDets))


def main_1(snapshotList):
    voc = VOC_Generator(snapshotList[0].runguid, output_dir)
    voc.create_folders(voc)

    sem_tmp = Semaphore(2)
    pool_tmp = ProcessPoolExecutor(2)
    lck_tmp = Lock()

    def on_done_tmp(x):
        snapshotIDList = x.result()
        sem_tmp.release()
        with lck_tmp:
            # coverage_debugger(updated_snapshotList)
            # voc = VOC_Generator(updated_snapshotList, "./outtt")
            voc.saveTrainTest(voc, snapshotIDList)
            # voc.save_snapshots()

    sem_tmp.acquire()
    result_tmp = pool_tmp.submit(main_fn, snapshotList, voc)
    result_tmp.add_done_callback(partial(on_done_tmp))
    pool_tmp.shutdown(wait=True)

def main_0():
    session_id_to_process = 58
    allRunIds = db.getAllRuns(sess_id=session_id_to_process)
    for i in allRunIds:
        print(i)
    exit()
    for i in allRunIds:
        snapshotListWithRunId = db.getSnapShotsFromSessionAndRun(session_id_to_process, i)
        # snapshotListWithRunId = snapshotListWithRunId[:10]
        print("\n\nFor runid="+str(i)+"\n\n")
        if len(snapshotListWithRunId)>0:
            main_1(snapshotListWithRunId)

if __name__ == "__main__":

    main_0()


# do_one() and debug_1() functions were created for debugging purpose.
def debug_1():
    ref_coded = sample_snapshot.refined_stencil_coded
    print(np.unique(ref_coded))
    return
    ref_colored = sample_snapshot.refined_stencil_colored
    deb = sample_snapshot.debug_image
    # ref_coded= cv2.resize(ref_coded, (int(sample_snapshot.width*0.5), int(sample_snapshot.height*0.5)))
    ref_colored = cv2.resize(ref_colored, (int(sample_snapshot.width * 0.5), int(sample_snapshot.height * 0.5)))
    deb = cv2.resize(deb, (int(sample_snapshot.width * 0.5), int(sample_snapshot.height * 0.5)))
    cv2.imshow("ref_colored", ref_colored)
    cv2.imshow("deb", deb)
    # cv2.imshow("ref_coded",ref_coded)
    cv2.waitKey()


def do_one():
    idx_to_analyze = 30
    snapshotList = db.getSnapShotsFromSession(session_id=1)
    sample_snapshot = snapshotList[idx_to_analyze]
    assert isinstance(sample_snapshot, Snapshot)

    processor = Processor(sample_snapshot, base_data_dir)
    processor.read_tiff()

    processor.refine_bbox()
    # debug_1()
    db.update_bestBoxes_processed(sample_snapshot, Path(pixel_path))  # This code was also saving stencil map.
    voc = VOC_Generator([sample_snapshot], "./outtt")
    voc.save_snapshots()
