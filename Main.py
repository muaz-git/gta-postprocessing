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
import os

base_data_dir = '/home/muaz/archives/' # Directory in which extracted zip files exist.
output_dir = './outtt/data4/'  # Directory in which processed data will be saved.

db = db_manager()
db.setFalseProcessed()

# process_snapshot refines bounding boxes
def process_snapshot(snapshot: Snapshot):
    processor = Processor(snapshot, base_data_dir)
    processor.read_tiff()
    processor.refine_bbox()
    usage_stat = psutil.virtual_memory()
    print("Available = ", math.ceil(usage_stat.available/1048576*100)/100, " MBs")
    return snapshot


# main_fn() iterates through individual snapshot call function to refine bounding boxes, function can be named process_snapshotList
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
            voc.saveTrainTest(voc, snapshotIDList)

    sem_tmp.acquire()
    result_tmp = pool_tmp.submit(main_fn, snapshotList, voc)
    result_tmp.add_done_callback(partial(on_done_tmp))
    pool_tmp.shutdown(wait=True)

def main_0():
    session_id_to_process = 59  # session id to be processed.
    allRunIds, allRunguIds = db.getAllRuns(sess_id=session_id_to_process) # fetches all runIds and runguids that belongs to same session.

    for runId, runGuid in zip(allRunIds, allRunguIds):
        if not os.path.isdir(base_data_dir + str(runGuid)):
            continue

        snapshotListWithRunId = db.getSnapShotsFromSessionAndRun(session_id_to_process, runId)  # form data from db in to data structure.
        # snapshotListWithRunId = snapshotListWithRunId[:10]
        print("\n\nFor runid="+str(runId)+"\n\n")
        if len(snapshotListWithRunId)>0:
            main_1(snapshotListWithRunId)

if __name__ == "__main__":

    main_0()