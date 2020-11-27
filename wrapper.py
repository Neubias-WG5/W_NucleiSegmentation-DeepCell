import sys
import os
from subprocess import call
from cytomine.models import Job
from biaflows import CLASS_OBJSEG
from biaflows.helpers import BiaflowsJob, prepare_data, upload_data, upload_metrics

def main(argv):
    base_path = "{}".format(os.getenv("HOME")) # Mandatory for Singularity
    problem_cls = CLASS_OBJSEG

    with BiaflowsJob.from_cli(argv) as bj:
        bj.job.update(status=Job.RUNNING, progress=0, statusComment="Initialisation...")
        
        # 1. Prepare data for workflow
        in_imgs, gt_imgs, in_path, gt_path, out_path, tmp_path = prepare_data(problem_cls, bj, is_2d=True, **bj.flags)

        # 2. Run image analysis workflow
        bj.job.update(progress=25, statusComment="Launching workflow...")
        shArgs = ["python", "/app/deepcell_script.py", in_path, tmp_path, out_path, str(bj.parameters.nuclei_min_size), str(bj.parameters.boundary_weight)]
        return_code = call(" ".join(shArgs), shell=True, cwd="/app/DeepCell/keras_version")

        # 3. Upload data to BIAFLOWS
        upload_data(problem_cls, bj, in_imgs, out_path, **bj.flags, monitor_params={
            "start": 60, "end": 90, "period": 0.1,
            "prefix": "Extracting and uploading polygons from masks"})
        
        # 4. Compute and upload metrics
        bj.job.update(progress=90, statusComment="Computing and uploading metrics...")
        upload_metrics(problem_cls, bj, in_imgs, gt_path, out_path, tmp_path, **bj.flags)

        # 5. Pipeline finished
        bj.job.update(progress=100, status=Job.TERMINATED, status_comment="Finished.")


if __name__ == "__main__":
    main(sys.argv[1:])
