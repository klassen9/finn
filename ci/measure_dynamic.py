import json
import os
import pandas as pd
import shutil
import subprocess
import sys

def delete_dir_contents(dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("ERROR: Failed to delete %s. Reason: %s" % (file_path, e))


if __name__ == "__main__":
    exit_code = 0
    print("SCANNING DEPLOYMENT PACKAGES IN BUILD ARTIFACTS..")
    # Find deployment packages from artifacts
    artifacts_in_dir = os.path.join("build_artifacts", "runs_output")
    artifacts_out_dir = os.path.join("measurement_artifacts", "runs_output")
    for run in os.listdir(artifacts_in_dir):
        run_in_dir = os.path.join(artifacts_in_dir, run)
        run_out_dir = os.path.join(artifacts_out_dir, run)
        reports_dir = os.path.join(run_out_dir, "reports")
        deploy_archive = os.path.join(run_in_dir, "deploy.zip")
        extract_dir = "measurement"
        if os.path.isfile(deploy_archive):
            print("FOUND DEPLOYMENT PACKAGE IN %s, EXTRACTING.." % run_in_dir)

            # Extract to temporary dir
            os.makedirs(extract_dir, exist_ok=True)
            delete_dir_contents(extract_dir)
            shutil.unpack_archive(deploy_archive, extract_dir)

            # run validate.py (from IODMA driver) if present, otherwise driver.py (instrumentation)
            # TODO: unify IODMA/instrumentation shell & driver
            if os.path.isfile(f"{extract_dir}/driver/validate.py"):
                driver_file = f"{extract_dir}/driver/validate.py"
                driver_args = {
                    "settingsfile": f"{extract_dir}/driver/settings.json",
                    "reportfile": f"{extract_dir}/validation.json",
                    "dataset_root": "/home/xilinx/datasets",  # TODO: env var
                    "batchsize": 100,
                    "platform": "zynq-iodma",
                    "runtime": 30,  # only relevant for idle baseline run
                    "frequency": 100.0,  # will be overwritten by settingsfile (TODO)
                    "device": 0,  # TODO: unnecessary?
                }
            else:
                driver_file = f"{extract_dir}/driver/driver.py"
                driver_dynamic_file = f"{extract_dir}/driver/driver_dynamic.py"
                driver_reset_file = f"{extract_dir}/driver/driver_reset.py"
                driver_args = {
                    "settingsfile": f"{extract_dir}/driver/settings.json",
                    "reportfile": f"{extract_dir}/measured_performance.json",
                    "runtime": 30,  # not relevant for live FIFO-Sizing
                    "frequency": 100.0,  # will be overwritten by settingsfile (TODO)
                    "seed": 1,
                    "device": 0,
                }

    sys.exit(exit_code)
