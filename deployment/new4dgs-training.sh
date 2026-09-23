#!/bin/bash
set -o pipefail
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
cd /workspace/new4dgs
./scripts/run_full_motion.sh 2>&1 | tee -a /workspace/new4dgs-full-motion.log
