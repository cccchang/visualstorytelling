#!/bin/bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Script to download and preprocess the MSCOCO data set.
#
# The outputs of this script are sharded TFRecord files containing serialized
# SequenceExample protocol buffers. See build_mscoco_data.py for details of how
# the SequenceExample protocol buffers are constructed.
#
# usage:
#  ./download_and_preprocess_mscoco.sh
set -e

if [ -z "$1" ]; then
  echo "usage preproces_vist.sh [data dir]"
  exit
fi

# Create the output directories.
OUTPUT_DIR="${1%/}"
SCRATCH_DIR="${OUTPUT_DIR}/raw-data"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SCRATCH_DIR}"
CURRENT_DIR=$(pwd)
WORK_DIR="$0.runfiles/__main__/im2story"

cd ${SCRATCH_DIR}

TRAIN_IMAGE_DIR="${SCRATCH_DIR}/images/train"
VAL_IMAGE_DIR="${SCRATCH_DIR}/images/val"
TEST_IMAGE_DIR="${SCRATCH_DIR}/images/test"

TRAIN_STORIES_FILE="${SCRATCH_DIR}/sis/train.story-in-sequence.json"
VAL_STORIES_FILE="${SCRATCH_DIR}/sis/val.story-in-sequence.json"
TEST_STORIES_FILE="${SCRATCH_DIR}/sis/test.story-in-sequence.json"

# Build TFRecords of the image data.
cd "${CURRENT_DIR}"
BUILD_SCRIPT="${WORK_DIR}/build_vist_data"
"${BUILD_SCRIPT}" \
  --train_image_dir="${TRAIN_IMAGE_DIR}" \
  --val_image_dir="${VAL_IMAGE_DIR}" \
  --test_image_dir="${TEST_IMAGE_DIR}"\
  --train_stories_file="${TRAIN_STORIES_FILE}" \
  --val_stories_file="${VAL_STORIES_FILE}" \
  --test_stories_file="${TEST_STORIES_FILE}"\
  --output_dir="${OUTPUT_DIR}" \
  --word_counts_output_file="${OUTPUT_DIR}/word_counts_1.txt" \
