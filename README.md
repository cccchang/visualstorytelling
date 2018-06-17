Visual Storytelling
=======
In visual storytelling tasks, we generate stories for given image sequences. This project implements two visual storytelling models. The first model is called direct link model, for it trains and predicts each image separately. Linking sentence for each image in the image sequence sequentially, we get a story for this image sequence. The second model is called image context model, for it considers information in both current image and its former image in the image sequence.

# model
In this project models are designed basing on [NIC](https://arxiv.org/abs/1609.06647)
And the code is based on [https://github.com/tensorflow/models/tree/master/research/im2txt](https://github.com/tensorflow/models/tree/master/research/im2txt)
You may read the paper and visit the website for more information.

## Direct link model
![Direct Link Model](https://github.com/cccchang/visualstorytelling/images/direct link model.png)

## Image Context model
![Image Context Model](https://github.com/cccchang/visualstorytelling/images/image context model.png)

# Getting Started
## A Note on Hardware and Training Time
This project is experimented on Nvidia Tesla P100. Using one GPU, initial training of a direct link model cost around a week and fine tuning of a direct link model cost around two weeks. Training time of a image context model is nearly 1.8 times of a direct link model.

## Install Required Packages
Please see [https://github.com/tensorflow/models/tree/master/research/im2txt](https://github.com/tensorflow/models/tree/master/research/im2txt) for the information.

## Prepare the Training Data
In this process, we used only 5 of 12 splits of VIST training dataset. The 5 splits and the TFRecord files of them cost around 250G. Before running the script, ensure you have enough space on your devices.
```Bash
#Location to save the VIST data
VIST_DIR="/raid/chuchang/vist"

#Build the preprocessing script.
cd /home/chuchang/im2story
bazel build //im2story:preprocess_vist

#Run the preprocessing script
bazel-bin/im2story/preprocess_vist "${VIST_DIR}"
```

## Download the Inception v3 Checkpoint
```Bash
#Loction to save Inception v3 checkpoint.
INCEPTION_DIR = "${HOME}/im2story/data"
mkdir -p ${INCEPTION_DIR}

wget "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
tar -xvf "inception_v3_2016_08_28.tar.gz" -C ${INCEPTION_DIR}
rm "inception_v3_2016_08_28.tar.gz" 
```

# Training a Model
## Initial Training
```Bash
# Directory containing preprocessed VIST data.
VIST_DIR="$/raid/chuchang/vist"

# Inception v3 checkpoint file.
INCEPTION_CHECKPOINT="${HOME}/im2story/data/inception_v3.ckpt"

# Directory to save the model.
MODEL_DIR="${HOME}/im2story/model"

# Build the model.
cd research/im2story
bazel build -c opt //im2story/...

# Run the training script.
bazel-bin/im2story/train \
  --input_file_pattern="${VIST_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=false \
  --number_of_steps=1000000
```
## Fine Tuning
```Bash
# Restart the training script with --train_inception=true.
bazel-bin/im2story/train \
  --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=true \
  --number_of_steps=3000000 
```
# Generating Stories
```Bash
# Path to checkpoint file or a directory containing checkpoint files. Passing
# a directory will only work if there is also a file named 'checkpoint' which
# lists the available checkpoints in the directory. It will not work if you
# point to a directory with just a copy of a model checkpoint: in that case,
# you will need to pass the checkpoint path explicitly.
CHECKPOINT_PATH="/home/chuchang/im2story/model/train"

# Vocabulary file generated by the preprocessing script.
VOCAB_FILE="/raid/chuchang/vist/word_counts.txt"

# JPEG image file to caption.
IMAGE_FILE="/raid/chuchang/vist/raw-data/images/train/355248088.jpg"

# Build the inference binary.
cd /home/chuchang/im2story
bazel build -c opt //im2story:run_inference

# Ignore GPU devices (only necessary if your GPU is currently memory
# constrained, for example, by running the training script).
export CUDA_VISIBLE_DEVICES=""

# Run inference to generate captions.
bazel-bin/im2story/run_inference \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --vocab_file=${VOCAB_FILE} \
  --input_files=${IMAGE_FILE}
```

# Results
In the results, the direct link model went through 800000 steps of initial training and 1500000 steps of fine tuning. While the image context model only went through 800000 steps of initial training and 600000 steps of fine tuning.
![example1](https://github.com/cccchang/visualstorytelling/images/example1.png)
![example2](https://github.com/cccchang/visualstorytelling/images/example2.png)
![example3](https://github.com/cccchang/visualstorytelling/images/example3.png)
