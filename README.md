**Overview**

This is pytorch code for NIPS adversarial attack/[defence](https://www.kaggle.com/c/nips-2017-defense-against-adversarial-attack).

Stack: Python3.5, Pytorch, [foolbox](https://github.com/bethgelab/foolbox)

[DockerImage](https://hub.docker.com/r/tyantov/nips-adv-pytorch/) based on def. pytorch, contains: miniconda 3.5, pytorch, foolbox

**Structure**

 * defence/ - code for the defence competition. For now simple inceptionv3 model on 6 TTA.
   * metadata.json uses my docker image with pytorch & foolbox (fork of pytorch default Dockerfile)
     * model_name point to a model in torch_defence/models.py
   * models/ contains pretrained CNN models

**Defence**

To download default model execute: `cd defence/models; wget https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth`
or specify path inside run-defense.sh 

Example of running the defence using Docker image (to replace prefixes):
```
OUTPUT_DATA=/home/tyantov/workspace/kaggle-nips-adversarial-attacks/output_torch
SUBMISSION_DIRECTORY=/home/tyantov/workspace/kaggle-nips-adversarial-attacks/defence
INPUT_IMAGES=/home/tyantov/workspace/kaggle-nips-adversarial-attacks/images_tf
DOCKER_CONTAINER_NAME=tyantov/nips-adv-pytorch

time nvidia-docker run --shm-size="1024m"\
  -v ${INPUT_IMAGES}:/input_images \
  -v ${OUTPUT_DATA}:/output_data \
  -v ${SUBMISSION_DIRECTORY}:/code \
  -w /code \
  ${DOCKER_CONTAINER_NAME} \
  ./run_defense.sh \
  /input_images \
  /output_data/result.csv
```

**Docker installation**

Some notes:
 * install [Docker CE](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/#install-using-the-repository)
   * to run docker without sudo: `sudo gpasswd -a <USERNAME docker`
 * install [Nvidia-docker](https://github.com/NVIDIA/nvidia-docker#quick-start) (nvidia drivers should be installes on the host)