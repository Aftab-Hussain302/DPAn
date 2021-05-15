# hash:sha256:3f0291ddc95ead1e9d1d589296a25a22805e8601c41439593e7daf04b016af03
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        pytorch==1.1.0 \
        torchvision==0.3.0 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    imageio==2.9.0 \
    matplotlib==3.3.3 \
    opencv-python==4.4.0.46 \
    pandas==1.1.4 \
    scikit-image==0.17.2 \
    scipy==1.5.4 \
    tqdm==4.54.0
