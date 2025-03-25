#FROM continuumio/anaconda3
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
COPY ./* .
RUN conda env create -f novobench.yaml
RUN conda activate novobench
#RUN python tests/casanovo.py --mode train --data_path parquet_path --model_path ckpt_path  --config_path config_path
