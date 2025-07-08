FROM ubuntu:22.04

WORKDIR /app

COPY ./server /app/server
RUN apt update && apt install -y \
    python3 python3-pip python3-dev \
    libavcodec-dev libavutil-dev libswscale-dev \
    libopencv-core-dev libopencv-imgproc-dev libopencv-objdetect-dev \
    libopencv-dnn-dev libopencv-calib3d-dev libtbb-dev build-essential cmake\
    && apt clean
RUN pip install --no-cache-dir -r /app/server/requirements.txt

ENV PYTHONPATH=/app/server

ENV LD_LIBRARY_PATH=/app/server/ai/rtp_process

WORKDIR /app/server
CMD ["python3", "-u", "websocket.py"]
