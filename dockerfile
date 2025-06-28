FROM ubuntu:22.04

# Tạo thư mục làm việc
WORKDIR /app

# Copy toàn bộ mã nguồn vào container (chỉ thư mục 'server')
COPY ./server /app/server
# Cài dependencies
RUN apt update && apt install -y \
    python3 python3-pip python3-dev \
    libavcodec-dev libavutil-dev libswscale-dev \
    libopencv-core-dev libopencv-imgproc-dev libopencv-objdetect-dev \
    libopencv-dnn-dev libopencv-calib3d-dev libtbb-dev build-essential cmake\
    && apt clean
RUN pip install --no-cache-dir -r /app/server/requirements.txt

# Thiết lập PYTHONPATH để import được module
ENV PYTHONPATH=/app/server

# Đảm bảo linker tìm được thư viện .so
ENV LD_LIBRARY_PATH=/app/server/ai/rtp_process

# Chạy đúng file
WORKDIR /app/server
CMD ["python3", "-u", "websocket.py"]
