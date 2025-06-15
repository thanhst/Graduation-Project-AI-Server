import asyncio
import websockets
import onnxruntime as ort
import numpy as np
import ssl
import json
from conn.conn import ConnectionManager

rtp_queue = asyncio.Queue()
manager = ConnectionManager()