import numpy as np
import json
from service.variable_service import rtp_queue, manager
from ai.rtp_process import rtpimageproc
import os
import tempfile
import cv2
import ffmpeg


async def ai_processor():
    while True:
        try:
            user_id, payload,codec = await rtp_queue.get()
            ws = await manager.get_connection(user_id)
            
            # print("Running for recognite")
            if(codec=="video/VP8"):
                codectype="VP8"
            else:
                codectype="H264"
            label = rtpimageproc.predict_emotion_from_rtp(payload,codectype)
            if ws:
                await ws.send(json.dumps({
                    "event": "emotion_result",
                    "user_id":user_id,
                    "payload": {
                        "emotion": label,
                    }
                }))
        except Exception as e:
            await ws.send(json.dumps({
                    "event": "emotion_result",
                    "user_id":user_id,
                    "payload": {
                        "emotion": "not found",
                    }
                }))
            print("AI Processor error:", e)
        