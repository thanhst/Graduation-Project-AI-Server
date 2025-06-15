import numpy as np
import json
from service.variable_service import rtp_queue, manager
from ai.rtp_process import rtpimageproc
async def ai_processor():
    while True:
        try:
            user_id, payload = await rtp_queue.get()
            ws = manager.get_connection(user_id)
            label = rtpimageproc.predict_emotion_from_rtp(payload)
            if ws:
                await ws.send(json.dumps({
                    "event": "emotion_result",
                    "user_id":user_id,
                    "payload": {
                        "emotion": label,
                    }
                }))
        except Exception as e:
            print("AI Processor error:", e)