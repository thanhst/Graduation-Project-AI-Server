import asyncio
import websockets
import onnxruntime as ort
import numpy as np
import ssl
import json
from conn.conn import ConnectionManager

# Load mô hình ONNX
# session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
# input_name = session.get_inputs()[0].name
rtp_queue = asyncio.Queue()
manager = ConnectionManager()
async def handle_connection(websocket):
    print("Client connected")
    try:
        init_msg = await websocket.recv()
        data = json.loads(init_msg)
        user_id = data.get("user_id")
        await manager.add(user_id=user_id,websocket=websocket)
        async for message in websocket:
            try:
                payload=message
            
            except Exception as e:
                print("Error:", e)
                await websocket.send("ERROR")

    except Exception as e:
        print("Error:",e)
        await websocket.send("ERROR")
    finally:
        if user_id:
            print(f"Client {user_id} disconnected")
            await manager.remove(user_id)

async def main():
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")

    async with websockets.serve(
        handle_connection,
        "0.0.0.0",
        8765,
        ssl=ssl_context
    ):
        print("WSS server listening on port 8765 with TLS")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
