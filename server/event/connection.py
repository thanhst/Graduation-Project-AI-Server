import json
from service.variable_service import manager,rtp_queue
from helper.dotenv import get_dotenv
import base64
async def handle_connection(websocket):
    try:
        init_msg = await websocket.recv()
        data = json.loads(init_msg)
        user_id = data.get("userId")
        print(f'Client {user_id} connected')
        await manager.add(user_id=user_id,websocket=websocket)
        async for message in websocket:
            msg = json.loads(message)
            try:
                event = msg["event"]
                if(event == "send_rtp"):
                    payload=msg["payload"]
                    codectype = payload["codec"]
                    encoded_data = payload["data"]
                    binary_data = base64.b64decode(encoded_data)
                    await rtp_queue.put((user_id,binary_data,codectype))
                    
                elif(event=="ping_ai"):
                    await manager.send(user_id,json.dumps({
                    "event": "pong_ai",
                    "payload": {}
                }))
                    
            except Exception as e:
                print("WebSocket closed or send failed:", e)

    except Exception as e:
            # await websocket.send("ERROR")
            print(e)
    finally:
        if user_id:
            print(f"Client {user_id} disconnected")
            await manager.remove(user_id)
            
async def check_origin(path, request_headers):
    origin = request_headers.get("Origin")
    allowed_origin = get_dotenv("MEDIA_URL")
    
    if origin != allowed_origin:
        print(f"Blocked connection from origin: {origin}")
        return (403, [], b"Forbidden: Invalid Origin\n")

    return None