import json
from service.variable_service import manager,rtp_queue

async def handle_connection(websocket):
    print("Client connected")
    try:
        init_msg = await websocket.recv()
        data = json.loads(init_msg)
        user_id = data.get("user_id")
        await manager.add(user_id=user_id,websocket=websocket)
        async for message in websocket:
            try:
                event = message["event"]
                if(event == "send_rtp"):
                    payload=message["payload"]
                    rtp_queue.put((user_id,payload))
                else:
                    await websocket.send(json.dumps({"error": "Unknown event"}))
            
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
            
async def check_origin(path, request_headers):
    origin = request_headers.get("Origin")
    allowed_origin = "https://yourdomain.com"
    
    if origin != allowed_origin:
        print(f"Blocked connection from origin: {origin}")
        return (403, [], b"Forbidden: Invalid Origin\n")

    return None