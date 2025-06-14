import asyncio

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, asyncio.Queue] = {}
        self.ws_connections: dict[str, any] = {}
        self.lock = asyncio.Lock()

    async def add(self, user_id: str, websocket):
        async with self.lock:
            self.ws_connections[user_id] = websocket
            self.active_connections[user_id] = asyncio.Queue()

    async def remove(self, user_id: str):
        async with self.lock:
            self.ws_connections.pop(user_id, None)
            self.active_connections.pop(user_id, None)

    async def get_connection(self, user_id: str):
        async with self.lock:
            return self.ws_connections.get(user_id, None)

    async def send(self, user_id: str, data: bytes):
        async with self.lock:
            ws = self.ws_connections.get(user_id)
            if ws:
                await ws.send(data)

    async def broadcast(self, data: bytes):
        async with self.lock:
            for ws in self.ws_connections.values():
                await ws.send(data)
