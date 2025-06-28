import asyncio
from conn.conn import ConnectionManager

rtp_queue = asyncio.Queue()
manager = ConnectionManager()