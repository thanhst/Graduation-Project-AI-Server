import asyncio
import websockets
from event.ai_process import ai_processor
from event.connection import handle_connection
from helper.dotenv import get_dotenv
async def main():
    asyncio.create_task(ai_processor())

    async with websockets.serve(
        handle_connection,
        "0.0.0.0",
        get_dotenv("PORT"),
    ):
        print("WSS server listening on port 8082 with TLS")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
