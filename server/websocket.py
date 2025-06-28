import asyncio
import websockets
from event.ai_process import ai_processor
from event.connection import handle_connection
from helper.dotenv import get_dotenv
import ssl

async def main():
    asyncio.create_task(ai_processor())
    # ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    # ssl_context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
    async with websockets.serve(
        handle_connection,
        "0.0.0.0",
        get_dotenv("PORT"),
        # ssl=ssl_context,
        ping_interval=None,
        ping_timeout=None
    ):
        print("WSS server listening on port 8082 with TLS")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
