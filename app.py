import os
from socket import socket
import time
import threading
from threading import Thread
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
# from fastapi_socketio import SocketManager
import socketio
import json
import uvicorn
import asyncio
from demo import demo
SOCKET_BACKEND_URL = 'http://192.168.0.167:6789'
PORT = 5678

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sio = socketio.AsyncClient(logger=True, engineio_logger=True)

'''
    API 
'''
@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to Core Detect URL Phising server1 API!"}

'''
    Server Fast API and SocketIO
'''
@app.on_event('startup')
async def startup():
    await sio.connect(SOCKET_BACKEND_URL)

@sio.event
async def connect():
    print('connection established')
    # await sio.emit('hello_from_core', "HEllo huyen xinh gai")

async def start_training(data):
    async for response in demo():
        await sio.emit(f'receive_training_process', json.dumps({
            "response": response,
            "sid": data["sid"]
        }))
        await sio.sleep(0.1)
        

@sio.on("start_training")
async def start_training_listener(data):
    Thread(target = await start_training(data)).start()
    # sio.start_background_task(start_training, data)
    # asyncio.create_task(start_training(data))

@sio.event
async def disconnect():
    print('disconnected from server')

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=True, debug=True, ws_ping_interval = 99999999, ws_ping_timeout = 99999999)