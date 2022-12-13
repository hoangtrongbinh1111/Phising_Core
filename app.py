import os
from socket import socket
import time
import threading
from threading import Thread
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
# from fastapi_sockeimtio import SocketManager
import socketio
import json
import uvicorn
import asyncio
from demo import demoTrain
from demo import demoTest
from demo import demoInfer

SOCKET_BACKEND_URL = 'http://192.168.0.103:6789'
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
    # print(data)
    async for response in demoTrain(data):
        await sio.emit(f'receive_training_process', json.dumps({
            "response": response,
            "sid": data["sid"],
        }))
        await sio.sleep(0.1)

async def start_testing(data):
    response = await demoTest(data)
    await sio.emit(f'receive_testing_process',json.dumps({
        "response": response,
        "sid" : data["sid"],
    }))
    await sio.sleep(0.1)   
async def start_infering(data):
    print(data)
    response = await demoInfer(data)
    await sio.emit(f'receive_infering_process',json.dumps({
        "response": response,
        "sid" : data["sid"],
    }))
    await sio.sleep(0.1)
        

@sio.on("start_testing")
async def start_testing_listener(data):
    Thread(target= await start_testing(data)).start()

@sio.on("start_training")
async def start_training_listener(data):
    Thread(target = await start_training(data)).start()
    # sio.start_background_task(start_training, data)
    # asyncio.create_task(start_training(data))

@sio.on("start_infering")
async def start_infering_listener(data):
    Thread(target= await start_infering(data)).start()


@sio.event
async def disconnect():
    print('disconnected from server')

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=True, debug=True, ws_ping_interval = 99999999, ws_ping_timeout = 99999999)