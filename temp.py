import asyncio
async def test(msg):
    for i in range(10):
        yield f'{i} ===> {msg}'
        # await asyncio.sleep(2)