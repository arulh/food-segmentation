import asyncio
import httpx


lan_servers = ["http://10.0.0.100:8000/batch_inference", "http://127.0.0.1:8000/batch_inference"]


async def load_balancer(servers):
    files = []
    for _ in range(500): # send two sample images 500 times
        files.append(('files', ('apple.jpg', open('apple.jpg', 'rb'), 'image/jpeg')))
        files.append(('files', ('banana.jpg', open('banana.jpg', 'rb'), 'image/jpeg')))

    async with httpx.AsyncClient() as client:
        server1 = client.post(lan_servers[0], files=files[:len(files)//2])
        server2 = client.post(lan_servers[1], files=files[len(files)//2:])

        response1, response2 = await asyncio.gather(server1, server2)

        print(response1.json())
        print(response2.json())


if __name__ == "__main__":
    asyncio.run(load_balancer(lan_servers))