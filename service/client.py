import numpy as np
import grpc
import pickle
import sys
import time
import asyncio

import matrix_op_pb2
import matrix_op_pb2_grpc

# handles np array pickling & unpickling
async def startmatmult(stub, a, b):
    #before = time.time()
    message = matrix_op_pb2.OpRequest(a=pickle.dumps(a), b=pickle.dumps(b))
    reply = await stub.MatMult(message)
    res =  pickle.loads(reply.res)
    print(f'pynq: {res} s')
    #after = time.time()
    #return (res, round(after - before, 2))


if (len(sys.argv) == 1):
    server_addr = 'localhost:50051'
else:
    server_addr = sys.argv[1]

with grpc.insecure_channel(server_addr) as channel:
    stub = matrix_op_pb2_grpc.MatrixOpStub(channel)
    before = time.time()
    for _ in range(800):
        a = np.random.rand(128, 128).astype(dtype=np.float32)
        b = np.random.rand(128, 128).astype(dtype=np.float32)
        asyncio.run(matmult(stub, a, b))
    after = time.time()
    lat1 = round((time.time() - before) * 1000000, 2)
    print(f'800: {lat1} microsec')

    before = time.time()
    res2 = np.matmul(a, b)
    lat = round((time.time() - before) * 1000000, 2)

    print(f'numpy: {lat} microsec')
    assert(np.array_equal(res1[0], res2))
