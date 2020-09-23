import numpy as np
import grpc
import pickle
import sys
import time

import matrix_op_pb2
import matrix_op_pb2_grpc

# handles np array pickling & unpickling
def matmult(stub, a, b):
    before = time.time()

    message = matrix_op_pb2.OpRequest(a=pickle.dumps(a), b=pickle.dumps(b))
    reply = stub.MatMult(message)
    res =  pickle.loads(reply.res)

    after = time.time()
    return (res, round(after - before, 2))


if (len(sys.argv) == 1):
    server_addr = 'localhost:50051'
else:
    server_addr = sys.argv[1]

with grpc.insecure_channel(server_addr) as channel:
    stub = matrix_op_pb2_grpc.MatrixOpStub(channel)

    a = np.random.rand(1, 1).astype(dtype=np.float32)
    b = np.random.rand(1, 1).astype(dtype=np.float32)

    res1 = matmult(stub, a, b)
    print(f'pynq: {res1[1]} s')

    before = time.time()
    res2 = np.matmul(a, b)
    lat = round((time.time() - before) * 1000000, 2)

    print(f'numpy: {lat} microsec')
    assert(np.array_equal(res1[0], res2))
