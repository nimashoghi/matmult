import numpy as np
import grpc
import pickle
import sys

import matrix_op_pb2
import matrix_op_pb2_grpc

# handles np array pickling & unpickling
def matmult(stub, a, b):
    message = matrix_op_pb2.OpRequest(a=pickle.dumps(a), b=pickle.dumps(b))
    return pickle.loads(stub.MatMult(message).res)

if (len(sys.argv) == 1):
    server_addr = 'localhost:50051'
else:
    server_addr = sys.argv[1]

with grpc.insecure_channel(server_addr) as channel:
    stub = matrix_op_pb2_grpc.MatrixOpStub(channel)
    a = np.random.rand(128, 128).astype(dtype=np.float32)
    b = np.random.rand(128, 128).astype(dtype=np.float32)

    res = matmult(stub, a, b)
    print(res)
    print(np.array_equal(np.matmul(a, b), res))
