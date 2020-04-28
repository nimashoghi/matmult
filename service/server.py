from concurrent import futures

from pynq import Overlay, Xlnk
import numpy as np
import grpc
import pickle

import matrix_op_pb2
import matrix_op_pb2_grpc

CTRL_REG = 0x00
AP_START = 1 << 0
AUTO_RESTART = 1 << 7

class MatrixOpServicer(matrix_op_pb2_grpc.MatrixOpServicer):
    DIM = 128

    def __init__(self):
        self.overlay = Overlay('/home/xilinx/matmult/overlay/matmult/matmult.bit')
        self.dma = self.overlay.dma
        self.mmult_ip = self.overlay.accel
        self.xlnk = Xlnk()

        self.in_buf = self.xlnk.cma_array(shape=(2, MatrixOpServicer.DIM, MatrixOpServicer.DIM), dtype=np.float32)
        self.out_buf = self.xlnk.cma_array(shape=(MatrixOpServicer.DIM, MatrixOpServicer.DIM), dtype=np.float32)
        

    def MatMult(self, request, context):
        print('request received: matrix mult')
        # load np arrays from bytes
        a = pickle.loads(request.a)
        b = pickle.loads(request.b)

        # run kernel
        self.in_buf[:] = np.stack((a, b))
        self.dma.sendchannel.transfer(self.in_buf)
        self.dma.recvchannel.transfer(self.out_buf)
        self.mmult_ip.write(CTRL_REG, (AP_START | AUTO_RESTART))

        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

        return matrix_op_pb2.OpReply(res=pickle.dumps(self.out_buf))
        

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    matrix_op_pb2_grpc.add_MatrixOpServicer_to_server(MatrixOpServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print('listening at 50051...')
    server.wait_for_termination()

if __name__ == '__main__':
    serve()

