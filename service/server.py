from concurrent import futures

from pynq import Overlay, Xlnk
import numpy as np
import grpc
import pickle
import time

import matrix_op_pb2
import matrix_op_pb2_grpc

CTRL_REG = 0x00
AP_START = 1 << 0
AUTO_RESTART = 1 << 7

class timer(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        elapsed_secs = time.time() - self.start
        elapsed = round(elapsed_secs * 1000000, 2)
        print('[%s] elapsed time: %f microssec' % (self.name, elapsed))


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
        with timer("unpickling"):
            a = pickle.loads(request.a)
            b = pickle.loads(request.b)

        # run kernel
        with timer("np.stack"):
            self.in_buf[:] = np.stack((a, b))

        with timer("channel transfers"):
            with timer("sendchannel transfer"):
                self.dma.sendchannel.transfer(self.in_buf)
            with timer("recvchannel transfer"):
                self.dma.recvchannel.transfer(self.out_buf)


        with timer("mmult_ip.write"):
            self.mmult_ip.write(CTRL_REG, (AP_START | AUTO_RESTART))

        with timer("channel waits"):
            with timer("sendchannel wait"):
                self.dma.sendchannel.wait()

            with timer("recvchannel wait"):
                self.dma.recvchannel.wait()

        with timer("matrix_op_pb2.OpReply"):
            ret = matrix_op_pb2.OpReply(res=pickle.dumps(np.array(self.out_buf)))

        return ret

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=[
        ('grpc.max_send_message_length', 50 * 1024 * 1024),
        ('grpc.max_receive_message_length', 50 * 1024 * 1024)
    ])
    matrix_op_pb2_grpc.add_MatrixOpServicer_to_server(MatrixOpServicer(), server)
    server.add_insecure_port('0.0.0.0:50051')
    server.start()
    print('listening at 50051...')
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
