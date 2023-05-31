from typing import Any
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np


class TensorRTInference:
    def __init__(self, model_path, batch_size:int=1, input_dim:int=256):
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        # Load the TensorRT engine
        with open(model_path, 'rb') as f:
            engine_data = f.read()

        # Create a runtime object
        self.runtime = trt.Runtime(self.TRT_LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        # Warmup
        input_data = np.random.rand(batch_size, 3, input_dim, input_dim).astype(np.float32)
        self.input_size = trt.volume(self.engine.get_binding_shape(0)) * input_data.itemsize
        self.output_size = trt.volume(self.engine.get_binding_shape(1)) * input_data.itemsize
        self.device_input = cuda.mem_alloc(self.input_size)
        self.device_output = cuda.mem_alloc(self.output_size)
        self.bindings = [int(self.device_input), int(self.device_output)]
        self.stream = cuda.Stream()

    def __call__(self, *args, **kwargs):
        return self.infer(*args, **kwargs)

    def infer(self, input_data):
        batch_size = input_data.shape[0]
        output = np.empty((batch_size, int(self.output_size / np.dtype(np.float32).itemsize)), dtype=np.float32)

        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(self.device_input, input_data, self.stream)
        # Run inference.
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(output, self.device_output, self.stream)
        # Synchronize the stream.
        self.stream.synchronize()

        return output
    
def ToTensorRT(model_path: str, batch_size: int = 1, input_dim: int = 256):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = batch_size
        
        with open(model_path + '.onnx', 'rb') as model:
            parser.parse(model.read())

        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP1)

        input_shape = [batch_size, 3, input_dim, input_dim]
        network.get_input(0).shape = input_shape
        engine = builder.build_engine(network, config)

        with open(model_path + '.trt', 'wb') as f:
            f.write(engine.serialize())
 

if __name__ == '__main__':
    model_path = 'efficientnetv2-s.engine'
    trt_inference = TensorRTInference(model_path)
    import numpy as np
    input_data = np.random.randn(1, 3, 256, 256).astype(np.float32)
    output_data = trt_inference.infer(input_data)
    print(output_data.shape)
    print(output_data)