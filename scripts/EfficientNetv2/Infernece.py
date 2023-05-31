import tensorrt as trt
import pycuda.driver as cuda

class TensorRTInference:
    def __init__(self, model_path):
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        # Load the TensorRT engine
        with open(model_path, 'rb') as f:
            engine_data = f.read()

        # Create a runtime object
        self.runtime = trt.Runtime(self.TRT_LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

    def infer(self, input_data):
        # Allocate memory for input and output data
        input_mem = cuda.mem_alloc(self.engine.get_binding_shape(0).numel() * input_data.itemsize)
        output_mem = cuda.mem_alloc(self.engine.get_binding_shape(1).numel() * input_data.itemsize)
        output_data = cuda.pagelocked_empty(self.engine.get_binding_shape(1), dtype=np.float32)

        # Create a stream to run inference
        stream = cuda.Stream()
        cuda.memcpy_htod_async(input_mem, input_data, stream)
        self.context.execute_async(bindings=[int(input_mem), int(output_mem)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_data, output_mem, stream)
        stream.synchronize()

        return output_data

def ToTensorRT(model_path:str):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = 1
        with open(model_path+'.onnx', 'rb') as model:
            parser.parse(model.read())
        
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)

        network.get_input(0).shape = [1, 3, 256, 256]
        engine = builder.build_serialized_network(network, config)
        engine = builder.build_engine(network, config)
        with open(model_path+'.trt', 'wb') as f:
            f.write(engine.serialize())   

if __name__ == '__main__':
    model_path = 'efficientnetv2-s.engine'
    trt_inference = TensorRTInference(model_path)
    import numpy as np
    input_data = np.random.randn(1, 3, 256, 256).astype(np.float32)
    output_data = trt_inference.infer(input_data)
    print(output_data.shape)
    print(output_data)