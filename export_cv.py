import torch
import torchvision
import onnx
from onnxruntime.quantization import QuantizationMode, quantize


model = torchvision.models.resnet50()

# The model needs to be in evaluation mode
model.eval()

torch.save(model, 'cv_eager.pt')

quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
torch.save(model, 'cv_eager_int8.pt')

traced_model = torch.jit.script(model)
torch.jit.save(traced_model, "cv_ts.pt")

# Input to the model
x = torch.randn(1, 3, 224, 224, requires_grad=True)
# torch_out = torch_model(x)

# Export the model
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "cv_onnx.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size', 2 : 'height', 3 : 'width'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

onnx_model = onnx.load('cv_onnx.onnx')

quantized_model = quantize(
        model=onnx_model,
        quantization_mode=QuantizationMode.IntegerOps,
        force_fusions=True,
        symmetric_weight=True,
    )

onnx.save_model(quantized_model, 'cv_onnx_int8.onnx')

model = model.cuda().half()
x = x.cuda().half()
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "cv_onnx_fp16.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size', 2 : 'height', 3 : 'width'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
