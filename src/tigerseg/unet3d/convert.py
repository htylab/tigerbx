import os
import logging

def convert_model_tf2onnx(model, output_path='', input_size=(1,128,128,128), opset=None):
    import tensorflow as tf
    import tf2onnx

    spec = (tf.TensorSpec((None, ) + tuple(input_size), tf.float64, name="modelInput"),)

    output_file = os.path.join(output_path, "model.onnx")
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=opset, output_path=output_file)
    
    logging.info('Model has been converted to ONNX') 
    return


def convert_model_pytorch2onnx(model, output_path='', input_size=(1,4,240,240,155), opset=12, dynamic_input=True):
    import torch.onnx

    model.eval() 
    model.cpu()
    
    dummy_input = torch.randn(tuple(input_size), requires_grad=True) 
    
    if dynamic_input:
        dynamic_axes={'modelInput' : {0 : 'batch_size', 2 : 'x', 3 : 'y', 4 : 'z'},
                        'modelOutput' : {0 : 'batch_size', 2 : 'x', 3 : 'y', 4 : 'z'}}
    else:
        dynamic_axes={'modelInput' : {0 : 'batch_size'},    
                        'modelOutput' : {0 : 'batch_size'}}

    output_file = os.path.join(output_path, "model.onnx")
    
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         output_file,       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=opset,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes=dynamic_axes)  # variable length axes 

    logging.info('Model has been converted to ONNX') 
    return