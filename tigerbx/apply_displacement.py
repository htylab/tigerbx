import sys
import os
from tigerbx import lib_tool
from tigerbx import lib_reg
from nilearn.image import reorder_img
import nibabel as nib
import numpy as np
from tigerbx import bx


# determine if application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(os.path.abspath(__file__))
    
def transform(image_path, warp_path, output_dir=None, GPU=False, interpolation='nearest'):
    
    displacement_dict = np.load(warp_path, allow_pickle=True)
    
    init_flow = displacement_dict['init_flow']
    rigid_matrix = displacement_dict['rigid_matrix']
    affine_matrix = displacement_dict['affine_matrix']
    dense_warp = displacement_dict['dense_warp']
    Fuse_dense_warp = displacement_dict['Fuse_dense_warp']
    
    if init_flow is None or (isinstance(init_flow, np.ndarray) and not np.any(init_flow)):
        raise ValueError("init_flow is None and the program cannot be executed.")    
    if (dense_warp is not None or Fuse_dense_warp is not None) and (affine_matrix is None or (isinstance(affine_matrix, np.ndarray) and not np.any(affine_matrix))):
        raise ValueError("affine_matrix is None or empty, and at least one of dense_warp or Fuse_dense_warp is not None.")
    
    ftemplate, f_output_dir = bx.get_template(image_path, output_dir, None)
    
    model_transform = lib_tool.get_model('mprage_transform_v002_near.onnx')
    model_transform_bili = lib_tool.get_model('mprage_transform_v002_bili.onnx')
    model_affine_transform = lib_tool.get_model('mprage_affinetransform_v002_near.onnx')
    model_affine_transform_bili = lib_tool.get_model('mprage_affinetransform_v002_bili.onnx')
        
    template_nib = lib_reg.get_template(None)
    template_nib = reorder_img(template_nib, resample='continuous')
    template_data = template_nib.get_fdata()
    template_data, pad_width = lib_reg.pad_to_shape(template_data, (256, 256, 256))
    
    input_nib = nib.load(image_path)
    input_nib = reorder_img(input_nib, resample=interpolation)
    input_data = input_nib.get_fdata().astype(np.float32)
    input_data, _ = lib_reg.pad_to_shape(input_data, (256, 256, 256))
    input_data, _ = lib_reg.crop_image(input_data, target_shape=(256, 256, 256))
    input_data = np.expand_dims(np.expand_dims(input_data, axis=0), axis=1)
    
    init_flow = init_flow.astype(np.float32)
    if rigid_matrix is not None and rigid_matrix.shape != ():
        rigid_matrix = np.expand_dims(rigid_matrix.astype(np.float32), axis=0)
        model = model_affine_transform if interpolation == 'nearest' else model_affine_transform_bili
        output = lib_tool.predict(model, [input_data, init_flow, rigid_matrix], GPU=GPU, mode='affine_transform')
        rigid = lib_reg.remove_padding(np.squeeze(output[0]), pad_width)
        rigid_nib = nib.Nifti1Image(rigid,
                                      template_nib.affine, template_nib.header)
        fn = bx.save_nib(rigid_nib, ftemplate, 'rigid')
    if affine_matrix is not None and affine_matrix.shape != ():
        affine_matrix = np.expand_dims(affine_matrix.astype(np.float32), axis=0)
        model = model_affine_transform if interpolation == 'nearest' else model_affine_transform_bili
        output = lib_tool.predict(model, [input_data, init_flow, affine_matrix], GPU=GPU, mode='affine_transform')
        affined = lib_reg.remove_padding(np.squeeze(output[0]), pad_width)
        affined_nib = nib.Nifti1Image(affined,
                                      template_nib.affine, template_nib.header)
        fn = bx.save_nib(affined_nib, ftemplate, 'Af')
    if dense_warp is not None and dense_warp.shape != ():
        affined_exp = np.expand_dims(np.expand_dims(affined, axis=0), axis=1)
        dense_warp = np.expand_dims(dense_warp.astype(np.float32), axis=0)
        model = model_transform if interpolation == 'nearest' else model_transform_bili
        output = lib_tool.predict(model, [affined_exp, dense_warp], GPU=GPU, mode='reg')
        reged = np.squeeze(output[0])
        reged_nib = nib.Nifti1Image(reged,
                                    template_nib.affine, template_nib.header)
        fn = bx.save_nib(reged_nib, ftemplate, 'reg')
    if Fuse_dense_warp is not None and Fuse_dense_warp.shape != ():
        affined_exp = np.expand_dims(np.expand_dims(affined, axis=0), axis=1)
        Fuse_dense_warp = np.expand_dims(Fuse_dense_warp.astype(np.float32), axis=0)
        model = model_transform if interpolation == 'nearest' else model_transform_bili
        output = lib_tool.predict(model, [affined_exp, Fuse_dense_warp], GPU=GPU, mode='reg')
        Fused = np.squeeze(output[0])
        Fused_nib = nib.Nifti1Image(Fused,
                                    template_nib.affine, template_nib.header)
        fn = bx.save_nib(Fused_nib, ftemplate, 'Fuse')
        
        
        
            
            
        