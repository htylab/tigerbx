import sys
import os
from os.path import basename, join, isdir, commonpath
import glob
import platform
import nibabel as nib
import numpy as np
import time
import warnings
from nilearn.image import resample_to_img, reorder_img, resample_img
warnings.simplefilter(action='ignore', category=FutureWarning)


from tigerbx import lib_tool
from tigerbx import lib_reg
from tigerbx.bx import produce_mask, save_nib, get_template


def reg(argstring, input=None, output=None, model=None, template=None, save_displacement=False, affine_type='C2FViT'):
    from types import SimpleNamespace as Namespace
    args = Namespace()
    if not isinstance(input, list):
        input = [input]
    args.input = input
    args.output = output
    args.model = model
    args.gpu = 'g' in argstring
    args.gz = 'z' in argstring
    args.bet = 'b' in argstring
    args.affine = 'A' in argstring
    args.registration = 'r' in argstring
    args.syn = 's' in argstring
    args.syncc = 'S' in argstring
    args.fusemorph = 'F' in argstring
    args.rigid = 'R' in argstring
    args.vbm = 'v' in argstring
    args.template = template
    args.save_displacement = save_displacement
    args.affine_type = affine_type
    return run_args(args)

def run_args(args):

    run_d = vars(args) #store all arg in dict
    if run_d['fusemorph']:
        run_d['aseg'] = True
        
    if run_d['vbm']:
        run_d['registration'] = run_d['cgw'] = True
        #run_d['fusemorph'] = run_d['cgw'] = run_d['aseg'] = True
 
    input_file_list = args.input
    if os.path.isdir(args.input[0]):
        input_file_list = glob.glob(join(args.input[0], '*.nii'))
        input_file_list += glob.glob(join(args.input[0], '*.nii.gz'))

    elif '*' in args.input[0]:
        input_file_list = glob.glob(args.input[0])

    output_dir = args.output
    omodel = dict()
    omodel['bet'] = 'mprage_bet_v005_mixsynthv4.onnx'
    omodel['aseg'] = 'mprage_aseg43_v007_16ksynth.onnx'
    omodel['cgw'] = 'mprage_cgw_v001_r111.onnx'
    omodel['reg'] = 'mprage_reg_v003_train.onnx'
    omodel['affine'] = 'mprage_affine_v002_train.onnx'
    omodel['rigid'] = 'mprage_rigid_v002_train.onnx'

 
    # if you want to use other models
    if isinstance(args.model, dict):
        for mm in args.model.keys():
            omodel[mm] = args.model[mm]
    elif isinstance(args.model, str):
        import ast
        model_dict = ast.literal_eval(args.model)
        for mm in model_dict.keys():
            omodel[mm] = model_dict[mm]


    print('Total nii files:', len(input_file_list))

    #check duplicate basename
    #for detail, check get_template
    base_ffs = [basename(f) for f in input_file_list]
    common_folder = None
    if len(base_ffs) != len(set(base_ffs)):
        common_folder = commonpath(input_file_list)
        
    count = 0
    result_all = []
    result_filedict = dict()
    for f in input_file_list:
        count += 1
        result_dict = dict()
        result_filedict = dict()

        print(f'{count} Preprocessing :', os.path.basename(f))
        t = time.time()
        ftemplate, f_output_dir = get_template(f, output_dir, args.gz, common_folder)    
        
        if run_d['vbm']:
            dir_path, filename = os.path.split(ftemplate)           
            prefix = filename.split('_@@@@')[0]          
            new_dir_path = os.path.join(dir_path, prefix)
            os.makedirs(new_dir_path, exist_ok=True)
            vbm_ftemplate = os.path.join(dir_path, prefix, filename)


        tbetmask_nib, qc_score = produce_mask(omodel['bet'], f, GPU=args.gpu, QC=True)
        input_nib = nib.load(f)
        tbet_nib = lib_tool.read_nib(input_nib) * lib_tool.read_nib(tbetmask_nib)

        if lib_tool.check_dtype(tbet_nib, input_nib.dataobj.dtype):
            tbet_nib = tbet_nib.astype(input_nib.dataobj.dtype)

        tbet_nib = nib.Nifti1Image(tbet_nib, input_nib.affine, input_nib.header)
        tbet_nib111 = lib_tool.resample_voxel(tbet_nib, (1, 1, 1),interpolation='continuous')
        tbet_nib111 = reorder_img(tbet_nib111, resample='continuous')

        zoom = tbet_nib.header.get_zooms() 

        if max(zoom) > 1.1 or min(zoom) < 0.9:
            tbet_seg = tbet_nib111
        else:
            tbet_seg = reorder_img(tbet_nib, resample='continuous')
        
        print('QC score:', qc_score)
        
        if run_d.get('bet', False):
            fn = save_nib(tbet_nib, ftemplate, 'tbet')
            result_dict['tbet'] = tbet_nib
            result_filedict['tbet'] = fn
            
        if run_d.get('aseg', False):
            result_nib = produce_mask(omodel['aseg'], f, GPU=args.gpu,
                                     brainmask_nib=tbetmask_nib, tbet111=tbet_seg)
            if not run_d['fusemorph']:
                fn = save_nib(result_nib, ftemplate, 'aseg')
                result_filedict['aseg'] = fn
            result_dict['aseg'] = result_nib

        if run_d.get('cgw', False): # FSL style segmentation of CSF, GM, WM
            model_ff = lib_tool.get_model(omodel['cgw'])
            normalize_factor = np.max(input_nib.get_fdata())
            #tbet_nib111 = lib_tool.resample_voxel(tbet_nib, (1, 1, 1),interpolation='linear')
            bet_img = lib_tool.read_nib(tbet_nib111)
            
            image = bet_img[None, ...][None, ...]
            image = image/normalize_factor
            cgw = lib_tool.predict(model_ff, image, args.gpu)[0]

            result_dict['cgw'] = []
            result_filedict['cgw'] = []
            for kk in [1, 2, 3]:
                pve = cgw[kk]
                pve = pve* (bet_img>0)

                pve_nib = nib.Nifti1Image(pve, tbet_nib111.affine, tbet_nib111.header)
                pve_nib = resample_to_img(
                    pve_nib, input_nib, interpolation="linear")

                pve_nib.header.set_data_dtype(float)
                
                if not run_d['vbm'] or kk==2:
                    print(ftemplate)
                    fn = save_nib(pve_nib, vbm_ftemplate, f'cgw_pve{kk-1}')
                    result_filedict['cgw'].append(fn)
                result_dict['cgw'].append(pve_nib)
                
        if run_d['affine'] or run_d['rigid'] or run_d['registration'] or run_d['fusemorph'] or run_d['syn'] or run_d['syncc']:
            displacement_dict = {
                "init_flow": None,
                "rigid_matrix": None,
                "affine_matrix": None,
                "reference_info": None,
                "dense_warp": None,
                "SyN_dense_warp": None,
                "SyNCC_dense_warp": None,
                "Fuse_dense_warp": None
            }
            
            bet = lib_tool.read_nib(input_nib) * lib_tool.read_nib(tbetmask_nib)
            bet = bet.astype(input_nib.dataobj.dtype)
            bet_nib = nib.Nifti1Image(bet, input_nib.affine, input_nib.header)
            
            bet_nib = reorder_img(bet_nib, resample='continuous')
            ori_affine = bet_nib.affine
            bet_data = bet_nib.get_fdata()
            
            template_nib = lib_reg.get_template(run_d['template'])
            template_nib = reorder_img(template_nib, resample='continuous')
            fixed_affine = template_nib.affine
            
            if run_d['rigid']:
                template_data = template_nib.get_fdata()
                bet_data_R, _ = lib_reg.pad_to_shape(bet_data, (256, 256, 256))
                bet_data_R, _ = lib_reg.crop_image(bet_data_R, target_shape=(256, 256, 256))
                template_data, pad_width = lib_reg.pad_to_shape(template_data, (256, 256, 256))
                
                moving = bet_data_R.astype(np.float32)[None, ...][None, ...]
                moving = lib_reg.min_max_norm(moving)
                if run_d['template'] == None:
                    template_data = np.clip(template_data, a_min=2500, a_max=np.max(template_data))
                fixed = template_data.astype(np.float32)[None, ...][None, ...]
                fixed = lib_reg.min_max_norm(fixed)
                
                model_ff = lib_tool.get_model(omodel['rigid'])
                output = lib_tool.predict(model_ff, [moving, fixed], GPU=args.gpu, mode='reg')
                rigided, rigid_matrix, init_flow = np.squeeze(output[0]), np.squeeze(output[1]), output[2]
                
                displacement_dict["init_flow"] = init_flow
                displacement_dict["rigid_matrix"] = rigid_matrix
                
                rigided = lib_reg.remove_padding(rigided, pad_width)
                
                rigid_nib = nib.Nifti1Image(rigided, fixed_affine)
                fn = save_nib(rigid_nib, ftemplate, 'rigid')
                result_dict['rigid'] = rigid_nib
                result_filedict['rigid'] = fn
            
            if run_d['affine'] or run_d['registration'] or run_d['fusemorph'] or run_d['syn'] or run_d['syncc']:
                template_data = template_nib.get_fdata()
                if run_d['affine_type'] != 'ANTs':
                    bet_data, _ = lib_reg.pad_to_shape(bet_data, (256, 256, 256))
                    bet_data, _ = lib_reg.crop_image(bet_data, target_shape=(256, 256, 256))
                    template_data, pad_width = lib_reg.pad_to_shape(template_data, (256, 256, 256))
                
                moving = bet_data.astype(np.float32)[None, ...][None, ...]
                moving = lib_reg.min_max_norm(moving)
                if run_d['template'] == None:
                    template_data = np.clip(template_data, a_min=2500, a_max=np.max(template_data))
                fixed = template_data.astype(np.float32)[None, ...][None, ...]
                fixed = lib_reg.min_max_norm(fixed)
                
                if run_d['affine_type'] == 'C2FViT':
                    model_ff = lib_tool.get_model(omodel['affine'])
                    output = lib_tool.predict(model_ff, [moving, fixed], GPU=args.gpu, mode='reg')
                    affined, affine_matrix, init_flow = np.squeeze(output[0]), np.squeeze(output[1]), output[2]
                    initflow_nib = nib.Nifti1Image(init_flow, ori_affine)
                    displacement_dict["init_flow"] = init_flow
                    result_dict['init_flow'] = initflow_nib
                    displacement_dict["affine_matrix"] = affine_matrix
                elif run_d['affine_type'] == 'ANTs':
                    ants_fixed, reference_info  = lib_reg.get_ants_info(template_data, fixed_affine)
                    ants_moving, _ = lib_reg.get_ants_info(bet_data, ori_affine)
                    
                    affined, affine_matrix = lib_reg.apply_ANTs_reg(ants_moving, ants_fixed, 'Affine')
                    affined = lib_reg.min_max_norm(affined)
                    displacement_dict.update(reference_info)
                    displacement_dict.update(affine_matrix)
                    
                result_dict['Affine_matrix'] = affine_matrix
                if run_d['affine_type'] != 'ANTs':
                    affined = lib_reg.remove_padding(affined, pad_width)
                affine_nib = nib.Nifti1Image(affined, fixed_affine)

                if run_d['affine']:
                    fn = save_nib(affine_nib, ftemplate, 'Af')
                    result_dict['Af'] = affine_nib
                    result_filedict['Af'] = fn
    
                if run_d['registration']:
                    template_data = template_nib.get_fdata()
                    
                    fixed_image = template_data.astype(np.float32)[None, ...][None, ...]
                    fixed_image = lib_reg.min_max_norm(fixed_image)
                    #fixed_image = fixed_image/np.max(fixed_image)
                    
                    Af_data = affine_nib.get_fdata()
                    moving_image = Af_data.astype(np.float32)[None, ...][None, ...]
                    #moving_image = moving_image/np.max(moving_image)
                    
                    model_ff = lib_tool.get_model(omodel['reg'])
                    
                    output = lib_tool.predict(model_ff, [moving_image, fixed_image], GPU=args.gpu, mode='reg')
                    moved, warp = np.squeeze(output[0]), np.squeeze(output[1])
                    moved_nib = nib.Nifti1Image(moved,
                                             fixed_affine, template_nib.header)
                    warp_nib = nib.Nifti1Image(warp,
                                             fixed_affine, template_nib.header)
                    
                    if not run_d['vbm']:
                        fn = save_nib(moved_nib, ftemplate, 'reg')
                        result_filedict['reg'] = fn
                    result_dict['reg'] = moved_nib        
                    
                    
                    displacement_dict["dense_warp"] = warp
                    result_dict['dense_warp'] = warp_nib
                for ants_reg_str in ['syn', 'syncc']:                    
                    if run_d[ants_reg_str]:
                        template_data = template_nib.get_fdata()
                        ants_fixed, reference_info  = lib_reg.get_ants_info(template_data, template_nib.affine)
                        bet_data = bet_nib.get_fdata()
                        ants_moving, _ = lib_reg.get_ants_info(bet_data, bet_nib.affine)
                        
                        if ants_reg_str == 'syn':
                            moved, ants_dict = lib_reg.apply_ANTs_reg(ants_moving, ants_fixed, 'SyN')
                        elif ants_reg_str == 'syncc':
                            moved, ants_dict = lib_reg.apply_ANTs_reg(ants_moving, ants_fixed, 'SyNCC')
                        moved = lib_reg.min_max_norm(moved)
                        displacement_dict.update(reference_info)
                        displacement_dict.update(ants_dict)
                        moved_nib = nib.Nifti1Image(moved,
                                                 fixed_affine, template_nib.header)
                        fn = save_nib(moved_nib, ftemplate, ants_reg_str)
                        result_dict[ants_reg_str] = moved_nib
                        result_dict[ants_reg_str + '_dense_warp'] = ants_dict
                    
                if run_d['fusemorph']:
                    model_affine_transform = lib_tool.get_model('mprage_affinetransform_v002_near.onnx')
                    model_transform = lib_tool.get_model('mprage_transform_v002_near.onnx')
                    model_transform_bili = lib_tool.get_model('mprage_transform_v002_bili.onnx')
                    
                    template_data = template_nib.get_fdata()
                    fixed_image = template_data.astype(np.float32)[None, ...][None, ...]
                    fixed_image = lib_reg.min_max_norm(fixed_image)

                                        
                    template_seg_nib = lib_reg.get_template_seg(run_d['template'])
                    template_seg_nib = reorder_img(template_seg_nib, resample='continuous')
                    template_seg_data = template_seg_nib.get_fdata()
                    fixed_seg_image = template_seg_data.astype(np.float32)[None, ...][None, ...]

                    
                    Af_data = affine_nib.get_fdata()
                    moving_image = Af_data.astype(np.float32)[None, ...][None, ...]
                    
                    moving_seg_nib = result_dict['aseg']
                    moving_seg_nib = reorder_img(moving_seg_nib, resample='nearest')
                    moving_seg_data = moving_seg_nib.get_fdata().astype(np.float32)
                    if run_d['affine_type'] != 'ANTs':
                        moving_seg_data, _ = lib_reg.pad_to_shape(moving_seg_data, (256, 256, 256))
                        moving_seg_data, _ = lib_reg.crop_image(moving_seg_data, target_shape=(256, 256, 256))
                        moving_seg = np.expand_dims(np.expand_dims(moving_seg_data, axis=0), axis=1)
                                
                        affine_matrix= np.expand_dims(affine_matrix, axis=0)
                        output = lib_tool.predict(model_affine_transform, [moving_seg, init_flow, affine_matrix], GPU=args.gpu, mode='affine_transform')
                        affine_matrix = np.squeeze(affine_matrix, axis=0)
                        moving_seg = np.squeeze(output[0])
                    
                        moving_seg = lib_reg.remove_padding(moving_seg, pad_width)
                    else:
                        ants_moving_seg, _ = lib_reg.get_ants_info(moving_seg_data, moving_seg_nib.affine)
                        moving_seg = lib_reg.ants_transform(ants_moving_seg, displacement_dict, mode='affine')
                        
                        
                    
                    model_ff = lib_tool.get_model(omodel['reg'])
                    moving_image_current = moving_image
                    moving_seg_current = np.expand_dims(np.expand_dims(moving_seg, axis=0), axis=1)
                    warps = []
                    
                    for i in range(1, 4):
                        output = lib_tool.predict(model_ff, [moving_image_current, fixed_image], GPU=args.gpu, mode='reg')
                        moved, warp = output[0], output[1]
                        
                        warps.append(warp)                    
                        output = lib_tool.predict(model_transform, [moving_seg_current, warp], GPU=args.gpu, mode='reg')
                        moved_seg = output[0]                        
                        moving_image_current = moved
                        moving_seg_current = moved_seg                    
                        # moving_nib = nib.Nifti1Image(np.squeeze(moved_seg), fixed_affine, template_nib.header)
                        # fn = save_nib(moving_nib, ftemplate, str(i))

                    _, _, best_warp = lib_reg.optimize_fusemorph(warps, moving_seg, model_transform, fixed_seg_image, args)
                    output = lib_tool.predict(model_transform_bili, [moving_image, best_warp], GPU=args.gpu, mode='reg')
                    
                    moved = np.squeeze(output[0])
                    warp = np.squeeze(best_warp)
                    moved_nib = nib.Nifti1Image(moved,
                                             fixed_affine, template_nib.header)
                    warp_nib = nib.Nifti1Image(warp,
                                             fixed_affine, template_nib.header)
                    
                    #if not run_d['vbm']:
                    fn = save_nib(moved_nib, ftemplate, 'Fuse')
                    result_filedict['Fuse'] = fn
                    result_dict['Fuse'] = moved_nib
                    
                    
                    displacement_dict["Fuse_dense_warp"] = warp
                    result_dict['dense_warp'] = warp_nib
                    
            if run_d['vbm']:
                model_transform = lib_tool.get_model('mprage_transform_v002_bili.onnx')
                model_affine_transform_bili = lib_tool.get_model('mprage_affinetransform_v002_bili.onnx')
                
                raw_GM_nib = reorder_img(result_dict['cgw'][1], resample='continuous')
                raw_GM = raw_GM_nib.get_fdata().astype(np.float32)
                if run_d['affine_type'] != 'ANTs':
                    raw_GM, _ = lib_reg.pad_to_shape(raw_GM, (256, 256, 256))
                    raw_GM, _ = lib_reg.crop_image(raw_GM, target_shape=(256, 256, 256))
                    raw_GM = np.expand_dims(np.expand_dims(raw_GM, axis=0), axis=1)
                
                    Affine_matrix= np.expand_dims(affine_matrix, axis=0)
                    output = lib_tool.predict(model_affine_transform_bili, [raw_GM, init_flow, Affine_matrix], GPU=None, mode='affine_transform')
                    affined_GM = np.squeeze(output[0])
                    affined_GM = lib_reg.remove_padding(affined_GM, pad_width)
                else:
                    ants_raw_GM, _ = lib_reg.get_ants_info(raw_GM, raw_GM_nib.affine)
                    affined_GM = lib_reg.ants_transform(ants_raw_GM, displacement_dict, interpolation='linear', mode='affine')
                
                affined_GM = np.expand_dims(np.expand_dims(affined_GM, axis=0), axis=1)
                warp = np.expand_dims(warp, axis=0)
                
                output = lib_tool.predict(model_transform, [affined_GM, warp], GPU=None, mode='reg')
                reg_GM = np.squeeze(output[0])
                reg_GM_nib = nib.Nifti1Image(reg_GM, template_nib.affine, template_nib.header)
                fn = save_nib(reg_GM_nib, vbm_ftemplate, 'RegGM')                
                result_dict['Reg_GM'] = reg_GM_nib
                result_filedict['Reg_GM'] = fn
                
                warp = warp.transpose(0, 2, 3, 4, 1).squeeze()
                warp_Jacobian = lib_reg.jacobian_determinant(warp)
                Modulated_GM = reg_GM*warp_Jacobian
                Modulated_GM_nib = nib.Nifti1Image(Modulated_GM, template_nib.affine, template_nib.header)
                fn = save_nib(Modulated_GM_nib, vbm_ftemplate, 'ModulatedGM')                
                result_dict['Modulated_GM'] = Modulated_GM_nib
                result_filedict['Modulated_GM'] = fn
                
                fwhm_value = 7.065
                Smoothed_GM = lib_reg.apply_gaussian_smoothing(Modulated_GM, fwhm=fwhm_value)
                Smoothed_GM_nib = nib.Nifti1Image(Smoothed_GM, template_nib.affine, template_nib.header)
                fn = save_nib(Smoothed_GM_nib, vbm_ftemplate, 'SmoothedGM')
                result_dict['Smoothed_GM'] = Smoothed_GM_nib
                result_filedict['Smoothed_GM'] = fn
            if run_d['save_displacement']:
                np.savez(ftemplate.replace('@@@@.nii.gz', 'warp') + '.npz', **displacement_dict)


     
        print('Processing time: %d seconds' %  (time.time() - t))
        if len(input_file_list) == 1:
            result_all = result_dict
        else:
            result_all.append(result_filedict) #storing output filenames
    return result_all


