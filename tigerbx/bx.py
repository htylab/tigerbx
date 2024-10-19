import sys
import os
from os.path import basename, join, isdir, dirname, commonpath, relpath
import argparse
import time
import numpy as np

import glob
import platform
import nibabel as nib

from tigerbx import lib_tool
from tigerbx import lib_bx
import copy
from nilearn.image import resample_to_img, reorder_img, resample_img
from itertools import product
import concurrent.futures

# determine if application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(os.path.abspath(__file__))
    
def produce_mask(model, f, GPU=False, QC=False, brainmask_nib=None, tbet111=None, patch=False):
    if not isinstance(model, list):
        model = [model]
    # for multi-model ensemble
    model_ff_list = []
    for mm in model:
        model_ff_list.append(lib_tool.get_model(mm))

    input_nib = nib.load(f)

    if tbet111 is None:
        input_nib_resp = lib_bx.read_file(model_ff_list[0], f)
    else:
        #input_nib_resp = lib_bx.reorient(tbet111)
        input_nib_resp = copy.deepcopy(tbet111) #using the copy to avoid modifying it.
        
        
    mask_nib_resp, prob_resp = lib_bx.run(
        model_ff_list, input_nib_resp,  GPU=GPU, patch=patch)
        
    mask_nib = resample_to_img(
        mask_nib_resp, input_nib, interpolation="nearest")


    if brainmask_nib is None:

        output = lib_bx.read_nib(mask_nib)

    else:
        output = lib_bx.read_nib(mask_nib) * lib_bx.read_nib(brainmask_nib)


    output = lib_bx.read_nib(mask_nib)

    if np.max(output) <=255:
        dtype = np.uint8
    else:
        dtype = np.int16

    output = output.astype(dtype)

    output_nib = nib.Nifti1Image(output, input_nib.affine, input_nib.header)
    output_nib.header.set_data_dtype(dtype)

    if QC:
        probmax = np.max(prob_resp, axis=0)
        qc_score = np.percentile(
            probmax[lib_bx.read_nib(mask_nib_resp) > 0], 1) - 0.5
        #qc = np.percentile(probmax, 1) - 0.5
        qc_score = min(int(qc_score * 1500), 100)
        return output_nib, qc_score
    else:

        return output_nib

def save_nib(data_nib, ftemplate, postfix):
    output_file = ftemplate.replace('@@@@', postfix)
    nib.save(data_nib, output_file)
    print('Writing output file: ', output_file)
    return output_file

def get_template(f, output_dir, get_z, common_folder=None):
    f_output_dir = output_dir
    ftemplate = basename(f).replace('.nii', f'_@@@@.nii').replace('.npz', f'_@@@@.nii.gz')

    if f_output_dir is None: #save the results in the same dir of T1_raw.nii.gz
        f_output_dir = os.path.dirname(os.path.abspath(f))
        
    else:
        os.makedirs(f_output_dir, exist_ok=True)
        #ftemplate = basename(f).replace('.nii', f'_@@@@.nii')
        # When we save results in the same directory, sometimes the result
        # filenames will all be the same, e.g., aseg.nii.gz, aseg.nii.gz.
        # In this case, the program tries to add a header to it.
        # For example, IXI001_aseg.nii.gz.
        if common_folder is not None:
            header = relpath(dirname(f), common_folder).replace(os.sep, '_')
            ftemplate = header + '_' + ftemplate
    
    if get_z and '.gz' not in ftemplate:
        ftemplate += '.gz'
    ftemplate = join(f_output_dir, ftemplate)

    return ftemplate, f_output_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, nargs='+', help='Path to the input image(s); can be a folder containing images in the specific format (nii.gz)')
    parser.add_argument('-o', '--output', default=None, help='File path for output segmentation (default: the directory of input files)')
    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPU')
    parser.add_argument('-m', '--betmask', action='store_true', help='Producing BET mask')
    parser.add_argument('-a', '--aseg', action='store_true', help='Producing ASEG mask')
    parser.add_argument('-b', '--bet', action='store_true', help='Producing BET images')
    parser.add_argument('-B', '--bam', action='store_true', help='Producing brain age mapping')
    parser.add_argument('-c', '--ct', action='store_true', help='Producing cortical thickness map')
    parser.add_argument('-C', '--cgw', action='store_true', help='Producing FSL-style PVE segmentation')
    parser.add_argument('-d', '--dgm', action='store_true', help='Producing deep GM mask')
    parser.add_argument('-k', '--dkt', action='store_true', help='Producing DKT mask')
    parser.add_argument('-S', '--syn', action='store_true', help='Producing ASEG mask using SynthSeg-like method')
    parser.add_argument('-w', '--wmp', action='store_true', help='Producing white matter parcellation')
    parser.add_argument('-W', '--wmh', action='store_true', help='Producing white matter hypo-intensity mask')
    parser.add_argument('-t', '--tumor', action='store_true', help='Producing tumor mask')
    parser.add_argument('-q', '--qc', action='store_true', help='Saving QC score (pay attention to results with QC scores less than 50)')
    parser.add_argument('-z', '--gz', action='store_true', help='Forcing storing in nii.gz format')
    parser.add_argument('-A', '--affine', action='store_true', help='Affining images to template')
    parser.add_argument('-r', '--registration', action='store_true', help='Registering images to template')
    parser.add_argument('-F', '--fusemorph', action='store_true', help='Registering images to template(FuseMorph)')
    parser.add_argument('-T', '--template', type=str, help='The template filename(default is MNI152)')
    parser.add_argument('-R', '--rigid', action='store_true', help='Rigid transforms images to template')
    parser.add_argument('-p', '--patch', action='store_true', help='patch inference')
    parser.add_argument('-v', '--vbm', action='store_true', help='vbm analysis')
    parser.add_argument('--model', default=None, type=str, help='Specifying the model name')
    parser.add_argument('--clean_onnx', action='store_true', help='Clean onnx models')
    parser.add_argument('--encode', action='store_true', help='Encoding a brain volume to its latent')
    parser.add_argument('--decode', action='store_true', help='Decoding a brain volume from its latent')
    args = parser.parse_args()
    run_args(args)


def run(argstring, input=None, output=None, model=None, template=None):
    from argparse import Namespace
    args = Namespace()
    if not isinstance(input, list):
        input = [input]
    args.input = input
    args.output = output
    args.model = model
    args.clean_onnx = 'clean_onnx' in argstring
    args.encode = 'encode' in argstring
    args.decode = 'decode' in argstring
    args.gpu = 'g' in argstring

    if (args.encode or args.decode or args.clean_onnx):
        argstring = ''
    args.betmask = 'm' in argstring
    args.aseg = 'a' in argstring
    args.bet = 'b' in argstring
    args.bam = 'B' in argstring
    args.ct = 'c' in argstring
    args.cgw = 'C' in argstring
    args.dgm = 'd' in argstring        
    args.dkt = 'k' in argstring
    args.wmh = 'W' in argstring    
    args.wmp = 'w' in argstring
    #args.seg3 = 's' in argstring
    args.syn = 'S' in argstring
    args.tumor = 't' in argstring
    args.qc = 'q' in argstring
    args.gz = 'z' in argstring
    args.affine = 'A' in argstring
    args.registration = 'r' in argstring
    args.fusemorph = 'F' in argstring
    args.rigid = 'R' in argstring
    args.patch = 'p' in argstring
    args.vbm = 'v' in argstring
    args.template = template
    return run_args(args)   


def run_args(args):

    run_d = vars(args) #store all arg in dict
    if True not in [run_d['betmask'], run_d['aseg'], run_d['bet'], run_d['dgm'],
                    run_d['dkt'], run_d['ct'], run_d['wmp'], run_d['qc'], 
                    run_d['wmh'], run_d['bam'], run_d['tumor'], run_d['cgw'], 
                    run_d['syn'], run_d['affine'], run_d['registration'],
                    run_d['fusemorph'], run_d['rigid'], run_d['template'],
                    run_d['encode'], run_d['decode'], run_d['patch'], 
                    run_d['vbm']]:
        run_d['bet'] = True
        # Producing extracted brain by default
        
    if run_d['fusemorph']:
        run_d['aseg'] = True
        
    if run_d['vbm']:
        run_d['registration'] = run_d['cgw'] = True
        
    if run_d['clean_onnx']:
        lib_tool.clean_onnx()
        print('Exiting...')
        return 1


    input_file_list = args.input
    if os.path.isdir(args.input[0]):
        if run_d['decode']:
            input_file_list = glob.glob(join(args.input[0], '*.npz'))
        else:
            input_file_list = glob.glob(join(args.input[0], '*.nii'))
            input_file_list += glob.glob(join(args.input[0], '*.nii.gz'))

    elif '*' in args.input[0]:
        input_file_list = glob.glob(args.input[0])

    output_dir = args.output
    omodel = dict()
    omodel['bet'] = 'mprage_bet_v005_mixsynthv4.onnx'
    omodel['aseg'] = 'mprage_aseg43_v007_16ksynth.onnx'    
    omodel['dkt'] = 'mprage_dkt_v002_train.onnx'
    omodel['ct'] = 'mprage_mix_ct.onnx'
    omodel['dgm'] = 'mprage_dgm12_v002_mix6.onnx'
    omodel['wmp'] = 'mprage_wmp_v003_14k8.onnx'
    omodel['wmh'] = 'mprage_wmh_v002_betr111.onnx'
    omodel['bam'] = 'mprage_bam_v002_betr111.onnx'
    omodel['tumor'] = 'mprage_tumor_v001_r111.onnx'
    omodel['cgw'] = 'mprage_cgw_v001_r111.onnx'
    omodel['syn'] = 'mprage_synthseg_v003_r111.onnx'
    omodel['reg'] = 'mprage_reg_v002_train.onnx'
    omodel['encode'] = 'mprage_encode_v2.onnx'
    omodel['decode'] = 'mprage_decode_v2.onnx'
    omodel['affine'] = 'mprage_affine_v001_train.onnx'
    omodel['rigid'] = 'mprage_rigid_v001_train.onnx'

 
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

        print(f'{count} Processing :', os.path.basename(f))
        t = time.time()

        ftemplate, f_output_dir = get_template(f, output_dir, args.gz, common_folder)

        

        if run_d['vbm']:
            dir_path, filename = os.path.split(ftemplate)           
            prefix = filename.split('_@@@@')[0]          
            new_dir_path = os.path.join(dir_path, prefix)
            os.makedirs(new_dir_path, exist_ok=True)
            ftemplate = os.path.join(dir_path, prefix, filename)
            

        if run_d['encode']:
            model_ff = lib_tool.get_model(omodel['encode'])
            input_nib = nib.load(f)
            input_vol = input_nib.get_fdata().squeeze()
            input_vol = input_vol/np.max(input_vol)
            z_mu, z_sigma = lib_tool.predict(model_ff, input_vol[None, ...][None, ...],
                                              GPU=args.gpu, mode='encode')

            npz = ftemplate.replace('.nii','').replace('.gz', '')
            npz = npz.replace('@@@@', f'encode.npz')
            np.savez_compressed(npz, z_mu=z_mu, z_sigma=z_sigma,
                                affine=input_nib.affine,
                                header=input_nib.header,
                                shape=input_nib.shape)
            
            result_dict['encode'] = np.load(npz)
            result_filedict['encode'] = npz


        if run_d['decode']:
            model_ff = lib_tool.get_model(omodel['decode'])
            latent = np.load(f, allow_pickle=True)
            sample = lib_tool.predict(model_ff, latent['z_mu'],
                                              GPU=args.gpu, mode='decode').squeeze()
            sample = np.clip(sample, 0, 1)
            sample = (sample * 255).astype(np.uint8)

            output_nib = nib.Nifti1Image(sample, latent['affine'], latent['header'].item())
            output_nib.header.set_data_dtype(np.uint8)

            fn = save_nib(output_nib, ftemplate, 'decode')
            result_dict['decode'] = output_nib
            result_filedict['decode'] = fn            
   


        if (not run_d['decode']) and (not run_d['encode']): #skipping the following operation for decode and encode
            tbetmask_nib, qc_score = produce_mask(omodel['bet'], f, GPU=args.gpu, QC=True)
            input_nib = nib.load(f)
            tbet_nib = lib_bx.read_nib(input_nib) * lib_bx.read_nib(tbetmask_nib)
            tbet_nib = tbet_nib.astype(input_nib.dataobj.dtype)
            tbet_nib = nib.Nifti1Image(tbet_nib, input_nib.affine,
                            input_nib.header)
            tbet_nib111 = lib_bx.resample_voxel(tbet_nib, (1, 1, 1),interpolation='continuous')
            tbet_nib111 = reorder_img(tbet_nib111, resample='continuous')

            zoom = tbet_nib.header.get_zooms() 

            if max(zoom) > 1.1 or min(zoom) < 0.9:
                tbet_seg = tbet_nib111
            else:
                tbet_seg = reorder_img(tbet_nib, resample='continuous')
            
            print('QC score:', qc_score)

            result_dict['QC'] = qc_score
            result_filedict['QC'] = qc_score
            if qc_score < 30:
                print('Pay attention to the result with QC < 30. ')
            if run_d['qc'] or qc_score < 30:
                qcfile = ftemplate.replace('.nii','').replace('.gz', '')
                qcfile = qcfile.replace('@@@@', f'qc-{qc_score}.log')
                with open(qcfile, 'a') as the_file:
                    the_file.write(f'QC: {qc_score} \n')
                print('Writing output file: ', qcfile)

        if run_d['betmask']:
            fn = save_nib(tbetmask_nib, ftemplate, 'tbetmask')
            result_dict['tbetmask'] = tbetmask_nib
            result_filedict['tbetmask'] = fn

        if run_d['bet']:
            fn = save_nib(tbet_nib, ftemplate, 'tbet')
            result_dict['tbet'] = tbet_nib
            result_filedict['tbet'] = fn
        
        for seg_str in ['aseg', 'dgm', 'dkt', 'wmp', 'wmh', 'tumor', 'syn']:
            if run_d[seg_str]:
                result_nib = produce_mask(omodel[seg_str], f, GPU=args.gpu,
                                         brainmask_nib=tbetmask_nib, tbet111=tbet_seg, patch=run_d['patch'])
                if not run_d['fusemorph']:
                    fn = save_nib(result_nib, ftemplate, seg_str)
                    result_filedict[seg_str] = fn
                result_dict[seg_str] = result_nib
        if run_d['bam']:
            model_ff = lib_tool.get_model(omodel['bam'])
            #input_nib = nib.load(f)
            
            bet_img = lib_bx.read_nib(tbet_nib111)
            
            image = bet_img[None, ...][None, ...]
            image = image/np.max(image)
            bam = lib_tool.predict(model_ff, image, args.gpu)[0, 0, ...]
            bam[bam < 0.5] = 0

            bam = bam * (bet_img>0)

            bam_nib = nib.Nifti1Image(bam, tbet_nib111.affine, tbet_nib111.header)
            bam_nib = resample_to_img(
                bam_nib, input_nib, interpolation="nearest")

            bam_nib.header.set_data_dtype(float)
            
            fn = save_nib(bam_nib, ftemplate, 'bam')
            result_dict['bam'] = bam_nib
            result_filedict['bam'] = fn

        if run_d['cgw']: # FSL style segmentation of CSF, GM, WM
            model_ff = lib_tool.get_model(omodel['cgw'])
            normalize_factor = np.max(input_nib.get_fdata())
            #tbet_nib111 = lib_bx.resample_voxel(tbet_nib, (1, 1, 1),interpolation='linear')
            bet_img = lib_bx.read_nib(tbet_nib111)
            
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
                    fn = save_nib(pve_nib, ftemplate, f'cgw_pve{kk-1}')
                    result_filedict['cgw'].append(fn)
                result_dict['cgw'].append(pve_nib)                       

        if run_d['ct']:
            model_ff = lib_tool.get_model(omodel['ct'])
            bet_img = lib_bx.read_nib(tbet_nib111)            
            image = bet_img[None, ...][None, ...]
            image = image/np.max(image)
            ct = lib_tool.predict(model_ff, image, args.gpu)[0, 0, ...]
            
            ct[ct < 0.2] = 0
            ct[ct > 5] = 5
            ct = ct * (bet_img > 0).astype(int)

            ct_nib = nib.Nifti1Image(ct, tbet_nib111.affine, tbet_nib111.header)
            ct_nib = resample_to_img(
                ct_nib, input_nib, interpolation="nearest")

            ct_nib.header.set_data_dtype(float)
            
            fn = save_nib(ct_nib, ftemplate, 'ct')
            result_dict['ct'] = ct_nib
            result_filedict['ct'] = fn
            
        if run_d['affine'] or run_d['rigid'] or run_d['registration'] or run_d['fusemorph']:            
            bet = lib_bx.read_nib(input_nib) * lib_bx.read_nib(tbetmask_nib)
            bet = bet.astype(input_nib.dataobj.dtype)
            bet_nib = nib.Nifti1Image(bet, input_nib.affine, input_nib.header)
            
            bet_nib = reorder_img(bet_nib, resample='continuous')
            ori_affine = bet_nib.affine
            bet_data = bet_nib.get_fdata()
            bet_data, _ = lib_bx.pad_to_shape(bet_data, (256, 256, 256))
            bet_data, _ = lib_bx.crop_image(bet_data, target_shape=(256, 256, 256))
            
            template_nib = lib_tool.get_template(run_d['template'])
            template_nib = reorder_img(template_nib, resample='continuous')
            
            fixed_affine = template_nib.affine
            template_data = template_nib.get_fdata()
            template_data, pad_width = lib_bx.pad_to_shape(template_data, (256, 256, 256))
            
            moving = bet_data.astype(np.float32)[None, ...][None, ...]
            moving = lib_bx.min_max_norm(moving)
            if run_d['template'] == None:
                template_data = np.clip(template_data, a_min=2500, a_max=np.max(template_data))
            fixed = template_data.astype(np.float32)[None, ...][None, ...]
            fixed = lib_bx.min_max_norm(fixed)
            
            if run_d['rigid']:
                model_ff = lib_tool.get_model(omodel['rigid'])
                output = lib_tool.predict(model_ff, [moving, fixed], GPU=args.gpu, mode='reg')
                rigided, regid_matrix = np.squeeze(output[0]), np.squeeze(output[1])
                rigided = lib_bx.remove_padding(rigided, pad_width)
                
                rigid_nib = nib.Nifti1Image(rigided, fixed_affine)
                fn = save_nib(rigid_nib, ftemplate, 'rigid')
                result_dict['rigid'] = rigid_nib
                result_filedict['rigid'] = fn
            
            if run_d['affine'] or run_d['registration'] or run_d['fusemorph']: 
                
                model_ff = lib_tool.get_model(omodel['affine'])
                output = lib_tool.predict(model_ff, [moving, fixed], GPU=args.gpu, mode='reg')
                affined, affine_matrix, init_flow = np.squeeze(output[0]), np.squeeze(output[1]), output[2]
                initflow_nib = nib.Nifti1Image(init_flow, ori_affine)
                result_dict['init_flow'] = initflow_nib
                affined = lib_bx.remove_padding(affined, pad_width)
                affine_nib = nib.Nifti1Image(affined, fixed_affine)

                result_dict['Affine_matrix'] = affine_matrix
                if run_d['affine']:
                    fn = save_nib(affine_nib, ftemplate, 'Af')
                    result_dict['Af'] = affine_nib
                    result_filedict['Af'] = fn
    
                if run_d['registration']:
                    template_data = template_nib.get_fdata()
                    
                    fixed_image = template_data.astype(np.float32)[None, ...][None, ...]
                    fixed_image = lib_bx.min_max_norm(fixed_image)
                    #fixed_image = fixed_image/np.max(fixed_image)
                    
                    Af_data = affine_nib.get_fdata()
                    moving_image = Af_data.astype(np.float32)[None, ...][None, ...]
                    #moving_image = moving_image/np.max(moving_image)
                    
                    model_ff = lib_tool.get_model(omodel['reg'])
                    
                    output = lib_tool.predict(model_ff, [moving_image, fixed_image], GPU=args.gpu, mode='reg')
                    moved = np.squeeze(output[0])
                    warp = np.squeeze(output[1])
                    moved_nib = nib.Nifti1Image(moved,
                                             fixed_affine, template_nib.header)
                    warp_nib = nib.Nifti1Image(warp,
                                             fixed_affine, template_nib.header)
                    
                    if not run_d['vbm']:
                        fn = save_nib(moved_nib, ftemplate, 'reg')
                        result_filedict['reg'] = fn
                    result_dict['reg'] = moved_nib        
                    
                    #fn = save_nib(warp_nib, ftemplate, 'dense_warp')
                    result_dict['dense_warp'] = warp_nib
                    #result_filedict['dense_warp'] = fn 
                if run_d['fusemorph']:
                    model_affine_transform = lib_tool.get_model('mprage_affine_transform_v001_train.onnx')
                    model_transform = lib_tool.get_model('mprage_transform.onnx')
                    model_transform_bili = lib_tool.get_model('mprage_transform_bili.onnx')
                    
                    template_data = template_nib.get_fdata()
                    fixed_image = template_data.astype(np.float32)[None, ...][None, ...]
                    fixed_image = lib_bx.min_max_norm(fixed_image)

                                        
                    template_seg_nib = lib_tool.get_template_seg(run_d['template'])
                    template_seg_nib = reorder_img(template_seg_nib, resample='continuous')
                    template_seg_data = template_seg_nib.get_fdata()
                    fixed_seg_image = template_seg_data.astype(np.float32)[None, ...][None, ...]

                    
                    Af_data = affine_nib.get_fdata()
                    moving_image = Af_data.astype(np.float32)[None, ...][None, ...]
                    
                    moving_seg_nib = result_dict['aseg']
                    moving_seg_nib = reorder_img(moving_seg_nib, resample='nearest')
                    moving_seg_data = moving_seg_nib.get_fdata().astype(np.float32)
                    moving_seg_data, _ = lib_bx.pad_to_shape(moving_seg_data, (256, 256, 256))
                    moving_seg_data, _ = lib_bx.crop_image(moving_seg_data, target_shape=(256, 256, 256))
                    moving_seg = np.expand_dims(np.expand_dims(moving_seg_data, axis=0), axis=1)
                                
                    affine_matrix= np.expand_dims(affine_matrix, axis=0)
                    output = lib_tool.predict(model_affine_transform, [moving_seg, init_flow, affine_matrix], GPU=args.gpu, mode='affine_transform')
                    moving_seg = np.squeeze(output[0])
                    
                    moving_seg = lib_bx.remove_padding(moving_seg, pad_width)

                    
                    model_ff = lib_tool.get_model(omodel['reg'])
                    moving_image_current = moving_image
                    moving_seg_current = np.expand_dims(np.expand_dims(moving_seg, axis=0), axis=1)
                    warps = []
                    
                    for i in range(1, 4):
                        output = lib_tool.predict(model_ff, [moving_image_current, fixed_image], GPU=args.gpu, mode='reg')
                        moved = output[0]
                        warp = output[1]
                        
                        warps.append(warp)                    
                        output = lib_tool.predict(model_transform, [moving_seg_current, warp], GPU=args.gpu, mode='reg')
                        moved_seg = output[0]                        
                        moving_image_current = moved
                        moving_seg_current = moved_seg                    
                        # moving_nib = nib.Nifti1Image(np.squeeze(moved_seg), fixed_affine, template_nib.header)
                        # fn = save_nib(moving_nib, ftemplate, str(i))
                    
                    x_values = [0.9, 1.0]
                    y_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                    z_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                    param_combinations = list(product(x_values, y_values, z_values))
                    
                    best_dice = float('-inf')
                    best_warp = None                    
                    moving_seg = np.expand_dims(np.expand_dims(moving_seg, axis=0), axis=1)     
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = [executor.submit(lib_bx.FuseMorph_evaluate_params, params, warps, moving_seg, model_transform, fixed_seg_image, args.gpu) for params in param_combinations]
                        
                        for future in concurrent.futures.as_completed(futures):
                            x, y, z, dice_score, warp = future.result()
                            if dice_score > best_dice:
                                best_dice = dice_score
                                best_warp = warp
                            
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
                    
                    #fn = save_nib(warp_nib, ftemplate, 'dense_warp')
                    result_dict['dense_warp'] = warp_nib
                    #result_filedict['dense_warp'] = fn 
                    
            if run_d['vbm']:
                raw_GM_nib = reorder_img(result_dict['cgw'][1], resample='continuous')
                raw_GM = raw_GM_nib.get_fdata().astype(np.float32)
                raw_GM, _ = lib_bx.pad_to_shape(raw_GM, (256, 256, 256))
                raw_GM, _ = lib_bx.crop_image(raw_GM, target_shape=(256, 256, 256))
                raw_GM = np.expand_dims(np.expand_dims(raw_GM, axis=0), axis=1)
                
                model_transform = lib_tool.get_model('mprage_transform_bili.onnx')
                model_affine_transform = lib_tool.get_model('mprage_affine_transform_v001_train_bili.onnx')
                
                Affine_matrix= np.expand_dims(affine_matrix, axis=0)
                output = lib_tool.predict(model_affine_transform, [raw_GM, init_flow, Affine_matrix], GPU=None, mode='affine_transform')
                affined_GM = np.squeeze(output[0])
                affined_GM = lib_bx.remove_padding(affined_GM, pad_width)
                
                affined_GM = np.expand_dims(np.expand_dims(affined_GM, axis=0), axis=1)
                warp = np.expand_dims(warp, axis=0)
                
                output = lib_tool.predict(model_transform, [affined_GM, warp], GPU=None, mode='reg')
                reg_GM = np.squeeze(output[0])
                reg_GM_nib = nib.Nifti1Image(reg_GM, template_nib.affine, template_nib.header)
                fn = save_nib(reg_GM_nib, ftemplate, 'RegGM')                
                result_dict['Reg_GM'] = reg_GM_nib
                result_filedict['Reg_GM'] = fn
                
                warp = warp.transpose(0, 2, 3, 4, 1).squeeze()
                warp_Jacobian = lib_bx.jacobian_determinant(warp)
                Modulated_GM = reg_GM*warp_Jacobian
                Modulated_GM_nib = nib.Nifti1Image(Modulated_GM, template_nib.affine, template_nib.header)
                fn = save_nib(Modulated_GM_nib, ftemplate, 'ModulatedGM')                
                result_dict['Modulated_GM'] = Modulated_GM_nib
                result_filedict['Modulated_GM'] = fn
                
                fwhm_value = 8.0
                Smoothed_GM = lib_bx.apply_gaussian_smoothing(Modulated_GM, fwhm=fwhm_value)
                Smoothed_GM_nib = nib.Nifti1Image(Smoothed_GM, template_nib.affine, template_nib.header)
                fn = save_nib(Smoothed_GM_nib, ftemplate, 'SmoothedGM')
                result_dict['Smoothed_GM'] = Smoothed_GM_nib
                result_filedict['Smoothed_GM'] = fn
                
                
        print('Processing time: %d seconds' %  (time.time() - t))
        if len(input_file_list) == 1:
            result_all = result_dict
        else:
            result_all.append(result_filedict) #storing output filenames
    return result_all


if __name__ == "__main__":
    main()
    if platform.system() == 'Windows':
        os.system('pause')

