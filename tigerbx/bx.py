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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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


def setup_parser(parser):
    #parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, nargs='+', help='Path to the input image(s); can be a folder containing images in the specific format (nii.gz)')
    parser.add_argument('-o', '--output', default=None, help='File path for output segmentation (default: the directory of input files)')
    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPU')
    parser.add_argument('-m', '--betmask', action='store_true', help='Producing BET mask')
    parser.add_argument('-a', '--aseg', action='store_true', help='Producing ASEG mask')
    parser.add_argument('-b', '--bet', action='store_true', help='Producing BET images')
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
    parser.add_argument('-p', '--patch', action='store_true', help='patch inference')
    parser.add_argument('--silent', action='store_true', help='Silent mode')
    parser.add_argument('--model', default=None, type=str, help='Specifying the model name')
    parser.add_argument('--clean_onnx', action='store_true', help='Clean onnx models')
    #args = parser.parse_args()
    #run_args(args)


def run(argstring, input=None, output=None, model=None, silent=False):

    from argparse import Namespace
    args = Namespace()
    if not isinstance(input, list):
        input = [input]
    args.input = input
    args.output = output
    args.model = model
    args.clean_onnx = 'clean_onnx' in argstring
    args.gpu = 'g' in argstring

    args.silent = silent

    if args.clean_onnx:
        argstring = ''
    args.betmask = 'm' in argstring
    args.aseg = 'a' in argstring
    args.bet = 'b' in argstring
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
    args.patch = 'p' in argstring
    return run_args(args)   


def run_args(args):

    run_d = vars(args) #store all arg in dict

    if run_d.get('silent', 0):
        printer = lambda *args, **kwargs: None
    else:
        printer = print

    if True not in [run_d['betmask'], run_d['aseg'], run_d['bet'], run_d['dgm'],
                    run_d['dkt'], run_d['ct'], run_d['wmp'], run_d['qc'], 
                    run_d['wmh'], run_d['tumor'], run_d['cgw'], 
                    run_d['syn'], run_d['patch']]:
        run_d['bet'] = True
        # Producing extracted brain by default
        
    if run_d['clean_onnx']:
        lib_tool.clean_onnx()
        printer('Exiting...')
        return 1


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
    omodel['dkt'] = 'mprage_dkt_v002_train.onnx'
    omodel['ct'] = 'mprage_mix_ct.onnx'
    omodel['dgm'] = 'mprage_dgm12_v002_mix6.onnx'
    omodel['wmp'] = 'mprage_wmp_v003_14k8.onnx'
    omodel['wmh'] = 'mprage_wmh_v002_betr111.onnx'
    omodel['tumor'] = 'mprage_tumor_v001_r111.onnx'
    omodel['cgw'] = 'mprage_cgw_v001_r111.onnx'
    omodel['syn'] = 'mprage_synthseg_v003_r111.onnx'

 
    # if you want to use other models
    if isinstance(args.model, dict):
        for mm in args.model.keys():
            omodel[mm] = args.model[mm]
    elif isinstance(args.model, str):
        import ast
        model_dict = ast.literal_eval(args.model)
        for mm in model_dict.keys():
            omodel[mm] = model_dict[mm]


    printer('Total nii files:', len(input_file_list))

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

        printer(f'{count} Processing :', os.path.basename(f))
        t = time.time()

        ftemplate, f_output_dir = get_template(f, output_dir, args.gz, common_folder)       

        tbetmask_nib, qc_score = produce_mask(omodel['bet'], f, GPU=args.gpu, QC=True)
        input_nib = nib.load(f)
        tbet_nib = lib_bx.read_nib(input_nib) * lib_bx.read_nib(tbetmask_nib)

        tbet_nib = nib.Nifti1Image(tbet_nib, input_nib.affine, input_nib.header)
        tbet_nib111 = lib_bx.resample_voxel(tbet_nib, (1, 1, 1),interpolation='continuous')
        tbet_nib111 = reorder_img(tbet_nib111, resample='continuous')

        zoom = tbet_nib.header.get_zooms() 

        if max(zoom) > 1.1 or min(zoom) < 0.9:
            tbet_seg = tbet_nib111
        else:
            tbet_seg = reorder_img(tbet_nib, resample='continuous')
        
        printer('QC score:', qc_score)

        result_dict['QC'] = qc_score
        result_filedict['QC'] = qc_score
        if qc_score < 50:
            printer('Pay attention to the result with QC < 50. ')
        if run_d['qc'] or qc_score < 50:
            qcfile = ftemplate.replace('.nii','').replace('.gz', '')
            qcfile = qcfile.replace('@@@@', f'qc-{qc_score}.log')
            with open(qcfile, 'a') as the_file:
                the_file.write(f'QC: {qc_score} \n')
            printer('Writing output file: ', qcfile)

        if run_d['betmask']:
            fn = save_nib(tbetmask_nib, ftemplate, 'tbetmask')
            printer('Writing output file: ', fn)
            result_dict['tbetmask'] = tbetmask_nib
            result_filedict['tbetmask'] = fn

        if run_d['bet']:

            imabet = tbet_nib.get_fdata()
            if lib_tool.check_dtype(imabet, input_nib.dataobj.dtype):
                imabet = imabet.astype(input_nib.dataobj.dtype)
            
            fn = save_nib(tbet_nib, ftemplate, 'tbet')
            printer('Writing output file: ', fn)
            result_dict['tbet'] = tbet_nib
            result_filedict['tbet'] = fn
        
        for seg_str in ['aseg', 'dgm', 'dkt', 'wmp', 'wmh', 'tumor', 'syn']:
            if run_d[seg_str]:
                result_nib = produce_mask(omodel[seg_str], f, GPU=args.gpu,
                                         brainmask_nib=tbetmask_nib, tbet111=tbet_seg, patch=run_d['patch'])
                
                fn = save_nib(result_nib, ftemplate, seg_str)
                printer('Writing output file: ', fn)
                result_filedict[seg_str] = fn
                result_dict[seg_str] = result_nib

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

                fn = save_nib(pve_nib, ftemplate, f'cgw_pve{kk-1}')
                printer('Writing output file: ', fn)
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
            printer('Writing output file: ', fn)
            result_dict['ct'] = ct_nib
            result_filedict['ct'] = fn
            
               
        printer('Processing time: %d seconds' %  (time.time() - t))
        if len(input_file_list) == 1:
            result_all = result_dict
        else:
            result_all.append(result_filedict) #storing output filenames
    return result_all


