import sys
import os
from os.path import basename, join, isdir
import argparse
import time
import numpy as np

import glob
import platform
import nibabel as nib

from tigerbx import lib_tool
from tigerbx import lib_bx

from nilearn.image import resample_to_img, reorder_img

def produce_mask(model, f, GPU=False, brainmask_nib=None, QC=False):

    model_ff = lib_tool.get_model(model)
    input_nib = nib.load(f)
    input_nib_resp = lib_bx.read_file(model_ff, f)
    mask_nib_resp, prob_resp = lib_bx.run(
        model_ff, input_nib_resp,  GPU=GPU)

    mask_nib = resample_to_img(
        mask_nib_resp, input_nib, interpolation="nearest")

    if brainmask_nib is None:

        output = lib_bx.read_nib(mask_nib)
    else:
        output = lib_bx.read_nib(mask_nib) * lib_bx.read_nib(brainmask_nib)

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
        qc_score = int(qc_score * 1000)
        return output_nib, qc_score
    else:

        return output_nib

def save_nib(data_nib, ftemplate, postfix):
    output_file = ftemplate.replace('@@@@', postfix)
    nib.save(data_nib, output_file)
    print('Writing output file: ', output_file)
    return output_file

def get_template(f, output_dir, get_z):
    f_output_dir = output_dir

    if f_output_dir is None:
        f_output_dir = os.path.dirname(os.path.abspath(f))
    else:
        os.makedirs(f_output_dir, exist_ok=True)

    ftemplate = basename(f).replace('.nii', f'_@@@@.nii')
    if get_z and '.gz' not in ftemplate:
        ftemplate += '.gz'
    ftemplate = join(f_output_dir, ftemplate)

    return ftemplate, f_output_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input',  type=str, nargs='+', help='Path to the input image, can be a folder for the specific format(nii.gz)')
    parser.add_argument('-o', '--output', default=None, help='File path for output segmentation, default: the directory of input files')
    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPU')
    parser.add_argument('-m', '--betmask', action='store_true', help='Producing bet mask')
    parser.add_argument('-a', '--aseg', action='store_true', help='Producing aseg mask')
    parser.add_argument('-b', '--bet', action='store_true', help='Producing bet images')
    parser.add_argument('-B', '--bam', action='store_true', help='Producing brain age mapping')
    parser.add_argument('-c', '--ct', action='store_true',  help='Producing cortical thickness map')
    parser.add_argument('-C', '--cgw', action='store_true', help='Producing FSL-style PVE segmentation')
    parser.add_argument('-d', '--dgm', action='store_true', help='Producing deepgm mask')    
    parser.add_argument('-k', '--dkt', action='store_true', help='Producing dkt mask')

    parser.add_argument('-S', '--syn', action='store_true', help='Producing SynthSeg mask (working in progress)')

    parser.add_argument('-w', '--wmp', action='store_true', help='Producing white matter parcellation')
    parser.add_argument('-W', '--wmh', action='store_true', help='Producing WMH lesion mask')

    parser.add_argument('-t', '--tumor', action='store_true',
                        help='Producing tumor mask')
    parser.add_argument('-q', '--qc', action='store_true',
                        help='Save QC score. Pay attention to the results with QC scores less than 50.')
    parser.add_argument('-z', '--gz', action='store_true', help='Force storing nii.gz format')

    parser.add_argument('--model', default=None, type=str, help='Specifies the modelname')
    #parser.add_argument('--report',default='True',type = strtobool, help='Produce additional reports')
    args = parser.parse_args()
    run_args(args)


def run(argstring, input, output=None, model=None):

    from argparse import Namespace
    args = Namespace()

    args.betmask = 'm' in argstring
    args.aseg = 'a' in argstring
    args.bet = 'b' in argstring
    args.bam = 'B' in argstring
    args.ct = 'c' in argstring
    args.cgw = 'C' in argstring
    args.dgm = 'd' in argstring
    args.gpu = 'g' in argstring
    args.dkt = 'k' in argstring
    args.wmh = 'W' in argstring    
    args.wmp = 'w' in argstring
    args.seg3 = 's' in argstring
    args.syn = 'S' in argstring
    args.tumor = 't' in argstring
    args.qc = 'q' in argstring
    args.gz = 'z' in argstring

    if not isinstance(input, list):
        input = [input]
    args.input = input
    args.output = output
    args.model = model
    return run_args(args)   


def run_args(args):

    run = vars(args) #store all arg in dict
    if True not in [run['betmask'], run['aseg'], run['bet'], run['dgm'],
                    run['dkt'], run['ct'], run['wmp'], run['qc'], 
                    run['wmh'], run['bam'], run['tumor'], run['cgw'], run['syn']]:
        run['bet'] = True
        # Producing extracted brain by default 

    input_file_list = args.input
    if os.path.isdir(args.input[0]):
        input_file_list = glob.glob(join(args.input[0], '*.nii'))
        input_file_list += glob.glob(join(args.input[0], '*.nii.gz'))

    elif '*' in args.input[0]:
        input_file_list = glob.glob(args.input[0])

    output_dir = args.output
    omodel = dict()
    omodel['bet'] = 'mprage_bet_v004_anisofocal.onnx'
    omodel['aseg'] = 'mprage_aseg43_v005_crop.onnx'
    omodel['dkt'] = 'mprage_dkt_v002_train.onnx'
    omodel['ct'] = 'mprage_mix_ct.onnx'
    omodel['dgm'] = 'mprage_dgm12_v002_mix6.onnx'
    omodel['wmp'] = 'mprage_wmp_v003_14k8.onnx'
    omodel['wmh'] = 'mprage_wmh_v001_T1r111.onnx'
    omodel['bam'] = 'mprage_bam_v002_betr111.onnx'
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


    print('Total nii files:', len(input_file_list))
    count = 0
    result_all = []
    result_filedict = dict()
    for f in input_file_list:
        count += 1
        result_dict = dict()
        result_filedict = dict()

        print(f'{count} Processing :', os.path.basename(f))
        t = time.time()

        ftemplate, f_output_dir = get_template(f, output_dir, args.gz)
        
        tbetmask_nib, qc_score = produce_mask(omodel['bet'], f, GPU=args.gpu, QC=True)
        input_nib = nib.load(f)
        tbet_nib = lib_bx.read_nib(input_nib) * lib_bx.read_nib(tbetmask_nib)
        tbet_nib = tbet_nib.astype(input_nib.dataobj.dtype)
        tbet_nib = nib.Nifti1Image(tbet_nib, input_nib.affine,
                        input_nib.header)
        
        print('QC score:', qc_score)

        result_dict['QC'] = qc_score
        result_filedict['QC'] = qc_score
        if qc_score < 50:
            print('Pay attention to the result with QC < 50. ')
        if run['qc'] or qc_score < 50:
            qcfile = basename(f).replace('.nii','').replace('.gz', '') + f'-qc-{qc_score}.log'
            qcfile = join(f_output_dir, qcfile)
            with open(qcfile, 'a') as the_file:
                the_file.write(f'QC: {qc_score} \n')

        if run['betmask']:
            fn = save_nib(tbetmask_nib, ftemplate, 'tbetmask')
            result_dict['tbetmask'] = tbetmask_nib
            result_filedict['tbetmask'] = fn

        if run['bet']:
            fn = save_nib(tbet_nib, ftemplate, 'tbet')
            result_dict['tbet'] = tbet_nib
            result_filedict['tbet'] = fn
        
        for seg_str in ['aseg', 'dgm', 'dkt', 'wmp', 'wmh', 'tumor', 'syn']:
            if run[seg_str]:
                if qc_score < 50:
                    result_nib = produce_mask(omodel[seg_str], f, GPU=args.gpu,
                                        brainmask_nib=tbetmask_nib)
                else:
                    result_nib = produce_mask(omodel[seg_str], f, GPU=args.gpu)
                fn = save_nib(result_nib, ftemplate, seg_str)
                result_dict[seg_str] = result_nib
                result_filedict[seg_str] = fn
        if run['bam']:
            model_ff = lib_tool.get_model(omodel['bam'])
            input_nib = nib.load(f)
            tbet_nib111 = lib_bx.resample_voxel(tbet_nib, (1, 1, 1),interpolation='continuous')
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

        if run['cgw']: # FSL style segmentation of CSF, GM, WM
            model_ff = lib_tool.get_model(omodel['cgw'])
            input_nib = nib.load(f)
            normalize_factor = np.max(input_nib.get_fdata())
            tbet_nib111 = lib_bx.resample_voxel(tbet_nib, (1, 1, 1),interpolation='linear')
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
                result_dict['cgw'].append(pve_nib)
                result_filedict['cgw'].append(fn)

        

        if run['ct']:

            model_ff = lib_tool.get_model(omodel['ct'])
            input_nib = nib.load(f)
            tbet_nib111 = lib_bx.resample_voxel(tbet_nib, (1, 1, 1),interpolation='continuous')
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

