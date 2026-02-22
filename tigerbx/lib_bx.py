from os.path import basename
import numpy as np
import nibabel as nib
from scipy.special import softmax
from nilearn.image import reorder_img, resample_img

from tigerbx import lib_tool

label_all = dict()
label_all['aseg43'] = (2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,24,26,28,30,31,41,42,43,
                44,46,47,49,50,51,52,53,54,58,60,62,63,77,85,251,252,253,254,255)
label_all['dkt'] = ( 1002, 1003,
               1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015,
               1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026,
               1027, 1028, 1029, 1030, 1031, 1034, 1035, 2002, 2003, 2005, 2006,
               2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
               2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028,
               2029, 2030, 2031, 2034, 2035)

label_all['wmp'] = (  251,  252,  253,  254,  255, 3001, 3002, 3003, 3005, 3006, 3007,
                     3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018,
                     3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3027, 3028, 3029,
                     3030, 3031, 3032, 3033, 3034, 3035, 4001, 4002, 4003, 4005, 4006,
                     4007, 4008, 4009, 4010, 4011, 4012, 4013, 4014, 4015, 4016, 4017,
                     4018, 4019, 4020, 4021, 4022, 4023, 4024, 4025, 4026, 4027, 4028,
                     4029, 4030, 4031, 4032, 4033, 4034, 4035)

label_all['synthseg'] = ( 2,  3,  4,  5,  7,  8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24,
                         26, 28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60)
#nib.Nifti1Header.quaternion_threshold = -100


def get_mode(model_ff):
    seg_mode, version, model_str = basename(model_ff).split('_')[1:4]  # aseg43, bet

    #print(seg_mode, version , model_str)

    return seg_mode, version, model_str

def getLarea(input_mask):
    from scipy import ndimage
    labeled_mask, cc_num = ndimage.label(input_mask)
    if cc_num > 0:
        mask = (labeled_mask == (np.bincount(
            labeled_mask.flat)[1:].argmax() + 1))
    else:
        mask = input_mask
    return mask

def get_affine(mat_size):

    target_shape = np.array((mat_size, mat_size, mat_size))
    new_resolution = [256/mat_size, ]*3
    new_affine = np.zeros((4, 4))
    new_affine[:3, :3] = np.diag(new_resolution)
    # putting point 0,0,0 in the middle of the new volume - this could be refined in the future
    new_affine[:3, 3] = target_shape * new_resolution/2.*-1
    new_affine[3, 3] = 1.
    #print(model_ff, target_shape)
    #print(new_affine, target_shape)
    return new_affine, target_shape


def get_mat_size(model_ff):
    import re
    tmp = re.compile(r'r\d{2,4}.onnx').findall(basename(model_ff))
    mat_size = -1
    if len(tmp) > 0:
        mat_size = int(tmp[0].replace('.onnx', '')[1:])
    #print(model_ff, mat_size)
    return mat_size

def logit_to_prob(logits, seg_mode):
    label_num = dict()
    label_num['bet'] = 2
    label_num['aseg43'] = 44
    label_num['dkt'] = 63
    label_num['dgm12'] = 13
    label_num['wmp'] = 74
    label_num['seg3'] = 4
    label_num['wmh'] = 2
    label_num['tumor'] = 2
    label_num['synthseg'] = 33
    #so far we only use sigmoid in tBET
    if label_num[seg_mode] > logits.shape[0]:
        #sigmoid
        th = 0.5
        from scipy.special import expit
        prob = expit(logits)
    else:
        #softmax mode
        #print(logits.shape)
        prob = softmax(logits, axis=0)
    return prob

def run(model_ff_list, input_nib, GPU, patch=False):

    if not isinstance(model_ff_list, list):
        model_ff_list = [model_ff_list]


    seg_mode, _ , model_str = get_mode(model_ff_list[0])

    #data = input_nib.get_fdata()
    data = lib_tool.read_nib(input_nib)

    image = data[None, ...][None, ...]
    if seg_mode == 'synthseg':
        rng = np.max(image) - np.min(image)
        if rng > 0:
            image = (image - np.min(image)) / rng
    else:
        mx = np.max(image)
        if mx > 0:
            image = image / mx

    prob = 0
    count = 0
    for model_ff in model_ff_list:
        count += 1
        if patch:
            logits = lib_tool.predict(model_ff, image, GPU, mode='patch')[0, ...]
        else:
            logits = lib_tool.predict(model_ff, image, GPU)[0, ...]
        prob += logit_to_prob(logits, seg_mode)
    prob = prob/count # average the prob

    if seg_mode =='bet': #sigmoid 1 channel
        th = 0.5
        mask_pred = np.ones(prob[0, ...].shape)
        mask_pred[prob[0, ...] < th] = 0
        mask_pred = getLarea(mask_pred)
    else:
        mask_pred = np.argmax(prob, axis=0)


    if seg_mode in ['aseg43', 'dkt', 'wmp', 'synthseg']:
        labels = label_all[seg_mode]
        lut = np.zeros(len(labels) + 2, dtype=np.int32)
        for ii, lbl in enumerate(labels):
            lut[ii + 1] = lbl
        mask_pred = lut[mask_pred]



    mask_pred = mask_pred.astype(int)


    output_nib = nib.Nifti1Image(
        mask_pred, input_nib.affine, input_nib.header)

    return output_nib, prob



def read_file(model_ff, input_file):

    mat_size = get_mat_size(model_ff)
    input_nib = nib.load(input_file)
    zoom = input_nib.header.get_zooms()

    if mat_size == -1 or mat_size == 111:

        if max(zoom) > 1.1 or min(zoom) < 0.9 or mat_size == 111:

            vol_nib = lib_tool.resample_voxel(input_nib, (1, 1, 1), interpolation='continuous')
            vol_nib = reorder_img(vol_nib, resample='continuous')
        else:
            vol_nib = reorder_img(input_nib, resample='continuous')
    else:
        affine, shape = get_affine(mat_size)
        vol_nib = resample_img(input_nib,
                           target_affine=affine, target_shape=shape)

    return vol_nib
