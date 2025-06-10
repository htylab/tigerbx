from os.path import basename, join, isdir, dirname, commonpath, relpath
import numpy as np
import glob
import nibabel as nib
import tigerbx
import pandas as pd
from tigerbx import lib_tool
from tigerbx import lib_reg
from nilearn.image import reorder_img
import sys
import os

# determine if application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(os.path.abspath(__file__))

def getdice(mask1, mask2):
    return 2*np.sum(mask1 & mask2)/(np.sum(mask1) + np.sum(mask2) + 1e-6)

def get_dice12(gt, pd, model_str):
    iigt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    if model_str == 'dgm':
        iipd = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    else:
        #aseg
        iipd = [10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54]
    d12 = []
    for ii in range(12):
        d12.append(getdice(gt==iigt[ii], pd==iipd[ii]))
    return np.array(d12)

#labels used by Synthmorph
def get_dice26(gt, pd):
    iigt = [2, 41, 3, 42, 4, 43, 7, 46, 8, 47, 10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54, 28, 60, 16, 24]
    d26 = []
    for ii in range(26):
        d26.append(getdice(gt==iigt[ii], pd==iigt[ii]))
    return np.array(d26)

def val(argstring, input_dir, output_dir=None, model=None, GPU=False, debug=False, template=None):
    #if model is not None:
    #    model = model.replace('\\','/')
    gpu_str = ''
    if GPU: gpu_str = 'g'

    if output_dir is None:
        output_dir = join(input_dir, 'tigerbx_validate_temp')
    
    print('Using output directory:', output_dir)

    if argstring == 'bet_synstrip':

        ffs = sorted(glob.glob(join(input_dir, '*', 'image.nii.gz')))
        if debug: ffs = ffs[:5]
        print(f'Total files: {len(ffs)}')
        tt_list = []
        dsc_list = []
        f_list = []
        cat_list = []
        count = 0
        for f in ffs:
            count += 1
            tt = basename(dirname(f)).split('_')[1]
            cat = '_'.join(basename(dirname(f)).split('_')[:2])
            f_list.append(f)
            tt_list.append(tt)
            cat_list.append(cat)
            result = tigerbx.run(gpu_str + 'm', f, output_dir, model=model)
            mask_pred = result['tbetmask'].get_fdata()
            
            mask_gt = nib.load(f.replace('image.nii.gz', 'mask.nii.gz')).get_fdata()
            mask1 = (mask_pred > 0).flatten()
            mask2 = (mask_gt > 0).flatten()

            dsc = getdice(mask1, mask2)
            dsc_list.append(dsc)

            print(count, len(ffs), f, f'Dice: {dsc:.3f}, mean Dice: {np.mean(dsc_list):.3f}')

        # Create a DataFrame
        result = {
            'Filename': f_list,
            'type': tt_list,
            'category': cat_list,
            'DICE': dsc_list
        }

        df = pd.DataFrame(result)
        df.to_csv(join(output_dir, 'val_bet_synstrip.csv'), index=False)

        average_dice_per_category = df.groupby('category')['DICE'].mean()

        # Display results
        print(average_dice_per_category)
        metric = df['DICE'].mean()
        print('mean Dice of all data:', df['DICE'].mean())

        return df, metric
    
    elif argstring == 'bet_NFBS':

        ffs = sorted(glob.glob(join(input_dir, '*', '*T1w.nii.gz')))
        if debug: ffs = ffs[:5]
        print(f'Total files: {len(ffs)}')
        dsc_list = []
        f_list = []
        count = 0
        for f in ffs:
            count += 1
            f_list.append(f)
            result = tigerbx.run(gpu_str + 'm', f, output_dir, model=model)

            mask_pred = result['tbetmask'].get_fdata()
            
            mask_gt = nib.load(f.replace('T1w.nii.gz', 'T1w_brainmask.nii.gz')).get_fdata()
            mask1 = (mask_pred > 0).flatten()
            mask2 = (mask_gt > 0).flatten()

            dsc = getdice(mask1, mask2)
            dsc_list.append(dsc)

            print(count, len(ffs), f, f'Dice: {dsc:.3f}, mean Dice: {np.mean(dsc_list):.3f}')

        # Create a DataFrame
        result = {
            'Filename': f_list,
            'DICE': dsc_list
        }

        df = pd.DataFrame(result)
        df.to_csv(join(output_dir, 'val_bet_NFBS.csv'), index=False)

        metric = df['DICE'].mean()
        print('mean Dice of all data:', df['DICE'].mean())

        return df, metric
    

    elif argstring == 'aseg_123' or argstring== 'dgm_123' or argstring== 'syn_123':

        if argstring == 'aseg_123':
            model_str = 'aseg'
            tigerrun_option = 'a'
        elif argstring== 'dgm_123':
            model_str = 'dgm'
            tigerrun_option = 'd'
        else:
            model_str = 'syn'
            tigerrun_option = 'S'

        ffs = sorted(glob.glob(join(input_dir, 'raw123', '*.nii.gz')))
        if debug: ffs = ffs[:5]

        print(f'Total files: {len(ffs)}')
        dsc_list = []
        f_list = []
        count = 0
        for f in ffs:
            count += 1
            f_list.append(f)
            result = tigerbx.run(gpu_str + tigerrun_option, f,
                                      output_dir, model=model)

            mask_pred = result[ model_str].get_fdata().astype(int)
            mask_gt = nib.load(f.replace('raw123', 'label123')).get_fdata().astype(int)
            dice12 = get_dice12(mask_gt, mask_pred, model_str)
            dsc_list.append(dice12)

            print(count, len(ffs), f, f'Dice: {np.mean(dice12):.3f}')
                # Create a DataFrame
        result = {
            'Filename': f_list,
            'DICE': dsc_list
        }

        column_names = ['Left-Thalamus', 'Right-Thalamus',
                        'Left-Caudate', 'Right-Caudate',
                        'Left-Putamen', 'Right-Putamen',
                        'Left-Pallidum', 'Right-Pallidum',
                        'Left-Hippocampus', 'Right-Hippocampus',
                        'Left-Amygdala', 'Right-Amygdala' ]

        # Convert dsc_list to a DataFrame
        df = pd.DataFrame(dsc_list, columns=column_names)

        # Add the IDs as the first column
        df.insert(0, 'Filename', f_list)
        df.to_csv(join(output_dir, f'val_{model_str}_123.csv'), index=False)
        print('mean Dice of all data:', np.mean(np.array(dsc_list).flatten()))
        mean_per_column = df.mean(numeric_only=True)
        print(mean_per_column)


        return df, mean_per_column
    elif argstring == 'reg_60':
        ffs = sorted(glob.glob(join(input_dir, 'raw60', '*.nii.gz')))
        if debug: ffs = ffs[:5]
        print(f'Total files: {len(ffs)}')        

        dsc_list = []
        f_list = []
        count = 0
        for f in ffs:
            count += 1
            f_list.append(f)
            result = tigerbx.reg(gpu_str + 'r', f, output_dir, model=model, template=template)
                
            model_transform = lib_tool.get_model('mprage_transform_v002_near.onnx')
            model_affine_transform = lib_tool.get_model('mprage_affinetransform_v002_near.onnx')
            
            template_nib = lib_reg.get_template(template)
            template_nib = reorder_img(template_nib, resample='continuous')
            template_data = template_nib.get_fdata()
            template_data, pad_width = lib_reg.pad_to_shape(template_data, (256, 256, 256))
            
            moving_seg_nib = nib.load(f.replace('raw60', 'label60'))
            moving_seg_nib = reorder_img(moving_seg_nib, resample='nearest')
            moving_seg_data = moving_seg_nib.get_fdata().astype(np.float32)
            moving_seg_data, _ = lib_reg.pad_to_shape(moving_seg_data, (256, 256, 256))
            moving_seg_data, _ = lib_reg.crop_image(moving_seg_data, target_shape=(256, 256, 256))
            moving_seg = np.expand_dims(np.expand_dims(moving_seg_data, axis=0), axis=1)

            init_flow = result['init_flow'].get_fdata().astype(np.float32)
            Affine_matrix = result['Affine_matrix'].astype(np.float32)            
            Affine_matrix= np.expand_dims(Affine_matrix, axis=0)
            
            output = lib_tool.predict(model_affine_transform, [moving_seg, init_flow, Affine_matrix], GPU=None, mode='affine_transform')
            moved_seg = np.squeeze(output[0])
            
            moved_seg = lib_reg.remove_padding(moved_seg, pad_width)
  
            moved_seg = np.expand_dims(np.expand_dims(moved_seg, axis=0), axis=1)
            warp = result['dense_warp'].get_fdata().astype(np.float32)
            warp = np.expand_dims(warp, axis=0)
            output = lib_tool.predict(model_transform, [moved_seg, warp], GPU=None, mode='reg')
            moved_seg = np.squeeze(output[0])
            moved_seg_nib = nib.Nifti1Image(moved_seg,
                                      template_nib.affine, template_nib.header)
    
            mask_pred = reorder_img(moved_seg_nib, resample='nearest').get_fdata().astype(int)
            template_seg = lib_reg.get_template_seg(template)
            mask_gt = reorder_img(template_seg, resample='nearest').get_fdata().astype(int)
  
            dice26 = get_dice26(mask_gt, mask_pred)
            dsc_list.append(dice26)

            print(count, len(ffs), f, f'Dice: {np.mean(dice26):.3f}')
                # Create a DataFrame
        result = {
            'Filename': f_list,
            'DICE': dsc_list
        }

        column_names = ['Left-Cerebral WM', 'Right-Cerebral WM', 'Left-Cerebral Cortex', 'Right-Cerebral Cortex', 'Left-Lateral Ventricle', 
                        'Right-Lateral Ventricle', 'Left-Cerebellum WM', 'Right-Cerebellum WM', 'Left-Cerebellum Cortex', 'Right-Cerebellum Cortex', 
                        'Left-Thalamus', 'Right-Thalamus', 'Left-Caudate', 'Right-Caudate', 'Left-Putamen', 'Right-Putamen',
                        'Left-Pallidum', 'Right-Pallidum', 'Left-Hippocampus', 'Right-Hippocampus', 'Left-Amygdala', 'Right-Amygdala', 
                        'Left-VentralDC', 'Right-VentralDC', 'Brain Stem', 'CSF']

        # Convert dsc_list to a DataFrame
        df = pd.DataFrame(dsc_list, columns=column_names)

        # Add the IDs as the first column
        df.insert(0, 'Filename', f_list)
        df.to_csv(join(output_dir, f'val_reg_60.csv'), index=False)
        print('mean Dice of all data:', np.mean(np.array(dsc_list).flatten()))
        mean_per_column = df.mean(numeric_only=True)
        print(mean_per_column)
        
        return df, mean_per_column



