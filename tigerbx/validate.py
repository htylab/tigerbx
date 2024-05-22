from os.path import basename, join, isdir, dirname, commonpath, relpath
import numpy as np
import glob
import nibabel as nib
import tigerbx
import pandas as pd


def getdice(mask1, mask2):
    return 2*np.sum(mask1 & mask2)/(np.sum(mask1) + np.sum(mask2) + 1e-6)

def val(argstring, input_dir, output_dir=None, model=None):

    if output_dir is None:
        output_dir = join(input_dir, 'temp')
    if argstring == 'bet_synstripv1.4':

        #/work/tyhuang0908/Dataset/synthstrip_data_v1.4     

        ffs = sorted(glob.glob(join(input_dir, '*', 'image.nii.gz')))
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
            if model is None:
                result = tigerbx.run('m', f, output_dir, model="{'bet':'%s'}" % model)
            else:
                result = tigerbx.run('m', f, output_dir)
            mask_pred = result['tbetmask'].get_fdata()
            
            mask_gt = nib.load(f.replace('image.nii.gz', 'mask.nii.gz')).get_fdata()
            mask1 = (mask_pred > 0).flatten()
            mask2 = (mask_gt > 0).flatten()

            dsc = getdice(mask1, mask2)
            dsc_list.append(dsc)

            print(count, len(ffs), f, dsc, np.mean(dsc_list))

            # Create a DataFrame
            result = {
                'Filename': f_list,
                'type': tt_list,
                'category': cat_list,
                'DICE': dsc_list
            }

            # Write the data to a CSV file
            with open(join(output_dir, 'val_bet_synstripv1.4.csv', 'w', encoding='utf-8') as file:
                # Write the header
                file.write('Filename,type,category,DICE\n')
                # Write the data rows
                for i in range(len(f_list)):
                    file.write(f'{f_list[i]},{tt_list[i]},{cat_list[i]},{dsc_list[i]}\n')

            df = pd.DataFrame(result)
            df.to_csv(join(output_dir, 'val_bet_synstripv1.4.csv'), index=False)

            average_dice_per_category = data.groupby('category')['DICE'].mean()

            # 顯示結果
            print(average_dice_per_category)
            print('mean Dice of all data:', data['DICE'].mean())

            return df

