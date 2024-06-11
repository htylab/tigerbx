### TigerBx: Tissue Mask Generation for Brain Extration

### Validation pipeline (after v0.1.16)

    import tigerbx
    import time
    t = time.time()
    # tigerbx.val(argstring, input_dir, output_dir=None, model=None, GPU=False, debug=False)
    df, metric1 = tigerbx.val('bet_NFBS', 'NFBS_Dataset','temp', GPU=True)
    df, metric2 = tigerbx.val('bet_synstrip', 'synthstrip_data_v1.4','temp', GPU=True)
    df, metric3 = tigerbx.val('aseg_123', 'aseg', 'temp', GPU=True)
    df, metric4 = tigerbx.val('dgm_123', 'aseg', 'temp', GPU=True)
    df, metric5 = tigerbx.val('syn_123', 'aseg', 'temp', GPU=True)
    df, metric6 = tigerbx.val('reg_60', 'aseg, 'temp', GPU=True, template='Template_T1_tbet.nii.gz')

    
    print('bet_NFBS', metric1)
    print('bet_synstrip', metric2)
    print('aseg_123', metric3)
    print('dgm_123', metric4)
    print('syn_123', metric5)
    print('reg_60', metric6)
    print('Time', time.time() - t)

### Validation Datasets
1. Skull Stripping: [NFBS_Dataset](http://preprocessed-connectomes-project.org/NFB_skullstripped)
2. Skull Stripping: [Synstrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip)
3. Deep gray matter: [MindBoggle-101](https://mindboggle.info/)
4. Deep gray matter: [CANDI](https://www.nitrc.org/projects/candi_share/)
5. Registration: [ABIDE](https://fcon_1000.projects.nitrc.org/indi/abide/)


### Internal Validation Sets:
1. aseg: A combination of MindBoggle-101 and CANDI (N = 123) with DGM labels

### Validation Results of Verion 0.1.16
#### Deep Gray matter: Dice coefficients

    | Structure  | L/R | aseg  | dgm  | syn  | L/R | aseg  | dgm  | syn  |
    |------------|-----|-------|------|------|-----|-------|------|------|
    | Thalamus   | L   | 0.879 | 0.898| 0.890| R   | 0.889 | 0.902| 0.884|
    | Caudate    | L   | 0.875 | 0.874| 0.850| R   | 0.875 | 0.872| 0.845|
    | Putamen    | L   | 0.873 | 0.885| 0.847| R   | 0.862 | 0.880| 0.829|
    | Pallidum   | L   | 0.827 | 0.827| 0.828| R   | 0.814 | 0.815| 0.794|
    | Hippocampus| L   | 0.808 | 0.828| 0.789| R   | 0.810 | 0.831| 0.779|
    | Amygdala   | L   | 0.737 | 0.764| 0.716| R   | 0.727 | 0.750| 0.711|
    | Mean       | L   | 0.833 | 0.846| 0.820| R   | 0.829 | 0.841| 0.807|
#### Registration:
    mean dice: 0.808

#### Skull Stripping
    bet_NFBS: 0.973
    bet_synstrip: 0.971





