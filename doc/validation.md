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
#### Registration: Dice coefficients

    | Structure                | Dice    |
    |--------------------------|---------|
    | Left-Cerebral WM         | 0.824747|
    | Right-Cerebral WM        | 0.826378|
    | Left-Cerebral Cortex     | 0.739047|
    | Right-Cerebral Cortex    | 0.741221|
    | Left-Lateral Ventricle   | 0.804508|
    | Right-Lateral Ventricle  | 0.803698|
    | Left-Cerebellum WM       | 0.838658|
    | Right-Cerebellum WM      | 0.849894|
    | Left-Cerebellum Cortex   | 0.902093|
    | Right-Cerebellum Cortex  | 0.909830|
    | Left-Thalamus            | 0.879016|
    | Right-Thalamus           | 0.878512|
    | Left-Caudate             | 0.822172|
    | Right-Caudate            | 0.834346|
    | Left-Putamen             | 0.875642|
    | Right-Putamen            | 0.883873|
    | Left-Pallidum            | 0.834932|
    | Right-Pallidum           | 0.833917|
    | Left-Hippocampus         | 0.835422|
    | Right-Hippocampus        | 0.848588|
    | Left-Amygdala            | 0.821882|
    | Right-Amygdala           | 0.834197|
    | Left-VentralDC           | 0.840949|
    | Right-VentralDC          | 0.842651|
    | Brain Stem               | 0.914912|
    | CSF                      | 0.661478|
#### Skull Stripping
    bet_NFBS: 0.969
    bet_synstrip: 0.971





