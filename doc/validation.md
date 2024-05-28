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
    df, metric6 = tigerbx.val('reg_123', input_dir, output_dir, GPU=True)

    
    print('bet_NFBS', metric1)
    print('bet_synstrip', metric2)
    print('aseg_123', metric3)
    print('dgm_123', metric4)
    print('syn_123', metric5)
    print('reg_50', metric6)
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
    | Left-Cerebral WM         | 0.857104|
    | Right-Cerebral WM        | 0.858439|
    | Left-Cerebral Cortex     | 0.728526|
    | Right-Cerebral Cortex    | 0.736365|
    | Left-Lateral Ventricle   | 0.803578|
    | Right-Lateral Ventricle  | 0.793350|
    | Left-Cerebellum WM       | 0.807485|
    | Right-Cerebellum WM      | 0.809267|
    | Left-Cerebellum Cortex   | 0.869220|
    | Right-Cerebellum Cortex  | 0.879554|
    | Left-Thalamus            | 0.851638|
    | Right-Thalamus           | 0.887088|
    | Left-Caudate             | 0.813050|
    | Right-Caudate            | 0.830947|
    | Left-Putamen             | 0.849082|
    | Right-Putamen            | 0.853028|
    | Left-Pallidum            | 0.786699|
    | Right-Pallidum           | 0.775344|
    | Left-Hippocampus         | 0.804197|
    | Right-Hippocampus        | 0.822381|
    | Left-Amygdala            | 0.773281|
    | Right-Amygdala           | 0.792968|
    | Left-VentralDC           | 0.805002|
    | Right-VentralDC          | 0.815977|
    | Brain Stem               | 0.886675|
    | CSF                      | 0.666240|
#### Skull Stripping
    bet_NFBS: 0.969
    bet_synstrip: 0.971





