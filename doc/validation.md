### TigerBx: Tissue Mask Generation for Brain Extration

### Validation pipeline (after v0.1.16)

    import tigerbx
    import time
    # tigerbx.val(argstring, input_dir, output_dir=None, model=None, GPU=False, debug=False)
    df, metric1 = tigerbx.val('bet_NFBS', 'NFBS_Dataset','temp', GPU=True)
    df, metric2 = tigerbx.val('bet_synstrip', 'synthstrip_data_v1.4','temp', GPU=True)
    df, metric3 = tigerbx.val('aseg_123', 'aseg', 'temp', GPU=True)
    df, metric4 = tigerbx.val('dgm_123', 'aseg', 'temp', GPU=True)
    df, metric5 = tigerbx.val('syn_123', 'aseg', 'temp', GPU=True)

    t = time.time()
    print('bet_NFBS', metric1)
    print('bet_synstrip', metric2)
    print('aseg_123', metric3)
    print('dgm_123', metric4)
    print('syn_123', metric5)
    print('Time', time.time() - t)

### Validation Datasets
1. Skull Striping: [NFBS_Dataset](http://preprocessed-connectomes-project.org/NFB_skullstripped)
2. Skull Striping: [Synstrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip)
3. Deep gray matter: [MindBoggle-101](https://mindboggle.info/)
4. Deep gray matter: [CANDI](https://www.nitrc.org/projects/candi_share/)

### Internal Validation Sets:
1. aseg: A combination of MindBoggle-101 and CANDI (N = 123) with DGM labels

