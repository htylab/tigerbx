"""The default model files and the sha256 of every model on the modelhub release.

One place for the names: the modules and pipelines take their defaults from here, `lib_tool.get_model()` checks a
cached or downloaded file against `SHA256`, and the executable's embedded models (`.github/workflows/exe.yml`
MODELS) must be listed here (tests/test_models.py). A new model: add its sha256, then change the default.
"""

# dual brain mask: inner = the tBX mask (the brain), outer (TSEG input; infer flag 'x') = with the surrounding CSF
TBX = 'mprage_tbxdual_v007_opt.onnx'
# whole-brain segmentation, 32 structures (FreeSurfer numbers); input = the tBX (outer) brain
TSEG = 'mprage_tseg_v1_opt.onnx'
# tigerbx.run(): the earlier models, outputs as in 0.2.x: tBET brain mask (_tbet, _tbetmask), ASEG (_aseg)
TBET = 'mprage_tbx_v005_opt.onnx'
ASEG = 'mprage_tseg_v0011_opt.onnx'
DGM = 'mprage_dgm12_v002_opt.onnx'
CT = 'mprage_mix_ct.onnx'
HLC = 'mprage_hlc_v004_opt.onnx'
# tReg: the deformable template registration (reg plan step 'D', norm, template).
# v001 = mprage_reg_v005_vfa (the same file; the full-resolution feature matching computes the 27 neighbour
# similarities one shift at a time: low peak memory), v002 = the same network trained for EPI as well, v003 = our own
# network (shared tBX U-Net features + per-level softmax matching), trained for EPI as well
# (_opt: the same weights exported for lower CPU memory and time; the results agree to 1e-3 voxel)
TREG = 'mprage_treg_v003_opt.onnx'
# embed: v002 = tReg v003's feature network up to its 2 mm and 4 mm decoder levels (v001 = the encoder of tReg v001)
EMBED = 'mprage_embed_v002.onnx'
# _opt: v002 with staged GroupNorm statistics (onnx_optimize): faster, and the CPU gives the GPU's result
GDM = 'vdm_unet3d_v002_opt.onnx'
INSPECT = 'inspect_v003.onnx'

SHA256 = {
    # base exports of the _opt models (the _opt files are rebuilt from them)
    'mprage_aseg43_v0011_allseg.onnx': 'c6163104e44277f8178575ebd62e097f8a02b40063176590b46d2581f7b6dfe1',
    'mprage_bet_v005_mixsynthv4.onnx': '7571ec093c0a54fa3a81f576063eaab0e3babac790f7946da00a6e23b4feda5d',
    'mprage_dgm12_v002_mix6.onnx': '2d9d5f09465f1883e6d0fcd4bf1972a450ecb6a49c033471c983b9f109f27d70',
    'mprage_hlc_v004_resunetplusT2.onnx': '00b56ad4fb3a008237216e07542b4b869c9e7003551d078c062437cc132b5ceb',
    'inspect_v003.onnx': 'afa33a88f814d6805152c2b0ffa6dc12f150ad0f1fc012203028935476c37b21',
    'mprage_dgm12_v002_opt.onnx': '77fb816a66ce4ae895f1ca044f34e3255fe4309cbc4d1c428fedf9f96a81a35e',
    'mprage_embed_v001_vfa.onnx': '833dd32a047238f147f4ae545ed3da32b97789d0cf8efed3300dc55650ef6c96',
    'mprage_embed_v002.onnx': '9a4277543f4c8b911b438319c1bfbcda10ef6ab6b5c68ffba296adabd1fc6cfc',
    'mprage_hlc_v004_opt.onnx': '7358caadcfc8ed51e2293c5eb5e154bb611c3b4ad21e60be067eea0ea026ddd1',
    'mprage_mix_ct.onnx': '48affdf980a1039fe440b3df86dd76f23318bc49486dcf60d775a8a500037b96',
    'mprage_reg_v004_vfa.onnx': '6a4f4eada8853367919da657716bbbd1ce8b2d378126bd3f8c7770352bb85138',
    'mprage_reg_v005_vfa.onnx': 'b29c4c466e06064b35aa97ec24773aac08b19f16c8923da8cf6c53ae6217a013',
    'mprage_tbxdual_v007_opt.onnx': 'c88199b70674c9db172932cbdc4211dfbaf7fd060928316ae204c786dea88c42',
    'mprage_treg_v003_opt.onnx': '8dbd58b6cf86af327918cb80e12e96a36f7734961bda332955a352e4f7f4b682',
    'mprage_tbx_v005_opt.onnx': '1fdb8ef2c36d1e2dc90e8987e4a9898d6399bb5f3b616d698bb5d192c8e5afa8',
    'mprage_tseg_v0011_opt.onnx': '67b7f553dfd99363f3f89268536986d24d1b79e9ce60760dc4e0ee969f755fe2',
    'vdm_unet3d_v002.onnx': '9357b40da479f4dc85efe2616cab02ad5ffe7f238221070a3769d767a968a97f',
    'vdm_unet3d_v002_opt.onnx': '230265338c5b9bf22bb0c38aeda2522969523368b02149f7ef441aa9c3005535',
    'mprage_tseg_v1_opt.onnx': 'af5e86abfe3069ef0e7c45ea4f45e79ad06792f247eb8f763a86de261a3a7592',
}
