from tigerbx.modules.bx import run
from tigerbx.validate import val
from tigerbx.lib_reg import transform
from tigerbx.gdmi import gdm
from tigerbx.modules.hlc import hlc
from tigerbx.modules.reg import reg
# pipelines (the recommended entry points); the modules (run, reg, hlc) live in tigerbx.modules
from tigerbx.pipelines import affine, bx, duope, inspect, norm, seg, template, vbm
from tigerbx.embed import embed
from tigerbx.eval import eval
from tigerbx.check import check
__version__ = '0.3.3'
