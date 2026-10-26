"""Rewrite exported ONNX models so ONNX Runtime runs them faster on CPU, without changing the result.

PyTorch exports ``nn.GroupNorm`` as ``Reshape(0, G, -1) -> InstanceNormalization -> Reshape(shape(x))
-> Mul(weight) -> Add(bias)``; ``nn.InstanceNorm3d`` becomes a bare ``InstanceNormalization``. ONNX Runtime's
CPU InstanceNormalization kernel is slow (about 30% of a TigerBx U-Net forward pass), so each of these is
replaced by ReduceMean / Sub / Mul / Add, which are multi-threaded and vectorised, and the per-channel affine
is folded into a single multiply-add. The variance stays two-pass (mean of squared deviations), as in
InstanceNormalization; the one-pass E[x^2] - E[x]^2 loses too much precision in float32. The means are taken in
stages (last spatial axis, then the next, then the rest): a flat CPU reduction over millions of voxels is off by
~1e-4 relative, which made CPU and GPU results differ (tBX ~0.1% of brain voxels); staged, they agree.

``--labels`` also adds a second output ``labels`` (uint8): the argmax of the logits over channels, or for a
single-channel (sigmoid) model such as tBX, logit >= 0 (probability >= 0.5). TigerBx asks models for that
output only, so the logits (1.15 GB for TSEG on a 172x212x179 crop) are never copied out of ONNX Runtime; the
logits output stays first, for patch inference. Models with more than 256 channels are refused.

``--heads name:start:end:kind,...`` adds one output per channel range of the logits, for models whose channels hold
several heads (HLC: parcellation, hemisphere, dorsal/ventral, thickness, PVE): ``argmax`` gives uint8 labels of
channels [start, end), ``slice`` gives those channels as they are. The caller then requests only the heads it needs.

Usage::

    python -m tigerbx.dev.onnx_optimize model.onnx model_opt.onnx            # rewrite + check
    python -m tigerbx.dev.onnx_optimize tseg.onnx tseg_opt.onnx --labels --shape 1,1,160,192,160

The check runs both models on the same random input and reports the largest output difference and the
speed-up; dynamic input dimensions default to 96 (``--shape`` overrides). Requires ``onnx`` (not a TigerBx
runtime dependency) and ``onnxruntime``.
"""

import argparse
import os
import time

import numpy as np


def _consts(graph):
    from onnx import numpy_helper

    val = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    for n in graph.node:
        if n.op_type == 'Constant' and n.attribute and n.attribute[0].name == 'value':
            val[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    return val


def _eps(node):
    for a in node.attribute:
        if a.name == 'epsilon':
            return a.f
    return 1e-5


class _Builder:
    """Collects new nodes/initializers for one replaced normalisation."""

    def __init__(self, prefix, opset):
        self.p, self.opset, self.nodes, self.inits = prefix, opset, [], []

    def const(self, name, arr):
        from onnx import numpy_helper

        self.inits.append(numpy_helper.from_array(np.asarray(arr), self.p + name))
        return self.p + name

    def node(self, op, inputs, out, **attrs):
        from onnx import helper

        self.nodes.append(helper.make_node(op, inputs, [out], name=out, **attrs))
        return out

    def t(self, name):
        return self.p + name

    def reduce_mean(self, x, axes, out):
        if self.opset >= 18:
            return self.node('ReduceMean', [x, self.const(out + '_axes', np.array(axes, np.int64))],
                             self.t(out), keepdims=1)
        return self.node('ReduceMean', [x], self.t(out), axes=list(axes), keepdims=1)


def _staged_mean(b, x, stages, out):
    """Mean over the axes of all ``stages`` (lists of axes), one stage after the other, the innermost first. The
    result is the same mean (every stage averages blocks of equal size), but ONNX Runtime's CPU ReduceMean adds
    up a flat reduction sequentially in float32, so over a few million voxels the mean is off by ~sqrt(N) eps
    (GroupNorm outputs ~1e-4 relative); short stages keep it at float32 precision, as the GPU's tree reduction."""
    for i, axes in enumerate(stages):
        x = b.reduce_mean(x, axes, out if i == len(stages) - 1 else f'{out}_s{i}')
    return x


def _normalise(b, x, stages, eps):
    """Return (d, inv_sd): d = x - mean, inv_sd = 1 / sqrt(var + eps), statistics over the axes of ``stages``."""
    m = _staged_mean(b, x, stages, 'mean')
    d = b.node('Sub', [x, m], b.t('d'))
    d2 = b.node('Mul', [d, d], b.t('d2'))
    var = _staged_mean(b, d2, stages, 'var')
    ve = b.node('Add', [var, b.const('eps', np.float32(eps))], b.t('ve'))
    sd = b.node('Sqrt', [ve], b.t('sd'))
    return d, b.node('Reciprocal', [sd], b.t('inv_sd'))


def _fuse_groupnorm(graph, opset, node, prod, cons, val, k, ranks):
    """GroupNorm pattern around ``node`` -> (nodes to remove, new nodes, new inits), or None."""
    r0 = prod.get(node.input[0])
    if r0 is None or r0.op_type != 'Reshape' or r0.input[1] not in val:
        return None
    gshape = val[r0.input[1]].reshape(-1).tolist()
    if len(gshape) != 3 or gshape[0] != 0 or gshape[2] != -1:
        return None
    groups = int(gshape[1])
    nxt = cons.get(node.output[0], [])
    if len(nxt) != 1 or nxt[0].op_type != 'Reshape':
        return None
    r1 = nxt[0]
    shp = prod.get(r1.input[1])
    if shp is None or shp.op_type != 'Shape' or shp.input[0] != r0.input[0]:
        return None
    x = r0.input[0]
    rank = ranks.get(x)
    s_g = val.get(node.input[1])
    b_g = val.get(node.input[2])
    if s_g is None or b_g is None or s_g.size != groups:
        return None
    remove = [node, r1]
    out = r1.output[0]
    w = b = None
    after = cons.get(out, [])
    if len(after) == 1 and after[0].op_type == 'Mul' and after[0].input[1] in val:
        mul = after[0]
        if rank is None and val[mul.input[1]].ndim >= 2:
            rank = val[mul.input[1]].ndim + 1          # PyTorch GroupNorm weight [C, 1, 1(, 1)]: input rank 4 or 5
        w = val[mul.input[1]].reshape(-1)
        remove.append(mul)
        out = mul.output[0]
        after = cons.get(out, [])
        if len(after) == 1 and after[0].op_type == 'Add' and after[0].input[1] in val:
            add = after[0]
            b = val[add.input[1]].reshape(-1)
            remove.append(add)
            out = add.output[0]
    if w is None:
        return None  # GroupNorm without affine is not exported this way by PyTorch; leave it alone
    ch = w.size
    if ch % groups:
        return None
    if b is None:
        b = np.zeros(ch, np.float32)
    if len(cons.get(r1.input[1], [])) == 1:
        remove.append(shp)
    per = ch // groups
    # y_c = w_c * (s_g * (x - m_g) * inv_sd_g + b_g) + b_c  =  d * (w_c s_g inv_sd_g) + (w_c b_g + b_c)
    scale_c = (w * np.repeat(s_g.reshape(-1), per)).astype(np.float32).reshape(1, ch, 1, 1, 1)
    bias_c = (w * np.repeat(b_g.reshape(-1), per) + b).astype(np.float32).reshape(1, ch, 1, 1, 1)
    bld = _Builder(f'_gn{k}_', opset)
    if rank == 5:
        # [1, G, per, X, Y, Z]: statistics over Z, then Y, then (per, X)
        xs = bld.node('Shape', [x], bld.t('xshape'))
        sp = bld.node('Slice', [xs, bld.const('s0', np.array([2], np.int64)), bld.const('s1', np.array([5], np.int64))],
                      bld.t('spatial'))
        s6 = bld.node('Concat', [bld.const('gper', np.array([1, groups, per], np.int64)), sp], bld.t('shape6'), axis=0)
        xr = bld.node('Reshape', [x, s6], bld.t('x6'))
        d, inv = _normalise(bld, xr, [[5], [4], [2, 3]], _eps(node))
        inv_c = bld.node('Reshape', [bld.node('Expand', [inv, bld.const('gshape', np.array([1, groups, per, 1, 1, 1],
                                                                                            np.int64))], bld.t('inv_x')),
                                     bld.const('cshape', np.array([1, ch, 1, 1, 1], np.int64))], bld.t('inv_c'))
    else:
        xr = r0.output[0]
        d, inv = _normalise(bld, xr, [[2]], _eps(node))
        inv_x = bld.node('Expand', [inv, bld.const('gshape', np.array([1, groups, per], np.int64))], bld.t('inv_x'))
        inv_c = bld.node('Reshape', [inv_x, bld.const('cshape', np.array([1, ch, 1, 1, 1], np.int64))], bld.t('inv_c'))
        xs = bld.node('Shape', [x], bld.t('xshape'))
    kk = bld.node('Mul', [inv_c, bld.const('scale', scale_c)], bld.t('k'))
    d5 = bld.node('Reshape', [d, xs], bld.t('d5'))
    y = bld.node('Mul', [d5, kk], bld.t('y'))
    bld.node('Add', [y, bld.const('bias', bias_c)], out)
    return remove, bld.nodes, bld.inits


def _fuse_instancenorm(graph, opset, node, rank, val, k):
    """Plain InstanceNormalization on an input of known ``rank`` (4 or 5)."""
    s = val.get(node.input[1])
    bias = val.get(node.input[2])
    if s is None or bias is None or rank not in (4, 5):
        return None
    ch = s.size
    shape = [1, ch] + [1] * (rank - 2)
    bld = _Builder(f'_in{k}_', opset)
    # statistics over the last spatial axis, then the next, then the rest (see _staged_mean)
    stages = [[rank - 1], [rank - 2]] + ([list(range(2, rank - 2))] if rank > 4 else [])
    d, inv = _normalise(bld, node.input[0], stages, _eps(node))
    kk = bld.node('Mul', [inv, bld.const('scale', s.astype(np.float32).reshape(shape))], bld.t('k'))
    y = bld.node('Mul', [d, kk], bld.t('y'))
    bld.node('Add', [y, bld.const('bias', bias.astype(np.float32).reshape(shape))], node.output[0])
    return [node], bld.nodes, bld.inits


def _ranks(model):
    """Rank of every tensor that shape inference can determine."""
    import onnx

    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        return {}
    ranks = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        t = vi.type.tensor_type
        if t.HasField('shape'):
            ranks[vi.name] = len(t.shape.dim)
    return ranks


def optimize_model(model):
    """Return (optimized ModelProto, stats dict). The input model is not modified."""
    import onnx

    model = onnx.ModelProto.FromString(model.SerializeToString())
    graph = model.graph
    opset = next((o.version for o in model.opset_import if o.domain in ('', 'ai.onnx')), 0)
    val = _consts(graph)
    nodes = list(graph.node)
    prod = {o: n for n in nodes for o in n.output}
    cons = {}
    for n in nodes:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    ranks = _ranks(model)

    remove, insert, inits = set(), {}, []
    stats = {'groupnorm': 0, 'instancenorm': 0, 'skipped': 0}
    for k, n in enumerate(nodes):
        if n.op_type != 'InstanceNormalization' or id(n) in remove:
            continue
        res = _fuse_groupnorm(graph, opset, n, prod, cons, val, k, ranks)
        kind = 'groupnorm'
        if res is None:
            res = _fuse_instancenorm(graph, opset, n, ranks.get(n.input[0]), val, k)
            kind = 'instancenorm'
        if res is None:
            stats['skipped'] += 1
            continue
        rm, new, ini = res
        # insert the replacement where the last removed node was, so every input already exists
        last = max(nodes.index(r) for r in rm)
        insert.setdefault(id(nodes[last]), []).extend(new)
        remove.update(id(r) for r in rm)
        inits += ini
        stats[kind] += 1

    out = []
    for n in nodes:
        if id(n) in insert:
            out += insert[id(n)]
        elif id(n) not in remove:
            out.append(n)
    del graph.node[:]
    graph.node.extend(out)
    graph.initializer.extend(inits)

    # drop nodes and initializers nothing uses any more
    live = {o.name for o in graph.output}
    keep = []
    for n in reversed(list(graph.node)):
        if any(o in live for o in n.output):
            keep.append(n)
            live.update(n.input)
    del graph.node[:]
    graph.node.extend(reversed(keep))
    used = [i for i in graph.initializer if i.name in live]
    del graph.initializer[:]
    graph.initializer.extend(used)

    onnx.checker.check_model(model)
    return model, stats


def add_label_output(model, name='labels'):
    """Return a copy of ``model`` with an extra uint8 output ``name``: argmax of output 0 over axis 1, or
    output 0 >= 0 when it has a single channel."""
    import onnx
    from onnx import TensorProto, helper

    model = onnx.ModelProto.FromString(model.SerializeToString())
    graph = model.graph
    if any(o.name == name for o in graph.output):
        raise ValueError(f'model already has an output named {name!r}')
    logits = graph.output[0]
    dims = logits.type.tensor_type.shape.dim
    if len(dims) >= 2 and dims[1].HasField('dim_value') and dims[1].dim_value > 256:
        raise ValueError(f'{dims[1].dim_value} channels do not fit in uint8 labels')
    single = len(dims) >= 2 and dims[1].HasField('dim_value') and dims[1].dim_value == 1
    if single:
        opset = next((o.version for o in model.opset_import if o.domain in ('', 'ai.onnx')), 0)
        if opset < 12:
            raise ValueError('a single-channel labels output needs opset >= 12 (GreaterOrEqual)')
        zero = name + '_zero'
        graph.initializer.append(onnx.numpy_helper.from_array(np.zeros((), np.float32), zero))
        if opset >= 18:
            ax = name + '_axes'
            graph.initializer.append(onnx.numpy_helper.from_array(np.array([1], np.int64), ax))
            sq = helper.make_node('ReduceMax', [logits.name, ax], [name + '_sq'], name=name + '_sq', keepdims=0)
        else:
            sq = helper.make_node('ReduceMax', [logits.name], [name + '_sq'], name=name + '_sq', axes=[1], keepdims=0)
        graph.node.extend([
            sq,
            helper.make_node('GreaterOrEqual', [name + '_sq', zero], [name + '_ge'], name=name + '_ge'),
            helper.make_node('Cast', [name + '_ge'], [name], name=name + '_cast', to=TensorProto.UINT8),
        ])
    else:
        idx = name + '_argmax'
        graph.node.extend([
            helper.make_node('ArgMax', [logits.name], [idx], name=idx, axis=1, keepdims=0),
            helper.make_node('Cast', [idx], [name], name=name + '_cast', to=TensorProto.UINT8),
        ])
    shape = [d.dim_param if d.HasField('dim_param') else (d.dim_value or None) for i, d in enumerate(dims) if i != 1]
    graph.output.append(helper.make_tensor_value_info(name, TensorProto.UINT8, shape or None))
    onnx.checker.check_model(model)
    return model


def add_head_outputs(model, heads):
    """Return a copy of ``model`` with one extra output per ``(name, start, end, kind)`` over channels [start, end)
    of output 0: kind ``'argmax'`` -> uint8 labels [N, X, Y, Z], ``'slice'`` -> float channels [N, end-start, X, Y, Z]."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.ModelProto.FromString(model.SerializeToString())
    graph = model.graph
    opset = next((o.version for o in model.opset_import if o.domain in ('', 'ai.onnx')), 0)
    if opset < 10:
        raise ValueError('channel heads need opset >= 10 (Slice with inputs)')
    logits = graph.output[0]
    dims = logits.type.tensor_type.shape.dim
    sym = [d.dim_param if d.HasField('dim_param') else (d.dim_value or None) for d in dims]
    names = {o.name for o in graph.output}
    for name, start, end, kind in heads:
        if name in names:
            raise ValueError(f'model already has an output named {name!r}')
        if kind not in ('argmax', 'slice') or not 0 <= start < end:
            raise ValueError(f'bad head {(name, start, end, kind)}')
        if kind == 'argmax' and end - start > 256:
            raise ValueError(f'head {name!r}: {end - start} channels do not fit in uint8 labels')
        names.add(name)
        c = {k: name + '_' + k for k in ('starts', 'ends', 'axes')}
        for k, v in (('starts', start), ('ends', end), ('axes', 1)):
            graph.initializer.append(numpy_helper.from_array(np.array([v], np.int64), c[k]))
        sl = name if kind == 'slice' else name + '_slice'
        graph.node.append(helper.make_node('Slice', [logits.name, c['starts'], c['ends'], c['axes']], [sl], name=sl))
        if kind == 'slice':
            shape = sym[:1] + [end - start] + sym[2:]
            graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))
        else:
            graph.node.extend([
                helper.make_node('ArgMax', [sl], [name + '_argmax'], name=name + '_argmax', axis=1, keepdims=0),
                helper.make_node('Cast', [name + '_argmax'], [name], name=name + '_cast', to=TensorProto.UINT8),
            ])
            graph.output.append(helper.make_tensor_value_info(name, TensorProto.UINT8, sym[:1] + sym[2:]))
    onnx.checker.check_model(model)
    return model


def _parse_heads(text):
    heads = []
    for item in text.split(','):
        name, start, end, kind = item.split(':')
        heads.append((name, int(start), int(end), kind))
    return heads


def _set_metadata(model, entries):
    """Add / replace metadata_props entries (provenance of the optimised model)."""
    keep = [p for p in model.metadata_props if p.key not in entries]
    del model.metadata_props[:]
    model.metadata_props.extend(keep)
    for k, v in entries.items():
        p = model.metadata_props.add()
        p.key, p.value = k, v


def _random_inputs(sess, shape=None, seed=0):
    rng = np.random.default_rng(seed)
    feeds = {}
    for inp in sess.get_inputs():
        if shape is not None and len(shape) == len(inp.shape):
            dims = list(shape)
        else:
            dims = [d if isinstance(d, int) and d > 0 else (1 if i == 0 else 96) for i, d in enumerate(inp.shape)]
        dtype = np.float16 if 'float16' in inp.type else np.float32
        feeds[inp.name] = rng.random(dims, dtype=np.float32).astype(dtype)
    return feeds


def check(original, optimized, shape=None, repeats=3, threads=None):
    """Run both models on one random input; return (max abs output difference, t_orig, t_opt)."""
    import onnxruntime as ort

    def session(path):
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.log_severity_level = 3
        return ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])

    def timed(sess, feeds):
        sess.run(None, feeds)
        ts = []
        for _ in range(repeats):
            t = time.perf_counter()
            out = sess.run(None, feeds)
            ts.append(time.perf_counter() - t)
        return out, float(np.median(ts))

    s0, s1 = session(original), session(optimized)
    feeds = _random_inputs(s0, shape)
    o0, t0 = timed(s0, feeds)
    o1, t1 = timed(s1, feeds)
    diff = float(np.max(np.abs(o0[0].astype(np.float64) - o1[0])))
    if len(o1) > 1 and o1[1].dtype == np.uint8:  # labels output: compare with the original logits
        ref = (o0[0][:, 0] >= 0) if o0[0].shape[1] == 1 else np.argmax(o0[0], axis=1)
        mismatch = int(np.count_nonzero(ref != o1[1]))
        if mismatch:
            print(f'warning: {mismatch} labels differ from argmax of the original logits')
    return diff, t0, t1


def main(argv=None):
    import onnx

    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('input')
    ap.add_argument('output')
    ap.add_argument('--shape', help='test input shape, e.g. 1,1,160,192,160 (default: 96 for dynamic dims)')
    ap.add_argument('--labels', action='store_true', help='add a uint8 "labels" output (argmax over channels)')
    ap.add_argument('--heads', help='extra outputs per channel range, e.g. parc:0:57:argmax,ct:63:64:slice')
    ap.add_argument('--no-check', action='store_true', help='skip the ONNX Runtime comparison')
    ap.add_argument('--threads', type=int, help='intra-op threads for the check')
    args = ap.parse_args(argv)

    model, stats = optimize_model(onnx.load(args.input))
    if args.labels:
        model = add_label_output(model)
    if args.heads:
        model = add_head_outputs(model, _parse_heads(args.heads))
    _set_metadata(model, {
        'tigerbx_optimized_from': os.path.basename(args.input),
        'tigerbx_optimize': ' '.join(filter(None, [
            f"groupnorm={stats['groupnorm']} instancenorm={stats['instancenorm']} stats=staged",
            'labels' if args.labels else '', f'heads={args.heads}' if args.heads else ''])),
    })
    onnx.save(model, args.output)
    print(f"{args.input} -> {args.output}: {stats['groupnorm']} GroupNorm, {stats['instancenorm']} "
          f"InstanceNorm rewritten, {stats['skipped']} left as is" + ('; labels output added' if args.labels else '')
          + (f'; heads {args.heads}' if args.heads else ''))
    if not args.no_check:
        shape = [int(s) for s in args.shape.split(',')] if args.shape else None
        diff, t0, t1 = check(args.input, args.output, shape, threads=args.threads)
        print(f'max |output difference| {diff:.3g}; run time {t0:.2f}s -> {t1:.2f}s ({t0 / t1:.2f}x)')


if __name__ == '__main__':
    main()
