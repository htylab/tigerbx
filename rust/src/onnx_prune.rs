//! Minimal ONNX protobuf surgery: keep only some graph outputs and the nodes they need.
//!
//! Used on the GPU, where the `labels` output of an optimised TigerBx model (ArgMax over the class axis) needs a
//! CUDA workspace of ~10x the logits; ONNX Runtime runs every node of the graph even when only the logits are
//! fetched, so the labels branch is removed from the model before the session is built.

/// (field number, wire type, the field's bytes including its key)
struct Field<'a> {
    num: u64,
    raw: &'a [u8],
    /// payload of a length-delimited field
    payload: Option<&'a [u8]>,
}

fn varint(b: &[u8], pos: &mut usize) -> Option<u64> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let byte = *b.get(*pos)?;
        *pos += 1;
        v |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Some(v);
        }
    }
    None
}

fn fields(b: &[u8]) -> Option<Vec<Field<'_>>> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < b.len() {
        let start = pos;
        let key = varint(b, &mut pos)?;
        let payload = match key & 7 {
            0 => {
                varint(b, &mut pos)?;
                None
            }
            1 => {
                pos += 8;
                None
            }
            2 => {
                let n = varint(b, &mut pos)? as usize;
                let p = b.get(pos..pos + n)?;
                pos += n;
                Some(p)
            }
            5 => {
                pos += 4;
                None
            }
            _ => return None,
        };
        if pos > b.len() {
            return None;
        }
        out.push(Field { num: key >> 3, raw: &b[start..pos], payload });
    }
    Some(out)
}

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Repeated string field `num` of a message.
fn strings(msg: &[u8], num: u64) -> Option<Vec<String>> {
    Some(
        fields(msg)?
            .iter()
            .filter(|f| f.num == num)
            .filter_map(|f| f.payload.map(|p| String::from_utf8_lossy(p).into_owned()))
            .collect(),
    )
}

/// Graph output names of a model (ModelProto.graph = 7, GraphProto.output = 12, ValueInfoProto.name = 1).
pub fn output_names(model: &[u8]) -> Option<Vec<String>> {
    let graph = fields(model)?.into_iter().find(|f| f.num == 7)?.payload?;
    let mut names = Vec::new();
    for f in fields(graph)?.iter().filter(|f| f.num == 12) {
        names.push(strings(f.payload?, 1)?.into_iter().next()?);
    }
    Some(names)
}

/// The model with only the graph outputs in `keep` and the nodes they depend on (nodes are topologically sorted in
/// ONNX, so one backward pass suffices). None if the model cannot be parsed.
pub fn keep_outputs(model: &[u8], keep: &[String]) -> Option<Vec<u8>> {
    let top = fields(model)?;
    let graph = top.iter().find(|f| f.num == 7)?.payload?;
    let gf = fields(graph)?;

    let mut needed: std::collections::HashSet<String> = keep.iter().cloned().collect();
    let mut keep_node = vec![false; gf.len()];
    for (i, f) in gf.iter().enumerate().rev() {
        if f.num != 1 {
            continue;
        }
        let node = f.payload?;
        if strings(node, 2)?.iter().any(|o| needed.contains(o)) {
            keep_node[i] = true;
            needed.extend(strings(node, 1)?.into_iter().filter(|s| !s.is_empty()));
        }
    }

    let mut new_graph = Vec::with_capacity(graph.len());
    for (i, f) in gf.iter().enumerate() {
        let drop = match f.num {
            1 => !keep_node[i],
            12 => !keep.contains(&strings(f.payload?, 1)?.into_iter().next()?),
            _ => false,
        };
        if !drop {
            new_graph.extend_from_slice(f.raw);
        }
    }

    let mut out = Vec::with_capacity(model.len());
    for f in &top {
        if f.num == 7 {
            put_varint(&mut out, (7 << 3) | 2);
            put_varint(&mut out, new_graph.len() as u64);
            out.extend_from_slice(&new_graph);
        } else {
            out.extend_from_slice(f.raw);
        }
    }
    Some(out)
}
