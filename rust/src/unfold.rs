//! Folding repair of the tReg warp (tigerbx.core.templatereg.unfold): where det J <= 0.05, the displacement is
//! replaced by its 3 x 3 x 3 mean on those voxels and their 6 neighbours, repeated until none is left (or 100
//! passes); run per folded cluster on a box around it, the rest of the warp untouched.

const THR: f64 = 0.05;
const MAX_ITER: usize = 100;

/// A displacement [3, X, Y, Z] (flat) on a box of shape `s`.
struct Field {
    d: Vec<f64>,
    s: [usize; 3],
}

impl Field {
    fn n(&self) -> usize {
        self.s[0] * self.s[1] * self.s[2]
    }
    fn at(&self, p: [usize; 3]) -> usize {
        (p[0] * self.s[1] + p[1]) * self.s[2] + p[2]
    }
}

/// np.gradient along axis `a` at p: central inside, one-sided at the edges.
fn grad(f: &Field, comp: usize, p: [usize; 3], a: usize) -> f64 {
    let n = f.n();
    let v = |q: [usize; 3]| f.d[comp * n + f.at(q)];
    let (mut lo, mut hi) = (p, p);
    let len = f.s[a];
    if len < 2 {
        return 0.0;
    }
    if p[a] == 0 {
        hi[a] = 1;
        v(hi) - v(lo)
    } else if p[a] == len - 1 {
        lo[a] = len - 2;
        v(hi) - v(lo)
    } else {
        lo[a] -= 1;
        hi[a] += 1;
        (v(hi) - v(lo)) / 2.0
    }
}

fn det(f: &Field) -> Vec<f64> {
    let mut out = vec![0.0; f.n()];
    for i in 0..f.s[0] {
        for j in 0..f.s[1] {
            for k in 0..f.s[2] {
                let p = [i, j, k];
                let mut m = [[0.0; 3]; 3];
                for (r, row) in m.iter_mut().enumerate() {
                    for (c, x) in row.iter_mut().enumerate() {
                        *x = grad(f, r, p, c) + if r == c { 1.0 } else { 0.0 };
                    }
                }
                out[f.at(p)] = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
            }
        }
    }
    out
}

/// 6-connected dilation, outside = false (scipy binary_dilation).
fn dilate(m: &[bool], s: [usize; 3]) -> Vec<bool> {
    let mut out = m.to_vec();
    let at = |p: [usize; 3]| (p[0] * s[1] + p[1]) * s[2] + p[2];
    for i in 0..s[0] {
        for j in 0..s[1] {
            for k in 0..s[2] {
                if !m[at([i, j, k])] {
                    continue;
                }
                for a in 0..3 {
                    let p = [i, j, k];
                    if p[a] > 0 {
                        let mut q = p;
                        q[a] -= 1;
                        out[at(q)] = true;
                    }
                    if p[a] + 1 < s[a] {
                        let mut q = p;
                        q[a] += 1;
                        out[at(q)] = true;
                    }
                }
            }
        }
    }
    out
}

/// scipy uniform_filter(size 3, mode 'reflect') of component `comp`: three 1-D means, axis 0, 1, 2.
fn mean3(f: &Field, comp: usize) -> Vec<f64> {
    let n = f.n();
    let mut x = f.d[comp * n..(comp + 1) * n].to_vec();
    for a in 0..3 {
        let mut y = vec![0.0; n];
        for i in 0..f.s[0] {
            for j in 0..f.s[1] {
                for k in 0..f.s[2] {
                    let p = [i, j, k];
                    let (mut lo, mut hi) = (p, p);
                    if p[a] > 0 {
                        lo[a] -= 1;
                    }
                    if p[a] + 1 < f.s[a] {
                        hi[a] += 1;
                    }
                    y[f.at(p)] = (x[f.at(lo)] + x[f.at(p)] + x[f.at(hi)]) / 3.0;
                }
            }
        }
        x = y;
    }
    x
}

/// unfold on a box; `cut[a]` = (low face, high face) of axis a is a cut (not the volume's edge). false when the
/// repair reaches 2 voxels from a cut face.
fn unfold_box(f: &mut Field, cut: [(bool, bool); 3]) -> bool {
    let s = f.s;
    let n = f.n();
    let near = |p: [usize; 3], w: usize| {
        (0..3).any(|a| (cut[a].0 && p[a] < w) || (cut[a].1 && p[a] + w >= s[a]))
    };
    let mut edge = vec![false; n];
    let mut rim = vec![false; n];
    for i in 0..s[0] {
        for j in 0..s[1] {
            for k in 0..s[2] {
                let p = [i, j, k];
                edge[f.at(p)] = near(p, 1);
                rim[f.at(p)] = near(p, 2);
            }
        }
    }
    for _ in 0..MAX_ITER {
        let dj = det(f);
        let bad: Vec<bool> = (0..n).map(|v| dj[v] <= THR && !edge[v]).collect();
        if !bad.iter().any(|&b| b) {
            return true;
        }
        let m = dilate(&bad, s);
        if m.iter().zip(&rim).any(|(&a, &b)| a && b) {
            return false;
        }
        let means: Vec<Vec<f64>> = (0..3).map(|c| mean3(f, c)).collect();
        for (c, mc) in means.iter().enumerate() {
            for v in 0..n {
                if m[v] {
                    f.d[c * n + v] = mc[v];
                }
            }
        }
    }
    true
}

/// 6-connected components of `m`: their bounding boxes [lo, hi).
fn boxes(m: &[bool], s: [usize; 3]) -> Vec<[(usize, usize); 3]> {
    let at = |p: [usize; 3]| (p[0] * s[1] + p[1]) * s[2] + p[2];
    let mut seen = vec![false; m.len()];
    let mut out = Vec::new();
    for i in 0..s[0] {
        for j in 0..s[1] {
            for k in 0..s[2] {
                if !m[at([i, j, k])] || seen[at([i, j, k])] {
                    continue;
                }
                let mut b = [(i, i + 1), (j, j + 1), (k, k + 1)];
                let mut stack = vec![[i, j, k]];
                seen[at([i, j, k])] = true;
                while let Some(p) = stack.pop() {
                    for a in 0..3 {
                        b[a] = (b[a].0.min(p[a]), b[a].1.max(p[a] + 1));
                        for up in [false, true] {
                            if (!up && p[a] == 0) || (up && p[a] + 1 >= s[a]) {
                                continue;
                            }
                            let mut q = p;
                            if up { q[a] += 1 } else { q[a] -= 1 }
                            if m[at(q)] && !seen[at(q)] {
                                seen[at(q)] = true;
                                stack.push(q);
                            }
                        }
                    }
                }
                out.push(b);
            }
        }
    }
    out
}

/// The tReg sampling grid [3, X, Y, Z] (flat, voxels) with its folds repaired, as templatereg.treg_grid.
pub fn unfold_grid(grid: &mut [f32], s: [usize; 3]) {
    let n = s[0] * s[1] * s[2];
    let id = |c: usize, v: usize| -> f32 {
        let (i, j, k) = (v / (s[1] * s[2]), (v / s[2]) % s[1], v % s[2]);
        [i, j, k][c] as f32
    };
    let mut f = Field { d: (0..3 * n).map(|x| grid[x] as f64 - id(x / n, x % n) as f64).collect(), s };
    let bad: Vec<bool> = det(&f).iter().map(|&v| v <= THR).collect();
    if !bad.iter().any(|&b| b) {
        return;
    }
    let mut grown = bad;
    for _ in 0..6 {
        grown = dilate(&grown, s);
    }
    for b in boxes(&grown, s) {
        let mut rim = 2;
        loop {
            let bx: [(usize, usize); 3] = std::array::from_fn(|a| (b[a].0.saturating_sub(rim), (b[a].1 + rim).min(s[a])));
            let cut: [(bool, bool); 3] = std::array::from_fn(|a| (bx[a].0 > 0, bx[a].1 < s[a]));
            let bs = [bx[0].1 - bx[0].0, bx[1].1 - bx[1].0, bx[2].1 - bx[2].0];
            let mut c = Field { d: vec![0.0; 3 * bs[0] * bs[1] * bs[2]], s: bs };
            let cn = c.n();
            for comp in 0..3 {
                for i in 0..bs[0] {
                    for j in 0..bs[1] {
                        for k in 0..bs[2] {
                            let v = c.at([i, j, k]);
                            c.d[comp * cn + v] = f.d[comp * n + f.at([i + bx[0].0, j + bx[1].0, k + bx[2].0])];
                        }
                    }
                }
            }
            if unfold_box(&mut c, cut) {
                for comp in 0..3 {
                    for i in 0..bs[0] {
                        for j in 0..bs[1] {
                            for k in 0..bs[2] {
                                let v = f.at([i + bx[0].0, j + bx[1].0, k + bx[2].0]);
                                f.d[comp * n + v] = c.d[comp * cn + c.at([i, j, k])];
                            }
                        }
                    }
                }
                break;
            }
            rim *= 2;
        }
    }
    for x in 0..3 * n {
        grid[x] = (id(x / n, x % n) as f64 + f.d[x]) as f32;
    }
}

#[cfg(test)]
mod tests {
    /// Parity with templatereg.unfold: TIGERBX_UNFOLD_IN = raw float32 grid [3, 160, 224, 192] (little-endian),
    /// written back repaired to TIGERBX_UNFOLD_OUT; skipped when unset.
    #[test]
    fn unfold_file() {
        let (Ok(src), Ok(dst)) = (std::env::var("TIGERBX_UNFOLD_IN"), std::env::var("TIGERBX_UNFOLD_OUT")) else {
            return;
        };
        let b = std::fs::read(src).unwrap();
        let mut g: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        super::unfold_grid(&mut g, [160, 224, 192]);
        std::fs::write(dst, g.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    }
}
