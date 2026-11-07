use ndarray::{ArrayD, IxDyn};
use std::collections::VecDeque;

/// Keep only the largest connected component in a 3D binary mask (6-connectivity BFS).
pub fn largest_connected_component_3d(mask: &ArrayD<u8>) -> ArrayD<u8> {
    let shape = mask.shape();
    assert_eq!(shape.len(), 3, "Expected 3D mask");
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);

    let mut labels = ArrayD::<u32>::zeros(IxDyn(shape));
    let mut current_label: u32 = 0;
    let mut best_label: u32 = 0;
    let mut best_size: usize = 0;

    // 6-connectivity offsets
    let offsets: [(i64, i64, i64); 6] = [
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1),
    ];

    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                if mask[[x, y, z]] == 0 || labels[[x, y, z]] > 0 {
                    continue;
                }
                // BFS from this voxel
                current_label += 1;
                let mut queue = VecDeque::new();
                queue.push_back((x, y, z));
                labels[[x, y, z]] = current_label;
                let mut component_size: usize = 0;

                while let Some((cx, cy, cz)) = queue.pop_front() {
                    component_size += 1;
                    for &(dx, dy, dz) in &offsets {
                        let nx2 = cx as i64 + dx;
                        let ny2 = cy as i64 + dy;
                        let nz2 = cz as i64 + dz;
                        if nx2 < 0 || nx2 >= nx as i64
                            || ny2 < 0 || ny2 >= ny as i64
                            || nz2 < 0 || nz2 >= nz as i64
                        {
                            continue;
                        }
                        let (ux, uy, uz) = (nx2 as usize, ny2 as usize, nz2 as usize);
                        if mask[[ux, uy, uz]] > 0 && labels[[ux, uy, uz]] == 0 {
                            labels[[ux, uy, uz]] = current_label;
                            queue.push_back((ux, uy, uz));
                        }
                    }
                }

                if component_size > best_size {
                    best_size = component_size;
                    best_label = current_label;
                }
            }
        }
    }

    // Keep only the largest component
    labels.mapv(|l| if l == best_label { 1u8 } else { 0u8 })
}

