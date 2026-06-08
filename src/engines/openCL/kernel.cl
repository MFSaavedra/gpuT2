inline uchar next_state(uchar alive, int neighbors) {
    if (alive) {
        return (neighbors == 2 || neighbors == 3) ? (uchar)1 : (uchar)0;
    }
    return (neighbors == 3 || neighbors == 6) ? (uchar)1 : (uchar)0;
}

__kernel void life_global(
    __global const uchar* src,
    __global uchar* dst,
    int rows,
    int cols,
    int wrap
) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    if (col >= cols || row >= rows) return;

    int n = 0;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;

            int nc = col + dx;
            int nr = row + dy;

            if (wrap) {
                nc = (nc + cols) % cols;
                nr = (nr + rows) % rows;
            } else {
                if (nc < 0 || nr < 0 || nc >= cols || nr >= rows) {
                    continue;
                }
            }

            n += src[nr * cols + nc];
        }
    }

    int idx = row * cols + col;
    dst[idx] = next_state(src[idx], n);
}

__kernel void life_local(
    __global const uchar* src,
    __global uchar* dst,
    int rows,
    int cols,
    int wrap,
    __local uchar* tile
) {
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    int local_w = get_local_size(0);
    int local_h = get_local_size(1);

    int tile_w = local_w + 2;
    int tile_h = local_h + 2;

    int group_x = get_group_id(0) * local_w;
    int group_y = get_group_id(1) * local_h;

    int col = get_global_id(0);
    int row = get_global_id(1);

    int lid = local_y * local_w + local_x;
    int group_size = local_w * local_h;
    int tile_size = tile_w * tile_h;

    // Cargar cooperativamente tile + halo.
    for (int t = lid; t < tile_size; t += group_size) {
        int tx = t % tile_w;
        int ty = t / tile_w;

        int gc = group_x + tx - 1;
        int gr = group_y + ty - 1;

        uchar value = (uchar)0;

        if (wrap) {
            int wc = (gc % cols + cols) % cols;
            int wr = (gr % rows + rows) % rows;
            value = src[wr * cols + wc];
        } else {
            if (gc >= 0 && gr >= 0 && gc < cols && gr < rows) {
                value = src[gr * cols + gc];
            }
        }

        tile[t] = value;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (col >= cols || row >= rows) return;

    int center_x = local_x + 1;
    int center_y = local_y + 1;

    int n = 0;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            n += tile[(center_y + dy) * tile_w + (center_x + dx)];
        }
    }

    int idx = row * cols + col;
    uchar alive = tile[center_y * tile_w + center_x];
    dst[idx] = next_state(alive, n);
}