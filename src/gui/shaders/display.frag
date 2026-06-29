#version 330 core

out vec4 fragColor;

// The board is a single-channel unsigned-integer texture (GL_R8UI): one texel per
// cell, value 0 (dead) or 1 (alive). usampler2D + texelFetch read it exactly,
// with no filtering (integer textures cannot be linearly filtered).
uniform usampler2D uBoard;

uniform vec2  uViewSize;  // framebuffer size in pixels
uniform float uScale;     // pixels per cell (zoom)
uniform vec2  uCenter;    // board cell shown at the screen centre (pan)
uniform ivec2 uBoardSize; // (cols, rows)
uniform int   uColorMode; // 0 = binary, 1 = neighbour count

// Read a cell, treating out-of-board coordinates as dead (matches bounded edges).
int cellAt(ivec2 c) {
    if (c.x < 0 || c.y < 0 || c.x >= uBoardSize.x || c.y >= uBoardSize.y) return 0;
    return int(texelFetch(uBoard, c, 0).r);
}

void main() {
    // gl_FragCoord origin is bottom-left; flip Y so board row 0 draws at the top.
    vec2 screen = gl_FragCoord.xy;
    screen.y = uViewSize.y - screen.y;

    // Screen pixel -> board cell, inverting the same transform the CPU uses for
    // mouse picking: cell = center + (pixelFromCentre) / pixelsPerCell.
    vec2 boardF = uCenter + (screen - 0.5 * uViewSize) / uScale;
    ivec2 cell = ivec2(floor(boardF));

    if (cell.x < 0 || cell.y < 0 || cell.x >= uBoardSize.x || cell.y >= uBoardSize.y) {
        fragColor = vec4(0.07, 0.07, 0.09, 1.0); // outside the board
        return;
    }

    int alive = cellAt(cell);

    if (uColorMode == 1) {
        // Colour by live-neighbour count (free: sampled straight from the texture).
        int n = cellAt(cell + ivec2(-1, -1)) + cellAt(cell + ivec2(0, -1)) + cellAt(cell + ivec2(1, -1))
              + cellAt(cell + ivec2(-1,  0))                                + cellAt(cell + ivec2(1,  0))
              + cellAt(cell + ivec2(-1,  1)) + cellAt(cell + ivec2(0,  1)) + cellAt(cell + ivec2(1,  1));
        float t = float(n) / 8.0;
        vec3 col = mix(vec3(0.0, 0.1, 0.3), vec3(1.0, 0.9, 0.2), t);
        col *= (alive == 1) ? 1.0 : 0.25; // dim dead cells, keep the field readable
        fragColor = vec4(col, 1.0);
        return;
    }

    // Binary: live cells bright, dead cells near-black.
    vec3 col = (alive == 1) ? vec3(0.85, 0.95, 0.85) : vec3(0.10, 0.10, 0.12);
    fragColor = vec4(col, 1.0);
}
