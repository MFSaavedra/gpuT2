#version 330 core

out vec4 fragColor;

// The board is a single-channel unsigned-integer texture (GL_R8UI): one texel per
// cell, value 0 (dead) or 1 (alive). usampler2D + texelFetch read it exactly,
// with no filtering (integer textures cannot be linearly filtered).
uniform usampler2D uBoard;
// Per-cell "generations survived" (GL_R8UI), maintained host-side; 0 = dead. Only
// sampled in the age-heatmap mode, but always bound to keep the sampler valid.
uniform usampler2D uAge;

uniform vec2  uViewSize;  // framebuffer size in pixels
uniform float uScale;     // pixels per cell (zoom)
uniform vec2  uCenter;    // board cell shown at the screen centre (pan)
uniform ivec2 uBoardSize; // (cols, rows)
uniform int   uColorMode; // 0 = binary, 1 = neighbour count, 2 = age heatmap
uniform int   uPalette;   // 0 phosphor, 1 amber, 2 grayscale (default), 3 magma, 4 ice
uniform float uAgeMax;    // age value mapped to the hot end of the ramp

const vec3 DEAD    = vec3(0.09, 0.09, 0.11); // dead cell inside the board
const vec3 OUTSIDE = vec3(0.07, 0.07, 0.09); // outside the board

// Read a cell, treating out-of-board coordinates as dead (matches bounded edges).
int cellAt(ivec2 c) {
    if (c.x < 0 || c.y < 0 || c.x >= uBoardSize.x || c.y >= uBoardSize.y) return 0;
    return int(texelFetch(uBoard, c, 0).r);
}

// Map a scalar t in [0,1] to an RGB ramp. Every palette shares the convention
// t=0 -> cool/dark end, t=1 -> hot/bright end, so the modes below can reuse them.
vec3 palette(int p, float t) {
    t = clamp(t, 0.0, 1.0);
    if (p == 1) {        // amber CRT
        return mix(vec3(0.06, 0.02, 0.0), vec3(1.0, 0.72, 0.18), pow(t, 0.75));
    } else if (p == 2) { // grayscale
        return mix(vec3(0.08), vec3(0.97), t);
    } else if (p == 3) { // magma-like (three stops)
        vec3 a = vec3(0.02, 0.01, 0.09);
        vec3 b = vec3(0.72, 0.19, 0.33);
        vec3 c = vec3(0.99, 0.92, 0.68);
        return t < 0.5 ? mix(a, b, t * 2.0) : mix(b, c, (t - 0.5) * 2.0);
    } else if (p == 4) { // ice (blue -> yellow)
        return mix(vec3(0.0, 0.12, 0.32), vec3(1.0, 0.92, 0.22), t);
    }
    return mix(vec3(0.0, 0.06, 0.02), vec3(0.55, 1.0, 0.6), t); // 0: phosphor (green)
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
        fragColor = vec4(OUTSIDE, 1.0); // outside the board
        return;
    }

    int alive = cellAt(cell);
    vec3 col;

    if (uColorMode == 2) {
        // Age heatmap: colour a live cell by how many generations it has survived.
        uint a = texelFetch(uAge, cell, 0).r;
        col = (a == 0u) ? DEAD : palette(uPalette, float(a) / uAgeMax);
    } else if (uColorMode == 1) {
        // Colour by live-neighbour count (free: sampled straight from the texture).
        int n = cellAt(cell + ivec2(-1, -1)) + cellAt(cell + ivec2(0, -1)) + cellAt(cell + ivec2(1, -1))
              + cellAt(cell + ivec2(-1,  0))                                + cellAt(cell + ivec2(1,  0))
              + cellAt(cell + ivec2(-1,  1)) + cellAt(cell + ivec2(0,  1)) + cellAt(cell + ivec2(1,  1));
        col = palette(uPalette, float(n) / 8.0);
        col *= (alive == 1) ? 1.0 : 0.25; // dim dead cells, keep the field readable
    } else {
        // Binary: dead cells dark, live cells the palette's hot colour.
        col = (alive == 1) ? palette(uPalette, 1.0) : DEAD;
    }

    // Grid lines: once zoomed in far enough, darken a ~1px border around each cell
    // so individual cells are legible. Cheap: derive the border from fract(boardF).
    if (uScale >= 4.0) {
        vec2 f = fract(boardF);
        vec2 dpix = min(f, 1.0 - f) * uScale; // pixels to the nearest cell edge
        float edge = min(dpix.x, dpix.y);
        col *= mix(0.35, 1.0, smoothstep(0.0, 1.0, edge));
    }

    fragColor = vec4(col, 1.0);
}
