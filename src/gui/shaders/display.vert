#version 330 core

// Fullscreen triangle: no vertex buffer needed, positions come from gl_VertexID.
// A single oversized triangle covers the screen with one draw (3 vertices).
void main() {
    vec2 verts[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0);
}
