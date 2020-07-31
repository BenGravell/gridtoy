import numpy as np
import numpy.random as npr
import moderngl
import moderngl_window as mglw


class Example(mglw.WindowConfig):
    gl_version = (3, 3)
    resizable = True
    samples = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def run(cls):
        mglw.run_window_config(cls)


class Diffuse(Example):
    title = "Diffusion"
    width, height = 3440, 1440
    window_size = (width, height)
    aspect_ratio = width / height

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        width, height = self.window_size
        canvas = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).astype('f4')
        pixels = npr.rand(width, height).astype('f4')
        grid = np.dstack(np.mgrid[0:height, 0:width][::-1]).astype('i4')

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330

                in vec2 in_vert;
                out vec2 v_text;

                void main() {
                    v_text = in_vert;
                    gl_Position = vec4(in_vert * 2.0 - 1.0, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330

                uniform sampler2D Texture;

                in vec2 v_text;
                out vec4 f_color;

                void main() {
                    f_color = texture(Texture, v_text);
                }
            ''',
        )

        self.transform = self.ctx.program(
            vertex_shader='''
                #version 330

                uniform sampler2D Texture;
                uniform int Width;
                uniform int Height;

                in ivec2 in_text;
                out float out_vert;

                #define RATE 0.2
                #define NOISE_AMOUNT 0.1
                #define NOISE_INERTIA 0.95
                
                // A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm (pseudorandom number generator)
                uint hash1(uint x) {
                    x += (x << 10u);
                    x ^= (x >>  6u);
                    x += (x <<  3u);
                    x ^= (x >> 11u);
                    x += (x << 15u);
                    return x;
                }
                
                // Compound versions of the hashing algorithm
                uint hash2( uvec2 v ) { return hash1( v.x ^ hash1(v.y)                         ); }
                uint hash3( uvec3 v ) { return hash1( v.x ^ hash1(v.y) ^ hash1(v.z)             ); }
                uint hash4( uvec4 v ) { return hash1( v.x ^ hash1(v.y) ^ hash1(v.z) ^ hash1(v.w) ); }
                
                // Construct a float with half-open range [0:1] using low 23 bits
                // All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0
                float floatConstruct(uint m) {
                    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
                    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32
                
                    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
                    m |= ieeeOne;                          // Add fractional part to 1.0
                
                    float  f = uintBitsToFloat( m );       // Range [1:2]
                    return f - 1.0;                        // Range [0:1]
                }
                
                // Pseudo-random value in half-open range [0:1]
                float random1( float x ) { return floatConstruct(hash1(floatBitsToUint(x))); }
                float random2( vec2  v ) { return floatConstruct(hash2(floatBitsToUint(v))); }
                float random3( vec3  v ) { return floatConstruct(hash3(floatBitsToUint(v))); }
                float random4( vec4  v ) { return floatConstruct(hash4(floatBitsToUint(v))); }

                // Get the cell value at x, y using wrap-around if x, y out of bounds
                float cell(int x, int y) {
                    return texelFetch(Texture, ivec2((x + Width) % Width, (y + Height) % Height), 0).r;
                }

                void main() {
                    float center = cell(in_text.x, in_text.y);
                    float noise;
                    float neighbors = 0;
                    neighbors += cell(in_text.x - 1, in_text.y);
                    neighbors += cell(in_text.x + 1, in_text.y);
                    neighbors += cell(in_text.x, in_text.y - 1);
                    neighbors += cell(in_text.x, in_text.y + 1);
                    
                    vec3 noise_inputs = vec3(in_text.x, in_text.y, int(center*1000000));
                    noise = NOISE_AMOUNT*(random3(noise_inputs)-0.5);
                    out_vert = ((1.0-4.0*RATE)*center) + (RATE*neighbors) + (noise);
                }
            ''',
            varyings=['out_vert']
        )

        self.transform['Width'].value = width
        self.transform['Height'].value = height

        self.texture = self.ctx.texture((width, height), 1, pixels.tobytes(), dtype='f4')
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.texture.swizzle = 'RRR1'
        self.texture.use()

        self.vbo = self.ctx.buffer(canvas)
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert')

        self.text = self.ctx.buffer(grid)
        self.tao = self.ctx.simple_vertex_array(self.transform, self.text, 'in_text')
        self.pbo = self.ctx.buffer(reserve=pixels.nbytes)

    def render(self, time, frame_time):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.tao.transform(self.pbo)
        self.texture.write(self.pbo)

        self.vao.render(moderngl.TRIANGLE_STRIP)


if __name__ == '__main__':
    Diffuse.run()
