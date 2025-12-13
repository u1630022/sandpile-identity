/* Sandpiles identity compute and render
 * Many piles are processed in parallel using AVX2 and OpenMP.
 *
 *   $ cc -O3 -march=native -fopenmp sandpiles.c
 *   $ ./a.out >identity.ppm
 *
 *   $ cc -O3 -march=native -DANIMATE -DN=64 -DSCALE=16 sandpiles.c
 *   $ ./a.out | mpv --no-correct-pts --fps=20 -
 *
 * Ref: https://www.youtube.com/watch?v=1MtEUErz7Gg
 * Ref: https://codegolf.stackexchange.com/a/106990
 * Ref: https://nullprogram.com/blog/2017/11/03/
 * Ref: https://nullprogram.com/blog/2015/07/10/
 * Ref: https://www.youtube.com/watch?v=hBdJB-BzudU
 */
#include <stdio.h>
#include <immintrin.h>
#include <emmintrin.h>

#ifndef N
#  define N 512
#endif
#ifndef SCALE
#  define SCALE 1
#endif

#define TILE_WIDTH 64
#define TILE_HEIGHT 8

/* Color palette */
#define C0 0xff9200
#define C1 0xf53d52
#define C2 0xfce315
#define C3 0x44c5cb
#define CA4 0xff0000
#define CX 0x000000

__attribute__((aligned(32))) static unsigned char state[2+N][N];
__attribute__((aligned(32))) static unsigned char state_copy[2+N][N];

static void
render(void)
{
    static unsigned char buf[3L*N*SCALE*N*SCALE];
    static const long colors[] = {C0, C1, C2, C3};
    for (int y = 0; y < N*SCALE; y++) {
        for (int x = 0; x < N*SCALE; x++) {
            int v = state[1+y/SCALE][x/SCALE];
            long c = CX;
            if (v > 3) {
                c = CA4;
            } else if (v >= 0) {
                c = colors[v];
            }
            buf[y*3L*SCALE*N + x*3L + 0] = c >> 16;
            buf[y*3L*SCALE*N + x*3L + 1] = c >>  8;
            buf[y*3L*SCALE*N + x*3L + 2] = c >>  0;
        }
    }
    printf("P6\n%d %d\n255\n", N*SCALE, N*SCALE);
    fwrite(buf, sizeof(buf), 1, stdout);
}

// Source - https://stackoverflow.com/a
// Posted by sergfc, modified by community. See post 'Timeline' for change history
// Retrieved 2025-12-11, License - CC BY-SA 3.0
static __m256i
m256_srl8_1(__m256i i) {
    // suppose i is [32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
    //               16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    //[87654321,  3231302928272625,   2423222120191817,   161514131211109]
    __m256i srl64_q = _mm256_permute4x64_epi64(i, _MM_SHUFFLE(0,3,2,1));

    //[ 1, 0, 0, 0      13, 0, 0, 0       9, 0, 0, 0       5, 0, 0, 0]
    __m256i srl64_m = _mm256_slli_epi64(srl64_q, 7*8);
    //[ 0, 16, 15, 14,  0, 12, 11, 10,    0, 8, 7, 6,      0, 4, 3, 2]
    __m256i srl16_z = _mm256_srli_epi64(i, 8);

    __m256i srl64 = _mm256_and_si256(srl64_m, _mm256_set_epi64x(0, ~0, ~0, ~0));
    __m256i r = _mm256_or_si256(srl64, srl16_z);

    return r;
}

static __m256i
m256_sll8_1(__m256i i) {
    // suppose i is [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    //[12, 11, 10, 9,   8, 7, 6, 5,    4, 3, 2, 1,    16, 15, 14, 13]
    __m256i srl64_q = _mm256_permute4x64_epi64(i, _MM_SHUFFLE(2,1,0,3));

    //[0, 0, 0, 12,   0, 0, 0, 8,    0, 0, 0, 4,    0, 0, 0, 16]
    __m256i srl64_m = _mm256_srli_epi64(srl64_q, 7*8);
    //[15, 14, 13, 0  11, 10, 9, 0    7, 6, 5, 0,      3, 2, 1, 0]
    __m256i srl16_z = _mm256_slli_epi64(i, 1*8);

    __m256i srl64 = _mm256_and_si256(srl64_m, _mm256_set_epi64x(~0, ~0, ~0, 0));
    __m256i r = _mm256_or_si256(srl64, srl16_z);

    return r;
}

__attribute__((aligned(32))) unsigned char sum_buffer[32];

static int
m256_hadd_all(__m256i i) {
    int sum = 0;
    _mm256_store_si256((void *) &sum_buffer, i);
    for (int i = 0; i < 32; i++) {
        sum += sum_buffer[i];
    }
    return sum;
}

static void
stabilize(void)
{
    long cellsChecked = 0;
    long spills = 0;
    long localSpills = 0;
    long totalSpills = 0;
    do {
        spills = 0;
        for (int xx = 0; xx < 2*N / TILE_WIDTH - 1; xx++) {
            for (int yy = 0; yy < 2*N / TILE_HEIGHT - 1; yy++) {
                do {
                    localSpills = 0;
                    __m256i vspills = _mm256_set1_epi8(0);
                    int ystart = (yy * TILE_HEIGHT) / 2;
                    int yend = ystart + TILE_HEIGHT;
                    for (int y = ystart; y < yend; y++) {
                        int xspills = 0;
                        int xstart = (xx * TILE_WIDTH) / 2;
                        int xend = xstart + TILE_WIDTH;

                        for (int x = xstart; x < xend; x += 32) {
                            __m256i vv = _mm256_load_si256((void *)&state[1+y][x]);
                            __m256i vvabove = _mm256_load_si256((void *)&state[y][x]);
                            __m256i vvbelow = _mm256_load_si256((void *)&state[2+y][x]);
                            __m256i vinc = _mm256_srl_epi16(vv, _mm_set1_epi64x(2));
                            vinc = _mm256_and_si256(vinc, _mm256_set1_epi8(0x3F));

                            vspills = _mm256_add_epi8(vinc, vspills);

                            _mm256_store_si256((void *)&state[y][x], _mm256_add_epi8(vvabove, vinc));
                            _mm256_store_si256((void *)&state[2+y][x], _mm256_add_epi8(vvbelow, vinc));

                            __m256i vinc_left = m256_srl8_1(vinc);
                            __m256i vinc_right = m256_sll8_1(vinc);

                            __m256i vinc4 = _mm256_sll_epi16(vinc, _mm_set1_epi64x(2));
                            __m256i vv_new = _mm256_add_epi8(vinc_left,
                                    _mm256_add_epi8(vinc_right,
                                    _mm256_sub_epi8(vv, vinc4)));
                            _mm256_store_si256((void *)&state[1+y][x], vv_new);

                            // tails
                            if (x - 1 > 0) {
                                int left_spill = _mm256_extract_epi8(vinc, 0);
                                state[1+y][x-1] += left_spill;
                            }
                            if (x + 32 < N) {
                                int right_spill = _mm256_extract_epi8(vinc, 31);
                                state[1+y][x+32] += right_spill;
                            }
                        }
                    }
                    cellsChecked += TILE_HEIGHT * TILE_WIDTH;
                    localSpills = m256_hadd_all(vspills);
                    spills += localSpills;
                } while (localSpills > TILE_HEIGHT * TILE_WIDTH / 4);
            }
        }
        totalSpills += spills;
        // render();
    } while (spills > 0);
    // 8915347868
    // 3251067552
    fprintf(stderr, "Total Cells Checked: %ld\tTotal Spills: %ld\n", cellsChecked, totalSpills);
}

int
states_eq(void)
{
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            if (state[1+y][x] != state_copy[1+y][x]) {
                return 0;
            }
        }
    }
    return 1;
}

void
states_copy(void)
{
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            state_copy[1+y][x] = state[1+y][x];
        }
    }
}

void
subtraction_algo(void)
{
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            state[1+y][x] = 6;
        }
    }
    stabilize();
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            state[1+y][x] = 6 - state[1+y][x];
        }
    }
    stabilize();
}

void
exp_burning_algo(void)
{
    // init with the burning config
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            state[1+y][x] = 0;
        }
    }
    for (int x = 0; x < N; x++) {
        state[1][x] = 1;
        state[N][x] = 1;
    }
    for (int y = 1; y < N-1; y++) {
        state[1+y][0]   = 1;
        state[1+y][N-1] = 1;
    }
    state[1][0]   = 2;
    state[1][N-1] = 2;
    state[N][0]   = 2;
    state[N][N-1] = 2;

    do {
        states_copy();
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                state[1+y][x] *= 2;
            }
        }
        stabilize();
        // render();
    } while (!states_eq());
}

int
main(void)
{
    exp_burning_algo();
    render();
}
