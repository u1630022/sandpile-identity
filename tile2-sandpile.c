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
#include <assert.h>
#include <limits.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <locale.h>
#include <math.h>
#include <stdio.h>
#include <omp.h>

#ifndef SCALE
#  define SCALE 4
#endif

#define TILE_WIDTH 64
#define TILE_HEIGHT 8

#define LEFT  0
#define RIGHT 1

/* Color palette */
#define C0 0x111111
#define C1 0x444444
#define C2 0x777777
#define C3 0xaaaaaa
#define CA4 0xff0000
#define CX 0x0000ff

static const long ALIGNMENT = 64;

// Inferno palette
static const long inferno[254] = {
    0x000006, 0x010007, 0x010109, 0x01010b,
    0x02010e, 0x020210, 0x030212, 0x040314,
    0x040316, 0x050418, 0x06041b, 0x07051d,
    0x08061f, 0x090621, 0x0a0724, 0x0c0726,
    0x0d0828, 0x0e082b, 0x0f092d, 0x10092f,
    0x120a32, 0x130a34, 0x140b37, 0x160b39,
    0x170b3b, 0x190b3e, 0x1a0c40, 0x1c0c43,
    0x1d0c45, 0x1f0c48, 0x210c4a, 0x220b4c,
    0x240b4e, 0x260b51, 0x270b53, 0x290b55,
    0x2b0a57, 0x2d0a59, 0x2e0a5a, 0x300a5c,
    0x32095e, 0x34095f, 0x360960, 0x370962,
    0x390963, 0x3b0964, 0x3c0965, 0x3e0966,
    0x400967, 0x420968, 0x430a68, 0x450a69,
    0x470a6a, 0x480b6a, 0x4a0b6b, 0x4c0c6b,
    0x4d0c6c, 0x4f0d6c, 0x500d6c, 0x520e6d,
    0x540e6d, 0x550f6d, 0x570f6d, 0x59106e,
    0x5a116e, 0x5c116e, 0x5d126e, 0x5f126e,
    0x61136e, 0x62146e, 0x64146e, 0x65156e,
    0x67156e, 0x68166e, 0x6a176e, 0x6c176e,
    0x6d186e, 0x6f186e, 0x70196e, 0x721a6e,
    0x741a6e, 0x751b6e, 0x771b6d, 0x781c6d,
    0x7a1c6d, 0x7c1d6d, 0x7d1d6c, 0x7f1e6c,
    0x801f6c, 0x821f6c, 0x84206b, 0x85206b,
    0x87216b, 0x88216a, 0x8a226a, 0x8c2369,
    0x8d2369, 0x8f2468, 0x902468, 0x922568,
    0x942567, 0x952667, 0x972766, 0x982765,
    0x9a2865, 0x9b2864, 0x9d2964, 0x9f2a63,
    0xa02a62, 0xa22b62, 0xa32b61, 0xa52c60,
    0xa72d60, 0xa82d5f, 0xaa2e5e, 0xab2f5d,
    0xad2f5d, 0xae305c, 0xb0315b, 0xb1315a,
    0xb33259, 0xb43359, 0xb63458, 0xb73457,
    0xb93556, 0xba3655, 0xbc3754, 0xbd3853,
    0xbf3852, 0xc03951, 0xc23a50, 0xc33b4f,
    0xc53c4e, 0xc63d4d, 0xc73e4c, 0xc93f4b,
    0xca404a, 0xcb4149, 0xcd4248, 0xce4347,
    0xcf4446, 0xd14545, 0xd24644, 0xd34743,
    0xd54841, 0xd64940, 0xd74a3f, 0xd84c3e,
    0xd94d3d, 0xdb4e3c, 0xdc4f3b, 0xdd5139,
    0xde5238, 0xdf5337, 0xe05536, 0xe15635,
    0xe25733, 0xe35932, 0xe45a31, 0xe55b30,
    0xe65d2f, 0xe75e2d, 0xe8602c, 0xe9612b,
    0xea632a, 0xeb6428, 0xec6627, 0xed6726,
    0xed6925, 0xee6a23, 0xef6c22, 0xf06e21,
    0xf16f20, 0xf1711e, 0xf2721d, 0xf3741c,
    0xf3761a, 0xf47719, 0xf47918, 0xf57b16,
    0xf67d15, 0xf67e14, 0xf78012, 0xf78211,
    0xf88410, 0xf8850e, 0xf8870d, 0xf9890c,
    0xf98b0b, 0xfa8d09, 0xfa8e08, 0xfa9008,
    0xfb9207, 0xfb9406, 0xfb9606, 0xfb9806,
    0xfc9906, 0xfc9b06, 0xfc9d06, 0xfc9f07,
    0xfca107, 0xfca308, 0xfca50a, 0xfca70b,
    0xfca90d, 0xfcaa0e, 0xfcac10, 0xfcae12,
    0xfcb014, 0xfcb216, 0xfcb418, 0xfcb61a,
    0xfcb81c, 0xfcba1e, 0xfbbc21, 0xfbbe23,
    0xfbc025, 0xfbc228, 0xfac42a, 0xfac62d,
    0xfac82f, 0xf9ca32, 0xf9cc34, 0xf9ce37,
    0xf8d03a, 0xf8d23d, 0xf7d43f, 0xf7d642,
    0xf6d845, 0xf6d949, 0xf5db4c, 0xf5dd4f,
    0xf4df52, 0xf4e156, 0xf4e359, 0xf3e55d,
    0xf3e761, 0xf2e965, 0xf2ea69, 0xf2ec6d,
    0xf2ee71, 0xf2ef75, 0xf2f179, 0xf2f37d,
    0xf3f482, 0xf3f586, 0xf4f78a, 0xf5f88e,
    0xf6f992, 0xf7fb96, 0xf8fc9a, 0xf9fd9d,
    0xfbfea1, 0xfdffa5
};

#define zero_messages_type(t) \
void \
zero_messages(int N, \
              t messages[N / TILE_WIDTH][2][N]) \
{ \
    for (int xx = 0; xx < N / TILE_WIDTH; xx++) { \
        for (int dir = LEFT; dir <= RIGHT; dir++) { \
            for (int i = 0; i < N; i++) { \
                messages[xx][dir][i] = 0; \
            } \
        } \
    } \
}

static void
render_colour(int N,
       unsigned char state[2+N][N])
{
    unsigned char buf[3L*N*SCALE*N*SCALE];
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

#define render(N, grid) unsigned char buf[3L*N*SCALE*N*SCALE]; \
                        double min = LONG_MAX; \
                        double max = LONG_MIN; \
                        for (int y = 0; y < N; y++) { \
                            for (int x = 0; x < N; x++) { \
                                long v = (long) grid[1+y][x]; \
                                if (v < min) { \
                                    min = v; \
                                } \
                                if (v > max) { \
                                    max = v; \
                                } \
                            } \
                        } \
                        for (int y = 0; y < N*SCALE; y++) { \
                            for (int x = 0; x < N*SCALE; x++) { \
                                long v = (long) grid[1+y/SCALE][x/SCALE]; \
                                int idx = (int) (253.0 * (v - min) / (max - min)); \
                                assert(idx >= 0); \
                                assert(idx < 254); \
                                long c = inferno[idx]; \
                                buf[y*3L*SCALE*N + x*3L + 0] = c >> 16; \
                                buf[y*3L*SCALE*N + x*3L + 1] = c >>  8; \
                                buf[y*3L*SCALE*N + x*3L + 2] = c >>  0; \
                            } \
                        } \
                        printf("P6\n%d %d\n255\n", N*SCALE, N*SCALE); \
                        fwrite(buf, sizeof(buf), 1, stdout);

void
render_d(int N, int M, double grid[N][M])
{
    unsigned char buf[3L*N*SCALE*N*SCALE];
    double min = 10000000.0;
    double max =-10000000.0;
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < M; x++) {
            double v = grid[y][x];
            if (v < min) {
                min = v;
            }
            if (v > max) {
                max = v;
            }
        }
    }
    for (int y = 0; y < N*SCALE; y++) {
        for (int x = 0; x < M*SCALE; x++) {
            double v = grid[y/SCALE][x/SCALE];
            int idx = (int) (253.0 * (v - min) / (max - min));
            assert(idx >= 0);
            assert(idx < 254);
            long c = inferno[idx];
            buf[y*3L*SCALE*N + x*3L + 0] = c >> 16;
            buf[y*3L*SCALE*N + x*3L + 1] = c >>  8;
            buf[y*3L*SCALE*N + x*3L + 2] = c >>  0;
        }
    }
    printf("P6\n%d %d\n255\n", N*SCALE, N*SCALE);
    fwrite(buf, sizeof(buf), 1, stdout);
}

void
render_i(int N, int grid[N][N])
{
    unsigned char buf[3L*(N-2)*SCALE*(N-2)*SCALE];
    double min = LONG_MAX;
    double max = LONG_MIN;
    for (int y = 1; y < N-1; y++) {
        for (int x = 1; x < N-1; x++) {
            double v = grid[y][x];
            if (v < min) {
                min = v;
            }
            if (v > max) {
                max = v;
            }
        }
    }
    for (int y = 0; y < (N-2)*SCALE; y++) {
        for (int x = 0; x < (N-2)*SCALE; x++) {
            int v = grid[1+y/SCALE][1+x/SCALE];
            int idx = (int) (253.0 * (v - min) / (max - min));
            assert(idx >= 0);
            assert(idx < 254);
            long c = inferno[idx];
            buf[y*3L*SCALE*(N-2) + x*3L + 0] = c >> 16;
            buf[y*3L*SCALE*(N-2) + x*3L + 1] = c >>  8;
            buf[y*3L*SCALE*(N-2) + x*3L + 2] = c >>  0;
        }
    }
    printf("P6\n%d %d\n255\n", (N-2)*SCALE, (N-2)*SCALE);
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
m256_srl32_1(__m256i i) {
    // suppose i is [8, 7, 6, 5, 4, 3, 2, 1]

    //[2 1   8 7     6 5     2 1]
    __m256i srl64_q = _mm256_permute4x64_epi64(i, _MM_SHUFFLE(0,3,2,1));

    //[1 0   7 0     5 0     3 0]
    __m256i srl64_m = _mm256_slli_epi64(srl64_q, 4*8);
    //[0 8   0 6     0 4     0 2]
    __m256i srl16_z = _mm256_srli_epi64(i, 4*8);

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

static __m256i
m256_sll32_1(__m256i i) {
    // suppose i is [8, 7, 6, 5, 4, 3, 2, 1]

    //[6 5    4 3    2 1    8 7]
    __m256i srl64_q = _mm256_permute4x64_epi64(i, _MM_SHUFFLE(2,1,0,3));

    //[0 6    0 4    0 2    0 8]
    __m256i srl64_m = _mm256_srli_epi64(srl64_q, 4*8);
    //[7 0    5 0    3 0    1 0]
    __m256i srl16_z = _mm256_slli_epi64(i, 4*8);

    __m256i srl64 = _mm256_and_si256(srl64_m, _mm256_set_epi64x(~0, ~0, ~0, 0));
    __m256i r = _mm256_or_si256(srl64, srl16_z);

    return r;
}

static int
m256_hadd_all(__m256i i) {
    int sum = 0;
    __attribute__((aligned(32))) unsigned char sum_buffer[32];
    _mm256_store_si256((void *) &sum_buffer, i);
    for (int i = 0; i < 32; i++) {
        sum += sum_buffer[i];
    }
    return sum;
}

// Assume surrounded by buffer
void
stabilize_generic(int N,
                  int M,
                  int state[N][M])
{
    long spills = 0;
    long totalSpills = 0;
    do {
        spills = 0;
        for (int y = 1; y < N - 1; y++) {
            for (int x = 1; x < M - 1; x++) {
                if (state[y][x] >= 4) {
                    state[y][x] -= 4;
                    state[y+1][x] += 1;
                    state[y-1][x] += 1;
                    state[y][x+1] += 1;
                    state[y][x-1] += 1;
                    spills++;
                }
            }
        }
        totalSpills += spills;
        // render_i(N, state);
    } while (spills > 0);
    // fprintf(stderr, "Total Spills: %ld", totalSpills);
}

static void
stabilize_small(int N,
                unsigned char state[2+N][N],
                unsigned char messages[N / TILE_WIDTH][2][N])
{
    long tileSpills = 0;
    long threadSpills = 0;
    long spills = 0;
    long totalSpills = 0;
    do {
        spills = 0;

        #pragma omp parallel for private(tileSpills, threadSpills) schedule(static, 1)
        for (int xx = 0; xx < N / TILE_WIDTH; xx++) {

            threadSpills = 0;

            // import from messages
            for (int y = 0; y < N; y++) {
                int tmp = 0;

                #pragma omp atomic capture
                {
                    tmp = messages[xx][LEFT][y];
                    messages[xx][LEFT][y] = 0;
                }
                state[1+y][xx * TILE_WIDTH] += tmp;

                #pragma omp atomic capture
                {
                    tmp = messages[xx][RIGHT][y];
                    messages[xx][RIGHT][y] = 0;
                }
                state[1+y][(xx + 1) * TILE_WIDTH - 1] += tmp;
            }

            for (int yy = 0; yy < 2*N / TILE_HEIGHT - 1; yy++) {
                do {
                    tileSpills = 0;
                    __m256i vspills = _mm256_set1_epi8(0);
                    int ystart = (yy * TILE_HEIGHT) / 2;
                    int yend = ystart + TILE_HEIGHT;
                    for (int y = ystart; y < yend; y++) {
                        int xspills = 0;
                        int xstart = (xx * TILE_WIDTH);
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
                            if (x > xstart) {
                                int left_spill = _mm256_extract_epi8(vinc, 0);
                                state[1+y][x - 1] += left_spill;
                            } else if (xx > 0) {
                                int left_spill = _mm256_extract_epi8(vinc, 0);

                                #pragma omp atomic update
                                messages[xx - 1][RIGHT][y] += left_spill;
                            }
                            if (x < xend - 32) {
                                int right_spill = _mm256_extract_epi8(vinc, 31);
                                state[1+y][x+32] += right_spill;
                            } else if (xx < N / TILE_WIDTH - 1) {
                                int right_spill = _mm256_extract_epi8(vinc, 31);

                                #pragma omp atomic update
                                messages[xx + 1][LEFT][y] += right_spill;
                            }
                        }
                    }
                    tileSpills = m256_hadd_all(vspills);
                    threadSpills += tileSpills;
                } while (tileSpills > TILE_HEIGHT * TILE_WIDTH / 2);
            }

            #pragma omp atomic update
            spills += threadSpills;
        }

        assert(spills >= 0);
        totalSpills += spills;
        // render();
    } while (spills > 0);

    for (int xx = 1; xx < N / TILE_WIDTH - 1; xx++) {
        for (int dir = LEFT; dir <= RIGHT; dir++) {
            for (int i = 0; i < N; i++) {
                assert(messages[xx][dir][i] == 0);
            }
        }
    }
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            assert(state[1+y][x] < 4);
            assert(state[1+y][x] >= 0);
        }
    }

    // 8915347868
    // 3251067552
    fprintf(stderr, "Total Spills: %'ld\n", totalSpills);
}

static void
stabilize_big(int N,
              int state[2+N][N],
              int messages[N / TILE_WIDTH][2][N])
{
    long tileSpills = 0;
    long threadSpills = 0;
    long spills = 0;
    long totalSpills = 0;
    do {
        spills = 0;

        #pragma omp parallel for private(tileSpills, threadSpills) schedule(static, 1)
        for (int xx = 0; xx < N / TILE_WIDTH; xx++) {

            threadSpills = 0;

            // import from messages
            for (int y = 0; y < N; y++) {
                int tmp = 0;

                #pragma omp atomic capture
                {
                    tmp = messages[xx][LEFT][y];
                    messages[xx][LEFT][y] = 0;
                }
                state[1+y][xx * TILE_WIDTH] += tmp;

                #pragma omp atomic capture
                {
                    tmp = messages[xx][RIGHT][y];
                    messages[xx][RIGHT][y] = 0;
                }
                state[1+y][(xx + 1) * TILE_WIDTH - 1] += tmp;
            }

            for (int yy = 0; yy < 2*N / TILE_HEIGHT - 1; yy++) {
                do {
                    tileSpills = 0;
                    __m256i vspills = _mm256_set1_epi8(0);
                    int ystart = (yy * TILE_HEIGHT) / 2;
                    int yend = ystart + TILE_HEIGHT;
                    for (int y = ystart; y < yend; y++) {
                        int xspills = 0;
                        int xstart = (xx * TILE_WIDTH);
                        int xend = xstart + TILE_WIDTH;

                        for (int x = xstart; x < xend; x += 8) {
                            __m256i vv = _mm256_load_si256((void *)&state[1+y][x]);
                            __m256i vvabove = _mm256_load_si256((void *)&state[y][x]);
                            __m256i vvbelow = _mm256_load_si256((void *)&state[2+y][x]);
                            __m256i vinc = _mm256_srl_epi16(vv, _mm_set1_epi64x(2));
                            vinc = _mm256_and_si256(vinc, _mm256_set1_epi8(0x3F));

                            vspills = _mm256_add_epi8(vinc, vspills);

                            _mm256_store_si256((void *)&state[y][x], _mm256_add_epi8(vvabove, vinc));
                            _mm256_store_si256((void *)&state[2+y][x], _mm256_add_epi8(vvbelow, vinc));

                            __m256i vinc_left = m256_srl32_1(vinc);
                            __m256i vinc_right = m256_sll32_1(vinc);

                            __m256i vinc4 = _mm256_sll_epi16(vinc, _mm_set1_epi64x(2));
                            __m256i vv_new = _mm256_add_epi8(vinc_left,
                                    _mm256_add_epi8(vinc_right,
                                    _mm256_sub_epi8(vv, vinc4)));
                            _mm256_store_si256((void *)&state[1+y][x], vv_new);

                            // tails
                            if (x > xstart) {
                                int left_spill = _mm256_extract_epi32(vinc, 0);
                                state[1+y][x - 1] += left_spill;
                            } else if (xx > 0) {
                                int left_spill = _mm256_extract_epi32(vinc, 0);

                                #pragma omp atomic update
                                messages[xx - 1][RIGHT][y] += left_spill;
                            }
                            if (x < xend - 8) {
                                int right_spill = _mm256_extract_epi32(vinc, 7);
                                state[1+y][x+8] += right_spill;
                            } else if (xx < N / TILE_WIDTH - 1) {
                                int right_spill = _mm256_extract_epi32(vinc, 7);

                                #pragma omp atomic update
                                messages[xx + 1][LEFT][y] += right_spill;
                            }
                        }
                    }
                    tileSpills = m256_hadd_all(vspills);
                    threadSpills += tileSpills;
                } while (tileSpills > TILE_HEIGHT * TILE_WIDTH / 2);
            }

            #pragma omp atomic update
            spills += threadSpills;
        }

        assert(spills >= 0);
        totalSpills += spills;
        // render();
    } while (spills > 0);

    for (int xx = 1; xx < N / TILE_WIDTH - 1; xx++) {
        for (int dir = LEFT; dir <= RIGHT; dir++) {
            for (int i = 0; i < N; i++) {
                assert(messages[xx][dir][i] == 0);
            }
        }
    }
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            assert(state[1+y][x] < 4);
            assert(state[1+y][x] >= 0);
        }
    }

    // 8915347868
    // 3251067552
    fprintf(stderr, "Total Spills: %'ld\n", totalSpills);
}

#define stabilize(dummy, N, state, messages) _Generic((dummy), \
        unsigned char: stabilize_small, \
        int: stabilize_big)(N, state, messages)


#define states_eq_type(t) \
int \
states_eq(int N, \
          t state[2+N][N], \
          t state_copy[2+N][N]) \
{ \
    for (int y = 0; y < N; y++) { \
        for (int x = 0; x < N; x++) { \
            if (state[1+y][x] != state_copy[1+y][x]) { \
                return 0; \
            } \
        } \
    } \
    return 1; \
}

#define states_copy_type(t) \
void \
states_copy(int N, \
            t state[2+N][N], \
            t state_copy[2+N][N]) \
{ \
    for (int y = 0; y < N; y++) { \
        for (int x = 0; x < N; x++) { \
            state_copy[1+y][x] = state[1+y][x]; \
        } \
    } \
}

#define add_burning_config_type(t) \
void \
add_burning_config(int N, \
                   t state[2+N][N]) \
{ \
    for (int x = 0; x < N; x++) { \
        state[1][x] += 1; \
        state[N][x] += 1; \
    } \
    for (int y = 1; y < N-1; y++) { \
        state[1+y][0]   += 1; \
        state[1+y][N-1] += 1; \
    } \
    state[1][0]   += 1; \
    state[1][N-1] += 1; \
    state[N][0]   += 1; \
    state[N][N-1] += 1; \
}

#define is_identity_type(t) \
int \
is_identity(int N, \
            t state[2+N][N], \
            t state_copy[2+N][N], \
            t messages[N / TILE_WIDTH][2][N]) \
{ \
    states_copy(N, state, state_copy); \
    add_burning_config(N, state); \
    t dummy; \
    stabilize(dummy, N, state, messages); \
    return states_eq(N, state, state_copy); \
}

#define subtraction_algo_type(t) \
void \
subtraction_algo(int N, \
                 t state[2+N][N], \
                 t messages[N / TILE_WIDTH][2][N]) \
{ \
    for (int y = 0; y < N; y++) { \
        for (int x = 0; x < N; x++) { \
            state[1+y][x] = 6; \
        } \
    } \
    t dummy; \
    stabilize(dummy, N, state, messages); \
    for (int y = 0; y < N; y++) { \
        for (int x = 0; x < N; x++) { \
            state[1+y][x] = 6 - state[1+y][x]; \
        } \
    } \
    stabilize(dummy, N, state, messages); \
}

#define exp_burning_algo_type(t) \
void \
exp_burning_algo(int N, \
                 t state[2+N][N], \
                 t state_copy[2+N][N], \
                 t messages[N / TILE_WIDTH][2][N]) \
{ \
    for (int y = 0; y < N; y++) { \
        for (int x = 0; x < N; x++) { \
            state[1+y][x] = 0; \
        } \
    } \
    zero_messages(N, messages); \
    add_burning_config(N, state); \
    t dummy; \
    do { \
        for (int y = 0; y < N; y++) { \
            for (int x = 0; x < N; x++) { \
                state[1+y][x] *= 2; \
            } \
        } \
        stabilize(dummy, N, state, messages); \
    } while (!is_identity(N, state, state_copy, messages)); \
}

// TODO:
// We want to approximate the odometer of the identity sandpile
// We do this by considering the equations Au = f (A: reduced laplacian, u: odometer, f: sandpile)
// We do this by upscaling a smaller identity by 2 and setting this as f and solving for u.
// u is cast to an integer array by rounding and true sandpile f is computed via laplacian.
// stabilize true sandpile f and fire the burning config until convergence.
//
// Functions needed:
// 1. Direct solver for small identity sandpiles (check)
// 2. Identity upscaler (Need to consider case of non-power of 2 upscale)
// 3. Rounding and converging of f.
// 4. FMG odometer calculator
// 4.1. Reduction operator
// 4.2. Interpolation operator
// 4.3. Relaxation operator (jacobi or gauss-seidel)
// 4.4. Recursive FMG scheme

// d2u/dx2 + d2u/dy2 = f
// d/dx [[u(x+h) - u(x)] / h] + d/dx [[u(y+h) - u(y)] / h] = f
// [d/dx u(x+h) - d/dx u(x)] / h + [d/dx u(y+h) - d/dx u(y)] / h = f
// [[u(x+2h) - u(x+h)] / h - [u(x+h) - u(x)] / h] / h + [[u(y+2h) - u(y+h)] / h - [u(y+h) - u(y)] / h] / h = f
// [u(x+2h) - 2*u(x+h) + u(x)] / h2 + [u(y+2h) - 2*u(y+h) + u(y)] / h2 = f
// [u_1,y - 2*u_0,y + u_-1,y + u_x,1 - 2*u_x,0 + u_x,-1] / h2 = f
// [u_x+1,y + u_x-1,y + u_x,y+1 + u_x,y-1 - 4*u_x,y] / h2 = f
// - 4*u_x,y = h2 * f - u_x+1,y - u_x-1,y - u_x,y+1 - u_x,y-1
// u_x,y = 0.25 * (u_x+1,y + u_x-1,y + u_x,y+1 + u_x,y-1 - h2 * f)

int FULL_N;
int FULL_M;

void
gauss_seidel(int n, int m, double u[n][m], double f[n][m], int iter)
{
    double h2 = ((FULL_N-1)*(FULL_M-1)) * 1.0 / ((n-1)*(m-1));
    for (int it = 0; it < iter; it++) {
        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < m-1; j++) {
                u[i][j] = 0.25 * (  u[i-1][j]
                                  + u[i+1][j]
                                  + u[i][j-1]
                                  + u[i][j+1]
                                  - h2 * f[i][j]);
            }
        }
        // Reverse direction
        for (int i = n - 2; i >= 1; i--) {
            for (int j = m - 2; j >= 1; j--) {
                u[i][j] = 0.25 * (  u[i-1][j]
                                  + u[i+1][j]
                                  + u[i][j-1]
                                  + u[i][j+1]
                                  - h2 * f[i][j]);
            }
        }
    }
}

// Full-weighting
void
inject(int n_fine,
       int m_fine,
       double fine[n_fine][m_fine],
       double coarse[n_fine / 2 + 1][m_fine / 2 + 1])
{
    int n_coarse = n_fine / 2 + 1;
    int m_coarse = m_fine / 2 + 1;
    for (int i = 1; i < n_coarse-1; i++) {
        for (int j = 1; j < m_coarse-1; j++) {
            coarse[i][j] =   0.25 * fine[2*i-1][2*j-1]
                           + 0.125 * (  fine[2*i-1][2*j] + fine[2*i-1][2*j+1]
                                      + fine[2*i][2*j-1] + fine[2*i+1][2*j-1])
                           + 0.0625 * (  fine[2*i][2*j]   + fine[2*i][2*j+1]
                                       + fine[2*i+1][2*j] + fine[2*i+1][2*j+1]);
        }
    }
}

// TODO: need to account for non power of two grids
void
interpolate(int n_coarse,
            int m_coarse,
            int n_fine,
            int m_fine,
            double coarse[n_coarse][m_coarse],
            double fine[n_fine][m_fine])
{
    double scale_x = (double)(n_coarse - 1) / (n_fine - 1);
    double scale_y = (double)(m_coarse - 1) / (m_fine - 1);

    for (int i = 1; i < n_fine - 1; i++) {
        for (int j = 1; j < m_fine - 1; j++) {
            double x = i * scale_x;
            double y = j * scale_y;

            int x0 = (int)floor(x);
            int y0 = (int)floor(y);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            double dx = x - x0;
            double dy = y - y0;

            // Bilinear interpolation
            fine[i][j] =
                (1 - dx) * (1 - dy) * coarse[x0][y0] +
                dx * (1 - dy) * coarse[x1][y0] +
                (1 - dx) * dy * coarse[x0][y1] +
                dx * dy * coarse[x1][y1];
        }
    }
}


void
residual(int n,
         int m,
         double u[n][m],
         double f[n][m],
         double res[n][m])
{
    double h2 = ((FULL_N-1)*(FULL_M-1)) * 1.0 / ((n-1)*(m-1));
    for (int i = 1; i < n-1; i++) {
        for (int j = 1; j < m-1; j++) {
            res[i][j] = - f[i][j] + (    u[i-1][j]
                                     +   u[i+1][j]
                                     +   u[i][j-1]
                                     +   u[i][j+1]
                                     - 4*u[i][j]  ) / h2;
        }
    }
}

double
max_norm(int n,
         int m,
         double arr[n][m])
{
    double max_norm = arr[0][0];
    double r;
    for (int i = 1; i < n-1; i++) {
        for (int j = 1; j < m-1; j++) {
            r = arr[i][j];
            if (max_norm < sqrt(r*r)) {
                max_norm = sqrt(r*r);
            }
        }
    }
    return max_norm;
}

double
dl2_norm(int n,
         int m,
         double arr[n][m])
{
    double norm = 0;
    double h2 = ((FULL_N-1)*(FULL_M-1)) * 1.0 / ((n-1)*(m-1));
    for (int i = 1; i < n-1; i++) {
        for (int j = 1; j < m-1; j++) {
            norm += arr[i][j] * arr[i][j];
        }
    }
    norm *= h2;
    norm = sqrt(norm);
    return norm;
}

void
vcycle(int n,
       int m,
       double u[n][m],
       double f[n][m],
       int depth)
{
    double res[n][m];
    double u_correction[n][m];
    double coarse_u[n/2 + 1][m/2 + 1];
    double coarse_f[n/2 + 1][m/2 + 1];

    if (n <= 3 || m <= 3) {
        assert(n == 3);
        assert(m == 3);

        double h2 = ((FULL_N-1)*(FULL_M-1)) * 1.0 / ((n-1)*(m-1));
        u[1][1] = - 0.25 * f[1][1] * h2;
        residual(n, m, u, f, res);
        return;
    }


    gauss_seidel(n, m, u, f, 2);
    residual(n, m, u, f, res);

    inject(n, m, res, coarse_f);
    for (int i = 0; i < n/2+1; i++) {
        for (int j = 0; j < m/2+1; j++) {
            coarse_u[i][j] = 0.0;
        }
    }

    vcycle(n/2+1, m/2+1, coarse_u, coarse_f, depth+1);

    interpolate(n/2+1, m/2+1, n, m, coarse_u, u_correction);
    for (int i = 1; i < n-1; i++) {
        for (int j = 1; j < m-1; j++) {
            u[i][j] -= u_correction[i][j];
        }
    }

    gauss_seidel(n, m, u, f, 2);
}

void
fmg(int n,
    int m,
    double u[n][m],
    double f[n][m],
    int depth)
{
    if (n <= 3 || m <= 3) {
        assert(n == 3);
        assert(m == 3);

        vcycle(n, m, u, f, 0);
        return;
    }

    double coarse_u[n/2 + 1][m/2 + 1];
    double coarse_f[n/2 + 1][m/2 + 1];
    for (int i = 0; i < n/2+1; i++) {
        for (int j = 0; j < m/2+1; j++) {
            coarse_u[i][j] = 0.0;
        }
    }

    inject(n, m, f, coarse_f);
    fmg(n/2+1, m/2+1, coarse_u, coarse_f, depth+1);
    interpolate(n/2+1, m/2+1, n, m, coarse_u, u);

    vcycle(n, m, u, f, 0);
}


#define settype(t) \
    zero_messages_type(t) \
    states_copy_type(t) \
    add_burning_config_type(t) \
    states_eq_type(t) \
    is_identity_type(t) \
    exp_burning_algo_type(t) \
    subtraction_algo_type(t)

// Either unsigned char or int
settype(unsigned char)

// TODO:
// 1. Rework functions to take pointers instead of variable arrays so that we can calculate larger identities
// 2. Implement odd even reduction and interpolation so that we can converge poisson solutions for all grid sizes.
// 3. Tweak base solver so that we can solve rectangular grids
// 4. Test recursive sandpile-poisson method
// 5. Benchmark methods
int
main(void)
{
    // setlocale(LC_ALL, "");
    // omp_set_num_threads(4);

    // int N = 256;
    // __attribute__((aligned(64))) unsigned char state[2+N][N];
    // __attribute__((aligned(64))) unsigned char state_copy[2+N][N];
    // __attribute__((aligned(64))) unsigned char messages[N / TILE_WIDTH][2][N];

    // exp_burning_algo(N, state, state_copy, messages);
    // render(N, state);

    int n = 257, m = 257;
    FULL_N = n;
    FULL_M = m;
    double u[n][m];
    double f[n][m];
    double res[n][m];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            f[i][j] = 2.125;
            u[i][j] = 0.0;
        }
    }

    fmg(n, m, u, f, 0);
    residual(n, m, u, f, res);
    fprintf(stderr, "Discrete L2 Norm: %lf \t| Max Norm: %lf\n",
            dl2_norm(n, m, res), max_norm(n, m, res));

    int s_buf[n][m];
    int s_copy[n][m];
    for (int i = 1; i < n-1; i++) {
        for (int j = 1; j < m-1; j++) {
            s_buf[i][j] =  (int) floor(u[i-1][j])
                      +    (int) floor(u[i+1][j])
                      +    (int) floor(u[i][j-1])
                      +    (int) floor(u[i][j+1])
                      - 4* (int) floor(u[i][j]);
        }
    }

    stabilize_generic(n, m, s_buf);

    int converged = 0;
    int burns = 0;
    do {
        // copy into s_copy
        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < m-1; j++) {
                s_copy[i][j] = s_buf[i][j];
            }
        }

        // Add burning config
        for (int x = 1; x < n-1; x++) {
            s_buf[1][x] += 1;
            s_buf[n-2][x] += 1;
        }
        for (int y = 1; y < n-1; y++) {
            s_buf[y][1]   += 1;
            s_buf[y][n-2] += 1;
        }

        stabilize_generic(n, m, s_buf);
        render_i(n, s_buf);
        burns++;

        // Check convergance
        converged = 1;
        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < m-1; j++) {
                if (s_copy[i][j] != s_buf[i][j]) {
                    converged = 0;
                }
            }
        }
    } while (!converged);

    fprintf(stderr, "Count burns: %d\n", burns);
    render_i(n, s_buf);
}
