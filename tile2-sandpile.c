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

__attribute__((aligned(32))) static unsigned short state[2+N][N];

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
m256_srl16_1(__m256i i) {
    // suppose i is [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    //[4, 3, 2, 1,      16, 15, 14, 13,   12, 11, 10, 9,   8, 7, 6, 5]
    __m256i srl64_q = _mm256_permute4x64_epi64(i, _MM_SHUFFLE(0,3,2,1));

    //[ 1, 0, 0, 0      13, 0, 0, 0       9, 0, 0, 0       5, 0, 0, 0]
    __m256i srl64_m = _mm256_slli_epi64(srl64_q, 3*16);
    //[ 0, 16, 15, 14,  0, 12, 11, 10,    0, 8, 7, 6,      0, 4, 3, 2]
    __m256i srl16_z = _mm256_srli_epi64(i, 1*16);

    __m256i srl64 = _mm256_and_si256(srl64_m, _mm256_set_epi64x(0, ~0, ~0, ~0));
    __m256i r = _mm256_or_si256(srl64, srl16_z);

    return r;
}

static __m256i
m256_sll16_1(__m256i i) {
    // suppose i is [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    //[12, 11, 10, 9,   8, 7, 6, 5,    4, 3, 2, 1,    16, 15, 14, 13]
    __m256i srl64_q = _mm256_permute4x64_epi64(i, _MM_SHUFFLE(2,1,0,3));

    //[0, 0, 0, 12,   0, 0, 0, 8,    0, 0, 0, 4,    0, 0, 0, 16]
    __m256i srl64_m = _mm256_srli_epi64(srl64_q, 3*16);
    //[15, 14, 13, 0  11, 10, 9, 0    7, 6, 5, 0,      3, 2, 1, 0]
    __m256i srl16_z = _mm256_slli_epi64(i, 1*16);

    __m256i srl64 = _mm256_and_si256(srl64_m, _mm256_set_epi64x(~0, ~0, ~0, 0));
    __m256i r = _mm256_or_si256(srl64, srl16_z);

    return r;
}

__attribute__((aligned(32))) unsigned short sum_buffer[16];

static int
m256_hadd_all(__m256i i) {
    int sum = 0;
    _mm256_store_si256((void *) &sum_buffer, i);
    for (int i = 0; i < 16; i++) {
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
                    __m256i vspills = _mm256_set1_epi16(0);
                    int ystart = (yy * TILE_HEIGHT) / 2;
                    int yend = ystart + TILE_HEIGHT;
                    for (int y = ystart; y < yend; y++) {
                        int xspills = 0;
                        int xstart = (xx * TILE_WIDTH) / 2;
                        int xend = xstart + TILE_WIDTH;

                        for (int x = xstart; x < xend; x += 16) {
                            __m256i vv = _mm256_load_si256((void *)&state[1+y][x]);
                            __m256i vvabove = _mm256_load_si256((void *)&state[y][x]);
                            __m256i vvbelow = _mm256_load_si256((void *)&state[2+y][x]);
                            __m256i vinc = _mm256_srl_epi16(vv, _mm_set1_epi64x(2));

                            vspills = _mm256_add_epi16(vinc, vspills);

                            _mm256_store_si256((void *)&state[y][x], _mm256_add_epi16(vvabove, vinc));
                            _mm256_store_si256((void *)&state[2+y][x], _mm256_add_epi16(vvbelow, vinc));

                            __m256i vinc_left = m256_srl16_1(vinc);
                            __m256i vinc_right = m256_sll16_1(vinc);

                            __m256i vinc4 = _mm256_sll_epi16(vinc, _mm_set1_epi64x(2));
                            __m256i vv_new = _mm256_add_epi16(vinc_left, 
                                    _mm256_add_epi16(vinc_right,
                                    _mm256_sub_epi16(vv, vinc4)));
                            _mm256_store_si256((void *)&state[1+y][x], vv_new);

                            // tails
                            if (x - 1 > 0) {
                                int left_spill = _mm256_extract_epi16(vinc, 0);
                                state[1+y][x-1] += left_spill;
                            }
                            if (x + 16 < N) {
                                int right_spill = _mm256_extract_epi16(vinc, 15);
                                state[1+y][x+16] += right_spill;
                            }
                        }
                    }
                    cellsChecked += TILE_HEIGHT * TILE_WIDTH;
                    localSpills = m256_hadd_all(vspills);
                    spills += localSpills;
                } while (localSpills > TILE_HEIGHT * TILE_WIDTH / 2);
            }
        }
        totalSpills += spills;
    } while (spills > 0);
    // 8915347868
    // 3251067552
    fprintf(stderr, "Total Cells Checked: %ld\tTotal Spills: %ld\n", cellsChecked, totalSpills);
}

int
main(void)
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
    render();
}
