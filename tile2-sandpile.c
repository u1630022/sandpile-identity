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
#include <stdio.h>
#include <immintrin.h>
#include <emmintrin.h>
#include <omp.h>
#include <locale.h>

#ifndef SCALE
#  define SCALE 1
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

void
zero_messages(int N,
              unsigned char messages[N / TILE_WIDTH][2][N])
{
    for (int xx = 0; xx < N / TILE_WIDTH; xx++) {
        for (int dir = LEFT; dir <= RIGHT; dir++) {
            for (int i = 0; i < N; i++) {
                messages[xx][dir][i] = 0;
            }
        }
    }
}

static void
render(int N,
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

static void
render_diff(int N,
            char diff[2+N][N])
{
    unsigned char buf[3L*N*SCALE*N*SCALE];
    static const long colors[] = {
        0x0d0887,
        0x5c01a6,
        0x5c01a6,
        0xcc4778,
        0xed7953,
        0xfdb42f,
        0xf0f921
    };
    for (int y = 0; y < N*SCALE; y++) {
        for (int x = 0; x < N*SCALE; x++) {
            int v = diff[1+y/SCALE][x/SCALE];
            long c = colors[v + 3];
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

static void
stabilize(int N,
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
                    assert(tileSpills >= 0);

                    threadSpills += tileSpills;

                    // #pragma omp atomic update
                    // cellsChecked += TILE_HEIGHT * TILE_WIDTH;
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

int
states_eq(int N,
          unsigned char state[2+N][N],
          unsigned char state_copy[2+N][N])
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
states_copy(int N,
            unsigned char state[2+N][N],
            unsigned char state_copy[2+N][N])
{
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            state_copy[1+y][x] = state[1+y][x];
        }
    }
}

void
add_burning_config(int N,
                   unsigned char state[2+N][N])
{
    for (int x = 0; x < N; x++) {
        state[1][x] += 1;
        state[N][x] += 1;
    }
    for (int y = 1; y < N-1; y++) {
        state[1+y][0]   += 1;
        state[1+y][N-1] += 1;
    }
    state[1][0]   += 1;
    state[1][N-1] += 1;
    state[N][0]   += 1;
    state[N][N-1] += 1;
}

int
is_identity(int N,
            unsigned char state[2+N][N],
            unsigned char state_copy[2+N][N],
            unsigned char messages[N / TILE_WIDTH][2][N])
{
    states_copy(N, state, state_copy);
    add_burning_config(N, state);
    stabilize(N, state, messages);
    return states_eq(N, state, state_copy);
}

void
subtraction_algo(int N,
                 unsigned char state[2+N][N],
                 unsigned char messages[N / TILE_WIDTH][2][N])
{
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            state[1+y][x] = 6;
        }
    }
    stabilize(N, state, messages);
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            state[1+y][x] = 6 - state[1+y][x];
        }
    }
    stabilize(N, state, messages);
}

void
exp_burning_algo(int N,
                 unsigned char state[2+N][N],
                 unsigned char state_copy[2+N][N],
                 unsigned char messages[N / TILE_WIDTH][2][N])
{
    // init with the burning config
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            state[1+y][x] = 0;
        }
    }
    zero_messages(N, messages);
    add_burning_config(N, state);

    do {
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                state[1+y][x] *= 2;
            }
        }
        stabilize(N, state, messages);
        // render();
    } while (!is_identity(N, state, state_copy, messages));
}

int
main(void)
{
    setlocale(LC_ALL, "");
    omp_set_num_threads(8);

    int N1 = 512;
    __attribute__((aligned(64))) unsigned char state[2+N1][N1];
    __attribute__((aligned(64))) unsigned char state_copy[2+N1][N1];
    __attribute__((aligned(64))) unsigned char messages[N1 / TILE_WIDTH][2][N1];

    int N2 = 256;
    __attribute__((aligned(64))) unsigned char state2[2+N2][N2];
    __attribute__((aligned(64))) unsigned char state_copy2[2+N2][N2];
    __attribute__((aligned(64))) unsigned char messages2[N2 / TILE_WIDTH][2][N2];

    exp_burning_algo(N1, state, state_copy, messages);
    exp_burning_algo(N2, state2, state_copy2, messages2);

    char diff[2+N1][N1];
    for (int y = 0; y < N1; y++) {
        for (int x = 0; x < N1; x++) {
            diff[1+y][x] = state[1+y][x] - state2[1+(y/2)][x/2];
        }
    }

    render_diff(N1, diff);
}
