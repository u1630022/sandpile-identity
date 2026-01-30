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
#  define SCALE 1
#endif

#define TILE_WIDTH 64
#define TILE_HEIGHT 8

#define LEFT  0
#define RIGHT 1

#define STACK_SIZE 512

/* Color palette */
#define C0 0x111111
#define C1 0x444444
#define C2 0x777777
#define C3 0xaaaaaa
#define CA4 0xff0000
#define CX 0x0000ff

static const int DIRECT_SOLVE_SIZE = 64;
static const long ALIGNMENT = 64;
static const double SPARSE_THRESHOLD = 0.03;
static const double FMG_CONVERGENCE_LIMIT = 0.0001;

int IDX(int i, int j, int n) { return i * n + j; }

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

// full buffer included in N and M
void
render_d(int N, int M, double* grid)
{
    unsigned char* buf = calloc(3L*(N-2)*SCALE*(M-2)*SCALE, sizeof(unsigned char));
    double min = LONG_MAX;
    double max = LONG_MIN;
    for (int y = 0; y < N-2; y++) {
        for (int x = 0; x < M-2; x++) {
            double v = grid[IDX(1+y, 1+x, M)];
            if (v < min) {
                min = v;
            }
            if (v > max) {
                max = v;
            }
        }
    }
    for (int y = 0; y < (N-2)*SCALE; y++) {
        for (int x = 0; x < (M-2)*SCALE; x++) {
            int idx;
            if (min == max) {
                idx = 127;
            } else {
                double v = grid[IDX(1+y/SCALE, 1+x/SCALE, M)];
                idx = (int) (253.0 * (v - min) / (max - min));
            }
            assert(idx >= 0);
            assert(idx < 254);
            long c = inferno[idx];
            buf[y*3L*SCALE*(N-2) + x*3L + 0] = c >> 16;
            buf[y*3L*SCALE*(N-2) + x*3L + 1] = c >>  8;
            buf[y*3L*SCALE*(N-2) + x*3L + 2] = c >>  0;
        }
    }
    printf("P6\n%d %d\n255\n", (N-2)*SCALE, (M-2)*SCALE);
    fwrite(buf, 3L*(N-2)*SCALE*(M-2)*SCALE * sizeof(unsigned char), 1, stdout);
    free(buf);
}

void
render_i(int N, char* grid)
{
    unsigned char* buf = calloc(3L*(N-2)*SCALE*(N-2)*SCALE, sizeof(unsigned char));
    double min = LONG_MAX;
    double max = LONG_MIN;
    for (int y = 0; y < N-2; y++) {
        for (int x = 0; x < N-2; x++) {
            double v = grid[IDX(1+y, x, N-2)];
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
            int idx;
            if (min == max) {
                idx = 127;
            } else {
                char v = grid[IDX(1+y/SCALE, x/SCALE, N-2)];
                idx = (int) (253.0 * (v - min) / (max - min));
            }
            assert(idx >= 0);
            assert(idx < 254);
            long c = inferno[idx];
            buf[y*3L*SCALE*(N-2) + x*3L + 0] = c >> 16;
            buf[y*3L*SCALE*(N-2) + x*3L + 1] = c >>  8;
            buf[y*3L*SCALE*(N-2) + x*3L + 2] = c >>  0;
        }
    }
    printf("P6\n%d %d\n255\n", (N-2)*SCALE, (N-2)*SCALE);
    fwrite(buf, 3L*(N-2)*SCALE*(N-2)*SCALE * sizeof(unsigned char), 1, stdout);
    free(buf);
}

typedef struct {
    int top;
    int size;
    int data[STACK_SIZE];
} IntStack;

IntStack* create_stack() {
    IntStack* stack = (IntStack*)aligned_alloc(ALIGNMENT, sizeof(IntStack));
    if (!stack) {
        perror("Failed to allocate memory for stack");
        exit(EXIT_FAILURE);
    }
    stack->size = 0;
    stack->top = 0;
    return stack;
}

void push(IntStack* stack, int value) {
    stack->top = (stack->top + 1) % STACK_SIZE;
    stack->data[stack->top] = value;
    if (stack->size < STACK_SIZE) {
        stack->size++;
    }
}

int pop(IntStack* stack) {
    if (stack->size == 0) {
        fprintf(stderr, "Stack underflow\n");
        exit(EXIT_FAILURE);
    }
    int val = stack->data[stack->top];
    stack->top = (stack->top + STACK_SIZE - 1) % STACK_SIZE;
    stack->size--;
    return val;
}

int is_empty(IntStack* stack) {
    return stack->size == 0;
}

void free_stack(IntStack* stack) {
    free(stack);
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

// buffered in n and m
void
enforce_symmetry(int n, int m, double* grid)
{
    assert(n % 2 == 0);
    assert(m % 2 == 0);
    for (int i = 1; i < n / 2; i++) {
        for (int j = 1; j < m / 2; j++) {
            double v = grid[IDX(i, j, m)];
            grid[IDX(i, m - j - 1, m)] = v;
            grid[IDX(n - i - 1, j, m)] = v;
            grid[IDX(n - i - 1, m - j - 1, m)] = v;
        }
    }
}

// buffered only in n
void
enforce_symmetry_i(int n, int m, char* grid)
{
    assert(n % 2 == 0);
    assert(m % 2 == 0);
    for (int i = 1; i < n / 2; i++) {
        for (int j = 0; j < m / 2; j++) {
            char v = grid[IDX(i, j, m)];
            grid[IDX(i, m - j - 1, m)] = v;
            grid[IDX(n - i - 1, j, m)] = v;
            grid[IDX(n - i - 1, m - j - 1, m)] = v;
        }
    }
}

// buffered only in n
void
assert_symmetry(int n, int m, char* grid)
{
    assert(n % 2 == 0);
    assert(m % 2 == 0);
    for (int i = 1; i < n / 2; i++) {
        for (int j = 0; j < m / 2; j++) {
            double v = grid[IDX(i, j, m)];
            assert(grid[IDX(i, m - j - 1, m)] == v);
            assert(grid[IDX(n - i - 1, j, m)] == v);
            assert(grid[IDX(n - i - 1, m - j - 1, m)] == v);
        }
    }
}


// Assume surrounded by buffer
void
stabilize_generic(int N,
                  int M,
                  int* state)
{
    long spills = 0;
    long totalSpills = 0;
    do {
        spills = 0;
        for (int y = 1; y < N - 1; y++) {
            for (int x = 1; x < M - 1; x++) {
                if (state[IDX(y, x, M)] >= 4) {
                    state[IDX(y, x, M)] -= 4;
                    state[IDX(y+1, x, M)] += 1;
                    state[IDX(y-1, x, M)] += 1;
                    state[IDX(y, x+1, M)] += 1;
                    state[IDX(y, x-1, M)] += 1;
                    spills++;
                }
            }
        }
        totalSpills += spills;
        // render_i(N, state);
    } while (spills > 0);
    fprintf(stderr, "Total Spills: %ld\n", totalSpills);
}

int
IDX3(int i, int j, int k, int width, int depth) {
    return i * (width * depth) + j * (depth) + k;
}



void
stabilize_sparse(int N,
                 int M,
                 char* state,
                 int render)
{
    long spills = 0;
    long checks = 0;
    long totalSpills = 0;
    char* messages = calloc(M/TILE_WIDTH * 2 * N, sizeof(char));
    double start_c = omp_get_wtime();

    // TODO: should be made private
    IntStack* stacks[M/TILE_WIDTH];
    for (int i = 0; i < M/TILE_WIDTH; i++) {
        stacks[i] = create_stack();
    }

    do {
        spills = 0;
        int first_iter = 1;

        #pragma omp parallel for schedule(static, 1)
        for (int xx = 0; xx < M / TILE_WIDTH; xx++) {
            int xstart = (xx * TILE_WIDTH);
            int xend = xstart + TILE_WIDTH;
            long localSpills = 0;
            long localChecks = 0;
            IntStack* stack = stacks[xx];

            // import from messages
            for (int y = 0; y < N; y++) {
                int tmp = 0;

                #pragma omp atomic capture
                {
                    tmp = messages[IDX3(xx, LEFT, y, 2, N)];
                    messages[IDX3(xx, LEFT, y, 2, N)] = 0;
                }
                state[IDX(1+y, xx * TILE_WIDTH, M)] += tmp;
                if (state[IDX(1+y, xx * TILE_WIDTH, M)] >= 4) {
                    push(stack, IDX(1+y, xx * TILE_WIDTH, M));
                }

                #pragma omp atomic capture
                {
                    tmp = messages[IDX3(xx, RIGHT, y, 2, N)];
                    messages[IDX3(xx, RIGHT, y, 2, N)] = 0;
                }
                state[IDX(1+y, (xx + 1) * TILE_WIDTH - 1, M)] += tmp;
                if (state[IDX(1+y, (xx + 1) * TILE_WIDTH - 1, M)] >= 4) {
                    push(stack, IDX(1+y, (xx + 1) * TILE_WIDTH - 1, M));
                }
            }

            if (stack->size < 128) {
                for (int y = 0; y < N; y++) {
                    for (int x = xend - 1; x >= xstart; x--) {
                        localChecks++;
                        if (state[IDX(1+y, x, M)] >= 4) {
                            push(stack, IDX(1+y, x, M));
                            break;
                        }
                    }
                    if (stack->size >= 128) {
                        break;
                    }
                }
            }

            while (!is_empty(stack)) {
                int idx = pop(stack);
                localChecks++;
                if (state[idx] >= 4) {
                    localSpills++;
                    state[idx] -= 4;
                    state[idx-M] += 1;
                    if ((idx > 2*M) && state[idx-M] >= 4) {
                        push(stack, idx-M);
                    }
                    state[idx+M] += 1;
                    if ((idx/M < N) && state[idx+M] >= 4) {
                        push(stack, idx+M);
                    }
                    if (idx % TILE_WIDTH != 0) {
                        state[idx-1] += 1;
                        if (state[idx-1] >= 4) {
                            push(stack, idx-1);
                        }
                    } else {
                        if (xx > 0) {
                            #pragma omp atomic update
                            messages[IDX3(xx - 1, RIGHT, idx / M - 1, 2, N)] += 1;
                        }
                    }
                    if ((idx + 1) % TILE_WIDTH != 0) {
                        state[idx+1] += 1;
                        if (state[idx+1] >= 4) {
                            push(stack, idx+1);
                        }
                    } else {
                        if (xx < (M / TILE_WIDTH) - 1) {
                            #pragma omp atomic update
                            messages[IDX3(xx + 1, LEFT, idx / M - 1, 2, N)] += 1;
                        }
                    }
                }
            }

            #pragma omp atomic update
            spills += localSpills;

            #pragma omp atomic update
            checks += localChecks;
        }

        totalSpills += spills;
    } while (spills > 0);

    for (int xx = 1; xx < N / TILE_WIDTH - 1; xx++) {
        for (int dir = LEFT; dir <= RIGHT; dir++) {
            for (int i = 0; i < N; i++) {
                assert(messages[IDX3(xx, dir, i, 2, N)] == 0);
            }
        }
    }
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < M; x++) {
            assert(state[IDX(1+y, x, M)] < 4);
        }
    }

    double end_c = omp_get_wtime();
    double time_s = end_c - start_c;

    fprintf(stderr, "Sparse: Total Spills: %'ld |\tChecks: %'ld |\t Efficiency: %lf |\t Spills/s: %'lf\n",
            totalSpills, checks, ((double) totalSpills) / checks, totalSpills / time_s);
    for (int i = 0; i < M/TILE_WIDTH; i++) {
        free_stack(stacks[i]);
    }
    free(messages);
}

void
stabilize_sparse_topleft(int N,
                         int M,
                         char* state,
                         int render)
{
    assert(N % 2 == 0);
    assert(M % 2 == 0);
    int TL_N = N / 2;
    int TL_M = M / 2;

    long spills = 0;
    long checks = 0;
    long totalSpills = 0;
    char* messages = calloc(TL_M/TILE_WIDTH * 2 * TL_N, sizeof(char));
    double start_c = omp_get_wtime();

    IntStack* stacks[TL_M/TILE_WIDTH];
    for (int i = 0; i < TL_M/TILE_WIDTH; i++) {
        stacks[i] = create_stack();
    }

    do {
        spills = 0;
        int first_iter = 1;

        #pragma omp parallel for schedule(static, 1)
        for (int xx = 0; xx < TL_M / TILE_WIDTH; xx++) {
            int xstart = (xx * TILE_WIDTH);
            int xend = xstart + TILE_WIDTH;
            long localSpills = 0;
            long localChecks = 0;
            IntStack* stack = stacks[xx];

            // import from messages
            for (int y = 0; y < TL_N; y++) {
                int tmp = 0;

                #pragma omp atomic capture
                {
                    tmp = messages[IDX3(xx, LEFT, y, 2, TL_N)];
                    messages[IDX3(xx, LEFT, y, 2, TL_N)] = 0;
                }
                state[IDX(1+y, xx * TILE_WIDTH, M)] += tmp;
                if (state[IDX(1+y, xx * TILE_WIDTH, M)] >= 4) {
                    push(stack, IDX(1+y, xx * TILE_WIDTH, M));
                }

                #pragma omp atomic capture
                {
                    tmp = messages[IDX3(xx, RIGHT, y, 2, TL_N)];
                    messages[IDX3(xx, RIGHT, y, 2, TL_N)] = 0;
                }
                state[IDX(1+y, (xx + 1) * TILE_WIDTH - 1, M)] += tmp;
                if (state[IDX(1+y, (xx + 1) * TILE_WIDTH - 1, M)] >= 4) {
                    push(stack, IDX(1+y, (xx + 1) * TILE_WIDTH - 1, M));
                }
            }

            if (stack->size < 128) {
                for (int y = 0; y < TL_N; y++) {
                    for (int x = xend - 1; x >= xstart; x--) {
                        localChecks++;
                        if (state[IDX(1+y, x, M)] >= 4) {
                            push(stack, IDX(1+y, x, M));
                            break;
                        }
                    }
                    if (stack->size >= 128) {
                        break;
                    }
                }
            }

            while (!is_empty(stack)) {
                int idx = pop(stack);
                localChecks++;
                if (state[idx] >= 4) {
                    char s = state[idx] / 4;
                    localSpills += s;
                    state[idx] -= 4 * s;
                    state[idx-M] += s;
                    if ((idx > 2*M) && state[idx-M] >= 4) {
                        push(stack, idx-M);
                    }
                    if (idx/M < TL_N) { // Not on bottom edge
                        state[idx+M] += s;
                        if (state[idx+M] >= 4) {
                            push(stack, idx+M);
                        }
                    } else {
                        state[idx] += s;
                    }
                    if (idx % TILE_WIDTH != 0) {
                        state[idx-1] += s;
                        if (state[idx-1] >= 4) {
                            push(stack, idx-1);
                        }
                    } else {
                        if (xx > 0) {
                            #pragma omp atomic update
                            messages[IDX3(xx - 1, RIGHT, idx / M - 1, 2, TL_N)] += s;
                        }
                    }
                    if ((idx + 1) % TILE_WIDTH != 0) {
                        state[idx+1] += s;
                        if (state[idx+1] >= 4) {
                            push(stack, idx+1);
                        }
                    } else {
                        if (xx < (TL_M / TILE_WIDTH) - 1) {
                            #pragma omp atomic update
                            messages[IDX3(xx + 1, LEFT, idx / M - 1, 2, TL_N)] += s;
                        } else {
                            state[idx] += s;
                        }
                    }
                }
            }

            #pragma omp atomic update
            spills += localSpills;

            #pragma omp atomic update
            checks += localChecks;
        }

        if (render) {
            render_i(N+2, state);
        }
        totalSpills += spills;
    } while (spills > 0);

    for (int xx = 1; xx < TL_N / TILE_WIDTH - 1; xx++) {
        for (int dir = LEFT; dir <= RIGHT; dir++) {
            for (int i = 0; i < TL_N; i++) {
                assert(messages[IDX3(xx, dir, i, 2, TL_N)] == 0);
            }
        }
    }

    enforce_symmetry_i(N+2, M, state);
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < M; x++) {
            assert(state[IDX(1+y, x, M)] < 4);
        }
    }

    double end_c = omp_get_wtime();
    double time_s = end_c - start_c;

    fprintf(stderr, "Sparse: Total Spills: %'ld |\tChecks: %'ld |\t Efficiency: %lf |\t Spills/s: %'lf\n",
            totalSpills, checks, ((double) totalSpills) / checks, totalSpills / time_s);
    for (int i = 0; i < TL_M/TILE_WIDTH; i++) {
        free_stack(stacks[i]);
    }
    free(messages);
}

static void
stabilize_dense(int N,
                int M,
                char* state,
                int render) // state is buffered only abovev and below
{
    long checks = 0;
    long tileSpills = 0;
    long threadSpills = 0;
    long spills = 0;
    long totalSpills = 0;
    char* messages = calloc(N/TILE_WIDTH * 2 * N, sizeof(char));
    double start_c = omp_get_wtime();

    do {
        spills = 0;
        double frame_start_c = omp_get_wtime();

        #pragma omp parallel for private(tileSpills, threadSpills) schedule(static, 1)
        for (int xx = 0; xx < N / TILE_WIDTH; xx++) {

            threadSpills = 0;

            // import from messages
            for (int y = 0; y < N; y++) {
                int tmp = 0;

                #pragma omp atomic capture
                {
                    tmp = messages[IDX3(xx, LEFT, y, 2, N)];
                    messages[IDX3(xx, LEFT, y, 2, N)] = 0;
                }
                state[IDX(1+y, xx * TILE_WIDTH, M)] += tmp;

                #pragma omp atomic capture
                {
                    tmp = messages[IDX3(xx, RIGHT, y, 2, N)];
                    messages[IDX3(xx, RIGHT, y, 2, N)] = 0;
                }
                state[IDX(1+y, (xx + 1) * TILE_WIDTH - 1, M)] += tmp;
            }

            for (int yy = 0; yy < 2*N / TILE_HEIGHT - 1; yy++) {
                tileSpills = 0;
                __m256i vspills = _mm256_set1_epi8(0);
                int ystart = (yy * TILE_HEIGHT) / 2;
                int yend = ystart + TILE_HEIGHT;
                for (int y = ystart; y < yend; y++) {
                    int xspills = 0;
                    int xstart = (xx * TILE_WIDTH);
                    int xend = xstart + TILE_WIDTH;

                    for (int x = xstart; x < xend; x += 32) {
                        __m256i vv = _mm256_load_si256((void *)&state[IDX(1+y, x, M)]);
                        __m256i vvabove = _mm256_load_si256((void *)&state[IDX(y, x, M)]);
                        __m256i vvbelow = _mm256_load_si256((void *)&state[IDX(2+y, x, M)]);
                        __m256i vpos = _mm256_cmpgt_epi8(vv, _mm256_set1_epi8(0));
                        __m256i vinc = _mm256_srl_epi16(vv, _mm_set1_epi64x(2));
                        vinc = _mm256_and_si256(vinc, _mm256_set1_epi8(0x3F));
                        vinc = _mm256_and_si256(vinc, vpos);

                        vspills = _mm256_add_epi8(vinc, vspills);

                        _mm256_store_si256((void *)&state[IDX(y, x, M)], _mm256_add_epi8(vvabove, vinc));
                        _mm256_store_si256((void *)&state[IDX(2+y, x, M)], _mm256_add_epi8(vvbelow, vinc));

                        __m256i vinc_left = m256_srl8_1(vinc);
                        __m256i vinc_right = m256_sll8_1(vinc);

                        __m256i vinc4 = _mm256_sll_epi16(vinc, _mm_set1_epi64x(2));
                        __m256i vv_new = _mm256_add_epi8(vinc_left,
                                _mm256_add_epi8(vinc_right,
                                _mm256_sub_epi8(vv, vinc4)));
                        _mm256_store_si256((void *)&state[IDX(1+y, x, M)], vv_new);

                        // tails
                        if (x > xstart) {
                            int left_spill = _mm256_extract_epi8(vinc, 0);
                            state[IDX(1+y, x - 1, M)] += left_spill;
                        } else if (xx > 0) {
                            int left_spill = _mm256_extract_epi8(vinc, 0);

                            #pragma omp atomic update
                            messages[IDX3(xx - 1, RIGHT, y, 2, N)] += left_spill;
                        }
                        if (x < xend - 32) {
                            int right_spill = _mm256_extract_epi8(vinc, 31);
                            state[IDX(1+y, x+32, M)] += right_spill;
                        } else if (xx < N / TILE_WIDTH - 1) {
                            int right_spill = _mm256_extract_epi8(vinc, 31);

                            #pragma omp atomic update
                            messages[IDX3(xx + 1, LEFT, y, 2, N)] += right_spill;
                        }
                    }
                }
                tileSpills = m256_hadd_all(vspills);
                threadSpills += tileSpills;
            }

            #pragma omp atomic update
            spills += threadSpills;
        }

        totalSpills += spills;
        checks += N * M;
        if (render) {
            render_i(N+2, state);
        }

        double current_efficiency = ((double) spills) / (N * M);
        if (current_efficiency < SPARSE_THRESHOLD) { // TODO: tune this const
            double end_c = omp_get_wtime();
            double time_s = end_c - start_c;
            double frame_time_s = end_c - frame_start_c;
            fprintf(stderr, "Switching to sparse stabilizer. Efficiency: %lf |\t Frame Spills/s: %'lf\n", 
                    current_efficiency, spills / frame_time_s);
            fprintf(stderr, "Dense: Total Spills: %'ld |\tChecks: %'ld |\t Efficiency: %lf |\t Spills/s %'lf\n",
                    totalSpills, checks, ((double) totalSpills) / checks, totalSpills / time_s);
            for (int xx = 0; xx < N / TILE_WIDTH; xx++) {
                for (int y = 0; y < N; y++) {
                    state[IDX(1+y, xx * TILE_WIDTH, M)] += messages[IDX3(xx, LEFT, y, 2, N)];;
                    state[IDX(1+y, (xx + 1) * TILE_WIDTH - 1, M)] += messages[IDX3(xx, RIGHT, y, 2, N)];
                }
            }
            stabilize_sparse(N, M, state, render);
            free(messages);
            return;
        }
    } while (spills > 0);

    for (int xx = 1; xx < N / TILE_WIDTH - 1; xx++) {
        for (int dir = LEFT; dir <= RIGHT; dir++) {
            for (int i = 0; i < N; i++) {
                assert(messages[IDX3(xx, dir, i, 2, N)] == 0);
            }
        }
    }
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < M; x++) {
            assert(state[IDX(1+y, x, M)] < 4);
        }
    }

    double end_c = omp_get_wtime();
    double time_s = end_c - start_c;

    // 8 915 347 868
    // 3 251 067 552
    fprintf(stderr, "Dense: Total Spills: %'ld |\tChecks: %'ld |\t Efficiency: %lf |\t Spills/s %'lf\n",
            totalSpills, checks, ((double) totalSpills) / checks, totalSpills / time_s);
    free(messages);
}

static void
stabilize_dense_topleft(int N,
                        int M,
                        char* state,
                        int render) // state is buffered only abovev and below
{
    assert(N % 2 == 0);
    assert(M % 2 == 0);
    int TL_N = N / 2;
    int TL_M = M / 2;
    assert(TL_M >= TILE_WIDTH);

    long checks = 0;
    long tileSpills = 0;
    long threadSpills = 0;
    long spills = 0;
    long totalSpills = 0;
    char* messages = calloc(TL_M/TILE_WIDTH * 2 * TL_N, sizeof(char));
    double start_c = omp_get_wtime();

    do {
        spills = 0;
        double frame_start_c = omp_get_wtime();

        #pragma omp parallel for private(tileSpills, threadSpills) schedule(static, 1)
        for (int xx = 0; xx < TL_N / TILE_WIDTH; xx++) {

            threadSpills = 0;

            // import from messages
            for (int y = 0; y < TL_N; y++) {
                int tmp = 0;

                #pragma omp atomic capture
                {
                    tmp = messages[IDX3(xx, LEFT, y, 2, TL_N)];
                    messages[IDX3(xx, LEFT, y, 2, TL_N)] = 0;
                }
                state[IDX(1+y, xx * TILE_WIDTH, M)] += tmp;

                #pragma omp atomic capture
                {
                    tmp = messages[IDX3(xx, RIGHT, y, 2, TL_N)];
                    messages[IDX3(xx, RIGHT, y, 2, TL_N)] = 0;
                }
                state[IDX(1+y, (xx + 1) * TILE_WIDTH - 1, M)] += tmp;
            }

            for (int yy = 0; yy < 2*TL_N / TILE_HEIGHT - 1; yy++) {
                tileSpills = 0;
                __m256i vspills = _mm256_set1_epi8(0);
                int ystart = (yy * TILE_HEIGHT) / 2;
                int yend = ystart + TILE_HEIGHT;
                for (int y = ystart; y < yend; y++) {
                    int xspills = 0;
                    int xstart = (xx * TILE_WIDTH);
                    int xend = xstart + TILE_WIDTH;
                    int isBottomRow = y == TL_N - 1;

                    for (int x = xstart; x < xend; x += 32) {
                        __m256i vv = _mm256_load_si256((void *)&state[IDX(1+y, x, M)]);
                        __m256i vvabove = _mm256_load_si256((void *)&state[IDX(y, x, M)]);
                        __m256i vpos = _mm256_cmpgt_epi8(vv, _mm256_set1_epi8(0));
                        __m256i vinc = _mm256_srl_epi16(vv, _mm_set1_epi64x(2));
                        vinc = _mm256_and_si256(vinc, _mm256_set1_epi8(0x3F));
                        vinc = _mm256_and_si256(vinc, vpos);

                        vspills = _mm256_add_epi8(vinc, vspills);

                        _mm256_store_si256((void *)&state[IDX(y, x, M)], _mm256_add_epi8(vvabove, vinc));
                        if (!isBottomRow) {
                            __m256i vvbelow = _mm256_load_si256((void *)&state[IDX(2+y, x, M)]);
                            _mm256_store_si256((void *)&state[IDX(2+y, x, M)], _mm256_add_epi8(vvbelow, vinc));
                        }

                        __m256i vinc_left = m256_srl8_1(vinc);
                        __m256i vinc_right = m256_sll8_1(vinc);

                        __m256i vv_new;
                        if (isBottomRow) {
                            __m256i vinc3 = _mm256_sub_epi8(_mm256_sll_epi16(vinc, _mm_set1_epi64x(2)), vinc);
                            vv_new = _mm256_add_epi8(vinc_left,
                                    _mm256_add_epi8(vinc_right,
                                    _mm256_sub_epi8(vv, vinc3)));
                        } else {
                            __m256i vinc4 = _mm256_sll_epi16(vinc, _mm_set1_epi64x(2));
                            vv_new = _mm256_add_epi8(vinc_left,
                                    _mm256_add_epi8(vinc_right,
                                    _mm256_sub_epi8(vv, vinc4)));
                        }
                        _mm256_store_si256((void *)&state[IDX(1+y, x, M)], vv_new);

                        // tails
                        if (x > xstart) {
                            char left_spill = _mm256_extract_epi8(vinc, 0);
                            state[IDX(1+y, x - 1, M)] += left_spill;
                        } else if (xx > 0) {
                            char left_spill = _mm256_extract_epi8(vinc, 0);

                            #pragma omp atomic update
                            messages[IDX3(xx - 1, RIGHT, y, 2, TL_N)] += left_spill;
                        }

                        char right_spill = _mm256_extract_epi8(vinc, 31);
                        if (x < xend - 32) { // Not at end of tile
                            state[IDX(1+y, x+32, M)] += right_spill;
                        } else if (xx < TL_N / TILE_WIDTH - 1) { // End of tile but not final tile
                            #pragma omp atomic update
                            messages[IDX3(xx + 1, LEFT, y, 2, TL_N)] += right_spill;
                        } else { // End of final tile
                            state[IDX(1+y, x + 31, M)] += right_spill;
                        }
                    }
                }
                tileSpills = m256_hadd_all(vspills);
                threadSpills += tileSpills;
            }

            #pragma omp atomic update
            spills += threadSpills;
        }

        totalSpills += spills;
        checks += TL_N * TL_M;
        if (render) {
            render_i(N+2, state);
        }

        double current_efficiency = ((double) spills) / (TL_N * TL_M);
        if (current_efficiency < SPARSE_THRESHOLD) {
            double end_c = omp_get_wtime();
            double time_s = end_c - start_c;
            double frame_time_s = end_c - frame_start_c;
            fprintf(stderr, "Switching to sparse stabilizer. Efficiency: %lf |\t Frame Spills/s: %'lf\n", 
                    current_efficiency, spills / frame_time_s);
            fprintf(stderr, "Dense: Total Spills: %'ld |\tChecks: %'ld |\t Efficiency: %lf |\t Spills/s %'lf\n",
                    totalSpills, checks, ((double) totalSpills) / checks, totalSpills / time_s);
            for (int xx = 0; xx < TL_M / TILE_WIDTH; xx++) {
                for (int y = 0; y < TL_N; y++) {
                    state[IDX(1+y, xx * TILE_WIDTH, M)] += messages[IDX3(xx, LEFT, y, 2, TL_N)];;
                    state[IDX(1+y, (xx + 1) * TILE_WIDTH - 1, M)] += messages[IDX3(xx, RIGHT, y, 2, TL_N)];
                }
            }
            stabilize_sparse_topleft(N, M, state, render);
            free(messages);
            return;
        }
    } while (spills > 0);

    for (int xx = 1; xx < TL_N / TILE_WIDTH - 1; xx++) {
        for (int dir = LEFT; dir <= RIGHT; dir++) {
            for (int i = 0; i < TL_N; i++) {
                assert(messages[IDX3(xx, dir, i, 2, TL_N)] == 0);
            }
        }
    }

    enforce_symmetry_i(N+2, M, state);
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < M; x++) {
            assert(state[IDX(1+y, x, M)] < 4);
        }
    }

    double end_c = omp_get_wtime();
    double time_s = end_c - start_c;

    // 8 915 347 868
    // 3 251 067 552
    fprintf(stderr, "Dense: Total Spills: %'ld |\tChecks: %'ld |\t Efficiency: %lf |\t Spills/s %'lf\n",
            totalSpills, checks, ((double) totalSpills) / checks, totalSpills / time_s);
    free(messages);
}


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


void gauss_seidel(int n, int m, double *u, double *f, int iter) {
    double h2 = ((FULL_N-1)*(FULL_M-1)) * 1.0 / ((n-1)*(m-1));
    for (int it = 0; it < iter; it++) {
        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < m-1; j++) {
                u[IDX(i, j, m)] = 0.25 * (
                    u[IDX(i-1, j, m)] + u[IDX(i+1, j, m)] +
                    u[IDX(i, j-1, m)] + u[IDX(i, j+1, m)] -
                    h2 * f[IDX(i, j, m)]);
            }
        }
        // Reverse direction
        for (int i = n - 2; i >= 1; i--) {
            for (int j = m - 2; j >= 1; j--) {
                u[IDX(i, j, m)] = 0.25 * (
                    u[IDX(i-1, j, m)] + u[IDX(i+1, j, m)] +
                    u[IDX(i, j-1, m)] + u[IDX(i, j+1, m)] -
                    h2 * f[IDX(i, j, m)]);
            }
        }
    }
}

// Consider n_fine = m_fine = 5
// then n_coarse = m_coarse = 3
// for 1,1 on the coarse grid, we should sample around 2,2 on the fine grid

// Full-weighting
void inject_odd(int n_fine, int m_fine, double *fine, double *coarse) {
    int n_coarse = (n_fine + 1) / 2;
    int m_coarse = (m_fine + 1) / 2;
    for (int i = 1; i < n_coarse-1; i++) {
        for (int j = 1; j < m_coarse-1; j++) {
            coarse[IDX(i, j, m_coarse)] =
                0.25 * fine[IDX(2*i, 2*j, m_fine)] +
                0.125 * (
                    fine[IDX(2*i-1, 2*j, m_fine)] + fine[IDX(2*i+1, 2*j, m_fine)] +
                    fine[IDX(2*i, 2*j-1, m_fine)] + fine[IDX(2*i, 2*j+1, m_fine)]) +
                0.0625 * (
                    fine[IDX(2*i-1, 2*j-1, m_fine)] + fine[IDX(2*i-1, 2*j+1, m_fine)] +
                    fine[IDX(2*i+1, 2*j-1, m_fine)] + fine[IDX(2*i+1, 2*j+1, m_fine)]);
        }
    }
}

// Consider n_fine = m_fine = 6
// then n_coarse = m_coarse = 3
// for 1,1 on the coarse grid, we should sample around 2,2 to 3,3 on the fine grid
// KERNEL = [[1 3 3 1]
//           [3 9 9 3]
//           [3 9 9 3]
//           [1 3 3 1]] * 1/64

void inject_even(int n_fine, int m_fine, double *fine, double *coarse) {
    int n_coarse = n_fine / 2;
    int m_coarse = m_fine / 2;
    for (int i = 1; i < n_coarse-1; i++) {
        for (int j = 1; j < m_coarse-1; j++) {
            coarse[IDX(i, j, m_coarse)] = 1.0/64 * (
                9 * (
                    fine[IDX(2*i, 2*j, m_fine)] + fine[IDX(2*i+1, 2*j, m_fine)] +
                    fine[IDX(2*i, 2*j+1, m_fine)] + fine[IDX(2*i+1, 2*j+1, m_fine)]) +
                3 * (
                    fine[IDX(2*i-1, 2*j, m_fine)] + fine[IDX(2*i-1, 2*j+1, m_fine)] + // TOP
                    fine[IDX(2*i, 2*j+2, m_fine)] + fine[IDX(2*i+1, 2*j+2, m_fine)] + // RIGHT
                    fine[IDX(2*i+2, 2*j, m_fine)] + fine[IDX(2*i+2, 2*j+1, m_fine)] + // BOTTOM
                    fine[IDX(2*i, 2*j-1, m_fine)] + fine[IDX(2*i+1, 2*j-1, m_fine)]) +// LEFT
                1 * (
                    fine[IDX(2*i-1, 2*j-1, m_fine)] + fine[IDX(2*i-1, 2*j+2, m_fine)] +
                    fine[IDX(2*i+2, 2*j-1, m_fine)] + fine[IDX(2*i+2, 2*j+2, m_fine)]));
        }
    }
}

void inject(int n_fine, int m_fine, double *fine, double *coarse) {
    if (n_fine % 2 == 0) {
        inject_even(n_fine, m_fine, fine, coarse);
    } else {
        inject_odd(n_fine, m_fine, fine, coarse);
    }
}

void interpolate(int n_coarse, int m_coarse, int n_fine, int m_fine, double *coarse, double *fine) {
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
            fine[IDX(i, j, m_fine)] =
                (1 - dx) * (1 - dy) * coarse[IDX(x0, y0, m_coarse)] +
                dx * (1 - dy) * coarse[IDX(x1, y0, m_coarse)] +
                (1 - dx) * dy * coarse[IDX(x0, y1, m_coarse)] +
                dx * dy * coarse[IDX(x1, y1, m_coarse)];
        }
    }
}

// fine includes the zero buffer
// sand is only buffered in n
void sandpile2f(int n_sand, int m_sand, int n_fine, int m_fine, char* coarse, double *fine) {
    double scale_y = ((double)(n_sand - 3.0001)) / (n_fine - 3);
    double scale_x = ((double)(m_sand - 1.0001)) / (m_fine - 3);

    for (int i = 1; i < n_fine - 1; i++) {
        for (int j = 1; j < m_fine - 1; j++) {
            double x = (j-1) * scale_x;
            double y = (i-1) * scale_y;

            int x0 = (int)floor(x);
            int y0 = (int)floor(y);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            double dx = x - x0;
            double dy = y - y0;

            // force nearest neighbour interpolation
            // dx = round(dx);
            // dy = round(dy);

            // Bilinear interpolation
            fine[IDX(i, j, m_fine)] =
                (1 - dx) * (1 - dy) * coarse[IDX(1+y0, x0, m_sand)] +
                dx * (1 - dy) * coarse[IDX(1+y0, x1, m_sand)] +
                (1 - dx) * dy * coarse[IDX(1+y1, x0, m_sand)] +
                dx * dy * coarse[IDX(1+y1, x1, m_sand)];
        }
    }
}

void residual(int n, int m, double *u, double *f, double *res) {
    double h2 = ((FULL_N-1)*(FULL_M-1)) * 1.0 / ((n-1)*(m-1));
    for (int i = 1; i < n-1; i++) {
        for (int j = 1; j < m-1; j++) {
            res[IDX(i, j, m)] = -f[IDX(i, j, m)] + (
                u[IDX(i-1, j, m)] + u[IDX(i+1, j, m)] +
                u[IDX(i, j-1, m)] + u[IDX(i, j+1, m)] -
                4 * u[IDX(i, j, m)]) / h2;
        }
    }
}

double max_norm(int n, int m, double *arr) {
    double max_norm = arr[IDX(0, 0, m)];
    double r;
    for (int i = 1; i < n-1; i++) {
        for (int j = 1; j < m-1; j++) {
            r = arr[IDX(i, j, m)];
            if (max_norm < sqrt(r*r)) {
                max_norm = sqrt(r*r);
            }
        }
    }
    return max_norm;
}

double dl2_norm(int n, int m, double *arr) {
    double norm = 0;
    double h2 = ((FULL_N-1)*(FULL_M-1)) * 1.0 / ((n-1)*(m-1));
    for (int i = 1; i < n-1; i++) {
        for (int j = 1; j < m-1; j++) {
            norm += arr[IDX(i, j, m)] * arr[IDX(i, j, m)];
        }
    }
    norm *= h2;
    norm = sqrt(norm);
    return norm;
}

// -4v11 + v12 + v21 = f11 * h2

void vcycle(int n, int m, double *u, double *f, int depth) {

    if (n <= 4 || m <= 4) {
        assert(n == m);
        assert(n >= 3);
        double h2 = ((FULL_N-1)*(FULL_M-1)) * 1.0 / ((n-1)*(m-1));
        if (n == 3) {
            u[IDX(1, 1, m)] = -0.25 * f[IDX(1, 1, m)] * h2;
        } else {
            // assume symmetric solution
            u[IDX(1, 1, m)] = -0.5 * f[IDX(1, 1, m)] * h2;
            u[IDX(1, 2, m)] = -0.5 * f[IDX(1, 2, m)] * h2;
            u[IDX(2, 1, m)] = -0.5 * f[IDX(2, 1, m)] * h2;
            u[IDX(2, 2, m)] = -0.5 * f[IDX(2, 2, m)] * h2;
        }
        return;
    }

    double *res = malloc(n * m * sizeof(double));
    double *u_correction = malloc(n * m * sizeof(double));
    int n_coarse = (n + 1)/2;
    int m_coarse = (m + 1)/2;
    double *coarse_u = malloc(n_coarse * m_coarse * sizeof(double));
    double *coarse_f = malloc(n_coarse * m_coarse * sizeof(double));

    gauss_seidel(n, m, u, f, 2);
    residual(n, m, u, f, res);

    inject(n, m, res, coarse_f);
    for (int i = 0; i < n_coarse; i++) {
        for (int j = 0; j < m_coarse; j++) {
            coarse_u[IDX(i, j, m_coarse)] = 0.0;
        }
    }

    vcycle(n_coarse, m_coarse, coarse_u, coarse_f, depth+1);

    interpolate(n_coarse, m_coarse, n, m, coarse_u, u_correction);
    for (int i = 1; i < n-1; i++) {
        for (int j = 1; j < m-1; j++) {
            u[IDX(i, j, m)] -= u_correction[IDX(i, j, m)];
        }
    }

    gauss_seidel(n, m, u, f, 1);

    free(res);
    free(u_correction);
    free(coarse_u);
    free(coarse_f);
}

void fmg(int n, int m, double *u, double *f, int depth) {
    if (n <= 4 || m <= 4) {
        assert(n == m);
        assert(n >= 3);
        vcycle(n, m, u, f, 0);
        return;
    }

    int n_coarse = (n + 1)/2;
    int m_coarse = (m + 1)/2;
    double *coarse_u = malloc(n_coarse * m_coarse * sizeof(double));
    double *coarse_f = malloc(n_coarse * m_coarse * sizeof(double));
    double* res = calloc(n * (m+2), sizeof(double));
    double norm = LONG_MAX;

    for (int i = 0; i < n_coarse; i++) {
        for (int j = 0; j < m_coarse; j++) {
            coarse_u[IDX(i, j, m_coarse)] = 0.0;
        }
    }

    inject(n, m, f, coarse_f);
    fmg(n_coarse, m_coarse, coarse_u, coarse_f, depth+1);
    interpolate(n_coarse, m_coarse, n, m, coarse_u, u);

    while (norm > FMG_CONVERGENCE_LIMIT) {
        vcycle(n, m, u, f, 0);
        residual(n, m, u, f, res);
        norm = max_norm(n, m, res);
    }

    free(res);
    free(coarse_u);
    free(coarse_f);
}

void
poisson_identity_helper(int n, int m,
                        int small_n, int small_m,
                        char* small_identity, char* identity,
                        double amplification, int step_size) {
    FULL_N = n;
    FULL_M = m+2;
    double* u = calloc(n * (m+2), sizeof(double));
    double* f = calloc(n * (m+2), sizeof(double));
    double* res = calloc(n * (m+2), sizeof(double));
    int* ident_copy = calloc(n * m, sizeof(int));
    sandpile2f(small_n, small_m, n, m+2, small_identity, f);
    // render_d(n, m+2, f);

    for (int i = 1; i < n-1; i++) {
        for (int j = 1; j < m+1; j++) {
            f[IDX(i, j, m+2)] *= amplification;
        }
    }

    fmg(n, m+2, u, f, 0);
    enforce_symmetry(n, m+2, u); // require u to have 4 fold symmetry
    residual(n, m+2, u, f, res);
    fprintf(stderr, "Discrete L2 Norm: %lf \t| Max Norm: %lf\n",
            dl2_norm(n, m+2, res), max_norm(n, m+2, res));
    // render_d(n, m+2, u);

    for (int i = 1; i < n-1; i++) {
        for (int j = 0; j < m; j++) {
            int chips = (int) (
                -4 * round(u[IDX(i, j+1, m+2)])
                +    round(u[IDX(i+1, j+1, m+2)])
                +    round(u[IDX(i, j+2, m+2)])
                +    round(u[IDX(i-1, j+1, m+2)])
                +    round(u[IDX(i, j+0, m+2)])
            );
            assert(chips >= -8);
            assert(chips < 12);
            identity[IDX(i, j, m)] = (char) (chips);
        }
    }

    // render_i(n, identity);
    assert_symmetry(n, m, identity);
    stabilize_dense_topleft(n-2, m, identity, 0);
    // render_i(n, identity);

    int converged = 0;
    int burns = 0;

    do {
        // copy into s_copy
        for (int i = 1; i < n-1; i++) {
            for (int j = 0; j < m; j++) {
                ident_copy[IDX(i, j, m)] = identity[IDX(i, j, m)];
            }
        }

        // Add burning config
        for (int x = 0; x < m; x++) {
            identity[IDX(1, x, m)] += step_size;
            identity[IDX(n-2, x, m)] += step_size;
        }
        for (int y = 1; y < n-1; y++) {
            identity[IDX(y, 0, m)]   += step_size;
            identity[IDX(y, m-1, m)] += step_size;
        }

        stabilize_dense_topleft(n-2, m, identity, 0);
        // render_i(n, identity);
        burns++;

        // Check convergance
        converged = 1;
        for (int i = 1; i < n-1; i++) {
            for (int j = 0; j < m; j++) {
                if (ident_copy[IDX(i, j, m)] != identity[IDX(i, j, m)]) {
                    converged = 0;
                }
            }
        }
    } while (!converged);

    free(u);
    free(f);
    free(res);
    free(ident_copy);
}

void
poisson_identity(int n, int m) {
    // 1.1 malloc all the identity grids we need
    // 1.2 maybe align them
    // 2.  Direct solve the smallest identity
    // 3.  Call helper func to solve all other identity grids

    // For now rectangular grids wont work
    assert(n == m);
    int grid_count = 1; // start at one to account for direct solve grid
    int tmp_n = n;
    while (tmp_n > DIRECT_SOLVE_SIZE) {
        tmp_n /= 2;
        grid_count++;
    }

    char** grids = malloc(sizeof(char*) * grid_count);
    int* ns = malloc(sizeof(int) * grid_count);
    int* ms = malloc(sizeof(int) * grid_count);
    int tmp_m = m;
    tmp_n = n;
    for (int i = 0; i < grid_count; i++) {
        grids[grid_count - i - 1] = aligned_alloc(ALIGNMENT, sizeof(char) * (tmp_n+2) * (tmp_m));
        ns[grid_count - i - 1] = tmp_n+2;
        ms[grid_count - i - 1] = tmp_m;
        tmp_n /= 2;
        tmp_m /= 2;
    }

    // Use subtraction algo to directly solve smallest identity
    char* direct_grid = grids[0];
    for (int i = 1; i < ns[0]-1; i++) {
        for (int j = 0; j < ms[0]; j++) {
            direct_grid[IDX(i, j, ms[0])] = 6;
        }
    }
    stabilize_dense(ns[0]-2, ms[0], direct_grid, 0);
    for (int i = 1; i < ns[0]-1; i++) {
        for (int j = 0; j < ms[0]; j++) {
            direct_grid[IDX(i, j, ms[0])] = 6 - direct_grid[IDX(i, j, ms[0])];
        }
    }
    stabilize_dense(ns[0]-2, ms[0], direct_grid, 0);

    // Iterate upto the largest grid
    for (int i = 1; i < grid_count; i++) {
        poisson_identity_helper(ns[i], ms[i], ns[i-1], ms[i-1], grids[i-1], grids[i], 1.0004, 40);
    }

    // Party!!!
    render_i(ns[grid_count-1], grids[grid_count-1]);

    free(ns);
    free(ms);
    for (int i = 0; i < grid_count; i++) {
        free(grids[i]);
    }
    free(grids);
}

// TODO:
// 3. Tweak base solver so that we can solve rectangular grids
int
main(void)
{
    setlocale(LC_ALL, "");
    omp_set_num_threads(8);
    poisson_identity(8192, 8192); // TODO: goal hit 16384
}
