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
#include <assert.h>
#include <immintrin.h>

#ifndef N
#  define N 510
#endif

#define TILE_WIDTH 85
#define TILE_HEIGHT 85
#define xTILES (N/TILE_WIDTH)
#define yTILES (N/TILE_HEIGHT)

#define TOP 0
#define LEFT 1
#define BOTTOM 2
#define RIGHT 3

/* Color palette */
#define C0 0x000000
#define C1 0x333333
#define C2 0x666666
#define C3 0x999999
#define CX 0xff0000

static unsigned short** tiles;
static int** messages;

static void
make_tiles(void)
{
    tiles = calloc(yTILES * xTILES, sizeof(unsigned short*));
    for (int yy = 0; yy < yTILES; yy++) {
        for (int xx = 0; xx < xTILES; xx++) {
            tiles[yy * xTILES + xx] = calloc(TILE_WIDTH * TILE_HEIGHT, sizeof(unsigned short));
        }
    }
}

static void
make_messages(void)
{
    messages = calloc(yTILES * xTILES * 4, sizeof(int*));
    for (int yy = 0; yy < yTILES; yy++) {
        for (int xx = 0; xx < xTILES; xx++) {
            for (int direction = 0; direction < 4; direction++) {
                int tileIdx = yy * xTILES + xx;
                size_t count;
                if (direction % 2 == 0) {
                    count = TILE_WIDTH;
                } else {
                    count = TILE_HEIGHT;
                }
                messages[tileIdx * 4 + direction] = calloc(count, sizeof(int));
            }
        }
    }
}

static inline int
message_idx(int tileIdx, int direction)
{
    return tileIdx * 4 + direction;
}

static inline int
tile_idx(int y, int x)
{
    return (y / TILE_HEIGHT) * xTILES + (x / TILE_WIDTH);
}

static inline int
local_idx(int y, int x)
{
    return (y % TILE_HEIGHT) * TILE_WIDTH + (x % TILE_WIDTH);
}

static void
render(void)
{
    static unsigned char buf[3L*N*N];
    static const long colors[] = {C0, C1, C2, C3};
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            int tileIdx = tile_idx(y, x);
            int localIdx = local_idx(y, x);
            unsigned short v = tiles[tileIdx][localIdx];
            long c = v < 4 && v >= 0 ? colors[v] : CX;
            buf[y*3L*N + x*3L + 0] = c >> 16;
            buf[y*3L*N + x*3L + 1] = c >>  8;
            buf[y*3L*N + x*3L + 2] = c >>  0;
        }
    }
    printf("P6\n%d %d\n255\n", N, N);
    fwrite(buf, sizeof(buf), 1, stdout);
}

static void
stabilize(void)
{
    long globalSpills;
    long localSpills;
    long gFrames = 0;
    do {
        globalSpills = 0;
        for (int tileIdx = 0; tileIdx < xTILES * yTILES; tileIdx++) {
            // import sand from messages
            int imports = 0;
            for (int direction = 0; direction < 4; direction++) {
                int importTileIdx = -1;
                switch (direction) {
                    case BOTTOM:
                        if (tileIdx < xTILES) {
                            break;
                        }
                        importTileIdx = tileIdx - xTILES;
                        break;
                    case TOP:
                        if (tileIdx >= yTILES * (xTILES - 1)) {
                            break;
                        }
                        importTileIdx = tileIdx + xTILES;
                        break;
                    case RIGHT:
                        if (tileIdx % xTILES == 0) {
                            break;
                        }
                        importTileIdx = tileIdx - 1;
                        break;
                    case LEFT:
                        if ((1 + tileIdx) % xTILES == 0) {
                            break;
                        }
                        importTileIdx = tileIdx + 1;
                        break;
                }

                if (importTileIdx == -1) {
                    continue;
                }

                assert(importTileIdx < xTILES * yTILES);
                int* msg = messages[message_idx(importTileIdx, direction)];
                switch (direction) {
                    case BOTTOM:
                        for (int i = 0; i < TILE_WIDTH; i++) {
                            int localIdx = i;
                            assert(msg[i] >= 0);
                            assert(msg[i] < 256 * 256);
                            tiles[tileIdx][localIdx] += msg[i];
                            msg[i] = 0;
                        }
                        break;
                    case TOP:
                        for (int i = 0; i < TILE_WIDTH; i++) {
                            int localIdx = TILE_WIDTH * (TILE_HEIGHT - 1) + i;
                            assert(msg[i] < 256 * 256);
                            assert(msg[i] >= 0);
                            tiles[tileIdx][localIdx] += msg[i];
                            msg[i] = 0;
                        }
                        break;
                    case LEFT:
                        for (int i = 0; i < TILE_HEIGHT; i++) {
                            int localIdx = TILE_WIDTH * (i + 1) - 1;
                            assert(msg[i] < 256 * 256);
                            assert(msg[i] >= 0);
                            tiles[tileIdx][localIdx] += msg[i];
                            msg[i] = 0;
                        }
                        break;
                    case RIGHT:
                        for (int i = 0; i < TILE_HEIGHT; i++) {
                            int localIdx = TILE_WIDTH * i;
                            assert(msg[i] < 256 * 256);
                            assert(msg[i] >= 0);
                            tiles[tileIdx][localIdx] += msg[i];
                            msg[i] = 0;
                        }
                        break;
                }
            }

            // Topple tile
            do {
                localSpills = 0;
                for (int localIdx = 0; localIdx < TILE_HEIGHT * TILE_WIDTH; localIdx++) {
                    unsigned short* v = &tiles[tileIdx][localIdx];
                    assert(*v >= 0);
                    if (*v < 4) {
                        continue;
                    }

                    localSpills++;
                    globalSpills++;
                    *v -= 4;
                    if (localIdx % TILE_WIDTH != TILE_WIDTH - 1) {
                        *(v + 1) += 1;
                    } else {
                        messages[message_idx(tileIdx, RIGHT)][localIdx / TILE_WIDTH] += 1;
                    }
                    if (localIdx % TILE_WIDTH != 0) {
                        *(v - 1) += 1;
                    } else {
                        messages[message_idx(tileIdx, LEFT)][localIdx / TILE_WIDTH] += 1;
                    }
                    if (localIdx < TILE_WIDTH * (TILE_HEIGHT - 1)) {
                        *(v + TILE_WIDTH) += 1;
                    } else {
                        messages[message_idx(tileIdx, BOTTOM)][localIdx % TILE_WIDTH] += 1;
                    }
                    if (localIdx >= TILE_WIDTH) {
                        *(v - TILE_WIDTH) += 1;
                    } else {
                        messages[message_idx(tileIdx, TOP)][localIdx % TILE_WIDTH] += 1;
                    }
                }
            } while (localSpills > 0);
        }

        fprintf(stderr, "Total Spills: %ld\tGlobal Frames: %ld\n", globalSpills, ++gFrames);

        // TODO: remove
        // render();
    } while (globalSpills > 0);
}

int
main(void)
{
    assert(N % TILE_HEIGHT == 0);
    assert(N % TILE_WIDTH == 0);

    make_tiles();
    make_messages();
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            tiles[tile_idx(y, x)][local_idx(y, x)] = 6;
        }
    }
    stabilize();
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            int tileIdx = tile_idx(y, x);
            int localIdx = local_idx(y, x);
            tiles[tileIdx][localIdx] = 6 - tiles[tileIdx][localIdx];
        }
    }
    stabilize();
    render();
}
