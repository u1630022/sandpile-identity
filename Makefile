.PHONY: runt2
.PHONY: t2dbg
.PHONY: clean

np: np-sandpile.c
	gcc -O3 -march=native -fopenmp np-sandpile.c -o np

tile2: tile2-sandpile.c
	gcc -O3 -march=native -fopenmp tile2-sandpile.c -lm -o tile2

runt2: tile2
	/usr/lib/linux-tools-6.8.0-88/perf stat -e 'cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,L1-icache-loads,L1-icache-misses,branch-misses,alignment-faults,stalled-cycles-frontend,stalled-cycles-backend,sse_avx_stalls,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,context-switches' ./tile2 > /dev/null

t2dbg: tile2-sandpile.c
	gcc -O3 -fsanitize=address -static-libasan -g3 -march=native -fopenmp tile2-sandpile.c -lm -o t2-debug
	gdb ./t2-debug

clean:
	rm -f t2-debug
	rm -f tile2
	rm -f np
	rm -f *.ppm
