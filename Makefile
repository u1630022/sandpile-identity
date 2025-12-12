.PHONY: run
.PHONY: clean
.PHONY: asan
.PHONY: debug

sand: my-sandpile.c
	gcc -O3 -march=native -fopenmp my-sandpile.c -o sand

np: np-sandpile.c
	gcc -O3 -march=native -fopenmp np-sandpile.c -o np

tile2: tile2-sandpile.c
	gcc -O3 -march=native -fopenmp tile2-sandpile.c -o tile2

debug: my-sandpile.c
	gcc -g3 -march=native -fopenmp my-sandpile.c -o sand-debug
	gdb ./sand-debug

asan: my-sandpile.c
	gcc -g -fsanitize=address -march=native -fopenmp my-sandpile.c -o sand-asan
	./sand-asan > /dev/null

run: sand
	/usr/lib/linux-tools-6.8.0-88/perf stat -e 'cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,L1-icache-loads,L1-icache-misses,branch-misses,alignment-faults,stalled-cycles-frontend,stalled-cycles-backend,sse_avx_stalls,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,context-switches' ./sand > /dev/null

runt2: tile2
	/usr/lib/linux-tools-6.8.0-88/perf stat -e 'cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,L1-icache-loads,L1-icache-misses,branch-misses,alignment-faults,stalled-cycles-frontend,stalled-cycles-backend,sse_avx_stalls,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,context-switches' ./tile2 > /dev/null

t2dbg: tile2-sandpile.c
	gcc -g3 -march=native -fopenmp tile2-sandpile.c -o t2-debug
	gdb ./t2-debug

clean:
	rm -f sand
	rm -f sand-debug
	rm -f t2-debug
	rm -f tile2
	rm -f np
	rm -f *.ppm
