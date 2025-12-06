.PHONY: run

sand: my-sandpile.c
	gcc -O3 -march=native -fopenmp my-sandpile.c -o sand

debug: my-sandpile.c
	gcc -g -march=native -fopenmp my-sandpile.c -o debug
	gdb ./debug

run: sand
	/usr/lib/linux-tools-6.8.0-88/perf stat -e 'cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,L1-icache-loads,L1-icache-misses,branch-misses,alignment-faults,stalled-cycles-frontend,stalled-cycles-backend,sse_avx_stalls,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,context-switches' ./sand > /dev/null

clean:
	rm -f sand
	rm -f debug
	rm -f *.ppm
