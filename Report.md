# Yes hello
This is my report for my amazing parallelized GPU implementation that I am writing at 21:00 as
I desperately try to finish my thesis while getting this done as well. I'm not writing
this in a PDF so enjoy the markdown.

I did, in fact, do a basic shared memory blocking strategy for this, but
that's about all the optimization I can muster right now. It uses
a 3\*18\*18 byte shared memory buffer for all the arithmetic. Also,
I learned that constant memory is a thing that NVIDIA GPUs have, which
can broadcast constant values across a warp in a single cycle, which is neat.


