# Yes hello
This is my report for my amazing parallelized ~~GPU~~ implementation that I am writing at 19:28 as
I desperately try to ~~finish my thesis while getting this done as well~~ cope with the ML math quiz I just took.

I'm not writing this in a PDF so enjoy the markdown.

I parallelized the for loop using openmp. Each iteration in the loop is
distributed across threads. `collapse(2)` merges i and j loops into a single iteration
space, which apparently helps balance the load on wide images. I didn't use `reduction`
since there's no data races to be concerned about, as all pixels being written to are unique.


