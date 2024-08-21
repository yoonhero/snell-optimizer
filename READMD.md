# Snell Optimizer

I supposed that you have ever heard of snell's law in physics class. Light travels across the shortest time path. This feature was used in proving the Bernoulli's problem.

I extend it to optimization idea. In `snell.py`, I implement the snell optimizer in 3d space. To cover N space, it needs to some edition of linear algebra part. (+efficient computation)

## Result

![result](./final.png)

I trained the simple network `mlp(3, 3, 3)` on Iris Datset and optimize it with Adam, SGD and mine. Following this result, I assume that my new proposal maybe useful in some trainings.

But universally, **Adam is no-doubting best option for you.**
