import timeit


no_loops = 100_000_000


def function():
    for i in range(no_loops):
        pass


timer = timeit.Timer(function)
time_taken = timer.timeit(number=1)

print(f"Time to run function with {no_loops} iterations: {time_taken:.6f} seconds")
print(f"Average time per iteration: {time_taken / no_loops:.2e} seconds")
