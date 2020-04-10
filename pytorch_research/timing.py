import time


class Timer():

    def __init__(self):
        self.start_time = self.elapsed_time = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.elapsed_time = time.time() - self.start_time
        print(f'finished in {self.elapsed_time * 1000:2f} ms')


# # Example
# with Timer() as t:
#     run_some_code()
#
# print(t.elapsed_time)
