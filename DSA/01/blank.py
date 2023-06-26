import time

def timer(func):
    def wrapper(*args, **kwargs):
        s = time.time()
        func(*args, **kwargs)
        e = time.time()
        print(f"t: {e-s:.5e}")
    return wrapper

@timer
def print_even(n):
    for i in range(2, n+1, 2):
        pass

if __name__ == "__main__":
    print_even(100000)
