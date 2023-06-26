import time

def func_time(func):
    def wrapper(*args, **kwargs):
        s = time.time()
        func(*args, **kwargs)
        e = time.time()
        print(f"t: {e-s:.5e}")
        print("===")
    return wrapper

@func_time
def print_even_1(n):
    for i in range(2, n+1):
        if i % 2 == 0:
            # print(i)
            pass

@func_time
def print_even_2(n):
    for i in range(2, n+1, 2):
        # print(i)
        pass

@func_time
def print_even_3(n):
    num = 2
    while num <= n:
        if num % 2 == 0:
            # print(num)
            pass
        num += 1
    
@func_time
def print_even_4(n):
    num = 2
    while num <= n:
        # print(num)
        pass
        num += 2

if __name__ == "__main__":
    print_even_1(1000000)
    print_even_2(1000000)
    print_even_3(1000000)
    print_even_4(1000000)
