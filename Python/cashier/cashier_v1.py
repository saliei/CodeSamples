import sys

cashmap = {"PENNY":0.01, "NICKEL":0.05, "DIME": .10, "QUARTER": 0.25,
        "HALF DOLLAR": 0.50, "ONE": 1.00, "TWO": 2.00, "FIVE": 5.00,
        "TEN": 10.00, "TWENTY": 20.00, "FIFTY": 50.00, "ONE HUNDRED": 100.00}

  
def printrm(l, d):
    for idx, val in enumerate(l):
        for s,n in cashmap.items():
            if val == n and d[idx] != 0:
                print(s, d[idx])

def findrm(rm):
    # in python 2.x no need to reverse!
    l = [val for val in cashmap.values() if val <= rm]
    l.reverse()
    idx = 0
    d = []
    while (idx < len(l)):
        v = int(rm / l[idx])
        d.append(v)
        rm = float("{0:.2f}".format(rm - v * l[idx]))
        idx += 1
    printrm(l, d)

if __name__ == "__main__":
    for line in sys.stdin:
        try:
            line = line.strip().split(";")
            try:
                pp = float(line[0])
            except ValueError:
                print("ERROR")
                exit(1)
            try:
                ch = float(line[1])
            except ValueError:
                print("ERROR")
                exit(1)
            if(ch - pp < 0.0):
                print("ERROR")
                exit(1)
            if(ch == pp):
                print("ZERO")
                exit()
            rm = float("{0:.2f}".format(ch - pp))
            findrm(rm)
        except KeyboardInterrupt:
            break
        if not line:
            break

