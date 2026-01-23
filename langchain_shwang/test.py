import sys

input = sys.stdin.readline
N = int(input().strip())
answer = 0
x = len(str(N)) - 1
for i in range(1, N + 1):
    for n in range(x):
        i += i // 10 ** (n)
    if i == N:
        answer = i
        break

print(answer)
