# *를 n개 출력하되 w개마다 줄바꿈하기

n = int(input('몇 개를 출력할까요?: '))
w = int(input('몇 개마다 줄바꿈할까요?: '))

# n // w 번 반복
for _ in range(n // w):
    print('*' * w)

# 나누고 남은 나머지를 출력
rest = n % w
if rest:
    print('*' * rest)
    