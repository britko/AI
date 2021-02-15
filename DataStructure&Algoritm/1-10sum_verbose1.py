a = int(input('정수 a를 입력: '))
b = int(input('정수 b를 입력: '))

if a > b:
    a, b = b, a

sum = 0
i = 1

for i in range(a, b+1):
    if i < b:
        print(f'{i} + ', end='')
    else:
        print(f'{i} = ', end='')
    sum += i
print(sum)