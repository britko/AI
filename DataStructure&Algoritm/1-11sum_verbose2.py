a = int(input('정수 a를 입력: '))
b = int(input('정수 b를 입력: '))

if a > b:
    a, b = b, a

sum = 0
for i in range(a, b):
    print(f'{i} + ', end='')
    sum += i

print(f'{b} = ', end='')
sum += b

print(sum)