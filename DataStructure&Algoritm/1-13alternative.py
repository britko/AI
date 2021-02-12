# +와 -를 번갈아 출력하기

n = int(input('출력 개수: '))

for _ in range(n // 2): # +-를 n//2개 출력, range()함수가 for문을 순환하며 반환하는 인덱스를 사용할 필요가 없기때문에 언더스코어(_)를 사용
    print('+-', end='')
if n % 2: # 홀수이면 마지막으로 +를 출력
    print('+',end='')
print()