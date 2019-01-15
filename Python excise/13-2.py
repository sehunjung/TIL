
import urllib.request, urllib.parse, urllib.error

# 생성 연결 요청 리턴까지 한번에 
fhand = urllib.request.urlopen('http://data.pr4e.org/romeo.txt')
#딕셔너리 생성
counts = dict()
#리턴 값을 순차적으로 리딩
for line in fhand:
    # 라인별로 디코딩 하면서 공백으로 나눠 워드에 쌓음...
    words = line.decode().split()
    #월드를 루프 돌면서 단어, 수량을 묶어서 카운트에 쌓음..
    for word in words:
        counts[word] = counts.get(word, 0) + 1
print(counts)