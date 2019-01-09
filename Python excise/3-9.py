

# file name input
fname = input('Enter File: ')
# 문자열 0 이하면 자동으로 샘플  파일 자동입력
if len(fname) < 1 :
    # fname = 'C:\==Task\Github\TIL\Python excise\clown.txt'
    fname = 'C:\==Task\Github\TIL\Python excise\intro.txt'
    
# 파일 조회..라인별로 메모리 로드    
hand = open(fname)

# 딕셔너리 선언.
di = dict()
# 전체 파일 내용 라인별로 실행
for lin in hand :
    # 오른쪽 공백 및 뉴라인 제거
    lin = lin.rstrip()
    # 라인을 ㅡ스패이스 단위로 눠서 리스트 wds에 저장.
    wds = lin.split()
    # 리스트 wds를 실행하면서
    for w in wds :
        # 라이별로 단어별 수량을 계산
        # 키밸류 비교를 자동으로 해주어서 수치가 자동 계산됨.
        di[w] = di.get(w,0) + 1

# 최종 정리할 리스트를 만듬.        
tmp = list()
# 두개의 필드를 가진 딕셔너리 di의 값을 k, v에 할당하면서 반복
for k, v in di.items() :
    # 튜플을 만들어서 키와 밸류를 나누어서 저장.()는 튜플..[]는 리스트..
    newt = (v,k)
    # 튜플을 tmp에 추가 
    tmp.append(newt)

# 밸류가 키로 바뀐 튜플이 저장된 리스트 tmp를 역순 정렬
tmp = sorted(tmp, reverse = True)
# 정렬이 되어 있으니 빈도수가 높은 단어 순으로 5개만 출력한다...
# 인덱스틑 0부터 시작이다...5는 5미만...하지만 0부터 시작해서 5개.
for v,k in tmp[:5] :
    # 빈도수 정렬을 위해 값이 앞에 있지만 출력시 순서를 변경한다.
    print(k,v)