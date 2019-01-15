
# 소켓 라이브러러 사용 선언...내장 라이브러리
import socket

#객체 생성
mysock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#커넥션 http 80 port
mysock.connect(('data.pr4e.org', 80))

# command GET(요청) 실행명령어 encode
cmd = 'GET http://data.pr4e.org/romeo.txt HTTP/1.0\r\n\r\n'.encode()
# 명령어 전송
mysock.send(cmd)

# 전송 완료 대기
while True:
    data = mysock.recv(512) # 크기는 상관 없음..
    if (len(data) < 1): # 전송 완료
        break

    # print(data)
    # print("=================")    
    print(data.decode(),end='')
    # print("=================")    
    # print(data.decode())

 # 완료후 소켓 닫기   
mysock.close()