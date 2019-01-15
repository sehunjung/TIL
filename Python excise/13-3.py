

import urllib.request, urllib.parse, urllib.error
# pip install bs4
from bs4 import BeautifulSoup
# ssl은 내장 라이브러리
import ssl

# Ignore SSL certificate errors =>  그냥 사용 함.
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# 주소 입력
url = input('Enter - ')
# 주소에 요청해서 결과 받음
html = urllib.request.urlopen(url, context=ctx).read()
# html 수정 및 정리
soup = BeautifulSoup(html, 'html.parser')

# Retrieve all of the anchor tags
tags = soup('a')
print(soup)
print(tags)
for tag in tags:
    print(tag.get('href', None))