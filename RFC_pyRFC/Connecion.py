import pyrfc

#amp 192.168.0.7
# ======================= IP OK
ASHOST='192.168.0.7'   
CLIENT='800'
SYSNR='60'
SYSID='AMP'
USER='sehun.jung'
PASSWD='qwer1234'

conn = pyrfc.Connection(ashost=ASHOST, sysnr=SYSNR, client=CLIENT, user=USER, passwd=PASSWD, sysid=SYSID)

# #======================= URL not working.
# GWHOST='armiq.mynetgear.com'
# # ASHOST=''   
# CLIENT='800'
# SYSNR='60'
# SYSID='AMP'
# USER='sehun.jung'
# PASSWD='qwer1234'
# trace='3'

# # SAPROUTER = 'H/armiq.mynetgear.com/S/3260'

# # conn = pyrfc.Connection(gwhost=GWHOST, client=CLIENT, user=USER, passwd=PASSWD, saprouter=SAPROUTER, trace=trace)
# conn = pyrfc.Connection(gwhost=GWHOST, client=CLIENT, user=USER, passwd=PASSWD, sysnr=SYSNR, sysid=SYSID)