
# str = 'X-DSPM-Confidence: 0.8475'

# ipos = str.find(':')
# piece = str[ipos+2:]
# value = float(piece)
# print(value)



# fname = input('Enter the file name:  ')
# try:
#     fhand = open(fname)
# except:
#     print('File cannot be opened: ', fname)
#     quit()

# count = 0
# for line in fhand:
#     if line.startswith('Subject:') :
#         count = count + 1
#         lx = line.rstrip()
#         print(lx.upper())
# print('There were', count, 'subject lines in', fname)



str1 = 'ccccc fffff'
print(len(str1))

str2 = 'ccccc\nfffff'
print(len(str2))