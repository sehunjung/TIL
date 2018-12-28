# python 실습..

hour = input("Enter Hours :")

try:
    isInt = int(hour)

    rate = input("Enter Rate :")

    try:
        isFloat = float(rate)

        if isInt > 40:
            addition = isInt - 40
            pay = ( 40 * isFloat )  + ( addition * isFloat * 1.5)
            
        else:
            pay = isInt * isFloat     

        print(pay)  
    except:
        isFloat = "Error, please enter numeric input"
        print(isFloat, "try again from start")
except:
    isInt = "Error, please enter numeric input"
    print(isInt, "try again from start")

print("Done")  




# astr ="MY"

# try:
#     print("hello")
#     islnt = int(astr)
#     print("world")  #except 이후는 실행안됨
# except:
#     islnt = "변환할수 없습니다"

# print("Done", islnt)