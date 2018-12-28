

count = 0
isNum = 0
avg = 0
total = 0

maxium = -1
minimum = None

while True:

    number = input("Enter a Number: ")

    if number == "done":
        break
    else:
        try:
            isNum = float(number)
        except:
            print("Error, please enter numeric input")
            print("Try again")

    total = isNum + total 
    count = count + 1

    if isNum > maxium:
        maxium = isNum
    if isNum < minimum:  #????
        manimium = isNum

if total > 0 and count > 0:
    avg = total/count

print(total, count, avg, maxium, minimum)


