def getStateNumber(state):
    number = 1 if state[0] == 'green' else 0
    number += 2 * (1 if state[1] else 0)
    number += 4 * (1 if state[2] else 0)
    number += 8 * (1 if state[3] else 0)
    number += 16 * (1 if state[5] else 0)
    if state[4] == 'left':
        number += 32*0
    elif state[4] == 'right':
        number += 32*1
    else :
        number +=  32*2

    return number

numbers=[]
for light in ['green', 'red']:
    for ongoing in [True, False]:
        for right in [True, False]:
            for left in [True, False]:
                for waypoint in ['right', 'left','forward']:
                    for deadline in [True, False]:
                        state=[light,ongoing,right,left,waypoint,deadline]
                        number = getStateNumber(state)
                        numbers.append(number)
                        print(" {} {}".format(state,number))

print sorted(numbers)
print(len(numbers))
print(len(set(numbers)))