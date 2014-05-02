'''
Mean_Average_Precision: 
  takes two arguments:
    the first argument is a list of the correct values
    the second argument is a list of tuples, the values in the tuple are the guesses
  in its current form this code is only suitable for use in the Best Buy competitionand the fact that it is MAP@5 is gauranteed if 5 guesses are provided for each known value

Max_Score:
  takes two arguments:
    the first argument is a list of the correct values
    the second argument is a list of tuples, the values in the tuple are the guesses
  This function computes the best possible score given the guesses provided (it treats any correct guess as a first guess).  This is useful for seeing if we are predominatly loosing points due to ordering or if it is due to not getting a right answer at all.

Sample Use:
act = [1,1,4,4,5]
pre = [(1,2,3,4,5),(1,2,3,4,5),(1,2,3,4,5),(1,2,3,4,5),(1,2,3,4,5)]

print Mean_Average_Precision(act,pre)
=0.54
print Max_Score(act,pre)
=1.0
'''
def Mean_Average_Precision(actual, predicted):
    if len(actual) != len(predicted):
        print "actual and predicted don't have same number of elements"
        return
    mean_average_precision = 0
    for i,p in zip(actual, predicted):
        n = float(1)
        average_precision = 0
        for guess in p:
            if i == guess:
                average_precision = 1/n
                break
            else:                    
                n += 1
        mean_average_precision += average_precision
    mean_average_precision /= len(actual)
    return mean_average_precision


def Max_Score(actual, predicted):
    if len(actual) != len(predicted):
        print "actual and predicted don't have same number of elements"
        return
    max_score = float(0)
    for i,p in zip(actual, predicted):
        if i in p:
            max_score += 1
    max_score /= len(actual)
    return max_score
