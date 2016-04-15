from collections import Counter
import random

# Create random array 
def randomArray(totalNumbers,min,max):
    """Creates a random array of numbers"""
    array = []
    while totalNumbers > 0:
        array.append(random.randrange(min,max))
        totalNumbers -= 1
    return array

# Check a string to see if it contains all unique characters

# O(n) using a dictionary
def unique(word):
    dic = Counter()
    if len(word) > 128:
        return False
    for char in word:
        dic[char] += 1
    for key,value in dic.items():
        if value > 1:
            return False
    return True

# O(n^2) using no data structures
def unique2(word):
    if len(word) > 128:
        return False
    for char in word:
        count = 0
        for charMatch in word:
            if char == charMatch:
                count += 1
            if count >= 2:
                return False
    return True

# Check two strings to see if one is a permutation of the other

# O(n log n) - This is the time complexity for Python's sort function
def permutation(word1, word2):
    if sorted(word1) == sorted(word2):
        return True
    else:
        return False

# Binary Search (recursive) - O(log n)
def binary(array,mini,maxi,num):
    mid = (mini+maxi)//2
    if mid == num:
        return True
    if mid < array[0] or mid > len(array) or num > len(array) or num < array[0]:
        return False
    if num > mid:
        binary(array,mid+1,len(array),num)
    elif num < mid:
        binary(array,mini,mid-1,num)

# O(n)? - Replace spaces in a string with '%20'
def replace(s):
    string = [w.replace(' ', '%20') for w in s]
    result = "".join(string)

    return result

# O(n) - Replace spaces in a string with '%20'
def replace2(s):
    spaceCount = 0
    for i in s:
        if i == " ":
            spaceCount += 1
    c = len(s) + spaceCount*2
    newS = [None] *c
    index = 0
    for i in s:
        if i != " ":
            newS[index] = i
            index += 1
        else:
            newS[index] = '%'
            newS[index+1] = "2"
            newS[index+2] = "0"
            index += 3
    result = "".join(newS)
    return result

# Check string to see if it a permutation of a palindrome

# O(n)
def palindrome(string):
    return sum(v % 2 for v in Counter(string).values()) <= 1

# O(n^2)
def palindrome2(string):
    s = [c.replace(' ', '') for c in string]
    merged = "".join(s)
    srt = sorted(merged)
    dic = {}
    singles = 0

    for i in srt:
        if i not in dic:
            dic[i] = 1
        else:
            for key, value in dic.items():
                if key == i:
                    dic[key] = 2 
    for key, value in dic.items():
        if value == 1:
            singles += 1
    if singles > 1:
        return False
    else:
        return True

# O(s*k) - String compression a-z (totals only)
def compress(string):
    s = string.lower()
    dic = {}
    for c in s:
        if c not in dic:
            dic[c] = 1
        else:
            for key, value in dic.items():
                if c == key:
                    dic[key] += 1
    compressed = ""
    for key, value in dic.items():
        compressed = compressed + key + str(value)
    return compressed

# O(n) - String compression a-z
def compress2(s):
    newS = []
    i = 0
    cnt = 1
    for c in s:
        if i == 0:
            newS.append(c)
            i += 1
        elif i >= 1:
            if c == newS[i-1]:
                cnt += 1
            else:
                newS.append(str(cnt))
                i += (1 + len(str(cnt)))
                cnt = 1
                newS.append(c)
    newS.append(str(cnt))
    result = ''.join(newS)
    return result

# Rotate matrix

# O(n^2) - Rotate matrix
def rotateMatrix2(m):
    layers = int(len(m) / 2)
    length = len(m) - 1
 
    for layer in range(layers): #for each layer
        for i in range(layer, length - layer): # loop through the elements we need to change at each layer
            temp = m[layer][i] #save the top element, it takes just one variable as extra memory
            #Left -> Top
            m[layer][i] = m[length - i][layer]
            #Bottom -> left
            m[length - i][layer] = m[length - layer][length - i]
            #Right -> bottom
            m[length - layer][length - i] = m[i][length - layer]
            #Top -> Right
            m[i][length - layer] = temp
    return m

# O(n^2) - Rotate matrix
def rotateMatrix3(matrix):
    size = len(matrix) 
    # init rotated matrix with None elements
    rotated_matrix = [[None]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            rotated_matrix[j][size-1-i] = matrix[i][j]
    return rotated_matrix

# Check if string is a rotation of another string

# O(s1*s2)
def substring(s1,s2):
    reord = []
    for i in s1:
        for j in s2:
            if i == j:
                reord.append(j)
                break
            else:
                reord = []
    result = str(''.join(reord))
    if result == "":
        return False
    elif result in s1:
        return True
    else:
        return False

# O(n)
def substring2(s1,s2):
    length = len(s1)
    if length == len(s2) and length > 0:
        s1s1 = s1 + s1
        return s2 in s1s1
    return False

# Stock trading backtest - O(n)
def backTest(s):
    maxProfit = 0
    min = s[0]
    for p in s:
        if p < min:
            min = p
        if p - min > maxProfit:
            maxProfit = p - min

    return maxProfit

# Check if a string is a palindrome - O(1)?
def palindrome(s):
    if s == s[::-1]:
        return True
    else:
        return False
            

# Create array with duplicate
def dupArray():
    array = []
    for i in range(1,101):
        if i == 100:
            array.append(i)
            array.append(i)
        else:
            array.append(i)
    return array

# Find duplicate number in array - O(n)
def dupNum(a):
    for i in range(1,len(a)+1):
        if i != a[i-1]:
            return i-1
