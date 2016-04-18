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

# LINKED LISTS

# Node class for singly linked list (Backwards linked)
class Node(object):

    def __init__(self, data=None, next_node=None):
        self.data = data
        self.next_node = next_node

    def get_data(self):
        return self.data

    def get_next(self):
        return self.next_node

    def set_next(self, new_next):
        self.next_node = new_next

# Singly linked list class (Backwards linked)
class LinkedListB(object):
    def __init__(self, head=None):
        self.head = head

    # Insert a node onto the front of the list
    def insert(self, data):
        new_node = Node(data)
        new_node.set_next(self.head)
        self.head = new_node

    # Return the size of a list
    def size(self):
        current = self.head
        count = 0
        while current:
            count += 1
            current = current.get_next()
        return count

    # Search for particular value in a list
    def search(self, data):
        current = self.head
        while current:
            if current.get_data() == data:
                return True
            else:
                current = current.get_next()
        if current is None:
            return False

    # Delete a particular value from a list
    def delete(self, data):
        current = self.head
        previous = None
        found = False
        while current and found is False:
            if current.get_data() == data:
                found = True
            else:
                previous = current
                current = current.get_next()
        if current is None:
            raise ValueError("Data not in list")
        if previous is None:
            self.head = current.get_next()
        else:
            previous.set_next(current.get_next())

    # Delete duplicates
    def deleteDups(self, data):
        dic = {}
        current = self.head
        while current:
            if current.get_data() in dic:
                self.delete(current.get_data())
            else:
                dic[current.get_data()] = 1
            current = current.get_next()


#Doubly Linked List

# Node for doubly linked list
class NodeD:
    def __init__(self, data=None, prev=None, next=None):
        self.data = data
        self.prev = prev
        self.next = next

    # Allow printing of the node
    def __repr__(self):
        return '<{}, {}>'.format(self.data, self.next)

# Doubly linked list class
class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    # Allow printing of the list
    def __repr__(self):
        return '<DoublyLinkedList {}>'.format(self.head)

    # Insert a node onto the front on the list
    def add_head(self, data=None):
        if self.head is None:
            self.head = self.tail = NodeD(data)  # the very fist node
        else:
            new_head = NodeD(data=data, next=self.head)  # prev is None
            self.head.prev = self.head = new_head

    # Insert a node onto the end of the list
    def add_tail(self, data=None):
        if self.tail is None:
            self.head = self.tail = NodeD(data)  # the very first node
        else:
            new_tail = NodeD(data=data, prev=self.tail)  # next is None
            self.tail.next = self.tail = new_tail

    # implements iteration from head to tail
    def __iter__(self):
        current = self.head
        while current is not None:
            yield current
            current = current.next

    # implements iteration from tail to head
    def __reversed__(self):
        current = self.tail
        while current  is not None:
            yield current
            current = current.prev


# Singly Linked List (Forwards linked)

# Singly linked list node class (Forwards linked)
class Node2:
    def __init__(self, data=None, next_node=None):
        self.data = data
        self.next_node = next_node

    # Allow printing of node
    def __repr__(self):
        return '<{}, {}>'.format(self.data, self.next_node)

    def set_data(self, data):
        self.data = data

    def set_next(self, new_next):
        self.next_node = new_next

    def get_data(self):
        return self.data

    def get_next(self):
        return self.next_node

# Singly linked list class (Forwards linked)
class LinkedList2:
    def __init__(self, head=None):
        self.head = head

    # Allow printing of list
    def __repr__(self):
        return '<LinkedList2 {}>'.format(self.head)

    # Create and insert a new node onto the end of a list
    def insert(self, data):
        new_node = Node2(data)
        if self.head is None:
            self.head = new_node
        else:
            current_node = self.head
            while current_node.get_next():
                current_node = current_node.get_next()
            current_node.set_next(new_node)

    # Insert an existing node onto the end of a list
    def insertNode(self, new_node):
        if self.head is None:
            self.head = new_node
        else:
            current_node = self.head
            while current_node.get_next():
                current_node = current_node.get_next()
            current_node.set_next(new_node)

    # Return the size of a list
    def size(self):
        current = self.head
        count = 0
        while current:
            count += 1
            current = current.get_next()
        return count

    # Return the data in the kth to last element in a list
    def kthLast(self,k):
        current = self.head
        size = self.size()
        for _ in range(size-k):
            current = current.get_next()
        return current.get_data()

    # Delete the middle node of a list
    def delMid(self):
        size = self.size()
        middle = int(size/2)
        current = self.head
        index = 0
        while current and index < middle:
            current = current.get_next()
            index += 1
        deleted = current.get_data()
        next_node = current.get_next()
        current.set_data(next_node.get_data())
        current.set_next(next_node.get_next())
        return deleted

    # Delete a particular node in a list
    def delNode(self,n):
        if n == None or n.get_next() == None:
            return False
        else:
            next_node = n.get_next()
            n.set_data(next_node.get_data())
            n.set_next(next_node.get_next())
            return True

    # Adds the values of two lists (assuming each node is a digit)
    def __add__(self,l):
        current = self.head
        sum1 = []
        while current:
            sum1.append(str(current.get_data()))
            current = current.get_next()
        num1 = ''.join(sum1)
        current2 = l.head
        sum2 = []
        while current2:
            sum2.append(str(current2.get_data()))
            current2 = current2.get_next()
        num2 = ''.join(sum2)
        totalSum = int(num1) + int(num2)
        ll = LinkedList2()
        for i in str(totalSum):
            ll.insert(i)
        return ll

    # Boolean that tests whether a list is a palindrome
    def palindrome(self):
        current = self.head
        fArr = []
        while current:
            fArr.append(str(current.get_data()))
            current = current.get_next()
        bStr = ''.join(fArr[::-1])
        fStr = ''.join(fArr)
        if fStr == bStr:
            return True
        else:
            return False

    # Returns the data in the insecting node of two lists
    def intersect(self, l):
        current = self.head
        dic = Counter()
        while current:
            dic[str(id(current))] += 1
            current = current.get_next()
        current = l.head
        while current:
            dic[str(id(current))] += 1
            if dic[str(id(current))] > 1:
                return current.data
            current = current.get_next()
        return False

    # Returns value and address of a circular list's first node, else False
    def circular(self):
        current = self.head
        dic = Counter()
        while current:
            dic[id(current)] += 1
            if dic[id(current)] > 1:
                return current.get_data(), id(current)
            current = current.get_next()
        return False

# Stacks and Queues

# Stack - LIFO
class Stack:
     def __init__(self):
         self.items = []

     def isEmpty(self):
         return self.items == []

     def push(self, item):
         self.items.append(item)

     def pop(self):
         return self.items.pop()

     def peek(self):
         return self.items[len(self.items)-1]

     def size(self):
         return len(self.items)

# Queue - FIFO
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items)-1]

    def size(self):
        return len(self.items)

# Set of stacks that can not contain more than ten elements
class SetOfStacks():
    def __init__(self):
        self.stacks = []

    def add(self, data):
        if not self.stacks:
            new_stack = Stack()
            new_stack.push(data)
            self.stacks.append(new_stack)
        if self.stacks[-1].size() > 10:
            new_stack = Stack()
            new_stack.push(data)
            self.stacks.append(new_stack)
        else:
            self.stacks[-1].push(data)

    def size(self):
        return len(self.stacks)

# Quick sort
def quickSort(array):
    less = []
    equal = []
    greater = []
    if len(array) > 1:
        pivot = array[0]
        for x in array:
            if x < pivot:
                less.append(x)
            elif x > pivot:
                greater.append(x)
            else: # x == pivot
                equal.append(x)
        return quickSort(less)+equal+quickSort(greater)
    else:
        return array

def mergeSort(items):
    if len(items) > 1:
        mid = len(items) // 2        # Determine the midpoint and split
        left = items[0:mid]
        right = items[mid:]
        mergeSort(left)            # Sort left list in-place
        mergeSort(right)           # Sort right list in-place
        l, r = 0, 0
        for i in range(len(items)):     # Merging the left and right list
            lval = left[l] if l < len(left) else None
            rval = right[r] if r < len(right) else None
            if (lval is not None and rval is not None and lval < rval) or rval is None:
                items[i] = lval
                l += 1
            elif (lval is not None and rval is not None and lval >= rval) or lval is None:
                items[i] = rval
                r += 1
        return items

# Binary search (linear)
def binarySearch(array,val):
    first = 0
    last = len(array)-1
    found = False
    while first <= last and not found:
        mid = first+last//2
        if array[mid] == val:
            found = True
        elif val < array[mid]:
            last = mid-1
        else:
            first = mid+1
    return found


def reverse(string):
    s = []
    s.extend(string)
    size = len(s)
    for i in range(0,size//2):
        s[i], s[size-1-i] = s[size-1-i], s[i]
    return ''.join(s)
