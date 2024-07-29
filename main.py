'''
#---------------LIST OPERATIONS--------------------

Matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(Matrix[0][1])
#Matrix.append(1)
Matrix.extend([[9, 9, 9]])
print(Matrix)
Matrix.pop(
  0
)  #removes index, bydefault - the last one, and also returns that thing. That is pop will remove and return as well
print(Matrix)
Matrix.remove([7, 8, 9])  #removes the value given
print(Matrix)
print([4, 5, 6] in Matrix)
print(Matrix.count([4, 5, 6]))
Matrix = [[9, 9, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]]
Matrix.sort()
print(Matrix)
Matrix = [[9, 9, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(Matrix)
print(sorted(Matrix))
print(
  f'\n\n The original Matrix is not sorted with this command, instead Matrix was sorted and printed- see:\t\n {Matrix} \n\n'
)

Matrix_2 = Matrix  #copy by reference

Matrix.sort()
print(Matrix_2)  #change in Matrix caused changes in Matrix_2
Matrix.append([10, 10, 10])
print(Matrix_2)  #change in Matrix caused changes in Matrix_2

Matrix_2 = Matrix.copy()  #copy by value or use Matrix[:]
Matrix.append([12, 12, 12])
print(Matrix_2)  #change in Matrix now doesn't cause changes in Matrix_2
Matrix.reverse()
print(Matrix)

Matrix[::
       -1]  #This List slicing will reverse the sequence on display, not in original list variable

print(Matrix)

print(f'this will show reversal in print only: {Matrix[::-1]} \n\n\n')

range(1, 100)  #create a range of numbers
print(list(range(1, 100)),
      '\n\n ******this is line 49')  #wrapping the range in the list data type
#range(100) will create range from 0 to 99 elements

#using join
sentence = '!'
new_sentence = sentence.join(['My', 'Name is', 'Jojo'])

print(new_sentence, '\n\n')  #! mark is joined with the sentence

print('  '.join([
  'My', 'Name is', 'Ishtiaq'
]))  #here string( space in our case)  is joined with a newly created list

#List Unpacking

a, b, c = [1, 2, 3]  # we can also assign variable like a,b,c=1,2,3

print(a)
print(b)
print(c)

#But unpacking of list gets more useful in this case:

a, b, c, *other, d = [1, 2, 3, 4, 5, 6, 7, 8]

print(a)
print(b)
print(c)
print(d)
print(other, '\t\n\n This is line 74')

weapons = None
print(weapons)

#Dictionary is data type and also a data structure in Python
#see PIAIC files for dictionary working

#***************SET******************

#it is also a data structure/ data type

mySet = {1, 2, 3, 4, 5, 5, 3,
         1}  #Notice, they dont have keys , like we had in dictionary

print(mySet)  # it only returns unique items
mySet.add(100)
mySet.add(2)  #2 will not be added as set already has it
print(mySet)

#we can wrap a LIST in a SET in order to return unique items in the LIST
#Set doesn't support indexing

my_set = {1, 2, 3, 4, 5}
your_set = {4, 5, 6, 7, 8, 9, 10}

print(
  my_set.difference(your_set)
)  #it works like a SET DIFFERENCE which we learned in mathematics. Here it showd that your_set don't have 1,2,3

my_set.discard(5)  #it removes 5 from the set
my_set.difference_update(your_set)
print(
  my_set
)  #it updated the set and removed the items which were present in your_set
my_set = {1, 2, 3, 4, 5}
print(my_set.intersection(your_set))
#prints the common elements on my_set with your_set

print(my_set.isdisjoint(your_set))
#return false if there is a common elements on my_set with your_set. Return True other wise

print(my_set.union(your_set))
#combines both sets

#now change my_set a bit to understand subset and superset understanding
my_set = {4, 5}

print(my_set.issubset(your_set))
#Now, my_set is sub-set of your set

print(your_set.issuperset(my_set))

#----------------IF STATEMENTS IN JUPYTER NOTE BOOK - PIAIC NOTES--------------

#----------------Truthy and Falsy values-----------

a = True
b = 5

#bool(5) or bool('hello') returns True in Python therefore below if condition gets true. This is called Truthy statement in Python
#however bool(0) or bool('') etc return false so it is called falsy

#----------------- Ternary Operator --------------

is_friend = True

CanMessage = 'MessageAllowed' if is_friend else 'You are not friend message is not allowed'

print(CanMessage)

if a and b:  #assuming both conditions as true
  print("ok")
else:
  print("not ok")

#----------------- Short Circuiting --------------

#its a way of working of Python that if one of the AND /OR conditions gets true or false then it ignors the second operations and perform the next task .. like

is_user = False
if is_friend or is_user:
  print("You can send message")

#in above expression, since is_friend is true therefore Python will not go in 'or' operation in order to save computation expense and start printing the thing in if block. The same happens when one of the end statement does nulify

# ----------------IS vs ==

print(10 == 10.0)
print(10 is 10.0)
print('1' is 1)
print([1, 2, 3] == [1, 2, 3])
print([1, 2, 3] is
      [1, 2, 3])  #False as these are two different lists created in memory

#-------------- ENUMERATE FUNCTION ------------------

for i, char in enumerate('Ishtiaq Ahmed'):

  print(i, char)  #so wtih enumerate, you get index counter alongwith iterable


# ------------- Programming Exercise -1 ----------------

pic = [[0, 0, 0, 1, 0, 0, 0], 
       [0, 0, 1, 1, 1, 0, 0], 
       [0, 1, 1, 1, 1, 1, 0],
       [1, 1, 1, 1, 1, 1, 1], 
       [0, 0, 0, 1, 0, 0, 0], 
       [0, 0, 0, 1, 0, 0, 0]]

display_row=''

for item in pic:
  
  for item2 in item:
    
    if item2==0:
      display_row=display_row+' '
    else:
      display_row=display_row+'x'
      
  print('\n',display_row)
  display_row=''

  #print('abc',end='') will remove addition of a new line 



# ------------- Programming Exercise - 2 ----------------

some_list = ['a', 'b', 'c', 'd', 'a', 'b', 'n', 'n']

unique_items = []
duplicate_items = []

for items in some_list:

  if items not in unique_items:
    unique_items.append(items)
  else:
    duplicate_items.append(items)

print(duplicate_items)

#-------------DOCSTRING------------


def test(a):


'''
#Info: This function prints whatever aurgument is given to it
#THE ABOVE COMMENTS WILL BE MADE WITH OUT THE HASH TAG SIGN, THE SIGN IS WRITTEN IN ORDER TO AVOID THE CONFUSION INTERPRETOR IS FACING BECAUSE OF COMMENTING THE WHOLE CODE
'''
  print(a)


  test('Hi')

#now whenever we call this function text, a string will display /dock
#this is used to provide information to the user of the function
#Also, if you use help(test) and run, this will display the docstring

print(test.__doc__)

#The above command will also print the Docstring

#-------------An Example of Clean Code-----------


def is_even(num):
  return (num % 2 == 0)
# checking if num modulus 2 (remainder of num, when divded by 2 equals to zero or not.)

print(is_even(51))



#-------------args and kwargs---------------

def super_func(*args,**kwargs):
  print(args)
  print(kwargs)
  
  print(sum(args)) #ars stores in tuple 
  total=0
  
  for items in kwargs.values():
    
    total+=items 
    
  print((total)) #kwargs stores in dictionary


print(super_func(1,2,3,4,5,num1=4,num2=6)) #args is stored in tuple 

#rule of sequence of aurguments:
# params, *args, defualt params, **kwargs


#------------Excerceise Highest Even ---------------

def highest_even(li):
  even_li=[]
    
  for item in li:
    if item %2 ==0:
      even_li.append(item)

  even_li.sort(reverse=True)  
  print(even_li[0],end=' is the highest even in the list \n')
  return(max(even_li #using of return will make the code smarter
             
  

print(highest_even([1,2,3,4,5,8,9,11,23,46]))

  

#----------------SCOPE -----------------

a = 1


def confusion():
  a = 5
  return a


print(confusion())
print(a)

#But if we remove a=5 from the confusion function, it will return a=1 because if interpretor dont find local variable, it will go to next level variable
#scope sequencing rules: 1) Start with Local, 2) if not found then go for Parent Local 3) if this not found then go to Global variable (main.py in this case) 4) built-in Python function, like

x = 10


def parent():
  x = 5

  def confusion2():
    return sum
    return x
    #this return will not be executed as function will exit on                 first return statement

  return confusion2()


print(parent())
print(x)


total=0

def total_func():
  global total
  total +=1


#------------NONLOCAL KEYWORD--------------


def outer():
  x = "local"

  def inner():
    nonlocal x  #nonlocal keyword wil make not create another variable                     but use the on created in outer/parent scope.
    x = " I am nonlocal X"
    print("Inner", x)

  inner()
  print("Outer", x)


outer()

#---------------------PURE FUNCTIONS, MAP,FILTER,ZIP AND REDUCE FUNCTIONS

#1) Pure functions are those who don't have any side effects. That is, they don't change the outside world - like printing something or changing a global variable
#2) Pure functions always yield consisten output. Pure Functions are easily readiable, testable and have high performance



#-----------MAP FUNCTION--------------

def multiply_by2(li):
  my_list=[]
  for items in li:
    my_list.append(items*2)
  return my_list

print(multiply_by2([1,2,3]))


#the same thing can be done using Map function. It is useful when you need to iterate some data, number or string (like changing case). (We iterated values of the list)

#Map in Python is a function that works as an iterator to return a result after applying a function to every item of an iterable (tuple, lists, etc.). 

my_list=[1,2,3]

def new_multiply_by2 (item):
  return item*2

map(new_multiply_by2,my_list) #returns list object

print(list(map(new_multiply_by2,my_list)))




#----------------FILTER---------------

#on the basis of Boolean statement, it filter out the items as per condition mentioned in the function

#Python's filter() is a built-in function that allows you to process an iterable and extract those items that satisfy a given condition.

my_list=[1,2,3,4,5,6,7,8]

def check_odd(item):
  return item % 2 !=0
  #will filter and return the items whose remainder is not 0

print(list(filter(check_odd,my_list)))


#----------------ZIP FUNCTION---------------
#Zips two iterables in to one without modifying the original iterrables, making them functionally immutable

my_list=[1,2,3]
your_list=[10,20,30,40]
their_list=[5,5,5]

print(list(zip(my_list,your_list,their_list)))
#items of two lists comibined in tuples when zipped
#even if your_list was a tuple, the things would get combine in a similar way




#----------------REDUCE FUNCTION---------------

#import related function library

from functools import reduce
my_list=[1,2,3]

def accumulator(acc,item):
  #we built accumulating/totaling function using reduce function, which will accumulate the list but the reduce function may be used for various purposes
  return acc+item

print (reduce(accumulator,my_list,0)) #zero here is like head start, the initial number which will be added in addition of item/list

#Note: map,zip, filter function use reduce under the hood.


#---------------LIST,SET AND DICTIONARY COMPREHENSIONS-----------------
#its a short cut to create and populate a iterable - that is list or set or dictionary
#it usually make the code less readible and complex so we need to avoid using comprehensions whenever we can if we think that code will get bit complex

my_list=[char for char in 'hello']
#my_list2=[num for num in range(1,100)]
#my_list3=[num**2 for num in range(1,100)]
my_list4=[num**2 for num in range(1,100) if num%2==0] #creates list of even numbers



print(my_list)
#print(my_list2)
#print(my_list3)
print(my_list4)

#set

my_set={char for char in 'hello dello'} #Unique chars will be fetched
my_set2={num**2 for num in range(1,100) if num%2==0} 
#creates list of even numbers

print(my_set)
print(my_set2)


#dictionary 


simple_dict={
  'a':1,
  'b':2,
  'c':4,
  
}

my_dict={key:value**2 for key,value in simple_dict.items() if value%2==0}

my_dict2={num:num**2 for num in [1,2,3,4,5] if num % 2==0}


print(my_dict)
print(my_dict2)



#------------Get List of Duplicate Items using Set Comprehension----------

some_list=['a','b','c','z','b','a']
duplicate_list1={char for char in some_list if some_list.count(char)>1}

#or this can also be done as:

duplicate_list2=list(set([char for char in some_list if some_list.count(char)>1]))



print(duplicate_list1)
print(duplicate_list2)

'''



#------------MODULES ---------------


import utility

print(utility)

#while running an imported module, interpreter will create '__pycache__' folder - with .pyc file. The c in pyc extention denotes the compiler is Python made in C language. The interpreter loads the complied version of the imported module, which we see in the .pyc file.

#main.py doesn't get compiled

print(utility.multiplication(2,3))



#------------PACKAGES ---------------
#Package act like a folder in Python that may contains various modules.
#In Replit, we create a folder for package and then create file for a module


import shopping.shoppingcart


print(shopping.shoppingcart.buy("apple"))


#Check PyCharm files - for more about packages and ways to import packages


