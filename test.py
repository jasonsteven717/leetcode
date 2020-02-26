# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:55:17 2019

@author: TsungYuan
"""
'''
def twoSum(nums, target):
    k = 0
    for i in nums:
        k += 1
        if target - i in nums[k:]:
            return(k - 1, nums[k:].index(target - i) + k)

ans = twoSum([2,7,11,15],9)
print(ans)
'''
'''
def reverse(x):
    if x < 0:      
        l = list(str(x)[1:]) 
        l.reverse() 
        if int(''.join(l)) * -1 < -2**31:
            return 0
        else:
            return int(''.join(l)) * -1
    else:
        l = list(str(x)) 
        l.reverse() 
        if int(''.join(l)) > 2**31 - 1:
            return 0
        else:
            return int(''.join(l))
ans = reverse(-1534236469)
print(ans)
'''
'''
def isPalindrome(x):
    k = 0
    if x < 0:
        return False
    if len(str(x)) == 1:
        return True
    if len(str(x))%2 == 0:
        k = 1
        m = round(len(str(x))/2)
    else:
        m = int(len(str(x))/2)+1
    if k == 1:
        l = str(x)[:m]
        r = list(str(x)[m:])
        print(l,r)
    else:
        l = str(x)[:m-1]
        r = list(str(x)[m:])
        print(l,r)
    r.reverse()
    if ''.join(r) == l:
        return True
    else :
        return False

ans = isPalindrome(888888)
print(ans)
'''
'''
def romanToInt(s):
    numerals = { "M": 1000, "D": 500, "C": 100, "L": 50, "X": 10, "V": 5, "I": 1 }
    sum=0
    s=s[::-1]
    last=None
    for x in s:
        if last and numerals[x]<last:
            sum-=2*numerals[x]
        sum+=numerals[x]
        last=numerals[x]
    return sum    
ans = romanToInt("LVIII")
print(ans)
''' 
'''
def longestCommonPrefix(strs):
    if not strs: 
        return ""
    s1 = min(strs)
    s2 = max(strs)  
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    return s1        
ans = longestCommonPrefix(["flower","flow","flight"])
print(ans) 
'''
'''
def isValid(s):
    stack = []
    left = {"(":1, "{":2, "[":3}
    right = {")":1, "}":2, "]":3}
    i,l,r = 0,0,0
    for x in s:
        if x in right and i==0:
            return False
        if x in left:
            l+=1
            stack.append(left[x])
        if x in right:
            r+=1
            if l<r:
                break
            if right[x]==stack[-1]:
                stack.pop()
        i += 1
    if stack == [] and l == r:
        return True
    else:
        return False
ans = isValid("[])")
print(ans) 
'''
'''
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
def mergeTwoLists(l1, l2):
        if l1 == None:
            return l2
        if l2 == None:
            return l1
        dummy = ListNode(0)
        tmp = dummy
        while l1 and l2:
            if l1.val <= l2.val:
                tmp.next = l1
                l1 = l1.next
                tmp = tmp.next
            else:
                tmp.next = l2
                l2 = l2.next
                tmp = tmp.next
        if l2 == None:
            tmp.next = l1
        else:
            tmp.next = l2
        return dummy.next
node1 = ListNode(1) 
node2 = ListNode(2) 
node3 = ListNode(4) 
node1.next = node2 
node2.next = node3
node4 = ListNode(1) 
node5 = ListNode(3) 
node6 = ListNode(4) 
node4.next = node5 
node5.next = node6
ans = mergeTwoLists(node1, node4)
print(ans)
'''
'''
def removeDuplicates(nums):
    j = 0
    for i in range(len(nums)-1):
        if nums[i] == nums[i+1]:
            nums[i] = []
    while(True):
        try:
            nums.remove([])
        except:
            break
    return nums           
ans = removeDuplicates([0,0,1,1,1,2,2,3,3,4])
print(ans) 
'''   
'''
def removeElement(nums, val):
    while(True):
        try:
            nums.remove(val)
        except:
            break
    return len(nums)
ans = removeElement([0,1,2,2,3,0,4,2], 2)
print(ans)    
'''
'''
def strStr(haystack, needle):
    ind = haystack.find(needle)
    return ind
ans = strStr("aaaaa", "ll")
print(ans)
'''
'''
def searchInsert(nums, target):
    for i in range(len(nums)):
        if nums[i] >= target:
            return i
            break
        if i == len(nums)-1:
            return i+1
ans = searchInsert([1,3,5,6], 0)
print(ans)
'''
'''
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
def addTwoNumbers(l1, l2):   
    num1 = ''
    num2 = ''
    while l1:
        num1 += str(l1.val)
        l1 = l1.next
        print(num1)
    while l2:
        num2 += str(l2.val)
        l2 = l2.next
    add = str(int(num1[::-1]) + int(num2[::-1]))[::-1]
    print(int(num1))
    head = ListNode(add[0])
    ans = head
    for i in range(1, len(add)):
        node = ListNode(add[i])
        head.next = node
        head = head.next
    return ans

node1 = ListNode(2) 
node2 = ListNode(4) 
node3 = ListNode(3) 
node1.next = node2 
node2.next = node3
node4 = ListNode(5) 
node5 = ListNode(6) 
node6 = ListNode(4) 
node4.next = node5 
node5.next = node6
ans = addTwoNumbers(node1, node4)
print(ans)
'''
'''
def lengthOfLongestSubstring(s):
    sub = ""
    lon,cnt,l = 0,0,0
    if len(s) == 0:
        return 0
    s = s + s[len(s)-1]
    for i in range(len(s)-1):
        sub = sub + str(s[i])
        if sub.find(s[i+1]) != -1:
            l = len(sub)
            if lon < l:
                lon = l
            sub = sub[sub.find(s[i+1])+1:]
            cnt += 1
        if i == len(s)-1-1 and cnt == 0:
            return len(s)
    return lon
        
ans = lengthOfLongestSubstring("dvdf")
print(ans)
'''
'''
def findMedianSortedArrays(nums1, nums2):
    num = nums1 + nums2
    nums = sorted(num)
    m = int(len(nums)/2)
    if len(nums)%2 == 0:
        return (nums[m-1] + nums[m]) / 2
    else :
        return nums[m]
ans = findMedianSortedArrays([3],[2])
print(ans)
'''
'''
def myAtoi(str):
    if str == '':
        return 0
    str = str + "!"
    for i in range(len(str)):
        if str[0].isdigit() or str[0] == '-' or str[0] == '+':
            break
        if str[i] == ' ' and str[i+1] != ' ':
            if str[i+1].isalpha() == 1:
                return 0
            if str[i+1].isdigit() or str[i+1] != '-' or str[i+1] != '+':
                break
    t = ""
    for i in range(len(str)):
        if str[i].isdigit():
            t = t + str[i]
            if str[i+1].isalpha() or str[i+1] == ' ' or str[i+1] == '-' or str[i+1] == '+':
                break
        if (str[i] == '+' or str[i] == '-' or str[i] == '.') and str[i+1].isdigit() == 0:
            return 0
        if str[i] == '+' or str[i] == '-' or str[i] == '.':
            t = t + str[i]
        if str[i].isalpha() and str[i+1].isdigit():
            return 0
    if t == "":
        return 0
    if t[0].isdigit() == 0 and str[0] != ' ' and str[0] != '-' and str[0] != '+':
        return 0
    if t.isdigit() == 0 and len(t) <= 1:
        return 0
    if '+' in t and '-' in t:
        return 0
    else:
        if int(float(t)) > 2**31 - 1:
            return 2**31 - 1
        if int(float(t)) < -2**31:
            return -2**31
        else :
            return int(float(t))
ans = myAtoi("w10")
print(ans)
'''
'''
def firstMissingPositive(nums):
    if nums == []:
        return 1
    nums=sorted(nums)
    for i in range(len(nums)):
        if nums[i] <= 0:
            nums[i] = ""
    for k in range(len(nums)):
        if "" in nums:
            nums.remove("")
    c = 1
    if nums == []:
        return 1
    else:
        print(nums)
        for j in range(len(nums)):
            if c in nums:
                for d in range(len(nums)):
                    if c in nums:
                        nums[d] = 0
                c += 1
                if nums[-1] == 0:
                    return c
            else:
                return c
        return c

ans = firstMissingPositive([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127])
print(ans)
'''
'''
def rotate(matrix):
    temp = matrix
    a = []
    for i in range(len(temp[0])):
        for j in range(len(temp[0])):
            a.append(temp[len(temp[0])-j-1][i])
    for k in range(len(a)):
        matrix[int(k/len(temp[0]))][k%len(temp[0])] = a[k]
ans = rotate([
  [ 5, 1, 9,11],
  [ 2, 4, 8,10],
  [13, 3, 6, 7],
  [15,14,12,16]
])
'''
'''
def myPow(x, n):
    return x**n
ans = myPow(-13.62608,3)
print(ans)
'''
'''
def letterCombinations(digits):
    from itertools import product
    phone = {'2': ['a', 'b', 'c'],
             '3': ['d', 'e', 'f'],
             '4': ['g', 'h', 'i'],
             '5': ['j', 'k', 'l'],
             '6': ['m', 'n', 'o'],
             '7': ['p', 'q', 'r', 's'],
             '8': ['t', 'u', 'v'],
             '9': ['w', 'x', 'y', 'z']}
    if digits == "":
        return []
    else:
        x=[]
        for i in range(len(digits)):
            if digits[i] in  phone:
                x.append(phone[digits[i]])
        ans = [''.join(s) for s in product(*x)]
        return ans    
ans = letterCombinations("")
print(ans)
'''
'''
def maxArea(height):
    start, end, maxArea = 0, len(height)-1, 0
    while start < end:
        area = (min(height[end],height[start]))*(end-start)
        maxArea = max(maxArea, area)
        if height[end] > height[start]:
            start += 1
        else:
            end -= 1
    return maxArea
ans = maxArea([1,2,1])
print(ans)
'''
'''
def groupAnagrams(strs):
    d = {}
    temp = [0]*len(strs)
    for i in range(len(strs)):            
        temp[i] = "".join(sorted(strs[i]))
    for j in range(len(temp)):
        if temp[j] not in d:
            d[temp[j]] = [strs[j]]
        else:
            d[temp[j]].append(strs[j])
    return d.values()
ans = groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
print(ans)
'''
'''
def permute(nums):
        import itertools
        return list(itertools.permutations(nums,len(nums)))
ans = permute([1,2,3])
print(ans) 
'''
'''
def permuteUnique(nums):
    import itertools
    allper = list(itertools.permutations(nums,len(nums)))
    ans = []
    for l in allper:
        if l not in ans:
            ans.append(l)
    print(ans)
ans = permuteUnique([1,2,1])
print(ans)
'''
'''
def maxSubArray(nums) :
    cursum,res = -1000000000000,-1000000000000
    for s in nums:
        cursum = max(cursum + s, s)
        res = max(res ,cursum)
    return res
ans = maxSubArray([-2,1,-3,4,-1,2,1,-5,4])
print(ans) 
'''
'''
def lengthOfLastWord(s):
    if s == "":
        return 0
    word = s.split(" ")
    k = 1
    for i in range(len(word)):
        if k == len(word) and word[0] == "":
            return 0
        if word[-(i+1)] == "":
            k += 1
        else:
            return len(word[-(i+1)])
ans = lengthOfLastWord("a ")
print(ans)
'''
'''
def spiralOrder(matrix):
    import numpy as np
    m = np.array(matrix)
    t = m.shape[0]%2
    if t != 0:
        t =int(m.shape[0]/2)+1
    else:
        t = int(m.shape[0]/2)
    ll,wl,lr,wr = 0,0,0,0
    M = np.empty((0,0),dtype=np.int)
    for i in range(t):
        if np.size(M) == m.shape[0]*m.shape[1]:
            break
        m1 = m[ll,wl:m.shape[1]-wr]
        m2 = m[ll+1:m.shape[0]-wr,m.shape[1]-1-lr]
        m3 = m[m.shape[0]-1-wr,m.shape[1]-wr-2:wl:-1]
        m4 = m[m.shape[0]-wr-1:ll:-1,wl]
        M = np.append(M,m1)
        if np.size(M) == m.shape[0]*m.shape[1]:
            break
        M = np.append(M,m2)
        if np.size(M) == m.shape[0]*m.shape[1]:
            break
        M = np.append(M,m3)
        if np.size(M) == m.shape[0]*m.shape[1]:
            break
        M = np.append(M,m4)
        if np.size(M) == m.shape[0]*m.shape[1]:
            break
        ll += 1
        wl += 1
        lr += 1
        wr += 1 
    return M
ans = spiralOrder(
[[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]])
print(ans)
'''
'''
def merge(intervals):
    t = 0
    intervals = sorted(intervals)
    while(1):
        for i in range(len(intervals)-1):
            if int(intervals[i][1]) >= int(intervals[i+1][0]):
                intervals[i][0] = min(intervals[i][0],intervals[i+1][0])
                intervals[i][1] = max(intervals[i+1][1],intervals[i][1])
                intervals[i+1] = [-1,-1]
            elif int(intervals[i+1][0]) < int(intervals[i][0]):
                intervals[i][0] = min(intervals[i+1][0],intervals[i][0])
                intervals[i][1] = max(intervals[i][1],intervals[i+1][1])
                intervals[i+1] = [-1,-1]
            elif intervals[i] == intervals[i+1]:
                intervals[i+1] = [-1,-1]
        ans = [k for k in intervals if (k != [-1, -1])]
        if t == len(ans):
            break
        intervals = ans
        t = len(ans)
    return ans
 
ans = merge([[1,4],[0,2],[3,5]])
print(ans)
'''
'''
def canJump(nums):
    x = 1
    for i in range(len(nums)-2, -1, -1):
        if nums[i] < x:
            x += 1
        else:
            x = 1
    return x == 1
ans = canJump([0,2,3])
print(ans)
'''
'''
def mySqrt(x):
    import math
    ans = math.sqrt(x)
    return int(ans)
ans = mySqrt(9)
print(ans)
'''
'''
def plusOne(digits):
    a = [0]
    digits = a+digits
    l = len(digits)-1
    digits[l] = digits[l] + 1
    for i in range(len(digits)):
        if digits[l-i] == 10:
            digits[l-i] = 0
            digits[l-i-1] += 1
    if digits[0] == 0:
        digits.remove(0)
    return digits
ans = plusOne([0])
print(ans)
'''
'''
def sortColors(nums):
    l, h, i = 0, len(nums) - 1, 0
    while i <= h:
        if nums[i] == 0:
            nums[i] = nums[l]
            nums[l] = 0
            l += 1
            i = max(l, i)
        elif nums[i] == 2:
            nums[i] = nums[h]
            nums[h] = 2
            h -= 1
        else: i += 1
sortColors([2,0,2,1,1,0])
'''
'''
def combine(n, k):
    N = []
    for i in range(n):
        N.append(i+1)
    from itertools import combinations
    return list(combinations(N, k))
ans = combine(4,2)
print(ans)
'''
'''
class ListNode(object):
    def __init__(self, x):
      self.val = x
      self.next = None               
def deleteDuplicates(head):
    if not head:
        return head
    cur_node = head
    while True:
        if cur_node.next == None:
            break
        if cur_node.val == cur_node.next.val:
            cur_node.next = cur_node.next.next
        else:
            cur_node = cur_node.next
    return head
node1 = ListNode(1) 
node2 = ListNode(1) 
node3 = ListNode(2) 
node1.next = node2 
node2.next = node3
ans = deleteDuplicates(node1)
print(ans)    
'''        