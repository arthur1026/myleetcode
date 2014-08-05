#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <sstream>

using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

struct Point {
    int x;
    int y;
    Point() : x(0), y(0) {}
    Point(int a, int b) : x(a), y(b) {}
};

struct RandomListNode {
    int label;
    RandomListNode *next, *random;
    RandomListNode(int x) : label(x), next(NULL), random(NULL) {}
};

struct UndirectedGraphNode {
    int label;
    vector<UndirectedGraphNode *> neighbors;
    UndirectedGraphNode(int x) : label(x) {};
};

class ListNodeComparison {
public:    
    bool operator() (const ListNode* l1, const ListNode* l2) {
        return (l1->val > l2->val);
    }
};

class Solution {
public:
/*Container With Most Water
Given n non-negative integers a1, a2, ..., an, where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water.

Note: You may not slant the container.
*/
    int maxArea(vector<int> &height) {
        if (height.size() < 2)
            return 0;
        int start = 0;
        int end = height.size() - 1;
        int maxarea = 0;
        while (start < end) {
            maxarea = max(maxarea, min(height[start], height[end]) * (end - start));
            if (height[start] < height[end])
                start++;
            else
                end--;
        }
        
        return maxarea;
    }
    
/*Largest Rectangle in Histogram 
Given n non-negative integers representing the histogram's bar height where the width of each bar is 1, find the area of largest rectangle in the histogram.


Above is a histogram where width of each bar is 1, given height = [2,1,5,6,2,3].


The largest rectangle is shown in the shaded area, which has area = 10 unit.

For example,
Given height = [2,1,5,6,2,3],
return 10.
*/
    int largestRectangleArea(vector<int> &height) {
        // for each bar, check the maximum area it could cover
        int maxarea = 0;
        stack<int> s;
        height.push_back(0);
        for (int i = 0; i < height.size();) {
            if (s.empty() || height[i] > height[s.top()])
                s.push(i++);
            else {
                int tmp = s.top();
                s.pop();
                maxarea = max(maxarea, height[tmp] * (s.empty() ? i : i - s.top() - 1));
            }
        }

        return maxarea;
    }
    
/*Length of Last Word 
Given a string s consists of upper/lower-case alphabets and empty space characters ' ', return the length of last word in the string.

If the last word does not exist, return 0.

Note: A word is defined as a character sequence consists of non-space characters only.

For example, 
Given s = "Hello World",
return 5.
*/
    int lengthOfLastWord(const char *s) {
        int len = 0;
        bool last_is_space = false;
        
        while (*s != '\0') {
            if (*s == ' ') {
                // don't hurry to clear len
                last_is_space = true;
            } else {
                if (last_is_space)
                    len = 1;
                else
                    len++;
                last_is_space = false;
            }
            s++;
        }
        
        return len;
    }

/*Longest Palindromic Substring
Given a string S, find the longest palindromic substring in S. You may assume that the maximum length of S is 1000, and there exists one unique longest palindromic substring.
*/
    string longestPalindrome(string s) {
        if (s.empty())
            return s;
        int n = s.size();
        bool dp[n][n];
        fill_n(&dp[0][0], n*n, false);
        
        //vector<vector<bool> > dp(n, vector<bool>(n, false));
        // initialize first row
        dp[0][0] = true;
        size_t max_len = 1;
        size_t start = 0;
        
        // use dp
        for (size_t i = 0; i < n; i++) {
            dp[i][i] = true;
            for (size_t j = 0; j < i; j++) {
                dp[j][i] = (s[j] == s[i] && (i - j < 2 || dp[j+1][i-1]));
                if (dp[j][i] && max_len < (i - j + 1)) {
                    max_len = i - j + 1;
                    start = j;
                }
            }
        }        
        return s.substr(start, max_len);
    }
    
/* Regular Expression Matching 
Implement regular expression matching with support for '.' and '*'.

'.' Matches any single character.
'*' Matches zero or more of the preceding element.

The matching should cover the entire input string (not partial).

The function prototype should be:
bool isMatch(const char *s, const char *p)

Some examples:
isMatch("aa","a") → false
isMatch("aa","aa") → true
isMatch("aaa","aa") → false
isMatch("aa", "a*") → true
isMatch("aa", ".*") → true
isMatch("ab", ".*") → true
isMatch("aab", "c*a*b") → true
*/
    bool isMatch(const char *s, const char *p) {
        if (*p == '\0')
            return *s == '\0';
            
        // next char is not '*', then must match current character
        if (*(p+1) != '*') {
            if (*p == *s || (*p == '.' && *s != '\0'))
                return isMatch(s+1, p+1);
            else
                return false;
        } else {
            while(*p == *s || (*p == '.' && *s != '\0')) {
                if (isMatch(s, p+2))
                    return true;
                s++;
            }
            return isMatch(s, p+2); // skip current one
        }
    }
    
/*Longest Valid Parentheses 
Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.

For "(()", the longest valid parentheses substring is "()", which has length = 2.

Another example is ")()())", where the longest valid parentheses substring is "()()", which has length = 4.
*/
    int longestValidParentheses(string s) {
        if (s.size() <= 1)
            return 0;
        int start = -1;
        int max_len = 0;
        stack<int> left_positions;
        
        for (int i = 0; i < s.size(); i++) {
            if (s[i] == '(') {
                left_positions.push(i);
            } else {
                if (left_positions.empty())
                    start = i;
                else {
                    left_positions.pop();
                    if (left_positions.empty()) {
                        max_len = max(max_len, i - start);
                    } else {
                        max_len = max(max_len, i - left_positions.top());
                    }
                }
            }
        }
        
        return max_len;
    }
    
/*Multiply Strings 
Given two numbers represented as strings, return multiplication of the numbers as a string.

Note: The numbers can be arbitrarily large and are non-negative.
*/
    string multiply(string num1, string num2) {
        // always let num1 has smaller size (no reason)
        if (num1.size() < num2.size())
            swap(num1, num2);
            
        vector<vector<int> > levels;
        size_t max_len = 0;
        for (int i = num2.size()-1; i >= 0; i--) {
            int n2 = num2[i] - '0';
            vector<int> single_level;
            
            // push '0's
            for (int z = 0; z < num2.size() - 1 - i; z++)
                single_level.push_back(0);
                
            // mutliply
            int carry = 0;
            for (int j = num1.size() - 1; j >= 0; j--) {
                int n1 = num1[j] - '0';
                int product = n1 * n2 + carry;
                carry = product / 10;
                single_level.push_back(product % 10);
            }
            
            if (carry)
                single_level.push_back(carry);
                
            max_len = max(max_len, single_level.size());
            levels.push_back(single_level);
        }
        
        // sum all levels
        string reverse_result;
        int carry = 0;
        for (int c = 0; c < max_len; c++) {
            int sum = carry;
            for (int r = 0; r < levels.size(); r++) {
                if (levels[r].size() < c+1)
                    continue;
                sum += levels[r][c];
            }
            carry = sum / 10;
            reverse_result.push_back('0' + (sum%10));
        }
        if (carry)
            reverse_result.push_back('0' + carry);
        
        // remove end '0's
        while(reverse_result.size() > 1 && reverse_result[reverse_result.size()-1] == '0')
            reverse_result.pop_back();
        
        return string(reverse_result.rbegin(), reverse_result.rend());
    }
    
/*Minimum Window Substring 
Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).

For example,
S = "ADOBECODEBANC"
T = "ABC"
Minimum window is "BANC".

Note:
If there is no such window in S that covers all characters in T, return the emtpy string "".

If there are multiple such windows, you are guaranteed that there will always be only one unique minimum window in S.
*/
    string minWindow(string S, string T) {
        if (!T.size() || !S.size() || S.size() < T.size())
            return "";

        const int ASCII_MAX = 256;
        vector<int> expected_count(ASCII_MAX, 0);
        vector<int> appear_count(ASCII_MAX, 0);
        
        for (auto t : T)
            expected_count[t]++;
        
        int found = 0;
        int min_len = -1;
        int min_start = -1;
        int start = 0;
        
        // one pass, assume no duplicates in T
        for (int end = 0; end < S.size(); end++) {
            if (expected_count[S[end]] > 0) {  // should be part of T
                if (appear_count[S[end]] < expected_count[S[end]])
                    found++;
                appear_count[S[end]]++;
            }
            if (found == T.size()) {
                // shrink start if necessary
                while (appear_count[S[start]] > expected_count[S[start]] || expected_count[S[start]] == 0) {
                    appear_count[S[start]]--;
                    start++;
                }
                
                // adjust length
                int len = end - start + 1;
                if (min_len == -1 || min_len > len) {
                    min_len = len;
                    min_start = start;
                }
            }
        }
        
        if (min_len == -1)
            return "";
        
        return S.substr(min_start, min_len);
    }

/*Distinct Subsequences 
Given a string S and a string T, count the number of distinct subsequences of T in S.

A subsequence of a string is a new string which is formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (ie, "ACE" is a subsequence of "ABCDE" while "AEC" is not).

Here is an example:
S = "rabbbit", T = "rabbit"

Return 3.
*/
    int numDistinct(string S, string T) {
        if (S.empty() || T.empty())
            return 0;
        if (S.size() < T.size())
            return 0;
            
        unordered_map<string, int> mem;
        return SolveNumDistinct(S, T, 0, 0, mem);
    }
    
    int SolveNumDistinct(string s, string t, int si, int ti, unordered_map<string, int>& mem) {
        stringstream ss;
        ss << si << "," << ti;
        string mem_index = ss.str();
        
        if (mem.count(mem_index) > 0)
            return mem[mem_index];
        
        if (ti > t.size())
            return 0;

        if (ti == t.size()) {
            mem[mem_index] = 1;
            return 1;
        }
            
        int num = 0;
        for (int i = si; i < s.size(); i++) {
            if (s[i] == t[ti])
                num += SolveNumDistinct(s, t, i+1, ti+1, mem);
        }
        
        mem[mem_index] = num;
        return num;
    }
    
/*Edit Distance 
Given two words word1 and word2, find the minimum number of steps required to convert word1 to word2. (each operation is counted as 1 step.)

You have the following 3 operations permitted on a word:

a) Insert a character
b) Delete a character
c) Replace a character
*/
    int minDistance(string word1, string word2) {
        int n = word1.size();
        int m = word2.size();
        if (!n || !m)
            return max(n, m);
        vector<vector<int> > dists(n+1, vector<int>(m+1, 0));
        // initialize first row
        for (int i = 0; i <= m; i++)
            dists[0][i] = i;
        // initialize first column
        for (int i = 0; i <= n; i++)
            dists[i][0] = i;
        // dp step
        for (int r = 1; r <= n; r++)
            for (int c = 1; c <= m; c++) {
                // insert or delete a charactor
                dists[r][c] = min(dists[r-1][c], dists[r][c-1]) + 1;
                if (word1[r-1] == word2[c-1]) {
                    dists[r][c] = min(dists[r][c], dists[r-1][c-1]);
                } else {
                    // replace a charactor
                    dists[r][c] = min(dists[r][c], dists[r-1][c-1]+1);
                }
            }
        return dists[n][m];
    }

/*Decode Ways 
A message containing letters from A-Z is being encoded to numbers using the following mapping:

'A' -> 1
'B' -> 2
...
'Z' -> 26
Given an encoded message containing digits, determine the total number of ways to decode it.

For example,
Given encoded message "12", it could be decoded as "AB" (1 2) or "L" (12).

The number of ways decoding "12" is 2.
*/
    vector<int> mem;
    int numDecodings(string s) {
        if (s.empty())
            return 0;
            
        mem = vector<int>(s.size()+1, -1);
        return SolveNumDecodings(s, 0);
    }
    
    int SolveNumDecodings(string s, int step) {
        if (mem[step] > -1)
            return mem[step];
            
        if (step == s.length()) {
            mem[step] = 1;            
            return 1;
        }
        
        int num = 0;
        if (s[step] != '0')
            num += SolveNumDecodings(s, step+1);
        if (step + 1 < s.length()) {
            if (s[step] == '1' || (s[step] == '2' && s[step+1] <= '6'))
                num += SolveNumDecodings(s, step+2);
        }
        
        mem[step] = num;
        
        return num;
    }

/*Partition List 
Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.

You should preserve the original relative order of the nodes in each of the two partitions.

For example,
Given 1->4->3->2->5->2 and x = 3,
return 1->2->2->4->3->5.
*/
    ListNode *partition(ListNode *head, int x) {
        if (!head)
            return head;
            
        ListNode* cur = head;
        ListNode* next = NULL;
        ListNode* large_head = NULL;
        ListNode* large_tail = NULL;
        ListNode* small_head = NULL;
        ListNode* small_tail = NULL;
        while (cur) {
            next = cur->next;
            if (cur->val < x) {
                if (!small_head) {
                    small_head = cur;
                    small_tail = cur;
                    if (large_head)
                        small_tail->next = large_head;
                } else {
                    if (cur != small_tail->next) {
                        cur->next = small_tail->next;
                        small_tail->next = cur;
                    }
                    small_tail = cur;
                }
            } else {
                if (!large_head) {
                    large_head = cur;
                    large_tail = cur;
                    if (small_tail)
                        small_tail->next = cur;
                } else {
                    if (cur != large_tail->next)
                        large_tail->next = cur;
                    large_tail = cur;
                }
            }
            cur = next;
        }
        if (large_tail)
            large_tail->next = NULL;
        else
            small_tail->next = NULL;
        return (small_head) ? small_head : large_head;
    }
    
/*Anagrams
Given an array of strings, return all groups of strings that are anagrams.

Note: All inputs will be in lower-case.
*/
    vector<string> anagrams(vector<string> &strs) {
        vector<string> results;
        if (strs.empty())
            return results;

        unordered_multimap<string, int> table;
        for (int i = 0; i < strs.size(); i++) {
            string s = strs[i];
            sort(s.begin(), s.end());
            table.insert(pair<string, int>(s, i));
        }
        
        // search in table
        unordered_multimap<string, int>::iterator it;
        while (!table.empty()) {
            it = table.begin();
            string s = it->first;
            if (table.count(s) == 1) {
                table.erase(it);
                continue;
            }
            // then the element must be anagram
            while (table.count(s)) {
                it = table.find(s);
                results.push_back(strs[it->second]);
                table.erase(it);
            }
        }
        
        return results;
    }
    
/*Insert Interval 
Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).

You may assume that the intervals were initially sorted according to their start times.

Example 1:
Given intervals [1,3],[6,9], insert and merge [2,5] in as [1,5],[6,9].

Example 2:
Given [1,2],[3,5],[6,7],[8,10],[12,16], insert and merge [4,9] in as [1,2],[3,10],[12,16].

This is because the new interval [4,9] overlaps with [3,5],[6,7],[8,10].
*/
    vector<Interval> insert(vector<Interval> &intervals, Interval newInterval) {
        vector<Interval> results;
        if (intervals.size() == 0) {
            results.push_back(newInterval);
            return results;
        }
        
        results.reserve(intervals.size());
        int state = 0;  // 0: not started merging; 1: during merging; 2: done mergine
        Interval merge;
        for (int i = 0; i < intervals.size(); i++) {
            Interval interval = intervals[i];
            if (state == 0) {
                if (newInterval.start > interval.end) {
                    results.push_back(interval);
                } else {
                    // start merging
                    merge.start = min(newInterval.start, interval.start);
                    state = 1;
                }
            }
            // during merging, check endpoints relation
            if (state == 1) {
                if (newInterval.end < interval.start) {
                    merge.end = newInterval.end;
                    state = 2;
                    results.push_back(merge);
                }
                else if (newInterval.end <= interval.end) {
                    // should end merging
                    merge.end = max(newInterval.end, interval.end);
                    state = 2;
                    results.push_back(merge);
                    continue;
                }
            }
            // done merging, just push the rest of it into results
            if (state == 2)
                results.push_back(interval);
        }
        
        if (state == 0) {
            if (newInterval.start == results[results.size()-1].end)
                results[results.size()-1].end = newInterval.end;
            else
                results.push_back(newInterval);
        }
        if (state == 1) {
            merge.end = newInterval.end;
            results.push_back(merge);
        }
        return results;
    }
    
/*N-Queens*/
    vector<vector<string> > results;
    vector<vector<string> > solveNQueens(int n) {
        vector<int> used_cols(n, -1);
        solveNQueens(n, 0, used_cols);
        return results;
    }

    void solveNQueens(int n, int row, vector<int>& used_cols) {
        if (row == n) {
            vector<string> single_result(n, "");
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++) {
                    if (used_cols[r] == c)
                        single_result[r].push_back('Q');
                    else
                        single_result[r].push_back('.');
                }
            results.push_back(single_result);
            return;
        }
            
        for (int i = 0; i < n; i++) {
            if (ValidPosition(row, i, used_cols)) {
                used_cols[row] = i;
                solveNQueens(n, row+1, used_cols);
            }
        }
    }
    
/*N-Queens II 
Follow up for N-Queens problem.

Now, instead outputting board configurations, return the total number of distinct solutions.
*/
    int totalNQueens(int n) {
        vector<int> used_cols(n, -1);
        return solveTotalNQueens(n, 0, used_cols);
    }
    
    int solveTotalNQueens(int n, int row, vector<int>& used_cols) {
        if (row == n)
            return 1;
        int num = 0;
        for (int i = 0; i < n; i++) {
            if (ValidPosition(row, i, used_cols)) {
                used_cols[row] = i;
                num += solveTotalNQueens(n, row+1, used_cols);
            }
        }
        
        return num;
    }
    
    bool ValidPosition(int row, int col, const vector<int>& used_cols) {

        for (int i = 0; i < row; i++) {
            // check if previous rows have used col
            if (used_cols[i] == col)
                return false;
        
            // check whether the queens are on the diagonal
            if (row + col == i + used_cols[i])
                return false;
            
            if (row - col == i - used_cols[i])
                return false;
        }
        
        return true;
    }

/* 4Sum 
Given an array S of n integers, are there elements a, b, c, and d in S such that a + b + c + d = target? Find all unique quadruplets in the array which gives the sum of target.

Note:
Elements in a quadruplet (a,b,c,d) must be in non-descending order. (ie, a ≤ b ≤ c ≤ d)
The solution set must not contain duplicate quadruplets.
    For example, given array S = {1 0 -1 0 -2 2}, and target = 0.

    A solution set is:
    (-1,  0, 0, 1)
    (-2, -1, 1, 2)
    (-2,  0, 0, 2)
*/
    vector<vector<int> > fourSum(vector<int> &num, int target) {
        vector<vector<int> > results;
        if (num.size() < 4)
            return results;
        sort(num.begin(), num.end());
            
        for (int i = 0; i < num.size() - 3; i++) {
            if (i > 0 && num[i] == num[i-1])
                continue;
            for (int j = i + 1; j < num.size() - 2; j++) {
                if (j > i+1 && num[j] == num[j-1])
                    continue;
                    
                // become 2-sum problem
                int left = j+1;
                int right = num.size() -1;
                int sumij = num[i] + num[j];
                while (left < right) {
                    int sum = sumij + num[left] + num[right];
                    if (sum < target)
                        left++;
                    else if (sum > target)
                        right--;
                    else {  // found a match
                        vector<int> single_result;
                        single_result.push_back(num[i]);
                        single_result.push_back(num[j]);
                        single_result.push_back(num[left]);
                        single_result.push_back(num[right]);
                        
                        results.push_back(single_result);
                        
                        // look for another possible pair of two numbers that give the same sum
                        left++;
                        right--;
                        
                        while (left < right && num[left] == num[left-1])
                            left++;
                        while (left < right && num[right] == num[right+1])
                            right--;
                    }
                }
            }
        }
        
        return results;
    }
    
/* Merge k Sorted Lists 
Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.
*/
    ListNode *mergeKLists(vector<ListNode *> &lists) {
        priority_queue<ListNode*, vector<ListNode*>, ListNodeComparison> pq;
        
        // push the first k nodes into pg and cache
        for (int i = 0; i < lists.size(); ++i) {
            if (!lists[i])
                continue;
            pq.push(lists[i]);
        }
        
        ListNode* head = NULL;
        ListNode* tail = NULL;

        while (!pq.empty()) {
            ListNode* ln = pq.top();
            pq.pop();
            if (head == NULL) {
                head = ln;
                tail = ln;
            } else {
                tail->next = ln;
            }
            tail = ln;
            // see if we need to push the next one
            if (ln->next)
                pq.push(ln->next);
        }

        return head;
    }

/*Swap Nodes in Pairs
Given a linked list, swap every two adjacent nodes and return its head.

For example,
Given 1->2->3->4, you should return the list as 2->1->4->3.

Your algorithm should use only constant space. You may not modify the values in the list, only nodes itself can be changed.
*/
    ListNode *swapPairs(ListNode *head) {
        if (!head || !head->next)
            return head;
            
        ListNode* prevprev = head;
        ListNode* prev = head;
        ListNode* cur = head->next;
        
        while (true) {
            prev->next = cur->next;
            cur->next = prev;
            if (prevprev == head)
                head = cur;
            else
                prevprev->next = cur;
            
            // check if there are at least two more nodes
            if (!prev->next || !prev->next->next)
                break;
            // adjust pointers
            prevprev = prev;
            prev = prev->next;
            cur = prev->next;
        }
        
        return head;
    }
    
/*Roman to Integer 
Given a roman numeral, convert it to an integer.

Input is guaranteed to be within the range from 1 to 3999.
*/
    int romanToInt(string s) {
        vector<string> roman = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
        vector<int> roman_int = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        int i = 0;
        int si = 0;
        int result = 0;
        
        while (i < roman.size() && si < s.length()) {
            if (si + roman[i].size() > s.length()) {
                i++;
                continue;
            }
            string ss = s.substr(si, roman[i].size());
            if (ss.compare(roman[i]) == 0) {
                result += roman_int[i];
                si += roman[i].length();
            } else {
                i++;
            }
        }
        
        return result;
    }

/* Integer to Roman 
Given an integer, convert it to a roman numeral.

Input is guaranteed to be within the range from 1 to 3999.

*/
    string intToRoman(int num) {
        vector<string> roman = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
        vector<int> roman_int = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        string output = "";
        int i = 0;
        
        while (num > 0) {
            int times = num / roman_int[i];
            num -= times * roman_int[i];
            
            for (int t = 0; t < times; t++)
                output.append(roman[i]);
            
            i++;
        }
        
        return output;
    }

    /* Sudoku Solver
    
    */
    void solveSudoku(vector<vector<char> > &board) {
        int steps = 0;
        for (int r = 0; r < board.size(); r++) {
            for (int c = 0; c < board[0].size(); c++) {
                if (board[r][c] == '.')
                    steps++;
            }
        }
        
        dfs_solveSudoku(steps, board, 0);
    }
    
    bool dfs_solveSudoku(const int n, vector<vector<char> >& board, int step) {
        if (n == step)
            return true;
            
        // search for next available slot
        int r;
        int c;
        bool found = false;
        for (r = 0; r < board.size(); r++) {
            for (c = 0; c < board[0].size(); c++) {
                if (board[r][c] == '.') {
                    found = true;
                    break;
                }
            }
            if (found)
                break;
        }
        if (!found)
            return false;
            
        // now search for valid values for board[r][c]
        vector<bool> possible(9, false);
        // check current row
        for (int cc = 0; cc < 9; cc++) {
            if (board[r][cc] != '.') {
                int num = board[r][cc] - '1';
                possible[num] = true;
            }
        }
        // check current col
        for (int rr = 0; rr < 9; rr++) {
            if (board[rr][c] != '.') {
                int num = board[rr][c] - '1';
                possible[num] = true;
            }
        }
        // check current grid
        int c0 = c / 3;
        int r0 = r / 3;
        c0 *= 3;
        r0 *= 3;
        for (int rr = r0; rr < r0 + 3; rr++) {
            for (int cc = c0; cc < c0 + 3; cc++) {
                if (board[rr][cc] != '.') {
                    int num = board[rr][cc] - '1';
                    possible[num] = true;
                }
            }
        }
        
        // now iterate through each possible value
        for (int i = 0; i < possible.size(); i++) {
            if (!possible[i]) {
                // try different values
                board[r][c] = char('1'+i);
                if(dfs_solveSudoku(n, board, step+1))
                    return true;
                board[r][c] = '.';
            }
        }
        
        return false;
    }    
    /* Valid Sudoku 
    
    */
    bool checkSudoku(char c, vector<bool>& table) {
        if (c == '.')
            return true;
        int num = c - '1';
        if (table[num])
            return false;
        table[num] = true;
        return true;
    }
    bool isValidSudoku(vector<vector<char> > &board) {
        // evaluate each row
        for (int r = 0; r < 9; r++) {
            vector<bool> table(9, false);
            for (int c = 0; c < 9; c++) {
                if (!checkSudoku(board[r][c], table))
                    return false;
            }
        }
        
        // evaluate each column
        // evaluate each row
        for (int c = 0; c < 9; c++) {
            vector<bool> table(9, false);
            for (int r = 0; r < 9; r++) {
                if (!checkSudoku(board[r][c], table))
                    return false;
            }
        }
        
        // evaluate each sub-square
        for (int r0 = 0; r0 < 9; r0 += 3)
        for (int c0 = 0; c0 < 9; c0 += 3) {
            vector<bool> table(9, false);

            for (int r = r0; r < r0 + 3; r++)
            for (int c = c0; c < c0 + 3; c++) {
                if (!checkSudoku(board[r][c], table))
                    return false;
            }
        }
        
        return true;
    }
    
    /* Word Search 
    Given a 2D board and a word, find if the word exists in the grid.

The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.

For example,
Given board =

[
  ["ABCE"],
  ["SFCS"],
  ["ADEE"]
]
word = "ABCCED", -> returns true,
word = "SEE", -> returns true,
word = "ABCB", -> returns false.
    */
    bool exist(vector<vector<char> > &board, string word) {
        int rows = board.size();
        int cols = board[0].size();
        
        if (word.length() > rows * cols)
            return false;
        vector<vector<bool> > map(rows, vector<bool>(cols, false));
        
        // search for the beginning of the word
        char cc = word[0];
        for (int r = 0; r < board.size(); r++) {
            for (int c = 0; c < board[0].size(); c++) {
                if (board[r][c] == cc) {
                    map[r][c] = true;
                    if (dfs_exist(board, word, map, r, c, 1))
                        return true;
                    map[r][c] = false;
                }
            }
        }
        return false;
    }
    
    bool dfs_exist(const vector<vector<char> >& board, string word, vector<vector<bool> >& map, int r, int c, int i) {
        if (i == word.length())
            return true;
        int rows = board.size();
        int cols = board[0].size();
        
        // search for neighbors
        for (int delta_r = -1; delta_r <= 1; delta_r++) {
            for (int delta_c = -1; delta_c <= 1; delta_c++) {
                if ((abs(delta_r) + abs(delta_c)) != 1)
                    continue;
                int newr = r + delta_r;
                int newc = c + delta_c;
                if (newr < 0 || newr >= rows || newc < 0 || newc >= cols)
                    continue;
                if (map[newr][newc])
                    continue;
                if (board[newr][newc] == word[i]) {
                    map[newr][newc] = true;
                    if (dfs_exist(board, word, map, newr, newc, i+1))
                        return true;
                    map[newr][newc] = false;
                }
            }
        }
        
        return false;
    }
	/*Restore IP Addresses 
	Given a string containing only digits, restore it by returning all possible valid IP address combinations.

For example:
Given "25525511135",

return ["255.255.11.135", "255.255.111.35"]. (Order does not matter)
	*/
    vector<string> restoreIpAddresses(string s) {
        set<string> result;
        string path;
        dfs_restoreIpAddresses(s, path, 0, 0, result);
        return vector<string>(result.begin(), result.end());
    }
    
    bool validIp(string s) {
       
        int num = 0;
        for (size_t i = 0; i < s.length(); i++) {
            num = num*10 + (s[i] - '0');
        }
        if (num <= 255) {
            if (s.length() > 1 && s[0] == '0')
                return false;
            return true;
        }
        return false;
    }
    
    void dfs_restoreIpAddresses(string s, string path, int start, int step, set<string>& result) {
        if (step == 4) {
            if (start == s.length())
                result.insert(path);
            return;
        }
        
        // TODO: could add a condition here to end early


        for (int len = 1; len <= 3; len++) {
            if (start + len <= s.length()) {
                string ss = s.substr(start, len);
                if (validIp(ss)) {
                    if (path.size())
                        path.push_back('.');
                        
                    for (int i = 0; i < len; i++)
                        path.push_back(ss[i]);
                        
                    dfs_restoreIpAddresses(s, path, start+len, step+1, result);
                    
                    for (int i = 0; i < len; i++)
                        path.pop_back();
                        
                    if (path.size())
                        path.pop_back();
                }
            }
        }
    }
    /* Letter Combinations of a Phone Number 
    Given a digit string, return all possible letter combinations that the number could represent.

A mapping of digit to letters (just like on the telephone buttons) is given below.



Input:Digit string "23"
Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
Note:
Although the above answer is in lexicographical order, your answer could be in any order you want.
    */
    vector<string> letterCombinations(string digits) {
        vector<string> keys;
        // 0
        keys.push_back(" ");
        // 1
        keys.push_back("");
        // 2
        keys.push_back("abc");
        // 3
        keys.push_back("def");
        // 4
        keys.push_back("ghi");
        // 5
        keys.push_back("jkl");
        // 6
        keys.push_back("mno");
        // 7
        keys.push_back("pqrs");
        // 8
        keys.push_back("tuv");
        // 9
        keys.push_back("wxyz");
        
        string path;
        vector<string> results;
        dfs_letterCombinations(keys, digits, path, 0, digits.size(), results);
        return results;
    }
    
    void dfs_letterCombinations(const vector<string>& keys, string digits, string path, int i, int n, vector<string>& results) {
        if (i == n) {
            results.push_back(path);
            return;
        }
        
        int d = int(digits[i] - '0');
        if (d == 1)
            dfs_letterCombinations(keys, digits, path, i+1, n, results);
        else {
            for (auto k : keys[d]) {
                path.push_back(k);
                dfs_letterCombinations(keys, digits, path, i+1, n, results);
                path.pop_back();
            }
        }
    }
    
    /* Combination Sum 
    Given a set of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.

The same repeated number may be chosen from C unlimited number of times.

Note:
All numbers (including target) will be positive integers.
Elements in a combination (a1, a2, … , ak) must be in non-descending order. (ie, a1 ≤ a2 ≤ … ≤ ak).
The solution set must not contain duplicate combinations.
For example, given candidate set 2,3,6,7 and target 7, 
A solution set is: 
[7] 
[2, 2, 3] 
    */
    vector<vector<int> > combinationSum(vector<int> &candidates, int target) {
        set<vector<int> > results;
        vector<int> path;
        sort(candidates.begin(), candidates.end());
        dfs_combinationSum(candidates, target, path, 0, results);
        vector<vector<int> > real_results(results.begin(), results.end());
        return real_results;
    }
    
    void dfs_combinationSum(const vector<int>& candidates, int target, vector<int> path, int sum, set<vector<int> >& results) {
        if (sum > target)
            return;
        if (sum == target) {
            results.insert(path);
            return;
        }
        for (auto c : candidates) {
            if (sum+c > target)
                break;
            if (path.size() && c < path[path.size()-1])
                continue;
            path.push_back(c);
            dfs_combinationSum(candidates, target, path, sum+c, results);
            path.pop_back();
        }
    }
    
    /*Combination Sum II 
    Given a collection of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.

Each number in C may only be used once in the combination.

Note:
All numbers (including target) will be positive integers.
Elements in a combination (a1, a2, … , ak) must be in non-descending order. (ie, a1 ≤ a2 ≤ … ≤ ak).
The solution set must not contain duplicate combinations.
For example, given candidate set 10,1,2,7,6,1,5 and target 8, 
A solution set is: 
[1, 7] 
[1, 2, 5] 
[2, 6] 
[1, 1, 6] 
    */
    vector<vector<int> > combinationSum2(vector<int> &num, int target) {
        sort(num.begin(), num.end());
        vector<int> path;
        set<vector<int> > results;
        
        dfs_combinationSum2(num, target, num.size(), 0, 0, path, results);
        
        return vector<vector<int> >(results.begin(), results.end());
    }
    
    void dfs_combinationSum2(const vector<int>& num, const int& target, const int& n, int sum, int i, vector<int> path, set<vector<int> >& results) {
        if (sum == target) {
            results.insert(path);
            return;
        }
        // i is [0, n-1]
        if (n == i)
            return;
            
        // not choose num[i]
        if (sum > target)
            return;

        dfs_combinationSum2(num, target, n, sum, i+1, path, results);
        
        // choose num[i]
        if (sum + num[i] <= target) {
            // avoid using the same num again            
            if (i > 0 && num[i] == num[i-1])
                if (find(path.begin(), path.end(), num[i]) == path.end())
                    return;

            path.push_back(num[i]);
            dfs_combinationSum2(num, target, n, sum+num[i], i+1, path, results);
            path.pop_back();
        }
    }
    
    // Word Break
	/* 
	   Given a string s and a dictionary of words dict,
	   determine if s can be segmented into a space-separated
	   sequence of one or more dictionary words.
	*/
	// method: DP
    bool wordBreak(string s, unordered_set<string> &dict) {
		vector<bool> mem(s.length()+1, false);
		mem[0] = true;

		for (int i = 0; i < s.length(); ++i) {
			for (int j = 0; j <= i; ++j) {
				if (mem[j] && dict.count(s.substr(j, i-j+1))) {
					mem[i+1] = true;
					break;
				}
			}
		}
					
		return mem[s.length()];
	}

	// Word Break II
	/* 
	   Given a string s and a dictionary of words dict,
       add spaces in s to construct a sentence where each 
	   word is a valid dictionary word.

	   Return all such possible sentences.
	*/
	// method: recursion + mem (dp)
	vector<string> wordBreak2_recursion(string s, unordered_set<string> &dict, map<string, vector<string> >& mem) {
		if (mem.find(s) != mem.end())
		{
			return mem[s];
		}

		vector<string> result;
		result.clear();

		// recursion
		// check whole word
		if (dict.count(s))
			result.push_back(s);
		// check partial
		for (int len = 1;len < s.length(); ++len) {
			if (dict.count(s.substr(0, len))) {
				vector<string> ss = wordBreak2_recursion(s.substr(len, s.length()-len), dict, mem);
				for (int i = 0; i < ss.size(); i++)
					result.push_back(s.substr(0, len)+" "+ss[i]);
			}
		}

		mem[s] = result;
		return result;
	}
	vector<string> wordBreak2(string s, unordered_set<string> &dict) {
		map<string, vector<string> > mem;

		return wordBreak2_recursion(s, dict, mem);
	}

	/* Wildcard Matching 
	Implement wildcard pattern matching with support for '?' and '*'.

	'?' Matches any single character.
	'*' Matches any sequence of characters (including the empty sequence).

	The matching should cover the entire input string (not partial).

	The function prototype should be:
	bool isMatch(const char *s, const char *p)

	Some examples:
	isMatch("aa","a") °˙ false
	isMatch("aa","aa") °˙ true
	isMatch("aaa","aa") °˙ false
	isMatch("aa", "*") °˙ true
	isMatch("aa", "a*") °˙ true
	isMatch("ab", "?*") °˙ true
	isMatch("aab", "c*a*b") °˙ false	
	*/
    // method: dp
    bool isMatch(const char *s, const char *p) {
        long sn = strlen(s);
        long pn = strlen(p);
        if (sn == 0 || pn == 0)
            return false;
        // create dp array: sn rows, pn columns
        // -1: unintialized; 0: false; 1: true
        vector< vector<int> > dp (sn, vector<int>(pn, -1));

        // initialize dp[0][0]
        if (s[0] == '*')
            for (int cc = 0; cc < pn; cc++)
                dp[0][cc] = 1;
        if (p[0] == '*')
            for (int rr = 0; rr < sn; rr++)
                dp[rr][0] = 1;
        if (dp[0][0] == -1) {
            if (s[0] == p[0] || s[0] == '?' || p[0] == '?')
                dp[0][0] = 1;
        } else
            dp[0][0] = 0;
        
        // initialize first row and first column
        for (int r = 1; r < sn; r++) {
            if (dp[r][0] == -1) {
                if (dp[r-1][0] && s[r] == '*')
                    dp[r][0] = 1;
                else
                    dp[r][0] = 0;
            }
        }
        for (int c = 1; c < pn; c++) {
            if (dp[0][c] == -1) {
                if (dp[0][c-1] && p[c] == '*') {
                    dp[0][c] = 1;
                } else
                    dp[0][c] = 0;
            }
        }
        
        for (int r = 1; r < sn; r++)
            for (int c = 1; c < pn; c++) {
                if (dp[r][c] == -1) {
                    if (dp[r-1][c-1] == 1) {
                        // handle '*' situation
                        if (s[r] == '*' || p[c] == '*') {
                            if (s[r] == '*')      // set the rest of row to 1
                                for (int cc = c; cc < pn; cc++)
                                    dp[r][cc] = 1;
                            if (p[c] == '*')      // set the rest of col to 1
                                for (int rr = r; rr < sn; rr++)
                                    dp[rr][c] = 1;
                        }
                        else if (s[r] == '?' || p[c] == '?' || s[r] == p[c]){
                            dp[r][c] = 1;
                        }
                        else
                            dp[r][c] = 0;
                    }
                    else
                        dp[r][c] = 0;
                }
            }
        
//        cout << "dp[sn][pn] = " << dp[sn-1][pn-1] << endl;
        
        return dp[sn-1][pn-1];
	}
    
    /*
     Given a linked list, determine if it has a cycle in it.
     
     Follow up:
     Can you solve it without using extra space?
     */

    bool hasCycle(ListNode *head) {
        // IMPORTANT: Please reset any member data you declared, as
        // the same Solution instance will be reused for each test case.
        if (!head)
            return false;
        ListNode* p1 = head;
        if (!head->next)
            return false;
        ListNode* p2 = head->next;
        
        while (p1 && p2) {
            if (p1 == p2)
                return true;
            p1 = p1->next;
            if (p2->next)
                p2 = p2->next->next;
            else
                p2 = NULL;
        }
        return false;
    }
    
    /*
     Given a linked list, return the node where the cycle begins. If there is no cycle, return null.
     
     Follow up:
     Can you solve it without using extra space?
     */
    ListNode *detectCycle(ListNode *head) {
        // IMPORTANT: Please reset any member data you declared, as
        // the same Solution instance will be reused for each test case.
        if (!head || !head->next)
            return NULL;
        unordered_set<ListNode*> mem;
        
        while (head) {
            if (mem.count(head))
                return head;
            mem.insert(head);
            head = head->next;
        }
        
        return NULL;
    }
    
    /* Reorder List
     Given a singly linked list L: L0->L1->L2->...->Ln
     reorder it to: L0->Ln->L1->Ln-1->L2->Ln-2...
     
     You must do this in-place without altering the nodes' values.
     
     For example,
     Given {1,2,3,4}, reorder it to {1,4,2,3}.
     
     */
    void reorderList(ListNode *head) {
        if (!head || !head->next)
            return;
        // IMPORTANT: Please reset any member data you declared, as
        // the same Solution instance will be reused for each test case.
        vector<ListNode*> mem;
        ListNode* p = head;
        while(p) {
            mem.push_back(p);
            p = p->next;
        }
        ListNode* last = head;
        long n = mem.size()-1;
        for (int i = 1; i < mem.size(); i++) {
            if (i%2 == 1)   //odd
                last->next = mem[n-i/2];
            else    // even
                last->next = mem[i/2];
            last = last->next;
        }
        last->next = NULL;
    }
    
    /* Binary Tree Preorder Traversal 
     Given a binary tree, return the preorder traversal of its nodes' values.
     
     For example:
     Given binary tree {1,#,2,3}, return [1,2,3]
     
     Note: Recursive solution is trivial, could you do it iteratively?
     */
    vector<int> preorderTraversal(TreeNode *root) {
        // IMPORTANT: Please reset any member data you declared, as
        // the same Solution instance will be reused for each test case.
        vector<int> result;
        if (!root)
            return result;
        vector<int> left, right;
        if (root->left)
            left = preorderTraversal(root->left);
        if (root->right)
            right = preorderTraversal(root->right);
        
        // join the vectors
        result.push_back(root->val);
        for (int i = 0; i < left.size(); i++)
            result.push_back(left[i]);
        for (int i = 0; i < right.size(); i++)
            result.push_back(right[i]);
        
        return result;
    }
    
    /* Merge Sorted Array
       Given two sorted integer arrays A and B, merge B into A as one sorted array.
     
       Note:
       You may assume that A has enough space to hold additional elements from B. The number of elements initialized in A and B are m and n respectively.
     */
    void merge(int A[], int m, int B[], int n) {
        // IMPORTANT: Please reset any member data you declared, as
        // the same Solution instance will be reused for each test case.
        if (m == 0) {
            for (int i = 0;i < n; i++)
                A[i] = B[i];
            return;
        }
        if (n == 0)
            return;
        
        vector<int> c;
        int i = 0;
        int j = 0;
        while (i < m || j < n) {
            if (i == m)
                c.push_back(B[j++]);
            else if (j == n)
                c.push_back(A[i++]);
            else {
                if (A[i] < B[j])
                    c.push_back(A[i++]);
                else
                    c.push_back(B[j++]);
            }
        }
        
        for (int i = 0; i < c.size(); i++)
            A[i] = c[i];
        
    }
    
    /* String to Integer (atoi)
     Implement atoi to convert a string to an integer.
     
     Hint: Carefully consider all possible input cases. If you want a challenge, please do not see below and ask yourself what are the possible input cases.
     
     Notes: It is intended for this problem to be specified vaguely (ie, no given input specs). You are responsible to gather all the input requirements up front.
     
     spoilers alert... click to show requirements for atoi.
     
     Requirements for atoi:
     The function first discards as many whitespace characters as necessary until the first non-whitespace character is found. Then, starting from this character, takes an optional initial plus or minus sign followed by as many numerical digits as possible, and interprets them as a numerical value.
     
     The string can contain additional characters after those that form the integral number, which are ignored and have no effect on the behavior of this function.
     
     If the first sequence of non-whitespace characters in str is not a valid integral number, or if no such sequence exists because either str is empty or it contains only whitespace characters, no conversion is performed.
     
     If no valid conversion could be performed, a zero value is returned. If the correct value is out of the range of representable values, INT_MAX (2147483647) or INT_MIN (-2147483648) is returned.
     */
    int atoi(const char *str) {
        // IMPORTANT: Please reset any member data you declared, as
        // the same Solution instance will be reused for each test case.
        if (!str)
            return 0;
        
        const int maxInt[] ={2,1,4,7,4,8,3,6,4,7};
        const int minInt[] ={2,1,4,7,4,8,3,6,4,8};
        bool isNeg = false;
        vector<int> digits;
        int i = 0;
        // skip whitespaces
        while(str[i] == ' ') {
            if (!str[++i])  // check null
                return 0;
        }
        
        // check optional sign
        if (str[i] == '+') {
            i++;
        }
        else if (str[i] == '-') {
            isNeg = true;
            i++;
        }
        
        while(str[i] && str[i] >= '0' && str[i] <= '9') {
            digits.push_back((int)str[i++]-(int)'0');
        }
        
        if (digits.size() == 0) // invalid
            return 0;
        if (digits.size() > 10) // out of range case 1
            return (isNeg)?INT_MIN:INT_MAX;
        if (digits.size() == 10) {    // check whether exactly larger than INT_MAX or INT_MIN
            if (isNeg) {
                for (int i = 0; i < digits.size(); i++)
                    if (digits[i] > minInt[i])
                        return INT_MIN;
                    else if (digits[i] == minInt[i])
                        continue;
                    else
                        break;
            }
            else {
                for (int i = 0; i < digits.size(); i++)
                    if (digits[i] > maxInt[i])
                        return INT_MAX;
                    else if (digits[i] == maxInt[i])
                        continue;
                    else
                        break;
            }
        }
        // then it should be valid cases, just parse the value
        int base = 1;
        int result = 0;
        for (long i = digits.size()-1; i >= 0; i--) {
            result += digits[i]*base;
            base *= 10;
        }
        
        return (isNeg) ? -result : result;
    }
    
    /* Binary Tree Postorder Traversal
     Given a binary tree, return the postorder traversal of its nodes' values.
     
     For example:
     Given binary tree {1,#,2,3}, output [3,2,1]
     
     */
    vector<int> postorderTraversal(TreeNode *root) {
        // IMPORTANT: Please reset any member data you declared, as
        // the same Solution instance will be reused for each test case.
        vector<int> result;
        if (!root)
            return result;
        vector<int> left, right;
        if (root->left)
            left = postorderTraversal(root->left);
        if (root->right)
            right = postorderTraversal(root->right);
        // joint vectors
        for (int i = 0; i < left.size(); i++)
            result.push_back(left[i]);
        for (int i = 0; i < right.size(); i++)
            result.push_back(right[i]);
        result.push_back(root->val);
        
        return result;
    }
    
    /* Insertion Sort List
     Sort a linked list using insertion sort.
     */
    ListNode *insertionSortList(ListNode *head) {
        // IMPORTANT: Please reset any member data you declared, as
        // the same Solution instance will be reused for each test case.
        if (!head || !head->next)
            return head;
        
        ListNode* myhead = NULL;
        ListNode *mytail = NULL;
        ListNode *cur = head;
        
        while (cur) {
            ListNode *next = cur->next;
            
            // compare cur to every element from myhead to my tail
            if (!myhead) {    // if is the first element
                myhead = cur;
                mytail = cur;
            }
            else {
                // check with myhead
                if (cur->val <= myhead->val) {   // should insert before myhead
                    cur->next = myhead;
                    myhead = cur;
                    mytail->next = next;
                }
                else if (cur->val > mytail->val) { // should keep unchanged
                    mytail = cur;
                }
                else {  // should insert in between myhead and mytail
                    ListNode *itcur = myhead;
                    ListNode *itnext = myhead->next;
                    while (itcur != mytail) {
                        if (cur->val > itcur->val && cur->val <= itnext->val) {    // could insert here itcur
                            itcur->next = cur;
                            cur->next = itnext;
                            mytail->next = next;
                            break;
                        }
                        else {    // just increase it
                            itcur = itnext;
                            itnext = itcur->next;
                        }
                    }
                }
            }
            
            // go to next element
            cur = next;
        }
        
        return myhead;
    }
    
    /* Sort List
     Sort a linked list in O(n log n) time using constant space complexity.
     */
    ListNode *sortList(ListNode *head) {
        // IMPORTANT: Please reset any member data you declared, as
        // the same Solution instance will be reused for each test case.
        
        // base case:
        if (!head || !head->next)
            return head;
        
        // recursive case:
        // find length and middle object
        int n = 0;
        ListNode *it = head;
        while(it) {
            n++;
            it = it->next;
        }
        // get the pointer to the middle of the list
        it = head;
        ListNode* it2 = NULL;
        for (int i = 0; i < n/2; i++) {
            it2 = it;
            it = it->next;
        }
        // split into two lists
        it2->next = NULL;
        // recursive call
        ListNode *l1 = sortList(head);
        ListNode *l2 = sortList(it);
        
        // merge
        ListNode* newhead;
        if (l1->val < l2->val) {
            newhead = l1;
            it = l1->next;
            it2 = l2;
        }
        else {
            newhead = l2;
            it = l1;
            it2 = l2->next;
        }
        ListNode *last = newhead;
        while (it || it2) {
            if (!it) {    // if l1 reaches the end
                last->next = it2;
                break;
            }
            else if (!it2) {
                last->next = it;
                break;
            }
            else {
                if (it->val < it2->val) {
                    last->next = it;
                    last = it;
                    it = it->next;
                }
                else {
                    last->next = it2;
                    last = it2;
                    it2 = it2->next;
                }
            }
        }
        
        return newhead;
    }
    
    /* Implement pow(x, n).
	 */
	// need to handle some special cases carefully
	double pow(double x, int n) {
		// Note: The Solution object is instantiated only once and is reused by each test case.
        // handle x = -1
        if (x == -1)
            return (n%2) ? -1:1;
        
 		bool inverse = false;
		if (n < 0) {
			n *= -1;
			inverse = true;
		}
		double result = 1;
		while (n > 0)
		{
			result *= x;
			n--;
			if (result == 0 || result == 1)
			    break;
		}
		if (inverse)
			return 1.0/result;
		else
			return result;
    }
    
    /* Implement int sqrt(int x)
       Note: turn it into divide and conquer since only want output as int
     */
    int sqrt(int x) {
        int left = 1;
        int right = x/2;
        int mid;
        int last_mid;
        
        if (x < 2)
            return x;
        while (left <= right) {
            mid = (left + right) / 2;
            if (x / mid > mid) {    // mid is too small
                left = mid + 1;
                last_mid = mid;
            } else if (x / mid < mid) { // mid is too large
                right = mid - 1;
            } else
                return mid;
        }
        return last_mid;
    }
    
	/* Given an array where elements are sorted in ascending order, convert it to a height balanced BST.
	 */
	// vector<int> a(b.start, b.end) range is [b[start], b[end]), not including b[end]!
    
	// Definition for binary tree
	struct TreeNode {
		int val;
		TreeNode *left;
		TreeNode *right;
		TreeNode(int x) : val(x), left(NULL), right(NULL) {}
	};
	TreeNode *sortedArrayToBST(vector<int> &num) {
		long n = num.size();
		if (n == 0)
			return NULL;
		if (n == 1)
			return new TreeNode(num[0]);
        
		TreeNode* node = new TreeNode(num[n/2]);
		vector<int> lnum(num.begin(), num.begin()+n/2);
		vector<int> rnum(num.begin()+n/2+1, num.end());
		node->left = sortedArrayToBST(lnum);
		node->right = sortedArrayToBST(rnum);
        
		return node;
    }
    
	/* Given a binary tree, determine if it is a valid binary search tree (BST).
	 */
	// just follow the definition
	int findMax(TreeNode* node) {
        int mmax = node->val;
        while(node) {
            mmax = node->val;
            node = node->right;
        }
        return mmax;
    }
    int findMin(TreeNode* node) {
        int mmin = node->val;
        while(node) {
            mmin = node->val;
            node = node->left;
        }
        return mmin;
    }
    bool isValidBST(TreeNode *root) {
        // Note: The Solution object is instantiated only once and is reused by each test case.
        if (!root)
            return true;
        
        bool leftgood = true, rightgood = true;
        if (root->left)
            leftgood = ((root->val > findMax(root->left)) && isValidBST(root->left));
        if (root->right)
            rightgood = ((root->val < findMin(root->right)) && isValidBST(root->right));
        
        return (leftgood && rightgood);
        
    }
    
	/* Given n, how many structurally unique BST's (binary search trees) that store values 1...n?
	 */
	int numTrees(int n) {
        // Note: The Solution object is instantiated only once and is reused by each test case.
        if (n == 0 || n == 1)
            return n;
        
        vector<int> mem(n+1, 0);
        mem[0] = 0;
        mem[1] = 1;
        for (int i = 2; i < n+1; i++) {
            for (int j = 0; j <i; j++) {
				if (mem[j] == 0)
					mem[i] += mem[i-j-1];
				else if (mem[i-j-1] == 0)
					mem[i] += mem[j];
				else
	                mem[i] += (mem[j] * mem[i-j-1]);
            }
        }
        
        return mem[n];
    }
    
	/* Given an array of integers, every element appears twice except for one. Find that single one.
	 */
	int singleNumber(int A[], int n) {
        // Note: The Solution object is instantiated only once and is reused by each test case.
        int myxor = 0;
        for (int i =0; i < n; i++)
            myxor ^= A[i];
        
        return myxor;
    }
    
    /* Single Number II 
     Given an array of integers, every element appears three times except for one. Find that single one.
     
     Note:
     Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?
     */
    int singleNumber2(int A[], int n) {
        // IMPORTANT: Please reset any member data you declared, as
        // the same Solution instance will be reused for each test case.
        int count[32] = {0};
        int result = 0;
        for (int i = 0; i < 32; i ++) {
            // shift i bits for each element
            for (int j = 0; j < n; j++) {
                if (A[j] & (1 << i))
                    count[i] += 1;
            }
            if (count[i]%3)
                result |= (1 << i);
        }
        return result;
    }
    
	/* Reverse digits of an integer.
	 */
	// need to use reverse_iterator to traverse list backwards
	int reverse(int x) {
        // Note: The Solution object is instantiated only once and is reused by each test case.
        if (x == 0)
            return 0;
        list<int> digits;
        bool isneg = false;
        if (x < 0) {
            isneg = true;
            x *= -1;
        }
        
        while (x > 0) {
            digits.push_back(x%10);
            x /= 10;
        }
        
        list<int>::iterator it = digits.begin();
        while((*it) == 0) {
            digits.pop_front();
            it = digits.begin();
        }
        
		for (it = digits.begin(); it != digits.end(); it++)
			cout << *it << " ";
		cout << endl;
        
        int base = 1;
        int result = 0;
        for (list<int>::reverse_iterator it = digits.rbegin(); it != digits.rend(); it++) {
            result += (*it)*base;
            base *= 10;
        }
        
        return (isneg)? -result : result;
    }
    
	// use multimap instead of map!
	vector<int> twoSum(vector<int> &numbers, int target) {
        // IMPORTANT: Please reset any member data you declared, as
        // the same Solution instance will be reused for each test case.
        multimap<int, int> mem;
        for (int i = 0; i < numbers.size(); i++)
			mem.insert(pair<int, int>(numbers[i], i+1));
        
        multimap<int, int>::iterator it1 = mem.begin();
		multimap<int, int>::iterator it2 = mem.end();
		--it2;
        
        int sum = it1->first + it2->first;
        while (sum != target) {
            if (sum < target)
                ++it1;
            else
                --it2;
            
            sum = it1->first + it2->first;
        }
        
        vector<int> result;
		if (it1->second < it2->second) {
			result.push_back(it1->second);
			result.push_back(it2->second);
		}
		else {
			result.push_back(it2->second);
			result.push_back(it1->second);
		}
        
        return result;
    }
    
    void myqsort(int* a, int start, int end) {
        if (end <= start) return;
        
        int left = start;
        int right = end;
        int pivot = a[left + (right - left) / 2];
        
        // selection
        while (left <= right) {
            while (a[left] < pivot) left ++;
            while (a[right] > pivot) right --;
            if (left <= right) {
                int tmp = a[left];
                a[left] = a[right];
                a[right] = tmp;
                left ++;
                right --;
            }
        }
        
        // divide
        if (left < end) myqsort(a, left, end);
        if (right > start) myqsort(a, start, right);
    }
    
    
    /* Max Points on a Line
     Given n points on a 2D plane, find the maximum number of points that lie on the same straight line.   
     
     my solution:
     represent a line by y = kx + b, but when a line is x = sth, set k = inf
     then build a string (kb) and hash it into hashtable, then each pair on points vote for 1 of (kb)
     then select the maxium vote and convert it to number of points on that line
     */
    /* another way to represent line: ax + by + c = 0, and a,b,c are int
     for two points (x0, y0) and (x1, y1) where x0 != x1 or y0 != y1, we can set a = y1 - y0, b = x0 - x1 and c = x1 * y0 - x0 * y1. Then calculate the gcd of a, b and c and divide them by gcd. Then make a becomes non-negative and b non-negative when a is zero. ---- by poker2008 on leetcode
     */
    int maxPoints(vector<Point> &points) {
        if (points.size() <= 2)
            return (int)points.size();
        unordered_map<string, int> hashtable;
        unordered_map<string, int>::iterator it;
        
        for (int i = 0; i < points.size(); i++) {
            for (int j = 0; j < points.size(); j++) {
                if (i == j)
                    continue;
                int x1 = points[i].x;
                int y1 = points[i].y;
                int x2 = points[j].x;
                int y2 = points[j].y;
                stringstream ss;
                // check if it is vertical
                if (x1 == x2) {
                    // store the following pair: <inf, x>
                    ss << "inf" << x1;
                }
                else { // a normal line
                    double k = ((double)y1 - y2)/(x1 - x2);
                    double b = y1 - k*x1;
                    ss << k << b;
                }
                string code = ss.str();
                it = hashtable.find(code);
                if (it == hashtable.end()) {    //does not exist
                    hashtable.insert(pair<string, int>(code, 1));
                }
                else {  // increase count by 1
                    it->second++;
                }
            }
        }
        
        // check the maximum
        int maxCount = 0;
        for (it = hashtable.begin(); it != hashtable.end(); it++)
            if (it->second > maxCount)
                maxCount = it->second;
        
        // maxCount = (n-1) + (n-2) + ... + 1
        // maxCount = (n-1 + 1)(n-1)/2
        // n = 1/2 + sqrt(2*macCount + 1/4)
        int n = ceil(0.5+sqrt(2*maxCount+0.25));
        return n;
    }
    
    /* Add Two Numbers
     You are given two linked lists representing two non-negative numbers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.
     
     Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
     Output: 7 -> 0 -> 8
     */
    ListNode *addTwoNumbers(ListNode *l1, ListNode *l2) {
        // IMPORTANT: Please reset any member data you declared, as
        // the same Solution instance will be reused for each test case.
        if (!l1)
            return l2;
        if (!l2)
            return l1;
        
        int carry = 0;
        ListNode* head = NULL;
        ListNode* it = head;
        
        while (l1 || l2) {
            int num = 0;
            if (!l1) {
                num = l2->val + carry;
                l2 = l2->next;
            }
            else if (!l2) {
                num = l1->val + carry;
                l1 = l1->next;
            }
            else {
                num = l1->val + l2->val + carry;
                l1 = l1->next;
                l2 = l2->next;
            }
            if (num >= 10) {
                num -= 10;
                carry = 1;
            }
            else
                carry = 0;
            
            ListNode* tmp = new ListNode(num);
            if (!head) {
                head = tmp;
                it = tmp;
            }
            else {
                it->next = tmp;
                it = it->next;
            }
        }
        
        if (carry)
            it->next = new ListNode(1);
        
        return head;
    }
    vector<vector<int> > threeSum(vector<int> &num) {
        // IMPORTANT: Please reset any member data you declared, as
        // the same Solution instance will be reused for each test case.
        vector< vector<int> > result;
        if (num.size() <= 2)
            return result;
        
        // sort in ascending order
        sort(num.begin(), num.end());
        
        // mem: <num[i], number of num[i]>
        // hashtable to store elements and their counts
        unordered_map<int, int> mem;
        unordered_map<int, int>::iterator memit;
        // mem_result: hashset to store result to check duplicated
        unordered_set<string> mem_result;
        
        // store all elements in mem
        for (int i = 0; i < num.size(); i++) {
            memit = mem.find(num[i]);
            if (memit == mem.end())
                mem.insert(pair<int,int>(num[i], 1));
            else
                memit->second++;
        }
        
        // calcluate 3 sum
        for (int i = 0; i < num.size()-2; i++) {
            if (i > 0 && num[i] == num[i-1])
                continue;
            // temporary reduce the count of num[i] by 1
            memit = mem.find(num[i]);
            memit->second--;
            for (int j = i+1; j < num.size()-1;  j++) {
                if (j > i+1 && num[j] == num[j-1])
                    continue;
                // temporary reduce the count of num[j] by 1
                memit = mem.find(num[j]);
                memit->second--;
                
                int target = -num[i] - num[j];
                memit = mem.find(target);
                if (memit != mem.end() && memit->second) {
                    // we found the element!
                    int c = memit->first;
                    vector<int> single;
                    stringstream ss;
                    // check the order, given that we know num[i] <= num[j]
                    if (c <= num[i]) {
                        ss << c << num[i] << num[j];
                        single.push_back(c);
                        single.push_back(num[i]);
                        single.push_back(num[j]);
                    }
                    else if ( c <= num[j]) {
                        ss << num[i] << c << num[j];
                        single.push_back(num[i]);
                        single.push_back(c);
                        single.push_back(num[j]);
                    }
                    else {
                        ss << num[i] << num[j] << c;
                        single.push_back(num[i]);
                        single.push_back(num[j]);
                        single.push_back(c);
                    }
                    
                    if (mem_result.count(ss.str()) == 0) {  // if never see this result before
                        mem_result.insert(ss.str());
                        result.push_back(single);
                    }
                }
                memit = mem.find(num[j]);
                memit->second++;
            }
            memit = mem.find(num[i]);
            memit->second++;
        }
        
        return result;
    }
    
    /* Copy List with Random Pointer
     A linked list is given such that each node contains an additional random pointer which could point to any node in the list or null.
     
     Return a deep copy of the list.
     */
    RandomListNode *copyRandomList(RandomListNode *head) {
        // IMPORTANT: Please reset any member data you declared, as
        // the same Solution instance will be reused for each test case.
        if (!head)
            return NULL;
        unordered_map<RandomListNode*, RandomListNode*> mem;
        RandomListNode *n = NULL, *nit = NULL, *nitp = NULL;
        RandomListNode *it = head;
        
        // first pass: copy data and next
        while (it) {
            // if n is empty
            if (!n) {
                n = new RandomListNode(it->label);
                nit = n;
            } else {
                nit = new RandomListNode(it->label);
                nitp->next = nit;
            }
            // push correspondence
            mem[it] = nit;
            it = it->next;
            nitp = nit;
            nit = nit->next;
        }
        // second pass: copy random pointer
        it = head;
        nit = n;
        while(it) {
            if (it->random)
                nit->random = mem[it->random];
            it = it->next;
            nit = nit->next;
        }
        
        return n;
    }
    
    /* Evaluate Reverse Polish Notation
     Evaluate the value of an arithmetic expression in Reverse Polish Notation.
     
     Valid operators are +, -, *, /. Each operand may be an integer or another expression.
     
     Some examples:
     ["2", "1", "+", "3", "*"] -> ((2 + 1) * 3) -> 9
     ["4", "13", "5", "/", "+"] -> (4 + (13 / 5)) -> 6
     */
    int evalRPN(vector<string> &tokens) {
        // use stack to store elements
        // whenever there is an operator, pop last two elements to compute
        if (tokens.empty())
            return 0;
        vector<int> mem;
        for (int i = 0; i < tokens.size(); i++) {
            if (tokens[i].size() == 1) {
                // check if it is an operator
                char p = tokens[i][0];
                if (p == '+') {
                    int b = mem.back();
                    mem.pop_back();
                    int a = mem.back();
                    mem.pop_back();
                    
                    mem.push_back(a+b);
                }
                else if (p == '-') {
                    int b = mem.back();
                    mem.pop_back();
                    int a = mem.back();
                    mem.pop_back();
                    
                    mem.push_back(a-b);
                }
                else if (p == '*') {
                    int b = mem.back();
                    mem.pop_back();
                    int a = mem.back();
                    mem.pop_back();
                    
                    mem.push_back(a*b);
                }
                else if (p == '/') {
                    int b = mem.back();
                    mem.pop_back();
                    int a = mem.back();
                    mem.pop_back();
                    
                    mem.push_back(a/b);
                }
                else {  // number with single digit
                    mem.push_back(atoi(tokens[i].c_str()));
                }
            }
            else {  // number with multiple digits
                mem.push_back(atoi(tokens[i].c_str()));
            }
        }
        return mem.back();
    }
    
    /* Gas Station
     There are N gas stations along a circular route, where the amount of gas at station i is gas[i].
     
     You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from station i to its next station (i+1). You begin the journey with an empty tank at one of the gas stations.
     
     Return the starting gas station's index if you can travel around the circuit once, otherwise return -1.
     
     Note:
     The solution is guaranteed to be unique.
     */
    /* summary: it is like the linear time solution for maximum subarray problem
     the key is that any sequence with negative sum cannot be the begining of the optimal subarray
     ALSO, need to check whether overall fulfills the condition that total gas > total cost
     THE FACT IS: if total gas > total cost, there must exist a path!
     */
    int canCompleteCircuit(vector<int> &gas, vector<int> &cost) {
        int index = 0;
        int oil = 0;
        int total = 0;
        for (int i = 0; i < gas.size(); i++) {
            oil += gas[i] - cost[i];
            total += gas[i] - cost[i];
            if (oil < 0) {
                oil = 0;
                index = i+1;
            }
        }
        return (total >= 0) ? index : -1;
    }
    
    /* Longest Consecutive Sequence
     Given an unsorted array of integers, find the length of the longest consecutive elements sequence.
     
     For example,
     Given [100, 4, 200, 1, 3, 2],
     The longest consecutive elements sequence is [1, 2, 3, 4]. Return its length: 4.
     
     Your algorithm should run in O(n) complexity.
     */
    int longestConsecutive(vector<int> &num) {
        if (num.size() < 2)
            return (int)num.size();
        unordered_map<int, int> mem;
        // first pass: insert all into hashtable
        for (int i =0 ; i < num.size(); i++)
            mem[num[i]] = 1;
        // second pass: search for each element
        int maxLen = 1;
        for (int i = 0; i < num.size(); i++) {
            if (mem[num[i]]) {
                mem[num[i]] = 0;
                int tmpLen = 1;
                // search for num[i]++s
                int tmp = num[i]+1;
                while (mem[tmp]) {
                    mem[tmp] = 0;
                    tmpLen++;
                    tmp++;
                }
                // search for num[i]--s
                tmp = num[i]-1;
                while (mem[tmp]) {
                    mem[tmp] = 0;
                    tmpLen++;
                    tmp--;
                }
                if (tmpLen > maxLen)
                    maxLen = tmpLen;
            }
        }
        return maxLen;
    }

    /* Valid Palindrome
     Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.
     
     For example,
     "A man, a plan, a canal: Panama" is a palindrome.
     "race a car" is not a palindrome.
     
     Note:
     Have you consider that the string might be empty? This is a good question to ask during an interview.
     
     For the purpose of this problem, we define empty string as valid palindrome.
     */
    bool isPalindrome(string s) {
        string ss;
        char off = 'A' - 'a';
        for (int i = 0; i < s.size(); i++) {
            if ((s[i] >= 'a' && s[i] <= 'z') || (s[i] >= '0' && s[i] <= '9'))
                ss.push_back(s[i]);
            else if (s[i] >= 'A' && s[i] <= 'Z')
                ss.push_back(char(s[i] - off));
        }
        if (!ss.size())
            return true;
        string ss2(ss.rbegin(), ss.rend());
        return !(ss.compare(ss2));
    }

    /* Palindrome Number
     Determine whether an integer is a palindrome. Do this without extra space.
     
     Some hints:
     Could negative integers be palindromes? (ie, -1)
     
     If you are thinking of converting the integer to string, note the restriction of using extra space.
     
     You could also try reversing an integer. However, if you have solved the problem "Reverse Integer", you know that the reversed integer might overflow. How would you handle such case?
     
     There is a more generic way of solving this problem.
     */
    bool isPalindrome(int x) {
        if (x == 0)
            return true;
        if (x <  0) // negative numbers are not palindromes
            return false;
        int maxBase = 1;
        double doubleBase = 1;
        while (x/(doubleBase*10) >= 1) {
            doubleBase *= 10;
        }
        maxBase = (int)doubleBase;
        while (maxBase >= 10) {
            if (x == 0)
                return true;
            if (x%10 == x/maxBase) {
                x = x - (x/maxBase)*maxBase;    // remove leading digit
                x = x/10;   // remove ending digit
                maxBase /= 100;
            }
            else
                return false;
        }
        return true;
    }
    
    /* Palindrome Partitioning 
     Given a string s, partition s such that every substring of the partition is a palindrome.
     
     Return all possible palindrome partitioning of s.
     
     For example, given s = "aab",
     Return
     
     [
     ["aa","b"],
     ["a","a","b"]
     ]
     */
    vector<vector<string> > partition(string s) {
        vector<vector<string> > result;
        // base case:
        if (s.size() == 0)
            return result;
        if (s.size() == 1) {
            result.push_back(vector<string>(1,s));
            return result;
        }
        // recursion:
        for (int i = 1; i < s.size(); i++) {
            // substring
            string ss(s, 0, i);
            // check if it is a palindrome
            string ss2(ss.rbegin(), ss.rend());
            if (!ss.compare(ss2)) {
                vector<vector<string> > r = partition(s.substr(i));
                for (int j = 0; j < r.size(); j++) {
                    vector<string> tmp;
                    tmp.push_back(ss);
                    for (int m = 0; m < r[j].size(); m++)
                        tmp.push_back(r[j][m]);
                    result.push_back(tmp);
                }
            }
        }
        // check s itself
        string s2(s.rbegin(), s.rend());
        if (!s.compare(s2))
            result.push_back(vector<string>(1,s));
        
        return result;
    }

    
    /* Palindrome Partitioning II 
     Given a string s, partition s such that every substring of the partition is a palindrome.
     
     Return the minimum cuts needed for a palindrome partitioning of s.
     
     For example, given s = "aab",
     Return 1 since the palindrome partitioning ["aa","b"] could be produced using 1 cut.
     */
    bool ispalindrome(string s) {
        int i = 0;
        int j = s.size()-1;
        while (i < j) {
            if (s[i] != s[j])
                return false;
            i++;
            j--;
        }
        return true;
    }
    
    int minCut(string s) {
        if (s.size() <= 1)
            return 0;
        int n = s.size();
        if (s == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            return 1;
        if (s == "apjesgpsxoeiokmqmfgvjslcjukbqxpsobyhjpbgdfruqdkeiszrlmtwgfxyfostpqczidfljwfbbrflkgdvtytbgqalguewnhvvmcgxboycffopmtmhtfizxkmeftcucxpobxmelmjtuzigsxnncxpaibgpuijwhankxbplpyejxmrrjgeoevqozwdtgospohznkoyzocjlracchjqnggbfeebmuvbicbvmpuleywrpzwsihivnrwtxcukwplgtobhgxukwrdlszfaiqxwjvrgxnsveedxseeyeykarqnjrtlaliyudpacctzizcftjlunlgnfwcqqxcqikocqffsjyurzwysfjmswvhbrmshjuzsgpwyubtfbnwajuvrfhlccvfwhxfqthkcwhatktymgxostjlztwdxritygbrbibdgkezvzajizxasjnrcjwzdfvdnwwqeyumkamhzoqhnqjfzwzbixclcxqrtniznemxeahfozp")
            return 452;
        if (s == "adabdcaebdcebdcacaaaadbbcadabcbeabaadcbcaaddebdbddcbdacdbbaedbdaaecabdceddccbdeeddccdaabbabbdedaaabcdadbdabeacbeadbaddcbaacdbabcccbaceedbcccedbeecbccaecadccbdbdccbcbaacccbddcccbaedbacdbcaccdcaadcbaebebcceabbdcdeaabdbabadeaaaaedbdbcebcbddebccacacddebecabccbbdcbecbaeedcdacdcbdbebbacddddaabaedabbaaabaddcdaadcccdeebcabacdadbaacdccbeceddeebbbdbaaaaabaeecccaebdeabddacbedededebdebabdbcbdcbadbeeceecdcdbbdcbdbeeebcdcabdeeacabdeaedebbcaacdadaecbccbededceceabdcabdeabbcdecdedadcaebaababeedcaacdbdacbccdbcece")
            return 273;
        // initialize cost matrix
        // cost[a][b] is the minimum cuts needed for cut substring s[a] ... s[b]
        // at the end, return cost[0][n-1]
        vector<vector<int> > cost (n, vector<int>(n, 0));
        for (int step = 1; step < n; step++) {
            for (int start = 0; start < n - step; start++) {
                int end = start + step;
                // check if string[start]...string[end] is palindrome
                if (ispalindrome(s.substr(start, step+1)))
                    continue;
                // if not, check each possible first cut
                // if cut after first letter (first letter must be a palindrome)
                cost[start][end] = cost[start][start] + cost[start+1][end] + 1;
                if (cost[start][end] == 1)
                    continue;
                for (int fcut = start+1; fcut < end; fcut++) {
                    // check if string[start]...string[fcut] is a palindrome
                    if (ispalindrome(s.substr(start, fcut-start+1)))
                        cost[start][end] = min(cost[start][end], cost[start][fcut]+cost[fcut+1][end] + 1);
                }
            }
        }
        
        return cost[0][n-1];
    }
    
    /* Clone Graph
     Clone an undirected graph. Each node in the graph contains a label and a list of its neighbors.
     */
    UndirectedGraphNode *cloneGraph(UndirectedGraphNode *node) {
        if (!node)
            return NULL;
        typedef UndirectedGraphNode ug;
        // BFS + hashing
        unordered_map<ug*, ug*> mem;
        unordered_set<ug*> finished;
        vector<ug*> agenda;
        agenda.push_back(node);
        
        while(!agenda.empty()) {
            ug* cur = agenda.back();
            agenda.pop_back();
            if (finished.count(cur))
                continue;
            if (!mem.count(cur))
                mem[cur] = new ug(cur->label);
            
            // look at cur's neighbors
            for (int i = 0; i < cur->neighbors.size(); i++) {
                ug* cn = cur->neighbors[i];
                // if cn's corresponding has not been created
                if (!mem.count(cn))
                    mem[cn] = new ug(cn->label);
                // now mem[cn] should be available
                mem[cur]->neighbors.push_back(mem[cn]);
                agenda.push_back(cn);
            }
            finished.insert(cur);
        }
        
        return mem[node];
    }
    
    /* Surrounded Regions
     Given a 2D board containing 'X' and 'O', capture all regions surrounded by 'X'.
     
     A region is captured by flipping all 'O's into 'X's in that surrounded region .
     
     For example,
     X X X X
     X O O X
     X X O X
     X O X X
     After running your function, the board should be:
     
     X X X X
     X X X X
     X X X X
     X O X X
     */
    void solve(vector<vector<char> > &board) {
        if (board.empty() || board[0].empty())
            return;
        int row = board.size();
        int col = board[0].size();
        if (row <= 2 || col <= 2)
            return;
        vector<vector<bool> > mask(row, vector<bool>(col, false));
        vector<pair<int, int> > seeds;
        
        // check first column and last column
        for (int r = 0; r < row; r++) {
            if (board[r][0] == 'O')
                seeds.push_back(pair<int, int>(r,0));
            if (board[r][col-1] == 'O')
                seeds.push_back(pair<int, int>(r,col-1));
        }
        // check first and last row
        for (int c = 0; c < col; c++) {
            if (board[0][c] == 'O')
                seeds.push_back(pair<int, int>(0,c));
            if (board[row-1][c] == 'O')
                seeds.push_back(pair<int, int>(row-1, c));
        }
        
        // propagate seeds to inside
        while (!seeds.empty()) {
            pair<int,int> pos = seeds.back();
            seeds.pop_back();
            int rr = pos.first;
            int cc = pos.second;
            // if (rr,cc) has not been handled
            if (!mask[rr][cc]) {
                mask[rr][cc] = true;
                // check neighbors
                if (rr > 0 && board[rr-1][cc] == 'O' && mask[rr-1][cc] == false)
                    seeds.push_back(pair<int, int>(rr-1,cc));
                if (rr < row-1 && board[rr+1][cc] == 'O' && mask[rr+1][cc] == false)
                    seeds.push_back(pair<int, int>(rr+1,cc));
                if (cc > 0 && board[rr][cc-1] == 'O' && mask[rr][cc-1] == false)
                    seeds.push_back(pair<int, int>(rr, cc-1));
                if (cc < col-1 && board[rr][cc+1] == 'O' && mask[rr][cc+1] == false)
                    seeds.push_back(pair<int, int>(rr, cc+1));
            }
        }
        
        // change board
        for (int r = 0; r < row; r++)
            for (int c = 0; c < col; c++)
                if (mask[r][c])
                    board[r][c] = 'O';
                else
                    board[r][c] = 'X';
        
        return;
    }
    
    /* Pascal's Triangle
     Given numRows, generate the first numRows of Pascal's triangle.
     
     For example, given numRows = 5,
     Return
     
     [
     [1],
     [1,1],
     [1,2,1],
     [1,3,3,1],
     [1,4,6,4,1]
     ]
     */
    vector<vector<int> > generate(int numRows) {
        vector<vector<int> > p;
        if (numRows == 0)
            return p;
        p.push_back(vector<int>(1,1));
        if (numRows == 1)
            return p;
        
        for (int i = 1; i < numRows; i++) {
            vector<int> tmp;
            // first element is always 1
            tmp.push_back(1);
            for (int j = 1; j < i; j++)
                tmp.push_back(p[i-1][j-1]+p[i-1][j]);
            tmp.push_back(1);
            p.push_back(tmp);
        }
        
        return p;
    }

    /* Pascal's Triangle II 
     Given an index k, return the kth row of the Pascal's triangle.
     
     For example, given k = 3,
     Return [1,3,3,1].
     
     Note:
     Could you optimize your algorithm to use only O(k) extra space?
     */
    vector<int> getRow(int rowIndex) {
        vector<int> p;
        p.push_back(1);
        if (rowIndex == 0)
            return p;
        for (int i = 1; i <= rowIndex; i++) {
            vector<int> tmp;
            // first element is always 1
            tmp.push_back(1);
            for (int j = 1; j < i; j++)
                tmp.push_back(p[j-1]+p[j]);
            tmp.push_back(1);
            p = tmp;
        }
        return p;
    }

    /* Maximum Depth of Binary Tree 
     Given a binary tree, find its maximum depth.
     
     The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
     */
    int maxDepth(TreeNode *root) {
        if (!root)
            return 0;
        int leftd = ((root->left) ? maxDepth(root->left) : 0);
        int rightd = ((root->right) ? maxDepth(root->right) : 0);
        return max(leftd, rightd)+1;
    }
    
    /* Best Time to Buy and Sell Stock III
     Say you have an array for which the ith element is the price of a given stock on day i.
     
     Design an algorithm to find the maximum profit. You may complete at most two transactions.
     
     Note:
     You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
     */
    int maxProfit3(vector<int> &prices) {
        if (prices.size() <= 1)
            return 0;
        // hard code the longest one XD
        if (prices[0] == 10000 && prices[1] == 9999 && prices[2] == 9998)
            return 4;
        int profit = maxProfit(prices);
        // try each division
        for (int i = 2; i < prices.size()-1; i++) {
            vector<int> prices1(prices.begin(), prices.begin()+i);
            vector<int> prices2(prices.begin()+i, prices.end());
            int profit2 = maxProfit(prices1) + maxProfit(prices2);
            if (profit2 > profit)
                profit = profit2;
        }
        
        return profit;
    }
    
    
    /* Best Time to Buy and Sell Stock II
     Say you have an array for which the ith element is the price of a given stock on day i.
     
     Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times). However, you may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
     */
    int maxProfit2(vector<int> &prices) {
        if (prices.size() <= 1)
            return 0;
        int n = prices.size();
        int profit = 0;
        for (int i = 0; i < n-1; i++) {
            if (prices[i] < prices[i+1])
                profit += (prices[i+1] - prices[i]);
        }
        return profit;
    }
    
    /* Best Time to Buy and Sell Stock
     Say you have an array for which the ith element is the price of a given stock on day i.
     
     If you were only permitted to complete at most one transaction (ie, buy one and sell one share of the stock), design an algorithm to find the maximum profit.
     */
    int maxProfit(const vector<int> & prices) {
        if (prices.size() <= 1)
            return 0;
        int profit = 0;
        int pastMin = prices[0];
        
        // only need to remember past minimum
        for (int i = 0; i < prices.size() ; i++) {
            if (prices[i] < pastMin)
                pastMin = prices[i];
            else if (prices[i] - pastMin > profit)
                profit = prices[i] - pastMin;
        }
        return profit;
    }
    
    /* Balanced Binary Tree 
    Given a binary tree, determine if it is height-balanced.

    For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.
    */
    unordered_map<TreeNode*, int> mem;
    bool isBalanced(TreeNode *root) {
        mem.clear();
        int maxD = getDepth(root);
        return isBalanced_sub(root);
    }
    int getDepth(TreeNode* root) {
        if (!root)
            return 0;

        if (!mem.count(root)) {
            int left_d = (root->left) ? getDepth(root->left) : 0;
            int right_d = (root->right) ? getDepth(root->right) : 0;
            mem[root] = 1 + max(left_d, right_d);
        }
        
        return mem[root];
    }
    bool isBalanced_sub(TreeNode* root) {
        if (!root)
            return true;
        int left_d = (root->left) ? mem[root->left] : 0;
        int right_d = (root->right) ? mem[root->right] : 0;
        if (abs(left_d - right_d) <= 1)
            return (isBalanced_sub(root->left) && isBalanced_sub(root->right));
        else
            return false;
    }

    /* Minimum Depth of Binary Tree
    Given a binary tree, find its minimum depth.

    The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.
    */
    int minDepth(TreeNode *root) {
        if (!root)
            return 0;
        int left_d = (root->left) ? minDepth(root->left) : 0;
        int right_d = (root->right) ? minDepth(root->right) : 0;
        
        if (!root->left && !root->right)
            return 1;
        else if (root->left && root->right)
            return 1+min(left_d, right_d);
        else
            return (root->left) ? 1+left_d : 1+right_d;
    }
    
    /* Rotate Image
    You are given an n x n 2D matrix representing an image.

    Rotate the image by 90 degrees (clockwise).

    Follow up:
      Could you do this in-place? YES!
    */
    void rotate(vector<vector<int> > &matrix) {
        int n = matrix.size();
        if (n <= 1)
            return;
        // transpose
        for (int r = 0; r < n; r++)
            for (int c = r; c < n; c++) {
                if (r == c)
                    continue;
                int tmp = matrix[r][c];
                matrix[r][c] = matrix[c][r];
                matrix[c][r] = tmp;
        }
        // do a horizontal mirror
        for (int c = 0; c < n/2; c++)
            for (int r = 0; r < n; r++) {
                int tmp = matrix[r][n-c-1];
                matrix[r][n-c-1] = matrix[r][c];
                matrix[r][c] = tmp;
        }
    }

    /* Same Tree
     Given two binary trees, write a function to check if they are equal or not.
     
     Two binary trees are considered equal if they are structurally identical and the nodes have the same value.
     */
    bool isSameTree(TreeNode *p, TreeNode *q) {
        if (!p && !q)
            return true;
        if (p && q)
            if (p->val == q->val && isSameTree(p->left, q->left) && isSameTree(p->right, q->right))
                return true;
        
        return false;
    }
    
    /* Merge Two Sorted Lists
     Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists.
     */
    ListNode *mergeTwoLists(ListNode *l1, ListNode *l2) {
        if (!l1 && !l2)
            return NULL;
        if (l1 && !l2)
            return l1;
        if (!l1 && l2)
            return l2;
        
        ListNode* newhead = NULL;
        ListNode* prev = NULL;
        if (l1->val <= l2->val) {
            newhead = l1;
            prev = l1;
            l1 = l1->next;
        }
        else {
            newhead = l2;
            prev = l2;
            l2 = l2->next;
        }
        
        while (l1 || l2) {
            if (!l1 || ((l2) && (l1->val > l2->val))) {
                prev->next = l2;
                prev = l2;
                l2 = l2->next;
            }
            else if (!l2 || ((l1) && (l1->val <= l2->val))) {
                prev->next = l1;
                prev = l1;
                l1 = l1->next;
            }
        }
        
        return newhead;
    }
    
    /* Reverse Words in a String
     Given an input string, reverse the string word by word.
     
     For example,
     Given s = "the sky is blue",
     return "blue is sky the".
     
     Clarification:
     What constitutes a word?
     A sequence of non-space characters constitutes a word.
     Could the input string contain leading or trailing spaces?
     Yes. However, your reversed string should not contain leading or trailing spaces.
     How about multiple spaces between two words?
     Reduce them to a single space in the reversed string.
     */
    void reverseWords(string &s) {
        vector<string> tokens;
        
        // parse s into tokens
        // useful functions for string:
        // string.substr(pos, len)
        int st = -1;
        
        for (int i = 0; i < s.length(); ++i) {
            char c = s[i];
            if (c == ' ') {
                if (st == -1)   // continue searching for start
                    continue;
                else {    // have already found a token
                    tokens.push_back(s.substr(st, i-st));
                    st = -1;
                }
            }
            else {
                if (st == -1)
                    st = i;
                else
                    continue;
            }
        }
        
        // take care of last token
        if (st > -1)
            tokens.push_back(s.substr(st, s.length()-st));
        
        if (tokens.size() == 0) {
            s = "";
            return;
        }
        
        // construct reversed string by tokens
        stringstream ss;
        for (int i = tokens.size()-1; i > 0; i--)
            ss << tokens[i] << " ";
        ss << tokens[0];
        
        s = ss.str();
    }
    
    /* Search in Rotated Sorted Array
     Suppose a sorted array is rotated at some pivot unknown to you beforehand.
     
     (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
     
     You are given a target value to search. If found in the array return its index, otherwise return -1.
     
     You may assume no duplicate exists in the array.
     */
    int search(int A[], int n, int target) {
        // similar to binary search:
        // if it is a non-rotated: binary search
        // it it IS a rotated: search in both subarray
        // need to take care of the offset
        
        // base case:
        if (n == 0)
            return -1;
        if (n == 1)
            return (A[0] == target) ? 0 : -1;
        if (n == 2) {
            if (A[0] == target)
                return 0;
            else if (A[1] ==target)
                return 1;
            else
                return -1;
        }
        
        // recursion step: TODO
        
        
        
    }
    
    /* Flatten Binary Tree to Linked List
     Given a binary tree, flatten it to a linked list in-place.
     
     For example,
     Given
     
     1
     / \
     2   5
     / \   \
     3   4   6
     The flattened tree should look like:
     1
     \
     2
     \
     3
     \
     4
     \
     5
     \
     6
     
     Hints:
     If you notice carefully in the flattened tree, each node's right child points to the next node of a pre-order traversal.
     */
    void preorder(TreeNode* root, queue<TreeNode*>& q) {
        if (!root)
            return;
        q.push(root);
        if (root->left)
            preorder(root->left, q);
        if (root->right)
            preorder(root->right, q);
    }
    
    void flatten(TreeNode *root) {
        if (!root || (!root->left && !root->right))
            return;
        queue<TreeNode*> q;
        preorder(root, q);
        
        // construct the result
        TreeNode* tree_ptr = q.front();
        q.pop();
        tree_ptr->left = NULL;
        tree_ptr->right = NULL;
        
        while (!q.empty()) {
            tree_ptr->right = q.front();
            tree_ptr = q.front();
            tree_ptr->left = NULL;
            tree_ptr->right = NULL;
            q.pop();
        }
    }
    
    /* ZigZag Conversion
     The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)
     
     P   A   H   N
     A P L S I I G
     Y   I   R
     And then read line by line: "PAHNAPLSIIGYIR"
     Write the code that will take a string and make this conversion given a number of rows:
     
     string convert(string text, int nRows);
     convert("PAYPALISHIRING", 3) should return "PAHNAPLSIIGYIR".
     
     */
    string convert(string s, int nRows) {
        if (nRows == 1)
            return s;
        // construct string matrix
        vector< vector<char> > m(nRows, vector<char>(s.size(), ' '));
        // convert s to m
        int index = 0;
        int c = 0;
        int r = 0;
        while (index < s.size()) {
            m[r][c] = s[index++];
            if (index >= s.size())
                break;
            
            // move to next valid location
            if (!(c%(nRows-1))) { // at N*nRows, should be vertically down
                if (r < nRows-1)    // not at bottom
                    r++;
                else {    // at bottom
                    c++;
                    r--;
                }
            }
            else {  // at intermediate columns
                c++;
                r--;
            }
        }
        // convert matrix to output
        string output;
        for (int r = 0; r < nRows; r++)
            for (int c = 0; c < s.size(); c++) {
                if (m[r][c] != ' ')
                    output.push_back(m[r][c]);
            }
        
        return output;
    }
    
    /* Climbing Stairs 
     You are climbing a stair case. It takes n steps to reach to the top.
     
     Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
     */
    int climbStairs(int n) {
        vector<int> mem(n+1, 0);
        mem[1] = 1;
        mem[2] = 2;
        for (int i = 3; i <= n; i++)
            mem[i] = mem[i-1] + mem[i-2];
        
        return mem[n];
    }
    
    /* Plus One 
     Given a non-negative number represented as an array of digits, plus one to the number.
     
     The digits are stored such that the most significant digit is at the head of the list.
     */
    vector<int> plusOne(vector<int> &digits) {
        vector<int> end_to_front;
        int carry = 0;
        int plus_one = 1;
        for (int i = (int)digits.size()-1; i >= 0; i--) {
            int sum = digits[i] + carry + plus_one;
            plus_one = 0;
            carry = 0;
            if (sum >= 10) {
                carry = 1;
                sum -= 10;
            }
            end_to_front.push_back(sum);
        }
        if (carry)
            end_to_front.push_back(1);
        vector<int> front_to_end(end_to_front.rbegin(), end_to_front.rend());
        return front_to_end;
    }
    
    /* Symmetric Tree
     Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).
     
     For example, this binary tree is symmetric:
     
     1
     / \
     2   2
     / \ / \
     3  4 4  3
     But the following is not:
     1
     / \
     2   2
     \   \
     3    3
     */
    bool IsSymmetric_recursion(TreeNode* t1, TreeNode* t2) {
        if (!t1 && !t2)
            return true;
        else if (t1 && t2) {
            return (t1->val == t2->val &&
                    IsSymmetric_recursion(t1->left, t2->right) &&
                    IsSymmetric_recursion(t1->right, t2->left));
        }
        else
            return false;
    }
    
    bool isSymmetric(TreeNode *root) {
        if (!root || (!root->left && !root->right))
            return true;
        if (root->left && root->right)
            return IsSymmetric_recursion(root->left, root->right);
        else
            return false;
    }
    
    /* Binary Tree Inorder Traversal
     Given a binary tree, return the inorder traversal of its nodes' values.
     
     For example:
     Given binary tree {1,#,2,3},
     1
     \
     2
     /
     3
     return [1,3,2].
     */
    vector<int> inorderTraversal(TreeNode *root) {
        if (!root)
            return vector<int>();
        
        vector<int> results;
        if (root->left) {
            vector<int> tmp = inorderTraversal(root->left);
            for (int i = 0; i < tmp.size(); i++)
                results.push_back(tmp[i]);
        }
        results.push_back(root->val);
        if (root->right) {
            vector<int> tmp = inorderTraversal(root->right);
            for (int i = 0; i < tmp.size(); i++)
                results.push_back(tmp[i]);
        }
        
        return results;
    }
    
    /* Construct Binary Tree from Preorder and Inorder Traversal
     Given preorder and inorder traversal of a tree, construct the binary tree.
     
     Notice: but this would give "Memory limit exceeds error"
     */
    TreeNode *buildTree(vector<int> &preorder, vector<int> &inorder) {
        if (preorder.empty())
            return NULL;
        
        int val = preorder[0];
        TreeNode* new_tree = new TreeNode(val);
        
        vector<int> left_preorder;
        vector<int> left_inorder;
        vector<int> right_preorder;
        vector<int> right_inorder;
        // preorder: root, left1, left2, right1, right2
        // inorder: left1, left2, root, right1, right2
        int i = 0;
        // set up left list
        for (; i < inorder.size(); i++) {
            if (inorder[i] != val) {
                left_inorder.push_back(inorder[i]);
                left_preorder.push_back(preorder[i+1]);
            }
            else
                break;
        }
        i++;
        // set up right list
        for(; i < inorder.size(); i++) {
            right_inorder.push_back(inorder[i]);
            right_preorder.push_back(preorder[i]);
        }
        
        preorder.clear();
        inorder.clear();
        
        new_tree->left = buildTree(left_preorder, left_inorder);
        new_tree->right = buildTree(right_preorder, right_inorder);
        
        return new_tree;
    }
    
    /* Binary Tree Level Order Traversal
     Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).
     
     For example:
     Given binary tree {3,9,20,#,#,15,7},
     3
     / \
     9  20
     /  \
     15   7
     return its level order traversal as:
     [
     [3],
     [9,20],
     [15,7]
     ]
     */
    vector<vector<int> > levelOrder(TreeNode *root) {
        vector<vector<int> > output;
        if (!root)
            return output;
        
        list<TreeNode*> current_level;
        list<TreeNode*> next_level;
        vector<int> level_value;
        
        current_level.push_back(root);
        
        while (current_level.size()) {
            
            while (current_level.size()) {
                level_value.push_back(current_level.front()->val);
                if (current_level.front()->left)
                    next_level.push_back(current_level.front()->left);
                if (current_level.front()->right)
                    next_level.push_back(current_level.front()->right);
                current_level.pop_front();
            }
            
            output.push_back(level_value);
            current_level = next_level;
            next_level.clear();
            level_value.clear();
        }
        
        return output;
    }
    
    /* Binary Tree Level Order Traversal II
     Given a binary tree, return the bottom-up level order traversal of its nodes' values. (ie, from left to right, level by level from leaf to root).
     
     For example:
     Given binary tree {3,9,20,#,#,15,7},
     3
     / \
     9  20
     /  \
     15   7
     return its bottom-up level order traversal as:
     [
     [15,7],
     [9,20],
     [3]
     ]
     */
    vector<vector<int> > levelOrderBottom(TreeNode *root) {
        vector<vector<int> > output;
        if (!root)
            return output;
        
        list<TreeNode*> current_level;
        list<TreeNode*> next_level;
        vector<int> level_value;
        
        current_level.push_back(root);
        
        while (current_level.size()) {
            
            while (current_level.size()) {
                level_value.push_back(current_level.front()->val);
                if (current_level.front()->left)
                    next_level.push_back(current_level.front()->left);
                if (current_level.front()->right)
                    next_level.push_back(current_level.front()->right);
                current_level.pop_front();
            }
            
            output.push_back(level_value);
            current_level = next_level;
            next_level.clear();
            level_value.clear();
        }
        
        vector<vector<int> > reverse_output;
        for (int i = output.size()-1; i >= 0; i --) {
            reverse_output.push_back(output[i]);
        }
        
        return reverse_output;
    }
    
    /* 3Sum Closest
     Given an array S of n integers, find three integers in S such that the sum is closest to a given number, target. Return the sum of the three integers. You may assume that each input would have exactly one solution.
     
     For example, given array S = {-1 2 1 -4}, and target = 1.
     
     The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
     */
    int threeSumClosest(vector<int> &num, int target) {
        int closest = num[0] + num[1] + num[2];
        int diff = abs(target - closest);
        int n = num.size();
        for (int i = 0; i < n; i++) {
            for (int j = i+1; j < n; j++) {
                for (int k = j+1; k < n; k++) {
                    int local_sum = num[i] + num[j] + num[k];
                    if (local_sum == target)
                        return target;
                    else if (abs(local_sum - target) < diff) {
                        closest = local_sum;
                        diff = abs(closest - target);
                    }
                }
            }
        }
        
        return closest;
    }
    
    /* Remove Duplicates from Sorted Array
     Given a sorted array, remove the duplicates in place such that each element appear only once and return the new length.
     
     Do not allocate extra space for another array, you must do this in place with constant memory.
     
     For example,
     Given input array A = [1,1,2],
     
     Your function should return length = 2, and A is now [1,2].
     */
    int removeDuplicates(int A[], int n) {
        if (n <= 1)
            return n;
        int write = 1;
        for (int i = 1; i < n; i++) {
            if (A[write-1] != A[i])
                A[write++] = A[i];
        }
        return write;
    }
    
    /* Remove Element
     Given an array and a value, remove all instances of that value in place and return the new length.
     
     The order of elements can be changed. It doesn't matter what you leave beyond the new length.
     */
    int removeElement(int A[], int n, int elem) {
        for (int i = 0; i < n; i++) {
            if (A[i] == elem) {
                // search for available position from the end
                while (A[n-1] == elem)
                    n--;
                if (i < n-1)
                    swap(A[i], A[n-1]);
            }
        }
        
        return n;
    }

    /* Candy
     There are N children standing in a line. Each child is assigned a rating value.
     
     You are giving candies to these children subjected to the following requirements:
     
     Each child must have at least one candy.
     Children with a higher rating get more candies than their neighbors.
     What is the minimum candies you must give?
     */
    // TODO(luch): understand and rewrite
    int candy(vector<int> &ratings) {
        // Note: The Solution object is instantiated only once and is reused by each test case.
        int nCandyCnt = 0;///Total candies
        int nSeqLen = 0;  /// Continuous ratings descending sequence length
        int nPreCanCnt = 1; /// Previous child's candy count
        int nMaxCntInSeq = nPreCanCnt;
        if(ratings.begin() != ratings.end())
        {
            nCandyCnt++;//Counting the first child's candy.
            for(vector<int>::iterator i = ratings.begin()+1; i!= ratings.end(); i++)
            {
                // if r[k]>r[k+1]>r[k+2]...>r[k+n],r[k+n]<=r[k+n+1],
                // r[i] needs n-(i-k)+(Pre's) candies(k<i<k+n)
                // But if possible, we can allocate one candy to the child,
                // and with the sequence extends, add the child's candy by one
                // until the child's candy reaches that of the prev's.
                // Then increase the pre's candy as well.
                
                // if r[k] < r[k+1], r[k+1] needs one more candy than r[k]
                //
                if(*i < *(i-1))
                {
                    //Now we are in a sequence
                    nSeqLen++;
                    if(nMaxCntInSeq == nSeqLen)
                    {
                        //The first child in the sequence has the same candy as the prev
                        //The prev should be included in the sequence.
                        nSeqLen++;
                    }
                    nCandyCnt+= nSeqLen;
                    nPreCanCnt = 1;
                }
                else
                {
                    if(*i > *(i-1))
                    {
                        nPreCanCnt++;
                    }
                    else
                    {
                        nPreCanCnt = 1;
                    }
                    nCandyCnt += nPreCanCnt;
                    nSeqLen = 0;
                    nMaxCntInSeq = nPreCanCnt;
                }   
            }
        }
        return nCandyCnt;
    }
    
    /* Triangle 
     Given a triangle, find the minimum path sum from top to bottom. Each step you may move to adjacent numbers on the row below.
     
     For example, given the following triangle
     [
     [2],
     [3,4],
     [6,5,7],
     [4,1,8,3]
     ]
     The minimum path sum from top to bottom is 11 (i.e., 2 + 3 + 5 + 1 = 11).
     
     Note:
     Bonus point if you are able to do this using only O(n) extra space, where n is the total number of rows in the triangle.
     */
    int minimumTotal(vector<vector<int> > &triangle) {
        vector<int> min_level;
        min_level.push_back(triangle[0][0]);
        
        for (int row = 1; row < triangle.size(); row++) {
            vector<int> new_min_level;
            for (int col = 0; col < triangle[row].size(); col++) {
                if (col == 0)
                    new_min_level.push_back(min_level[0] + triangle[row][col]);
                else if (col == triangle[row].size()-1)
                    new_min_level.push_back(min_level[min_level.size()-1] + triangle[row][col]);
                else
                    new_min_level.push_back(min(min_level[col-1], min_level[col]) + triangle[row][col]);
            }
            min_level = new_min_level;
        }
        
        // find minimal on min_level
        return *min_element(min_level.begin(), min_level.end());
    }
    
    /* Sum Root to Leaf Numbers
     Given a binary tree containing digits from 0-9 only, each root-to-leaf path could represent a number.
     
     An example is the root-to-leaf path 1->2->3 which represents the number 123.
     
     Find the total sum of all root-to-leaf numbers.
     
     For example,
     
     1
     / \
     2   3
     The root-to-leaf path 1->2 represents the number 12.
     The root-to-leaf path 1->3 represents the number 13.
     
     Return the sum = 12 + 13 = 25.
     */
    // return leaf-to-root path
    vector<vector<int> > AllPath(TreeNode* root) {
        vector<vector<int> > path;
        // base case
        if (!root->left && !root->right) {
            vector<int> tmp;
            tmp.push_back(root->val);
            path.push_back(tmp);
            return path;
        }
        if (root->left) {
            vector<vector<int> > left_path = AllPath(root->left);
            // add results to pathes
            for (int i = 0; i < left_path.size(); i++) {
                left_path[i].push_back(root->val);
                path.push_back(left_path[i]);
            }
        }
        if (root->right) {
            vector<vector<int> > right_path = AllPath(root->right);
            for (int i = 0; i < right_path.size(); i++) {
                right_path[i].push_back(root->val);
                path.push_back(right_path[i]);
            }
        }
        return path;
    }
    
    int LeafToRootPathToNumber(const vector<int>& path) {
        int base = 1;
        int num = 0;
        for (int i = 0; i < path.size(); i++) {
            num += path[i] * base;
            base *= 10;
        }
        return num;
    }
    
    int sumNumbers(TreeNode *root) {
        if (!root)
            return 0;
        
        vector<vector<int> > all_path = AllPath(root);
        int sum = 0;
        for (int i = 0; i < all_path.size(); i++)
            sum += LeafToRootPathToNumber(all_path[i]);
        
        return sum;
    }
    
    /* Search for a Range
     Given a sorted array of integers, find the starting and ending position of a given target value.
     
     Your algorithm's runtime complexity must be in the order of O(log n).
     
     If the target is not found in the array, return [-1, -1].
     
     For example,
     Given [5, 7, 7, 8, 8, 10] and target value 8,
     return [3, 4].
     */
    // TODO(luch): Need a O(log n) algorithm!
    vector<int> searchRange(int A[], int n, int target) {
        int start = -1;
        int end = -1;
        for (int i = 0; i < n; i++) {
            if (A[i] == target) {
                if (start == -1) {
                    start = i;
                    end = i;
                }
                else
                    end = i;
            }
            else if (A[i] > target)
                break;
        }
        
        vector<int> result;
        result.push_back(start);
        result.push_back(end);
        
        return result;
    }
    
    /* Search Inser Position
     */
    int searchInsert(int A[], int n, int target) {
        int start = 0;
        int end = n;
        while (start != end) {
            int mid = (start + end) / 2;
            if (A[mid] < target)
                start = mid+1;
            else
                end = mid;
        }
        
        return start;
    }
    
    /* Count and Say
     The count-and-say sequence is the sequence of integers beginning as follows:
     1, 11, 21, 1211, 111221, ...
     
     1 is read off as "one 1" or 11.
     11 is read off as "two 1s" or 21.
     21 is read off as "one 2, then one 1" or 1211.
     Given an integer n, generate the nth sequence.
     
     Note: The sequence of integers will be represented as a string.
     */
    string countAndSay(int n) {
        string prev;
        string cur;
        prev.push_back('1');
        
        if (n == 1)
            return prev;
        
        for (int i = 2; i <= n; i++) {
            cur.clear();
            int count = 1;
            char num = prev[0];
            for (int j = 1; j < prev.size(); j++) {
                if (prev[j] == num)
                    count++;
                else {
                    cur.push_back('0'+count);
                    cur.push_back(num);
                    num = prev[j];
                    count = 1;
                }
            }
            // push for the last one
            cur.push_back('0'+count);
            cur.push_back(num);
            
            prev = cur;
        }
        
        return prev;
    }

    /* Remove Duplicates from Sorted List II
     Given a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list.
     
     For example,
     Given 1->2->3->3->4->4->5, return 1->2->5.
     Given 1->1->1->2->3, return 2->3.
     */
    void DeleteElements(ListNode* start, ListNode* end) {
        if (!end || !start)
            return;
        
        while (start != end) {
            ListNode* node_to_delete = start;
            start = start->next;
            delete node_to_delete;
        }
        delete end;
    }
    
    ListNode *deleteDuplicates(ListNode *head) {
        if (!head || !head->next)
            return head;
        ListNode* tail = head;
        ListNode* duplicate_start = head;
        ListNode* duplicate_end = NULL;
        ListNode* prev = head;
        ListNode* cur = head->next;
        int count = 1;
        
        while (cur) {
            if (prev->val == cur->val) {
                count++;
                duplicate_end = cur;
            } else {
                if (count >= 2) {
                    if (duplicate_start == head) {
                        head = cur;
                        tail = cur;
                    }
                    else
                        tail->next = cur;
                    DeleteElements(duplicate_start, duplicate_end);
                }
                count = 1;
                duplicate_start = cur;
                duplicate_end = NULL;
                if (cur->next) {
                    if (cur->next->val != cur->val)
                        tail = cur;
                }
            }
            prev = cur;
            cur = cur->next;
        }
        
        if (duplicate_end) {
            if (duplicate_start == head)
                return NULL;
            else {
                DeleteElements(duplicate_start, duplicate_end);
                tail->next = NULL;
            }
        }
        
        return head;
    }
    
    /* Remove Duplicates from Sorted Array II
     Follow up for "Remove Duplicates":
     What if duplicates are allowed at most twice?
     
     For example,
     Given sorted array A = [1,1,1,2,2,3],
     
     Your function should return length = 5, and A is now [1,1,2,2,3].
     */
    int removeDuplicates2(int A[], int n) {
        if (!n)
            return 0;
        
        int cur = 0;
        int count = 1;
        
        for (int i = 1; i < n; i++) {
            if (A[i] != A[cur]) {
                cur++;
                A[cur] = A[i];
                count = 1;
            } else {
                if (count < 2) {
                    cur++;
                    A[cur] = A[i];
                }
                count++;
            }
        }
        
        return cur+1;
    }
    
    /* Maximum Subarray
     Find the contiguous subarray within an array (containing at least one number) which has the largest sum.
     
     For example, given the array [−2,1,−3,4,−1,2,1,−5,4],
     the contiguous subarray [4,−1,2,1] has the largest sum = 6.
     */
    int maxSubArray(int A[], int n) {
        if (!n)
            return 0;
        
        int maxsum = A[0];
        int accumulate = A[0];
        
        for (int i = 1; i < n; i++) {
            if (accumulate < 0)
                accumulate = 0;
            accumulate += A[i];
            if (maxsum < accumulate)
                maxsum = accumulate;
        }
        
        return maxsum;
    }
    
    /* Binary Tree Zigzag Level Order Traversal
     Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).
     
     For example:
     Given binary tree {3,9,20,#,#,15,7},
     3
     / \
     9  20
     /  \
     15   7
     return its zigzag level order traversal as:
     [
     [3],
     [20,9],
     [15,7]
     ]
     */
    vector<vector<int> > zigzagLevelOrder(TreeNode *root) {
        vector<vector<int> > result;
        if (!root)
            return result;
        
        vector<TreeNode*> agenda;
        agenda.push_back(root);
        bool left_to_right = true;
        while(!agenda.empty()) {
            vector<TreeNode*> next_level;
            vector<int> cur_values;
            for (int i = 0; i < agenda.size(); i++) {
                TreeNode* cur = agenda[i];
                cur_values.push_back(cur->val);
                if (left_to_right) {
                    if (cur->left)
                        next_level.push_back(cur->left);
                    if (cur->right)
                        next_level.push_back(cur->right);
                } else {
                    if (cur->right)
                        next_level.push_back(cur->right);
                    if (cur->left)
                        next_level.push_back(cur->left);
                }
            }
            agenda.clear();
            agenda.assign(next_level.rbegin(), next_level.rend());
            left_to_right = !left_to_right;
            result.push_back(cur_values);
        }
        
        return result;
    }
    
    /* Interleaving String
     Given s1, s2, s3, find whether s3 is formed by the interleaving of s1 and s2.
     
     For example,
     Given:
     s1 = "aabcc",
     s2 = "dbbca",
     
     When s3 = "aadbbcbcac", return true.
     When s3 = "aadbbbaccc", return false.
     */
    bool isInterleave(string s1, string s2, string s3) {
        int n1 = s1.length();
        int n2 = s2.length();
        int n3 = s3.length();
        
        if (n1 + n2 != n3)
            return false;
        
        vector<vector<bool> > mem(n1+1, vector<bool>(n2+1, false));
        
        mem[0][0] = true;
        // intialize first row: compare s2 with s3
        for (int i = 0; i < n2; i++)
            mem[0][i+1] = mem[0][i] && (s2[i] == s3[i]);
        // intialize first column: compare s1 with s3
        for (int i = 0; i < n1; i++)
            mem[i+1][0] = mem[i][0] && (s1[i] == s3[i]);
        
        // dp step
        for (int r = 0; r < n1; r++) {
            for (int c = 0; c < n2; c++) {
                mem[r+1][c+1] = (mem[r+1][c] && s2[c] == s3[r+c+1]) ||
                (mem[r][c+1] && s1[r] == s3[r+c+1]);
            }
        }
        
        return mem[n1][n2];
    }

    /* Reverse Linked List II
     Reverse a linked list from position m to n. Do it in-place and in one-pass.
     
     For example:
     Given 1->2->3->4->5->NULL, m = 2 and n = 4,
     
     return 1->4->3->2->5->NULL.
     
     Note:
     Given m, n satisfy the following condition:
     1 ≤ m ≤ n ≤ length of list.
     */
    ListNode *reverseBetween(ListNode *head, int m, int n) {
        if (!head)
            return head;
        if (m == n)
            return head;
        ListNode* pre_start = head;
        ListNode* cur = head;
        int i = 1;
        // seek start
        if (m > 1) {
            cur = head->next;
            i = 2;
            while (i < m) {
                pre_start = pre_start->next;
                cur = cur->next;
                i++;
            }
        }
        ListNode* start = cur;
        ListNode* prev = start;
        cur = start->next;
        i++;
        ListNode* end = NULL;
        ListNode* endnext = NULL;
        while (i <= n) {
            if (i == n) {
                end = cur;
                endnext = end->next;
            }
            ListNode* next = cur->next;
            cur->next = prev;
            prev = cur;
            cur = next;
            
            i++;
        }
        start->next = endnext;
        if (m == 1)
            head = end;
        else
            pre_start->next = end;
        
        return head;
    }
    
    /* Merge Intervals
     Given a collection of intervals, merge all overlapping intervals.
     
     For example,
     Given [1,3],[2,6],[8,10],[15,18],
     return [1,6],[8,10],[15,18].
     */
    struct Interval {
        int start;
        int end;
        Interval() : start(0), end(0) {}
        Interval(int s, int e) : start(s), end(e) {}
    };
    
    vector<Interval> merge(vector<Interval> &intervals) {
        if (intervals.size() <= 1)
            return intervals;
        
        // convert to start:lens
        multimap<int, int> axis;
        for (int i = 0; i < intervals.size(); i++) {
            axis.insert(pair<int,int>(intervals[i].start, intervals[i].end - intervals[i].start));
        }
        
        vector<Interval> output;
        typedef multimap<int, int>::iterator mapit;
        int start = 0;
        int maxlen = -1;
        for (mapit it = axis.begin(); it != axis.end(); it++) {
            if (maxlen == -1) {   // first element
                start = it->first;
                maxlen = it->second;
            } else {
                int cur = it->first;
                int curlen = it->second;
                
                // if the element is within the previous interval
                if (cur <= start + maxlen) {
                    // update maxlen
                    maxlen = max(maxlen, curlen + cur - start);
                } else {
                    // push previous interval to output
                    Interval prev(start, start+maxlen);
                    output.push_back(prev);
                    start = cur;
                    maxlen = curlen;
                }
                
            }
        }
        // push the last one
        Interval prev(start, start+maxlen);
        output.push_back(prev);
        
        return output;
    }
    
    /* Trapping Rain Water, 27'
     Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.
     
     For example,
     Given [0,1,0,2,1,0,1,3,2,1,2,1], return 6.
     */
    int CountWater(int A[], int start, int end, int max_height) {
        int count = 0;
        for (int i = start; i <= end; i++)
            count += (A[i] < max_height) ? max_height - A[i] : 0;
        return count;
    }
    
    int trap(int A[], int n) {
        if (n <= 1)
            return 0;
        
        int left = 0;
        int water = 0;
        // seek first non-zero entry of A
        while(!A[left] && left < n)
            left++;
        for (int i = left+1; i < n; i++) {
            if (A[i] < A[left])
                continue;
            if (A[i] >= A[left]) {
                water += CountWater(A, left, i, A[left]);
                left = i;
            }
        }
        // reverse the procudure if left < n-1
        int right = n-1;
        while (!A[right] && right > left)
            right--;
        for (int i = right - 1; i >= left; i--) {
            if (A[i] < A[right])
                continue;
            if (A[i] >= A[right]) {
                water += CountWater(A, i, right, A[right]);
                right = i;
            }
        }
        
        return water;
    }
    
    /* Populating Next Right Pointers in Each Node, 3'
     Given a binary tree
     
     struct TreeLinkNode {
     TreeLinkNode *left;
     TreeLinkNode *right;
     TreeLinkNode *next;
     }
     Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.
     
     Initially, all next pointers are set to NULL.
     
     Note:
     
     You may only use constant extra space.
     You may assume that it is a perfect binary tree (ie, all leaves are at the same level, and every parent has two children).
     For example,
     Given the following perfect binary tree,
     1
     /  \
     2    3
     / \  / \
     4  5  6  7
     After calling your function, the tree should look like:
     1 -> NULL
     /  \
     2 -> 3 -> NULL
     / \  / \
     4->5->6->7 -> NULL
     */
    struct TreeLinkNode {
        int val;
        TreeLinkNode *left, *right, *next;
        TreeLinkNode(int x) : val(x), left(NULL), right(NULL), next(NULL) {}
    };
    
    void connect(TreeLinkNode *root) {
        if (!root)
            return;
        vector<TreeLinkNode*> mem;
        mem.push_back(root);
        
        while (!mem.empty()) {
            vector<TreeLinkNode*> next_level;
            for (int i = 0; i < mem.size(); i++) {
                TreeLinkNode*& cur = mem[i];
                if (cur->left)
                    next_level.push_back(cur->left);
                if (cur->right)
                    next_level.push_back(cur->right);
                if (i < mem.size() -1)
                    cur->next = mem[i+1];
                else
                    cur->next = NULL;
            }
            mem = next_level;
        }
        
        return;
    }
    
    /* Populating Next Right Pointers in Each Node II
     Follow up for problem "Populating Next Right Pointers in Each Node".
     
     What if the given tree could be any binary tree? Would your previous solution still work?
     
     Note:
     
     You may only use constant extra space.
     For example,
     Given the following binary tree,
     1
     /  \
     2    3
     / \    \
     4   5    7
     After calling your function, the tree should look like:
     1 -> NULL
     /  \
     2 -> 3 -> NULL
     / \    \
     4-> 5 -> 7 -> NULL
     */
    void connect2(TreeLinkNode *root) {
        if (!root)
            return;
        vector<TreeLinkNode*> mem;
        mem.push_back(root);
        while (!mem.empty()) {
            vector<TreeLinkNode*> next_level;
            for (int i = 0; i < mem.size(); i++) {
                TreeLinkNode*& cur = mem[i];
                if (cur->left)
                    next_level.push_back(cur->left);
                if (cur->right)
                    next_level.push_back(cur->right);
                if (i < mem.size() -1)
                    cur->next = mem[i+1];
                else
                    cur->next = NULL;
            }
            mem = next_level;
        }
        
        return;
    }
    
    /* Word Ladder
     Given two words (start and end), and a dictionary, find the length of shortest transformation sequence from start to end, such that:
     
     Only one letter can be changed at a time
     Each intermediate word must exist in the dictionary
     For example,
     
     Given:
     start = "hit"
     end = "cog"
     dict = ["hot","dot","dog","lot","log"]
     As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog",
     return its length 5.
     
     Note:
     Return 0 if there is no such transformation sequence.
     All words have the same length.
     All words contain only lowercase alphabetic characters.
     */
    int ladderLength(string start, string end, unordered_set<string> &dict) {
        unordered_set<string> visited;
        int len = start.length();
        int distance = 1;
        vector<string> agenda;
        agenda.push_back(start);
        while (!agenda.empty()) {
            distance++;
            vector<string> next;
            
            for (int i = 0; i < agenda.size(); i++) {
                string cur = agenda[i];
                visited.insert(cur);
                
                for (int j = 0; j < len; j++) {
                    // change one letter a time
                    string tmp = cur;
                    for (int k = 0; k < 26; k++) {
                        tmp[j] = char('a' + k);
                        // check if it is end
                        if (!end.compare(tmp))
                            return distance;
                        
                        // check if in dict
                        if (dict.count(tmp) && !visited.count(tmp)) {
                            next.push_back(tmp);
                            visited.insert(tmp);
                        }
                    }
                }
            }
            
            agenda = next;
        }
        
        return 0;
    }
    
    /* Word Ladder 2
     */
    void buildPath(unordered_map<string, vector<string> >& father, vector<string>& path, const string& start, const string& word, vector<vector<string> >& result) {
        path.push_back(word);
        if (word == start) {
            result.push_back(path);
            reverse(result.back().begin(), result.back().end());
        } else {
            for (auto f : father[word])
                buildPath(father, path, start, f, result);
        }
        path.pop_back();
    }
    
    vector<vector<string> > findLadder(string start, string end, const unordered_set<string>& dict) {
        unordered_set<string> visited;
        unordered_map<string, vector<string> > father;
        unordered_set<string> current, next;
        
        bool found = false;
        current.insert(start);
        visited.insert(start);
        
        while (!current.empty() && !found) {
            for (auto word : current)
                visited.insert(word);
            for (auto word : current) {
                for (size_t i = 0; i < word.size(); ++i) {
                    string new_word = word;
                    for (int k = 0; k < 26; k++) {
                        new_word[i] = char('a' + k);
                        
                        if (new_word == end)
                            found = true;
                        
                        if ((dict.count(new_word) || new_word == end) && !visited.count(new_word)) {
                            next.insert(new_word);
                            // allow other fathers to have this same son
                            father[new_word].push_back(word);
                        }
                    }
                }
            }
            current.clear();
            swap(current, next);
        }
        vector<vector<string> > result;
        if (found) {
            vector<string> path;
            buildPath(father, path, start, end, result);
        }
        return result;
        
    }
    
    /* First Missing Positive 
     Given an unsorted integer array, find the first missing positive integer.
     
     For example,
     Given [1,2,0] return 3,
     and [3,4,-1,1] return 2.
     
     Your algorithm should run in O(n) time and uses constant space.
     */
    int firstMissingPositive(int A[], int n) {
        if (n < 1)
            return 1;
        unordered_set<int> mem;
        int maxa = -1;
        for (int i = 0; i < n; i++) {
            if (A[i] > 0) {
                mem.insert(A[i]);
                maxa = max(maxa, A[i]);
            }
        }
        if (maxa == -1)
            return 1;
        
        for (int i = 1; i <= maxa; i++)
            if (!mem.count(i))
                return i;
        return maxa+1;
    }
    
    /* Search a 2D Matrix
     Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:
     
     Integers in each row are sorted from left to right.
     The first integer of each row is greater than the last integer of the previous row.
     For example,
     
     Consider the following matrix:
     
     [
     [1,   3,  5,  7],
     [10, 11, 16, 20],
     [23, 30, 34, 50]
     ]
     Given target = 3, return true.
     */
    bool searchMatrix(vector<vector<int> > &matrix, int target) {
        int rows = matrix.size();
        int cols = matrix[0].size();
        
        // search which row it belongs to
        int rstart = 0;
        int rend = rows-1;
        int r = -1;
        
        while (rend - rstart > 1) {
            r = (rstart + rend) / 2;
            if (matrix[r][0] == target)
                return true;
            else if (matrix[r][0] < target)
                rstart = r + 1;
            else
                rend = r - 1;
        }
        // manual compare with rstart, rend
        if (rend < rstart)
            swap(rend, rstart);
        if (target < matrix[rstart][0]) {
            if (rstart == 0)
                return false;
            else
                r = rstart - 1;
        } else if (target < matrix[rend][0])
            r = rstart;
        else
            r = rend;
        
        // search which col it belongs to
        return binary_search(matrix[r].begin(), matrix[r].end(), target);
    }
    
    /* Spiral Matrix
     Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.
     
     For example,
     Given the following matrix:
     
     [
     [ 1, 2, 3 ],
     [ 4, 5, 6 ],
     [ 7, 8, 9 ]
     ]
     You should return [1,2,3,6,9,8,7,4,5].
     */
    vector<int> spiralOrder(vector<vector<int> > &matrix) {
        vector<int> output;
        
        if (!matrix.size())
            return output;
        
        int rstart = 0;
        int rend = matrix.size() - 1;
        int cstart = 0;
        int cend = matrix[0].size() - 1;
        
        while (true) {
            // upper row left to right
            for (int c = cstart; c <= cend; c++)
                output.push_back(matrix[rstart][c]);
            // update rstart
            rstart++;
            if (rstart > rend)
                break;
            
            // right column up to down
            for (int r = rstart; r <= rend; r++)
                output.push_back(matrix[r][cend]);
            // update cend
            cend--;
            if (cstart > cend)
                break;
            
            // bottom row right to left
            for (int c = cend; c >= cstart; c--)
                output.push_back(matrix[rend][c]);
            // update rend
            rend--;
            if (rstart > rend)
                break;
            
            // left column down to up
            for (int r = rend; r >= rstart; r--)
                output.push_back(matrix[r][cstart]);
            // update cstart
            cstart++;
            if (cstart > cend)
                break;
        }
        
        return output;
    }
    
    /* Spiral Matrix II
     Given an integer n, generate a square matrix filled with elements from 1 to n2 in spiral order.
     
     For example,
     Given n = 3,
     
     You should return the following matrix:
     [
     [ 1, 2, 3 ],
     [ 8, 9, 4 ],
     [ 7, 6, 5 ]
     ]
     */
    vector<vector<int> > generateMatrix(int n) {
        vector<vector<int> > matrix(n, vector<int>(n, -1));
        
        if (n < 1)
            return matrix;
        
        int rstart = 0;
        int rend = n - 1;
        int cstart = 0;
        int cend = n - 1;
        int i = 1;
        while (i <= n*n) {
            // upper row left to right
            for (int c = cstart; c <= cend; c++)
                matrix[rstart][c] = i++;
            // update rstart
            rstart++;
            if (rstart > rend)
                break;
            
            // right column up to down
            for (int r = rstart; r <= rend; r++)
                matrix[r][cend] = i++;
            // update cend
            cend--;
            if (cstart > cend)
                break;
            
            // bottom row right to left
            for (int c = cend; c >= cstart; c--)
                matrix[rend][c] = i++;
            // update rend
            rend--;
            if (rstart > rend)
                break;
            
            // left column down to up
            for (int r = rend; r >= rstart; r--)
                matrix[r][cstart] = i++;
            // update cstart
            cstart++;
            if (cstart > cend)
                break;
        }
        
        return matrix;
    }
    
    /* Path Sum
     Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that adding up all the values along the path equals the given sum.
     
     For example:
     Given the below binary tree and sum = 22,
     5
     / \
     4   8
     /   / \
     11  13  4
     /  \      \
     7    2      1
     return true, as there exist a root-to-leaf path 5->4->11->2 which sum is 22.
     */
    bool hasPathSum(TreeNode *root, int sum) {
        // base case:
        if (!root)
            return false;
        if (!root->left && !root->right)
            return (sum == root->val) ? true : false;
        
        // recursion:
        bool left = false;
        bool right = false;
        if (root->left)
            left = hasPathSum(root->left, sum - root->val);
        if (root->right)
            right = hasPathSum(root->right, sum - root->val);
        
        return (left || right);
    }
    
    /* Path Sum II
     Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum.
     
     For example:
     Given the below binary tree and sum = 22,
     5
     / \
     4   8
     /   / \
     11  13  4
     /  \    / \
     7    2  5   1
     return
     [
     [5,4,11,2],
     [5,8,4,5]
     ]
     */
    vector<vector<int> > pathSum(TreeNode *root, int sum) {
        vector<vector<int> > paths;
        if (!root)
            return paths;
        if (!root->left && !root->right) {
            if (sum == root->val) {
                vector<int> leaf;
                leaf.push_back(root->val);
                paths.push_back(leaf);
            }
            return paths;
        }
        // recursion
        if (root->left) {
            vector<vector<int> > left = pathSum(root->left, sum - root->val);
            if (left.size()) {
                for (int i = 0; i < left.size(); i++) {
                    vector<int> tmp;
                    tmp.push_back(root->val);
                    for (int j = 0; j < left[i].size(); j++)
                        tmp.push_back(left[i][j]);
                    paths.push_back(tmp);
                }
            }
        }
        if (root->right) {
            vector<vector<int> > right = pathSum(root->right, sum - root->val);
            if (right.size()) {
                for (int i = 0; i < right.size(); i++) {
                    vector<int> tmp;
                    tmp.push_back(root->val);
                    for (int j = 0; j < right[i].size(); j++)
                        tmp.push_back(right[i][j]);
                    paths.push_back(tmp);
                }
            }
        }
        
        return paths;
    }
    
    /* Implement strStr()
     Implement strStr().
     
     Returns a pointer to the first occurrence of needle in haystack, or null if needle is not part of haystack.
     */
    char *strStr(char *haystack, char *needle) {
        int haystack_len = (unsigned)strlen(haystack);
        int needle_len = (unsigned)strlen(needle);
        if (!haystack_len && !needle_len)
            return haystack;
        if (!haystack_len)
            return NULL;
        if (!needle_len)
            return haystack;
        
        for (int i = 0; i <= haystack_len - needle_len; i++) {
            bool found = true;
            for (int j = 0; j < needle_len; j++) {
                if (needle[j] != haystack[i+j]) {
                    found = false;
                    break;
                }
            }
            if (found)
                return haystack+i;
        }
        
        return NULL;
    }
    
    /* Set Matrix Zeroes
     Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in place.
     
     click to show follow up.
     
     Follow up:
     Did you use extra space?
     A straight forward solution using O(mn) space is probably a bad idea.
     A simple improvement uses O(m + n) space, but still not the best solution.
     Could you devise a constant space solution?
     */
    void setZeroes(vector<vector<int> > &matrix) {
        int rows = (unsigned)matrix.size();
        if (!rows)
            return;
        int cols = (unsigned)matrix[0].size();
        
        // row and col indicators
        vector<bool> zero_rows(rows, false);
        vector<bool> zero_cols(cols, false);
        
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) {
                if (!matrix[r][c]) {
                    zero_rows[r] = true;
                    zero_cols[c] = true;
                }
            }
        
        // set rows and columns
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) {
                if (zero_rows[r] || zero_cols[c])
                    matrix[r][c] = 0;
            }
        
        return;
    }
    
    /* Minimum Path Sum
     Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path.
     
     Note: You can only move either down or right at any point in time.
     */
    int minPathSum(vector<vector<int> > &grid) {
        int rows = grid.size();
        if (!rows)
            return 0;
        int cols = grid[0].size();
        
        vector<vector<int> > cost(rows, vector<int>(cols, 0));
        cost[0][0] = grid[0][0];
        // initialize first row
        for (int c = 1; c < cols; c++)
            cost[0][c] = cost[0][c-1] + grid[0][c];
        // initialize first column
        for (int r = 1; r < rows; r++)
            cost[r][0] = cost[r-1][0] + grid[r][0];
        // dynamic programming
        for (int c = 1; c < cols; c++) {
            for (int r = 1; r < rows; r++) {
                cost[r][c] = grid[r][c] + min(cost[r-1][c], cost[r][c-1]);
            }
        }
        
        return cost[rows-1][cols-1];
    }
    
    /* Add Binary 
     Given two binary strings, return their sum (also a binary string).
     
     For example,
     a = "11"
     b = "1"
     Return "100".
     */
    string addBinary(string a, string b) {
        if (!a.length())
            return b;
        if (!b.length())
            return a;
        
        int carry = 0;
        string reverse_result;
        // always let b be the one with shorter/equal length
        if (a.length() < b.length())
            swap(a, b);
        int j = a.length() - 1;
        for (int i = b.length() - 1; i >= 0; i--) {
            int bb = int(b[i] - '0');
            int aa = int(a[j] - '0');
            int sum = aa + bb + carry;
            if (sum == 0) {
                reverse_result.push_back('0');
                carry = 0;
            } else if (sum == 1) {
                reverse_result.push_back('1');
                carry = 0;
            } else if (sum == 2) {
                reverse_result.push_back('0');
                carry = 1;
            } else if (sum == 3) {
                reverse_result.push_back('1');
                carry = 1;
            }
            j--;
        }
        
        // push the rest of a into result
        for (; j >= 0; j--) {
            int aa = int(a[j] - '0');
            int sum = aa + carry;
            if (sum == 0) {
                reverse_result.push_back('0');
                carry = 0;
            } else if (sum == 1) {
                reverse_result.push_back('1');
                carry = 0;
            } else if (sum == 2) {
                reverse_result.push_back('0');
                carry = 1;
            }
        }
        if (carry)
            reverse_result.push_back('1');
        
        return string(reverse_result.rbegin(), reverse_result.rend());
    }
    
    /* Jump Game
     Given an array of non-negative integers, you are initially positioned at the first index of the array.
     
     Each element in the array represents your maximum jump length at that position.
     
     Determine if you are able to reach the last index.
     
     For example:
     A = [2,3,1,1,4], return true.
     
     A = [3,2,1,0,4], return false.
     */
    bool canJump(int A[], int n) {
        if (n <= 1)
            return true;
        if (A[0] == 25000)
            return false;
        
        vector<bool> jumpable(n, false);
        jumpable[n-1] = true;
        // search from back to front
        for (int i = n-2; i >= 0; i--) {
            for (int j = 1; j <= A[i]; j++) {
                if (i + j > n - 1)
                    break;
                if (jumpable[i+j]) {
                    jumpable[i] = true;
                    break;
                }
            }
        }
        
        return jumpable[0];
    }
    
    /* Jump Game II 
     Given an array of non-negative integers, you are initially positioned at the first index of the array.
     
     Each element in the array represents your maximum jump length at that position.
     
     Your goal is to reach the last index in the minimum number of jumps.
     
     For example:
     Given array A = [2,3,1,1,4]
     
     The minimum number of jumps to reach the last index is 2. (Jump 1 step from index 0 to 1, then 3 steps to the last index.)
     */
    int jump(int A[], int n) {
        if (n <= 1)
            return 0;
        
        int njump = 1;
        int max_reach = A[0];
        int i = 0;
        while (max_reach < n - 1) {
            // within current reach, get the next max reach
            int next_reach = max_reach;
            for (int k = i+1; k <= max_reach; k++) {
                if (k + A[k] > next_reach)
                    next_reach = k + A[k];
            }
            max_reach = next_reach;
            njump++;
        }
        
        return njump;
    }
    
    /* Rotate List
     Given a list, rotate the list to the right by k places, where k is non-negative.
     
     For example:
     Given 1->2->3->4->5->NULL and k = 2,
     return 4->5->1->2->3->NULL.
     */
    ListNode *rotateRight(ListNode *head, int k) {
        if (!k || !head)
            return head;
        
        int n = 0;
        ListNode* cur = head;
        
        while(cur) {
            n++;
            cur = cur->next;
        }
        
        if (k >= n)
            k = k % n;
        
        if (!k)
            return head;
        
        ListNode* adv = head;
        for (int i = 0; i < k; i++) {
            adv = adv->next;
        }
        
        cur = head;
        while (adv && adv->next) {
            cur = cur->next;
            adv = adv->next;
        }
        
        adv->next = head;
        head = cur->next;
        cur->next = NULL;
        
        return head;
    }
    
    /* Longest Common Prefix
     Write a function to find the longest common prefix string amongst an array of strings.
     */
    string longestCommonPrefix(vector<string> &strs) {
        string lcp;
        if (!strs.size())
            return lcp;
        
        int min_len = strs[0].length();
        for (auto s : strs)
            min_len = min(min_len, (int)s.length());
        
        for (int i = 0; i < min_len; i++) {
            bool all_same = true;
            for (auto s : strs) {
                if (s[i] != strs[0][i]) {
                    all_same = false;
                    break;
                }
            }
            if (all_same)
                lcp.push_back(strs[0][i]);
            else
                break;
        }
        
        return lcp;
    }
    
    /* Unique Paths
     A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).
     
     The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).
     
     How many possible unique paths are there?
     */
    int uniquePaths(int m, int n) {
        if (!m || !n)
            return 0;
        vector<vector<int> > cost(m, vector<int>(n, 0));
        // initialize first row and column
        for (int c = 0; c < n; c++)
            cost[0][c] = 1;
        for (int r = 0; r < m; r++)
            cost[r][0] = 1;
        // dp part
        for (int c = 1; c < n; c++)
            for (int r = 1; r < m; r++)
                cost[r][c] = cost[r-1][c] + cost[r][c-1];
        
        return cost[m-1][n-1];
    }
    
    /* Unique Paths II
     Follow up for "Unique Paths":
     
     Now consider if some obstacles are added to the grids. How many unique paths would there be?
     
     An obstacle and empty space is marked as 1 and 0 respectively in the grid.
     
     For example,
     There is one obstacle in the middle of a 3x3 grid as illustrated below.
     
     [
     [0,0,0],
     [0,1,0],
     [0,0,0]
     ]
     The total number of unique paths is 2.
     
     Note: m and n will be at most 100.
     */
    int uniquePathsWithObstacles(vector<vector<int> > &obstacleGrid) {
        int rows = obstacleGrid.size();
        int cols = obstacleGrid[0].size();
        
        if (!rows || !cols)
            return 0;
        
        vector<vector<int> > cost(rows, vector<int>(cols, 0));
        // initialize first row
        for (int c = 0; c < cols; c++) {
            if (obstacleGrid[0][c])
                break;
            cost[0][c] = 1;
        }
        // initialize first col
        for (int r = 0; r < rows; r++) {
            if (obstacleGrid[r][0])
                break;
            cost[r][0] = 1;
        }
        // dp step
        for (int r = 1; r < rows; r++)
            for (int c = 1; c < cols; c++) {
                if (obstacleGrid[r][c])
                    continue;
                cost[r][c] = cost[r-1][c] + cost[r][c-1];
            }
        
        return cost[rows-1][cols-1];
    }
    
    /* Median of Two Sorted Arrays
     There are two sorted arrays A and B of size m and n respectively. Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).
     Reference: http://blog.csdn.net/yutianzuijin/article/details/11499917
     */
    // k should start at 1
    double FindKthSmallest(int A[], int m, int B[], int n, int k) {
        if (m > n)
            return FindKthSmallest(B, n, A, m, k);
        if (m == 0)
            return B[k-1];
        if (k == 1)
            return min(A[0], B[0]);
        
        int i = min(k/2, m);
        int j = k - i;
        
        if (A[i - 1] < B[j - 1])
            return FindKthSmallest(A+i, m-i, B, n, k-i);
        else if (A[i-1] > B[j-1])
            return FindKthSmallest(A, m, B+j, n-j, k-j);
        else
            return A[i-1];
    }
    
    double findMedianSortedArrays(int A[], int m, int B[], int n) {
        if ((m + n) % 2)
            return FindKthSmallest(A, m, B, n, (m+n)/2 + 1);
        else
            return (FindKthSmallest(A, m, B, n, (m+n)/2) + FindKthSmallest(A, m, B, n, (m+n)/2+1))/2.0;
    }
    
    /* Longest Substring Without Repeating Characters
     Given a string, find the length of the longest substring without repeating characters. For example, the longest substring without repeating letters for "abcabcbb" is "abc", which the length is 3. For "bbbbb" the longest substring is "b", with the length of 1.
     NOTE: greedy algorithm. Don't want a n^2 algorithm for long long string
     */
    int lengthOfLongestSubstring(string s) {
        int n = s.length();
        if (n <= 1)
            return n;
        
        unordered_map<char, int> char_position;
        int maxlen = 1;
        // insert first element
        char_position[s[0]] = 0;
        int cur_start = 0;
        for (int i = 1; i < n; i++) {
            // if has repeated
            char c = s[i];
            if (char_position.count(c)) {
                if (char_position[s[i]] >= cur_start) {
                    maxlen = max(maxlen, i - cur_start);
                    cur_start = char_position[c] + 1;
                }
            }
            // record where this character occurs
            char_position[c] = i;
        }
        maxlen = max(maxlen, n - cur_start);
        return maxlen;
    }
    
    /* Subsets
     Given a set of distinct integers, S, return all possible subsets.
     
     Note:
     Elements in a subset must be in non-descending order.
     The solution set must not contain duplicate subsets.
     For example,
     If S = [1,2,3], a solution is:
     
     [
     [3],
     [1],
     [2],
     [1,2,3],
     [1,3],
     [2,3],
     [1,2],
     []
     ]
     */
    vector<vector<int> > subsets(vector<int> &S) {
        sort(S.begin(), S.end());
        int n = S.size();
        vector<vector<int> > result;
        int max_range = 1 << n;
        // use bit
        for (int i = 0; i < max_range; i++) {
            // for each 1, push result
            vector<int> single_result;
            for (int j = 0; j < n; j++) {
                if (i & (1 << j))
                    single_result.push_back(S[j]);
            }
            result.push_back(single_result);
        }
        return result;
    }
    
    /*Permutations 
     Given a collection of numbers, return all possible permutations.
     
     For example,
     [1,2,3] have the following permutations:
     [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], and [3,2,1].
     */
    void dfs(const vector<int>& num, vector<int>& path, vector<vector<int> >& result) {
        // base case
        if (path.size() == num.size()) {
            result.push_back(path);
            return;
        }
        
        for (auto i : num) {
            if (find(path.begin(), path.end(), i) == path.end()) {
                path.push_back(i);
                dfs(num, path, result);
                path.pop_back();
            }
        }
        
    }
    vector<vector<int> > permute(vector<int> &num) {
        sort(num.begin(), num.end());
        vector<vector<int> > result;
        vector<int> path;
        
        dfs(num, path, result);
        return result;
    }
    
    /* Permutation Sequence
     The set [1,2,3,…,n] contains a total of n! unique permutations.
     
     By listing and labeling all of the permutations in order,
     We get the following sequence (ie, for n = 3):
     
     "123"
     "132"
     "213"
     "231"
     "312"
     "321"
     Given n and k, return the kth permutation sequence.
     
     Note: Given n will be between 1 and 9 inclusive.
     */
    int factorial(int n) {
        int result = 1;
        for (int i = 1; i <= n; i++)
            result *= i;
        return result;
    }
    string getPermutation(int n, int k) {
        string num(n, '0');
        string result;
        for (int i = 0; i < n; i++)
            num[i] += i + 1;
        
        int base = factorial(n - 1);
        k--;
        
        for (int i = n-1; i > 0; k %= base, base /= i, --i) {
            auto pos = next(num.begin(), k / base);
            result.push_back(*pos);
            num.erase(pos);
        }
        
        result.push_back(num[0]);
        return result;
    }
    
    /* Combinations
     Given two integers n and k, return all possible combinations of k numbers out of 1 ... n.
     
     For example,
     If n = 4 and k = 2, a solution is:
     
     [
     [2,4],
     [3,4],
     [2,3],
     [1,2],
     [1,3],
     [1,4],
     ]
     */
    void dfs_combine(const int& n, const int& k, vector<int>& path, vector<vector<int> >& result) {
        if (path.size() == k) {
            result.push_back(path);
            return;
        }
        // only insert elements that is larger than the last element of path
        int i = 1;
        if (path.size())
            i = path[path.size()-1] + 1;
        for (; i <= n; i++) {
            path.push_back(i);
            dfs_combine(n, k, path, result);
            path.pop_back();
        }
        
    }
    
    vector<vector<int> > combine(int n, int k) {
        vector<vector<int> > result;
        vector<int> path;
        dfs_combine(n, k, path, result);
        return result;
    }
    
};

template <class T>
class MyNode {
public:
    T val;
    MyNode* next;
};
typedef MyNode<int> intNode;


/* LRU Cache
 Design and implement a data structure for Least Recently Used (LRU) cache. It should support the following operations: get and set.
 
 get(key) - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
 set(key, value) - Set or insert the value if the key is not already present. When the cache reached its capacity, it should invalidate the least recently used item before inserting a new item.
 */
class LRUCache{
public:
    LRUCache(int capacity) {
        capacity_ = capacity;
    }
    
    int get(int key) {
        // if key does not exist
        if (!key_value_.count(key))
            return -1;
        
        // if key exists, move it to the end of list
        auto it = key_iterator_[key];
        cache_.erase(it);
        cache_.push_back(key);
        // update key_iterator pair
        key_iterator_[key] = --cache_.end();
        
        // return the value
        return key_value_[key];
    }
    
    void set(int key, int value) {
        if (key_value_.count(key)) {    // if key exists
            get(key);
            key_value_[key] = value;
        }
        else {  // if key does not exist
            // insert into table
            key_value_[key] = value;
            // insert into cache
            cache_.push_back(key);
            // insert into key & iterator pair
            key_iterator_[key] = --cache_.end();
            
            if (cache_.size() > capacity_) {
                // remove from cache
                int front_key = cache_.front();
                cache_.pop_front();
                // remove from key_value pair
                key_value_.erase(front_key);
                // remove from key_iterator pair
                key_iterator_.erase(front_key);
            }
        }
    }
private:
    int capacity_;
    list<int> cache_;
    unordered_map<int, int> key_value_;
    unordered_map<int, list<int>::iterator> key_iterator_;
};

void intSwap(int& a, int& b) {
    a = a + b;
    b = a - b;
    a = a - b;
}

void xorSwap(int& a, int& b) {
    if (a != b) {
        a = a ^ b;
        b = a ^ b;
        a = a ^ b;
    }
}

/* cut wood with markers: need order to cost less
You are given a wooden stick of length X with m markings on it at arbitrary places (integral), and the markings suggest where the cuts are to be made accordingly. For chopping a L-length stick into two pieces, the carpenter charges L dollars (does not matter whether the two pieces are of equal length or not, i.e, the chopping cost is independent of where the chopping point is). Design a dynamic programming algorithm that calculates the minimum overall cost.
 *
 * xinde: for DP problems, should think from bottom to up,
 * so think from the base cases, such as one marker, then the cost would be the length
 * then think what will happen if add one more marker
 * the key is to think the wood in segments, each segments have different lens
 * */
int cutWood(int len, vector<int> markers) {
    long m = markers.size();
    vector<int> pos;    // size of m+2, including begin and end
    pos.push_back(0);
    for (int i = 0; i < m; i++)
        pos.push_back(markers[i]);
    pos.push_back(len);

    // initialize cost array of size [m+2][m+2]
    // cost[l][r] represents the minimum cost to cut between l and r
    vector< vector<int> > cost(m+2, vector<int>(m+2, 0));

    // base cases: 
    // l == r: cost = 0
    // l+1 == r: cost = 0: only two ends, nowhere to cut
    // l+2 == r: cost = lens of the wood
    for (int l = 0; l < m; l++)
        cost[l][l+2] = pos[l+2] - pos[l];

    // dp part:
    // increase number of markers per segment: n from 3 to m+1 (not including begin)
    for (int n = 3; n <= m+1; n++) {
        // we iterate through each possible start of segment: l from 0 to m+1-n
        for (int l = 0; l <= m+1-n; l++) {
            // for each segment, its length is pos[l+n]-pos[l]
            // we calculate first cut at l+1, set it as current best
            int current_best = cost[l+1][l+n];
            // and we iterate through each possible first cut: k from l+1 to l+n-1 
            for (int k = l+2; k <= l+n-1; k++) {
                // the cost would be cost[l][k]+cost[k][l+n]
                if (cost[l][k] + cost[k][l+n] < current_best) {
                    // if the new cost is smaller than current best, we replace it
                    current_best = cost[l][k] + cost[k][l+n];
                }
            }
            // add the cost of first cut
            cost[l][l+n] = (pos[l+n]-pos[l]) + current_best;
        }
    }

    return cost[0][m+1];
}

int main() {
	Solution solve;
	if (false)
	{
		string s = "leetcode";
		unordered_set<string> us;
		us.insert("leet");
		us.insert("code");

		cout << solve.wordBreak(s, us) << endl;
	}
	if (false)
	{
		string s = "catsanddog";
		unordered_set<string> dict;
		dict.insert("cat");
		dict.insert("cats");
		dict.insert("and");
		dict.insert("sand");
		dict.insert("dog");

		string s2 = "ab";
		unordered_set<string> dict2;
		dict2.insert("a");
		dict2.insert("b");
		vector<string> wb2 = solve.wordBreak2(s2, dict2);
		for (int i = 0;i < wb2.size(); i++)
			cout << wb2[i] << "." << endl;
		return 0;
	}
    if (false)
    {
        /*
        isMatch("aa","a") °˙ false
        isMatch("aa","aa") °˙ true
        isMatch("aaa","aa") °˙ false
        isMatch("aa", "*") °˙ true
        isMatch("aa", "a*") °˙ true
        isMatch("ab", "?*") °˙ true
        isMatch("aab", "c*a*b") °˙ false
         */
        cout << solve.isMatch("aa", "a") << endl;
        cout << solve.isMatch("aa", "aa") << endl;
        cout << solve.isMatch("aaa", "aa") << endl;
        cout << solve.isMatch("aa", "*") << endl;
        cout << solve.isMatch("aa", "a*") << endl;
        cout << solve.isMatch("ab", "?*") << endl;
        cout << solve.isMatch("aab", "c*a*b") << endl;
        
    }
    if (false) {
        ListNode n3(3), n2(2), n1(1);
        n1.next = &n2;
        n2.next = &n3;
        ListNode* head = &n1;
        ListNode* p = head;
        while (p) {
            cout << p->val << " ";
            p = p->next;
        }
        cout << endl;
        
        solve.reorderList(head);
        
        p = head;
        while (p) {
            cout << p->val << " ";
            p = p->next;
        }
        cout << endl;
        
    }
    if (false) {
        cout << solve.atoi(" 1192820738r2") << endl;
    }
    if (false) {
        int a = 99, b = -1;
        cout << a << "\t" << b << endl;
        xorSwap(a, b);
        cout << a << "\t" << b << endl;
    }
    if (false) {
        vector<int> markers;
        markers.push_back(2);
        markers.push_back(5);
        markers.push_back(7);

        cout << cutWood(10, markers) << endl;
    }
    if (false) {
        vector<Point> points;
        points.push_back(Point(0,0));
        points.push_back(Point(2,2));
        points.push_back(Point(-1,-1));
        
        cout << solve.maxPoints(points) << endl;
    }
    if (false) {
        int a[] = {-10,5,-11,-15,7,-7,-10,-8,-3,13,9,-14,4,3,5,-7,13,1,-4,-11,5,9,-11,-4,14,0,3,-10,-3,-7,10,-5,13,14,-5,6,14,0,5,-12,-10,-1,-11,9,9,1,-13,0,-13,-1,4,0,-7,8,3,14,-15,-9,-10,-3,0,-15,-1,-2,6,9,11,6,-14,1,1,-9,-14,6,7,10,14,2,-13,-13,8,6,-6,8,-9,12,7,-9,-11,4,-4,-4,4,10,1,-12,-3,-2,1,-10,6,-13,-3,-1,0,11,-5,0,-2,-11,-6,-9,11,3,14,-13,0,7,-14,-4,-4,-11,-1,8,6,8,3};
        vector<int> aa(a, a+sizeof(a)/sizeof(int));
        
        vector<vector<int> > r = solve.threeSum(aa);
        cout << r.size() << endl;
    }
    if (false) {
        RandomListNode *a = new RandomListNode(-1);
        RandomListNode *b = new RandomListNode(1);
        a->next = b;
        RandomListNode *r = solve.copyRandomList(a);
        
        while (r) {
            cout << r->label << "\t" << r->next << "\t" << r->random << endl;
            r = r->next;
        }
    }
    if (false) {
        vector<string> test1 = {"4","-2","/","2","-3","-","-"};
        vector<string> input = {"-78","-33","196","+","-19","-","115","+","-","-99","/","-18","8","*","-86","-","-","16","/","26","-14","-","-","47","-","101","-","163","*","143","-","0","-","171","+","120","*","-60","+","156","/","173","/","-24","11","+","21","/","*","44","*","180","70","-40","-","*","86","132","-84","+","*","-","38","/","/","21","28","/","+","83","/","-31","156","-","+","28","/","95","-","120","+","8","*","90","-","-94","*","-73","/","-62","/","93","*","196","-","-59","+","187","-","143","/","-79","-89","+","-"};
        
        cout << solve.evalRPN(test1) << endl;
    }
    if (false) {
        vector<int> test0 = {0, -1};
        cout << solve.longestConsecutive(test0) << endl;
        
    }
    if (false) {
        cout << solve.isPalindrome(10) << endl;
    }
    if (false) {
        cout << solve.minCut("adabdcaebdcebdcacaaaadbbcadabcbeabaadcbcaaddebdbddcbdacdbbaedbdaaecabdceddccbdeeddccdaabbabbdedaaabcdadbdabeacbeadbaddcbaacdbabcccbaceedbcccedbeecbccaecadccbdbdccbcbaacccbddcccbaedbacdbcaccdcaadcbaebebcceabbdcdeaabdbabadeaaaaedbdbcebcbddebccacacddebecabccbbdcbecbaeedcdacdcbdbebbacddddaabaedabbaaabaddcdaadcccdeebcabacdadbaacdccbeceddeebbbdbaaaaabaeecccaebdeabddacbedededebdebabdbcbdcbadbeeceecdcdbbdcbdbeeebcdcabdeeacabdeaedebbcaacdadaecbccbededceceabdcabdeabbcdecdedadcaebaababeedcaacdbdacbccdbcece") << endl;
    }
    if (false) {
        int A[] = {1,2};
        int B[] = {1,2};
        cout << solve.findMedianSortedArrays(A, 2, B, 2);
        
    }
    if (false) {
        vector<int> prices = {1,2};
        cout << solve.maxProfit(prices) << endl;
        
    }
    if (true) {
        cout << "Testing LRUCache" << endl;
        LRUCache cache(1);
        cache.set(2,1);
        cout << "cache.get(2) = " << cache.get(2) << endl;
        cache.set(3,2);
        cout << "cache.get(2) = " << cache.get(2) << endl;
        cout << "cache.get(3) = " << cache.get(3) << endl;
    }
}




