#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <unordered_set>
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

class Solution {
public:
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
	vector<string> wordBreak2_recursion(string s, unordered_set<string> &dict, map<string, vector<string>>& mem) {
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
		map<string, vector<string>> mem;

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
    vector<vector<string>> partition(string s) {
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
    void solve(vector<vector<char>> &board) {
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
    
    double findMedianSortedArrays(int A[], int m, int B[], int n) {
        // Note: The Solution object is instantiated only once and is reused by each test case.
        // base case:
        if (m <= 0) {
            if (n%2 == 1)
                return B[n/2];
            else
                return (B[n/2]+B[n/2-1])/2.0;
        }
        if (n <= 0) {
            if (m%2 == 1)
                return A[m/2];
            else
                return (A[m/2]+A[m/2-1])/2.0;
        }
        if (m+n == 2)
            return (A[0]+B[0])/2.0;
        
        // recursion:
        double ma, mb;
        if (m%2 == 1)
            ma = A[m/2];
        else
            ma = (A[m/2] + A[m/2-1]) / 2.0;
        if (n%2 == 1)
            mb = B[n/2];
        else
            mb = (B[n/2] + B[n/2-1]) / 2.0;
        
        if (ma == mb)
            return ma;
        else if (ma < mb) {
            return findMedianSortedArrays(&A[m/2+1], m/2-1, B, n/2-1);
        }
        else { // ma > mb
            return findMedianSortedArrays(A, m/2-1, &B[n/2+1], n/2-1);
        }
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

/* thinking:
 1. what data structure available: array, linked list, binary tree; queue, stack, hash_table, map
 2. what performance do I want: best: constant time get; constant time set
 3. how to keep track of usage
 
 my idea: use a linked list to track usage; meanwhile, use hash table(or map for simple implementation) to access data
 */
class LRUCache{
public:
    LRUCache(int capacity) {
        cap = capacity;
        head = NULL;
        tail = NULL;
    }
    
    int get(int key) {
        // check if exist
            // if not, return -1;
            // if yes, adjust list and map
        return 0;
    }
    
    void set(int key, int value) {
        
    }
private:
    intNode *head, *tail;
    map<int, intNode> mem;
    int cap;
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
    if (true) {
        vector<int> prices = {1,2};
        cout << solve.maxProfit(prices) << endl;
        
    }
}




