/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 *
 * @author sangeetdas
 */
public class DSA450Questions {

    class GraphEdge {

        int vertex;
        int weight;

        public GraphEdge(int vertex, int weight) {
            this.vertex = vertex;
            this.weight = weight;
        }

    }

    public void reverseArray(int[] a) {

        int len = a.length;

        //.....................O(N)
        for (int i = 0; i < len / 2; i++) {
            int temp = a[i];
            a[i] = a[len - i - 1];
            a[len - i - 1] = temp;
        }

        //output
        for (int x : a) {
            System.out.print(x + " ");
        }
        System.out.println();

    }

    public void arrayElementMoreThan_NDivK(int[] a, int K) {

        int N = a.length;
        int count = N / K;
        Map<Integer, Integer> map = new HashMap<>();
        for (int x : a) {
            map.put(x, map.getOrDefault(x, 0) + 1);
        }

        map.entrySet().stream()
                .filter(e -> e.getValue() > count)
                .collect(Collectors.toMap(e -> e.getKey(), e -> e.getValue()))
                .entrySet()
                .stream()
                .forEach(e -> System.out.println(e.getKey()));

    }

    public void minMaxInArray_1(int[] a) {

        //...................T: O(N)
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;

        for (int i = 0; i < a.length; i++) {
            max = Math.max(max, a[i]);
            min = Math.min(min, a[i]);
        }

        //outpur
        System.out.println("Min and max value in array: " + min + " " + max);
    }

    public void minMaxInArray_2(int[] a) {

        //https://www.geeksforgeeks.org/maximum-and-minimum-in-an-array/
        //...................T: O(N)
        //Min no of comparision
        int min = Integer.MIN_VALUE;
        int max = Integer.MAX_VALUE;

        int n = a.length;
        int itr = 0;
        //check if array is even/odd
        if (n % 2 == 0) {
            max = Math.max(a[0], a[1]);
            min = Math.min(a[0], a[1]);
            //in case of even choose min & max from first two element
            //and set itr to start from 2nd index i.e(3rd element) in pair wise
            itr = 2;
        } else {
            max = a[0];
            min = a[0];
            //in case of odd choose first element as min & max both
            //set itr to 1 i.e, 2nd element
            itr = 1;
        }

        //since we checking itr and itr+1 value in loop 
        //so run loop to n-1 so that itr+1th element corresponds to n-1th element
        while (itr < n - 1) {

            //check current itr and itr+1 element
            if (a[itr] > a[itr + 1]) {
                max = Math.max(max, a[itr]);
                min = Math.min(min, a[itr + 1]);
            } else {
                max = Math.max(max, a[itr + 1]);
                min = Math.min(min, a[itr]);
            }
            itr++;
        }

        //outpur
        System.out.println("Min and max value in array: " + min + " " + max);

    }

    public void kThSmallestElementInArray(int[] arr, int K) {

        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(
                (o1, o2) -> o2.compareTo(o1)
        );

        for (int val : arr) {
            maxHeap.add(val);
            if (maxHeap.size() > K) {
                maxHeap.poll();
            }
        }
        //output
        System.out.println(K + " th smallest element: " + maxHeap.peek());
    }

    public void kThLargestElementInArray(int[] arr, int K) {
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();

        for (int val : arr) {
            minHeap.add(val);
            if (minHeap.size() > K) {
                minHeap.poll();
            }
        }
        //output
        System.out.println(K + " th largest element: " + minHeap.peek());
    }

    public void sortArrayOf012_1(int[] arr) {

        //.............T: O(N)
        //.............S: O(3)
        Map<Integer, Integer> map = new HashMap<>();
        for (int val : arr) {
            map.put(val, map.getOrDefault(val, 0) + 1);
        }

        //creating array
        int k = 0;
        for (int val = 0; val <= 2; val++) {
            int freq = map.get(val);
            while (freq != 0) {
                arr[k++] = val;
                freq--;
            }
        }

        //output
        for (int val : arr) {
            System.out.print(val + " ");
        }
        System.out.println();
    }

    private void swapIntArray(int[] a, int x, int y) {
        int temp;
        temp = a[x];
        a[x] = a[y];
        a[y] = temp;
    }

    public void sortArrayOf012_2(int[] arr) {

        //.............T: O(N)
        //.............S: O(1)
        //https://leetcode.com/problems/sort-colors/
        //https://www.geeksforgeeks.org/sort-an-array-of-0s-1s-and-2s/
        //based on Dutch National Flag Algorithm
        int start = 0;
        int mid = 0;
        int end = arr.length - 1;
        while (mid <= end) {
            switch (arr[mid]) {
                case 0: {
                    swapIntArray(arr, start, mid);
                    start++;
                    mid++;
                    break;
                }
                case 1:
                    mid++;
                    break;
                case 2: {
                    swapIntArray(arr, mid, end);
                    end--;
                    break;
                }
            }
        }
        //output
        for (int val : arr) {
            System.out.print(val + " ");
        }
        System.out.println();
    }

    private void nextPermutation_Print(int[] nums) {
        for (int x : nums) {
            System.out.print(x);
        }
        System.out.println();
    }

    public void nextPermutation(int[] nums) {

        int N = nums.length;

        //length == 1
        if (N == 1) {
            //output:
            nextPermutation_Print(nums);
            return;
        }

        //check if desc sorted
        boolean descSorted = true;
        for (int i = 1; i < N; i++) {
            if (nums[i - 1] < nums[i]) {
                descSorted = false;
                break;
            }
        }

        if (descSorted) {
            for (int i = 0; i < N / 2; i++) {
                swapIntArray(nums, i, N - i - 1);
            }
            nextPermutation_Print(nums);
            return;
        }

        //any other cases
        int firstDecNumIndex = N - 1;
        for (int i = N - 2; i >= 0; i--) {
            if (nums[i] < nums[firstDecNumIndex]) {
                firstDecNumIndex = i;
                break;
            }
            firstDecNumIndex = i;
        }

        int justGreaterNumIndex = firstDecNumIndex + 1;
        int diff = nums[justGreaterNumIndex] - nums[firstDecNumIndex];
        for (int i = justGreaterNumIndex + 1; i < N; i++) {
            int currDiff = nums[i] - nums[firstDecNumIndex];
            if (currDiff > 0 && currDiff < diff) {
                diff = currDiff;
                justGreaterNumIndex = i;
            }
        }

        swapIntArray(nums, firstDecNumIndex, justGreaterNumIndex);
        Arrays.sort(nums, firstDecNumIndex + 1, N);

        //output:
        nextPermutation_Print(nums);
    }

    private int factorialLargeNumber_Multiply(int x, int[] res, int resSize) {

        int carry = 0;
        for (int i = 0; i < resSize; i++) {
            int prod = res[i] * x + carry;
            res[i] = prod % 10;
            carry = prod / 10;
        }

        while (carry != 0) {

            res[resSize] = carry % 10;
            carry = carry / 10;
            resSize++;
        }

        return resSize;
    }

    public void factorialLargeNumber(int N) {
        int[] res = new int[Integer.MAX_VALUE / 200];
        res[0] = 1;

        int resSize = 1;
        for (int x = 2; x <= N; x++) {
            resSize = factorialLargeNumber_Multiply(x, res, resSize);
        }

        //output
        for (int i = resSize - 1; i >= 0; i--) {
            System.out.print(res[i]);
        }
        System.out.println();
    }

    public void rainWaterTrappingUsingStack(int[] height) {

        //https://leetcode.com/problems/trapping-rain-water/solution/
        //..................T: O(N)
        //..................S: O(N)
        int ans = 0;
        int current = 0;
        int N = height.length;
        Stack<Integer> s = new Stack<>();
        while (current < N) {

            while (!s.isEmpty() && height[current] > height[s.peek()]) {

                int top = s.pop();
                if (s.isEmpty()) {
                    break;
                }

                int distance = current - s.peek() - 1;
                int boundedHeight = Math.min(height[current], height[s.peek()]) - height[top];
                ans += distance * boundedHeight;
            }
            s.push(current++);
        }

        //output
        System.out.println("Rain water trapping using stack: " + ans);
    }

    public void rainWaterTrappingUsingTwoPointers(int[] height) {
        //https://leetcode.com/problems/trapping-rain-water
        //OPTIMISED than stack
        //..................T: O(N)
        //..................S: O(1)
        int left = 0;
        int right = height.length - 1;
        int ans = 0;
        int leftMax = 0;
        int rightMax = 0;
        while (right > left) {
            if (height[left] < height[right]) {

                if (height[left] >= leftMax) {
                    leftMax = height[left];
                } else {
                    ans += (leftMax - height[left]);
                }

                left++;
            } else {

                if (height[right] >= rightMax) {
                    rightMax = height[right];
                } else {
                    ans += (rightMax - height[right]);
                }

                right--;
            }
        }

        //output
        System.out.println("Rain water trapping using tow pointers: " + ans);
    }

    public void findMaximumProductSubarray(int[] arr) {
        //https://leetcode.com/problems/maximum-product-subarray/
        //Explanation: https://www.youtube.com/watch?v=lXVy6YWFcRM
        int result = arr[0];
        int currMax = 1;
        int currMin = 1;
        for (int i = 0; i < arr.length; i++) {

            if (arr[i] == 0) {
                //just reset
                currMax = 1;
                currMin = 1;
                result = Math.max(arr[i], result);
                continue;
            }

            int tempCurrMax = arr[i] * currMax;
            currMax = Math.max(Math.max(arr[i] * currMax, arr[i] * currMin), arr[i]);
            currMin = Math.min(Math.min(tempCurrMax, arr[i] * currMin), arr[i]);
            result = Math.max(currMax, result);
        }

        //output:
        System.out.println("Maximum product subarray: " + result);
    }

    public int kadaneAlgorithm(int[] arr) {

        //for finding maximum sum subarray
        int maxSum = arr[0];
        int currMaxSum = arr[0];
        for (int i = 1; i < arr.length; i++) {
            currMaxSum = Math.max(arr[i], currMaxSum + arr[i]);
            maxSum = Math.max(maxSum, currMaxSum);
        }
        //output
        return maxSum;
    }

    public void kadaneAlgorithm_PointingIndexes(int[] arr) {

        int maxSum = 0;
        int currMaxSum = 0;
        int maxElement = Integer.MIN_VALUE;
        int maxElementIndex = 0;

        int start = 0;
        int end = 0;
        int index = 0;
        while (index < arr.length) {

            currMaxSum += arr[index];
            //case to handle all negative element
            if (arr[index] > maxElement) {
                maxElement = arr[index];
                maxElementIndex = index;
            }
            if (currMaxSum < 0) {
                currMaxSum = 0;
                start = index + 1;
            }
            if (maxSum < currMaxSum) {
                maxSum = currMaxSum;
                end = index;
            }
            index++;
        }

        //output:
        System.out.println("Max sum subarray with start & end: "
                + (maxSum == 0
                        ? maxElement + " Start: " + maxElementIndex + " end: " + maxElementIndex
                        : maxSum + " Start: " + start + " end: " + end)
        );
    }

    public void moveNegativeElementsToOneSideOfArray(int[] arr) {

        //Two pointer approach
        //...........................T: O(N)
        //actual:
        for (int x : arr) {
            System.out.print(x + " ");
        }
        System.out.println();

        int n = arr.length;
        int negativeIndex = 0;
        int positiveIndex = n - 1;

        while (positiveIndex > negativeIndex) {

            //if any element in left side id already a -ve then that element should be taken into consideration
            //move to next element
            while (negativeIndex < positiveIndex && arr[negativeIndex] < 0) {
                negativeIndex++;
            }

            //same way any +ve no on the right side should not be counted and move to next element
            while (negativeIndex < positiveIndex && arr[positiveIndex] > 0) {
                positiveIndex--;
            }

            //as we are planning to shift all the -ve elements to left side of array
            //after above while loops we will be having
            // +ve element (arr[f] > 0) in left side AND any -ve element(arr[h] <0) on right side should be swapped 
            swapIntArray(arr, negativeIndex, positiveIndex);
            negativeIndex++;
            positiveIndex--;
        }

        //output:
        for (int x : arr) {
            System.out.print(x + " ");
        }
        System.out.println();
    }

    public void findUnionAndIntersectionOfTwoArrays(int[] a, int[] b) {

        int m = a.length;
        int n = b.length;
        int maxLen = Math.max(m, n);
        Set<Integer> unionSet = new HashSet<>();
        for (int i = 0; i < maxLen; i++) {

            if (i < m) {
                unionSet.add(a[i]);
            }

            if (i < n) {
                unionSet.add(b[i]);
            }
        }

        //output
        System.out.println("No of union element: " + unionSet.size() + " elements: " + unionSet);

        //finding intersection of two array
        Set<Integer> aSet = new HashSet<>();
        Set<Integer> bSet = new HashSet<>();
        for (int x : a) {
            aSet.add(x);
        }
        for (int x : b) {
            bSet.add(x);
        }

        Set<Integer> intersectionSet = new HashSet<>();
        for (int i = 0; i < maxLen; i++) {

            if (i < m) {
                if (aSet.contains(a[i]) && bSet.contains(a[i])) {
                    intersectionSet.add(a[i]);
                }
            }

            if (i < n) {
                if (aSet.contains(b[i]) && bSet.contains(b[i])) {
                    intersectionSet.add(b[i]);
                }
            }
        }

        //output
        System.out.println("No of intersection element: " + intersectionSet.size() + " elements: " + intersectionSet);
    }

    public void rotateArrayByK_BruteForce(int[] arr, int k) {
        //.......................T: O(N^2)
        //https://leetcode.com/problems/rotate-array/
        //actual:
        for (int x : arr) {
            System.out.print(x + " ");
        }
        System.out.println();
        int actualK = k; //just for output purpose
        int n = arr.length;

        while (k-- != 0) {
            int last = arr[n - 1];
            for (int i = n - 1; i >= 1; i--) {
                arr[i] = arr[i - 1];
            }

            arr[0] = last;
        }

        //output:
        System.out.println("Rotate array by " + actualK + " steps output brute force: ");
        for (int x : arr) {
            System.out.print(x + " ");
        }
        System.out.println();
    }

    private void rotateArrayByK_ReverseArray(int[] arr, int start, int end) {

        while (end > start) {
            int temp = arr[start];
            arr[start] = arr[end];
            arr[end] = temp;
            start++;
            end--;
        }
    }

    public void rotateArrayByK(int[] arr, int k) {
        //......................T: O(N)
        //https://leetcode.com/problems/rotate-array/
        //explanation: https://youtu.be/BHr381Guz3Y
        //actual:
        for (int x : arr) {
            System.out.print(x + " ");
        }
        System.out.println();

        int len = arr.length;
        //if k > len, mod it with len so that k always fall under range of len
        k = k % len;
        int start = 0;
        //reverse the array
        rotateArrayByK_ReverseArray(arr, start, len - 1);

        //reverse first [0 to k] elements
        rotateArrayByK_ReverseArray(arr, start, k - 1);

        //reverse remaining [k to len] elements
        rotateArrayByK_ReverseArray(arr, k, len - 1);

        //output:
        System.out.println("Rotate array by " + k + " steps output approach2: ");
        for (int x : arr) {
            System.out.print(x + " ");
        }
        System.out.println();
    }

    public void minimizeDifferenceBetweenHeights(int[] arr, int k) {

        //problem statement: https://practice.geeksforgeeks.org/problems/minimize-the-heights3351/1
        //sol: https://www.geeksforgeeks.org/minimize-the-maximum-difference-between-the-heights/
        int n = arr.length;

        Arrays.sort(arr);

        int ans = arr[n - 1] - arr[0];

        int big = arr[0] + k;
        int small = arr[n - 1] - k;

        int temp = big;
        big = Math.max(big, small);
        small = Math.min(temp, small);

        //all in between a[0] to a[n-1] i.e, a[1] -> a[n-2]
        for (int i = 1; i < n - 1; i++) {

            int subtract = arr[i] - k;
            int add = arr[i] + k;

            // If both subtraction and addition 
            // do not change diff 
            if (subtract >= small || add <= big) {
                continue;
            }

            // Either subtraction causes a smaller 
            // number or addition causes a greater 
            // number. Update small or big using 
            // greedy approach (If big - subtract 
            // causes smaller diff, update small 
            // Else update big) 
            if (big - subtract <= add - small) {
                small = subtract;
            } else {
                big = add;
            }
        }
        //output:
        System.out.println("Min height: " + Math.min(ans, big - small));
    }

    public void bestProfitToBuySellStock(int[] prices) {

        //https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
        int buy = Integer.MAX_VALUE;
        int maxProfit = 0;
        for (int i = 0; i < prices.length; i++) {
            //buy any stock at min price, so find a price < minPrice
            if (prices[i] < buy) {
                buy = prices[i];
            }

            //if any price > minPrice, we can sell that stock to earn profit
            //maxProfit = max(maxProfit, price - minPrice)
            maxProfit = Math.max(maxProfit, prices[i] - buy);
        }

        //output:
        System.out.println("Maximum profit from buying and selling the stocks: " + maxProfit);
    }

    public void countAllPairsInArrayThatSumIsK(int[] arr, int K) {
        /*
         //brute force apprach
         //........................T: O(N^2)
         int pairCount = 0;
         for(int i=0; i<arr.length; i++){
         for(int j=i+1; j<arr.length; j++){
         if(arr[i] + arr[j] == K){
         pairCount++;
         }
         }
         }
        
         System.out.println("Count of pairs whose sum is equal to K: "+pairCount);
         */

        //Time optimised approach
        //https://www.geeksforgeeks.org/count-pairs-with-given-sum/
        //.......................T: O(N)
        //.......................S: O(N)
        Map<Integer, Integer> map = new HashMap<>();
        for (int val : arr) {
            map.put(val, map.getOrDefault(val, 0) + 1);
        }

        int pairCount = 0;
        for (int val : arr) {
            pairCount += map.getOrDefault(K - val, 0);

            if (K - val == val) {
                pairCount--;
            }
        }

        System.out.println("Count of pairs whose sum is equal to K: " + pairCount / 2);
    }

    public boolean checkIfSubarrayWithSum0(int[] arr) {

        int n = arr.length;
        int sum = 0;
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < n; i++) {
            sum += arr[i];

            if (arr[i] == 0 || sum == 0 || set.contains(sum)) {
                return true;
            }
            set.add(sum);
        }
        return false;
    }

    public void bestProfitToBuySellStockCanHoldAtmostOneStock(int[] prices) {
        //https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/
        //https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/discuss/2058747/JAVA-or-Single-loop-solution
        int n = prices.length;
        int min = prices[0];
        int max = prices[0];
        int profit = 0;
        for (int price : prices) {
            if (price > max) {
                max = price;
            } else {
                profit += max - min;
                min = price;
                max = price;
            }
        }
        //output:
        System.out.println("Max profit frm buying selling stock atmost twice: " + (profit + max - min));
    }

    public void bestProfitToBuySellStockAtMostTwice(int[] prices) {

        //https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/
        int n = prices.length;
        int[] maxProfits = new int[n];

        int maxSellingPrice = prices[n - 1];
        //looping all prices and checking if currBuyPrice will
        //give max profit from if sold at maxSellingPrice
        for (int i = n - 2; i >= 0; i--) {
            int currBuyPrice = prices[i];
            if (currBuyPrice > maxSellingPrice) {
                maxSellingPrice = currBuyPrice;
            }
            //choosing the max profits, that we already have seen(maxProfit[i + 1])
            //or if we buy at currBuyPrice and sell at maxSellingPrice
            maxProfits[i] = Math.max(maxProfits[i + 1], maxSellingPrice - currBuyPrice);
        }

        int minBuyPrice = prices[0];
        //looping all prices and checking if currSellPrice will
        //give max profit from if previously bought at minBuyPrice
        for (int i = 1; i < n; i++) {
            int currSellPrice = prices[i];
            if (minBuyPrice > currSellPrice) {
                minBuyPrice = currSellPrice;
            }
            //choosing the max profits, that we already have seen(maxProfit[i - 1])
            //or if we sell at currSellPrice that was bought at minBuyPrice
            //in addition with the previous max profit((bought and sold from above loop)maxProfit[i])
            maxProfits[i] = Math.max(maxProfits[i - 1], maxProfits[i] + (currSellPrice - minBuyPrice));
        }

        //output:
        System.out.println("Max profit frm buying selling stock atmost twice: " + maxProfits[n - 1]);
    }

    public void mergeIntervals_1(int[][] intervals) {

        //.................................T: O(N.LogN)
        //https://leetcode.com/problems/merge-intervals/
        //https://leetcode.com/problems/non-overlapping-intervals/
        System.out.println("approach 1");
        List<int[]> result = new ArrayList<>();

        if (intervals == null || intervals.length == 0) {
            //return result.toArray(new int[0][]);
            return;
        }

        Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
        int prevStart = intervals[0][0];
        int prevEnd = intervals[0][1];

        for (int i = 1; i < intervals.length; i++) {

            int currStart = intervals[i][0];
            int currEnd = intervals[i][1];

            if (currStart <= prevEnd) {
                //overlapping situation, choose the max end time out of two that can cover most intervals
                prevEnd = Math.max(prevEnd, currEnd);
            } else if (currStart > prevEnd) {
                //no overlapp situation
                result.add(new int[]{prevStart, prevEnd});
                prevStart = currStart;
                prevEnd = currEnd;
            }
        }

        //final pair
        result.add(new int[]{prevStart, prevEnd});
        //output:
        int[][] output = result.toArray(new int[result.size()][]);
        for (int[] r : output) {
            System.out.print("[");
            for (int c : r) {
                System.out.print(c + " ");
            }
            System.out.println("]");
            System.out.println();
        }
    }

    public void mergeIntervals_2(int[][] intervals) {

        //.................................T: O(N.LogN)
        //https://leetcode.com/problems/merge-intervals/
        System.out.println("approach 2");
        Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
        LinkedList<int[]> merged = new LinkedList<>();
        for (int[] interval : intervals) {

            if (merged.isEmpty() || merged.getLast()[1] < interval[0]) {
                merged.add(interval);
            } else {
                merged.getLast()[1] = Math.max(merged.getLast()[1], interval[1]);
            }
        }

        //output:
        int[][] output = merged.toArray(new int[merged.size()][]);
        for (int[] r : output) {
            System.out.print("[");
            for (int c : r) {
                System.out.print(c + " ");
            }
            System.out.println("]");
            System.out.println();
        }
    }

    public void minOperationsToMakeArrayPallindrome(int[] arr) {

        //TWO POINTERS
        int n = arr.length;
        int start = 0;
        int end = n - 1;
        int minOpr = 0;
        while (end >= start) {

            if (arr[start] == arr[end]) {
                start++;
                end--;
            } else if (arr[start] > arr[end]) {
                end--;
                arr[end] += arr[end + 1];
                minOpr++;
            } else {
                start++;
                arr[start] += arr[start - 1];
                minOpr++;
            }
        }

        //output:
        System.out.println("Minimum operation to make array pallindrome: " + minOpr);
    }

    public void productOfArrayExcludingElementItself_BruteForce(int[] arr) {

        //.....................T: O(N)
        int n = arr.length;
        int[] result = new int[n];
        int prod = 1;

        for (int i = 0; i < n; i++) {
            prod *= arr[i];
        }

        for (int i = 0; i < n; i++) {
            result[i] = prod / arr[i];
        }

        //output:
        for (int x : result) {
            System.out.print(x + " ");
        }
        System.out.println();
    }

    public void productOfArrayExcludingElementItself_Optimised1(int[] arr) {

        //.....................T: O(N)
        //.....................S: O(N)
        int n = arr.length;
        int[] result = new int[n];
        int[] leftProd = new int[n];
        int[] rightProd = new int[n];

        leftProd[0] = 1;
        rightProd[n - 1] = 1;

        for (int i = 1; i < n; i++) {
            leftProd[i] = leftProd[i - 1] * arr[i - 1];
        }

        for (int i = n - 2; i >= 0; i--) {
            rightProd[i] = rightProd[i + 1] * arr[i + 1];
        }

        for (int i = 0; i < n; i++) {
            result[i] = leftProd[i] * rightProd[i];
        }

        //output:
        for (int x : result) {
            System.out.print(x + " ");
        }
        System.out.println();
    }

    public void productOfArrayExcludingElementItself_Optimised2(int[] arr) {

        //.....................T: O(N)
        //.....................S: O(1) //result[] is needed to save output 
        int n = arr.length;
        int[] result = new int[n];

        result[0] = 1;
        for (int i = 1; i < n; i++) {
            result[i] = result[i - 1] * arr[i - 1];
        }

        int right = 1;
        for (int i = n - 1; i >= 0; i--) {
            result[i] = result[i] * right;
            right *= arr[i];
        }

        //output:
        for (int x : result) {
            System.out.print(x + " ");
        }
        System.out.println();
    }

    public void maximumOfAllSubArrayOfSizeK(int[] arr, int K) {
        //https://leetcode.com/problems/sliding-window-maximum/
        //https://www.geeksforgeeks.org/sliding-window-maximum-maximum-of-all-subarrays-of-size-k/
        List<Integer> result = new ArrayList<>();
        int n = arr.length;
        int index;
        Deque<Integer> queue = new LinkedList<>();
        for (index = 0; index < K; index++) {
            //check curr element >= last added element
            while (!queue.isEmpty() && arr[queue.peekLast()] <= arr[index]) {
                queue.removeLast();
            }
            queue.addLast(index);
        }

        for (; index < n; index++) {

            result.add(arr[queue.peekFirst()]);

            //index => endIndex, q.peekFirst() = startIndex
            //endIndex - startIndex >= window(K) then maintain window size
            while (!queue.isEmpty() && index - queue.peekFirst() >= K) {
                queue.removeFirst();
            }

            while (!queue.isEmpty() && arr[queue.peekLast()] <= arr[index]) {
                queue.removeLast();
            }
            queue.addLast(index);
        }
        result.add(arr[queue.peekFirst()]);

        //output:
        System.out.println("Max of all subarrays of size K: " + result);
    }

    public void averageWaitingTime(int[][] customers) {
        //https://leetcode.com/problems/average-waiting-time
        double n = customers.length;
        double timeAtPrevOrderEnd = 0;
        double totalTime = 0;
        for (int[] order : customers) {
            int arrivalTime = order[0];
            int prepTime = order[1];
            timeAtPrevOrderEnd = (timeAtPrevOrderEnd > arrivalTime ? timeAtPrevOrderEnd : arrivalTime)
                    + prepTime;
            totalTime += (timeAtPrevOrderEnd - arrivalTime);
        }        //output:
        System.out.println("Average waitig time: " + (totalTime / n));
    }

    public void threeSum(int[] arr) {

        //https://leetcode.com/problems/3sum/
        //https://www.geeksforgeeks.org/java-program-to-find-all-triplets-with-zero-sum/
        //explanation: https://youtu.be/qJSPYnS35SE
        int n = arr.length;
        Arrays.sort(arr);
        List<List<Integer>> result = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            //if the current arr element is not same as prev 
            if (i == 0 || (i > 0 && arr[i] != arr[i - 1])) {

                int start = i + 1;
                int end = n - 1;
                int sum = 0 - arr[i];

                while (end > start) {

                    if (arr[start] + arr[end] == sum) {
                        result.add(Arrays.asList(arr[i], arr[start], arr[end]));

                        //before updating start and end
                        //check wheather there is any duplicates are there or not
                        //if there are duplactes update start and end until we reach the last
                        //duplicated element
                        while (end > start && arr[start] == arr[start + 1]) {
                            start++;
                        }

                        while (end > start && arr[end] == arr[end - 1]) {
                            end--;
                        }

                        //update start and end to the last duplicated element
                        start++;
                        end--;

                    } else if (arr[start] + arr[end] < sum) {
                        start++;
                    } else {
                        end--;
                    }
                }
            }
        }

        //output:
        System.out.println("All the triplets that sum to 0: " + result);
    }

    public int minimumTimeDifference(List<String> timePoints) {

        //problem statement: https://leetcode.com/problems/minimum-time-difference/
        //explanation: https://youtu.be/c5ecNf7JM1Q
        //1 hour = 60 min
        //24 hour = 24 * 60 min == 1440 min
        boolean[] everyMinute = new boolean[24 * 60]; //[1440]

        //convert the string "HH:MM" time format to minute only 
        for (String time : timePoints) {
            String[] timeSplit = time.split(":");
            int hour = Integer.parseInt(timeSplit[0]);
            int minute = Integer.parseInt(timeSplit[1]);

            //total minute in an hour = hour * 60 + minute
            //that total minute is minute based index in everyMinute
            int minuteIndex = (hour * 60) + minute;
            //if a minute is already seen and that same timePoint comes again
            //the diff b/w these two timePoints is minimum
            //ex: "10:02", "10:02"
            if (everyMinute[minuteIndex]) {
                return 0;
            }
            everyMinute[minuteIndex] = true;
        }

        int firstTime = -1;
        int prevTime = -1;
        int minDiff = Integer.MAX_VALUE;

        for (int i = 0; i < 1440; i++) {

            if (everyMinute[i]) {

                if (firstTime == -1) {

                    firstTime = i;
                    prevTime = i;
                } else {

                    minDiff = Math.min(minDiff, Math.min(
                            i - prevTime, //clockwise dir
                            1440 - i + prevTime //anti clockwise dir
                    ));

                    prevTime = i;
                }
            }
        }

        minDiff = Math.min(minDiff, Math.min(
                prevTime - firstTime, //clockwise dir
                1440 - prevTime + firstTime //anti clockwise dir
        ));

        return minDiff;
    }

    public void contigousArrayWithEqualZeroAndOne(int[] arr) {

        //https://leetcode.com/problems/contiguous-array/
        Map<Integer, Integer> prefixSumIndexes = new HashMap<>();
        prefixSumIndexes.put(0, -1);
        int prefixSum = 0;
        int maxLen = 0;

        for (int i = 0; i < arr.length; i++) {

            prefixSum += arr[i] == 1 ? 1 : -1;

            if (prefixSumIndexes.containsKey(prefixSum)) {
                maxLen = Math.max(maxLen, i - prefixSumIndexes.get(prefixSum));
            } else {
                prefixSumIndexes.put(prefixSum, i);
            }
        }
        //output:
        System.out.println("Max length: " + maxLen);
    }

    public void maxSumPathInTwoSortedArrays(int[] arr1, int[] arr2) {

        //https://www.geeksforgeeks.org/maximum-sum-path-across-two-arrays/
        int m = arr1.length;
        int n = arr2.length;

        int result = 0;
        int arrSum1 = 0;
        int arrSum2 = 0;

        int i = 0; // for arr1
        int j = 0; // for arr2

        while (i < m && j < n) {

            if (arr1[i] < arr2[j]) {
                arrSum1 += arr1[i++];
            } else if (arr1[i] > arr2[j]) {
                arrSum2 += arr2[j++];
            } else {
                //common point
                result += Math.max(arrSum1, arrSum2);

                arrSum1 = 0;
                arrSum2 = 0;

                int temp = i;
                while (i < m && arr1[i] == arr2[j]) {
                    arrSum1 += arr1[i++];
                }

                while (j < n && arr1[temp] == arr2[j]) {
                    arrSum2 += arr2[j++];
                }

                result += Math.max(arrSum1, arrSum2);

                arrSum1 = 0;
                arrSum2 = 0;
            }
        }

        while (i < m) {
            arrSum1 += arr1[i++];
        }

        while (j < n) {
            arrSum2 += arr2[j++];
        }

        result += Math.max(arrSum1, arrSum2);

        //output:
        System.out.println("Max path sum: " + result);
    }

    public void asteroidCollision(int[] asteroids) {

        //https://leetcode.com/problems/asteroid-collision/
        //explanation: https://youtu.be/6GGTBM7mwfs

        /*
         cond when two collide
         peek = -ve, incoming = -ve = left, left dir no collision //1 if cond in while()
         peek = -ve, incoming = +ve = left, right dir no collision //1 if cond in for()
         peek = +ve, incoming = +ve = right, right dir no collision //1 if cond in for()
         peek = +ve, incoming = -ve = right, left dir will collision
         if(abs(incoming) > peek) all peek will be destroyed and incoming will remain in sack //last else cond   
         if(abs(incoming) < peek) incoming will be destroyed and stack remain same //3 else if cond
         if(abs(incoming) == peek) both will be destroyed and stack need to pop out peek value //2 else if cond
        
         */
        Stack<Integer> stack = new Stack<>();

        for (int stone : asteroids) {
            if (stack.isEmpty() || stone > 0) {
                stack.push(stone);
            } else {
                while (true) {
                    int prevStone = stack.peek();
                    if (prevStone < 0) {
                        //prevStone = -ve, stone = -ve => left, left
                        //prevStone = -ve, stone = +ve => left, right
                        //both cases not collision will happen and we
                        //can add stone to our stack
                        stack.push(stone);
                        break;
                    } else if (prevStone == -stone) {
                        //prevStone = +ve, stone = -ve => 
                        //prevStone == abs(stone) => right, left & size same
                        stack.pop();
                        break;
                    } else if (prevStone > -stone) {
                        //prevStone = +ve, stone = -ve =>
                        //prevStone > abs(stone) => right, left, stone get destroyed
                        break;
                    } else {
                        //prevStone = +ve, stone = -ve => right, left
                        //but abs(stone) > prevStone then all prevStone under this situation will get destroyed
                        stack.pop();
                        if (stack.isEmpty()) {
                            //once all such prevStone get destroyed, add stone
                            stack.push(stone);
                            break;
                        }
                    }
                }
            }
        }

        //output:
        //int[] output = stack.stream().mapToInt(stone -> stone).toArray();
        int[] output = new int[stack.size()];
        int index = stack.size() - 1;
        while (!stack.isEmpty()) {
            output[index--] = stack.pop();
        }

        for (int x : output) {
            System.out.print(x + " ");
        }

        System.out.println();
    }

    public void jumpGame(int[] nums) {

        //https://leetcode.com/problems/jump-game/
        //Explanation: https://youtu.be/muDPTDrpS28
        int reachablePoint = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i > reachablePoint) {
                System.out.println("Can we reach the end of nums array from 0th index: NO");
                return;
            }
            reachablePoint = Math.max(reachablePoint, i + nums[i]);
        }

        System.out.println("can we reach the end of nums array from 0th index: YES");
    }

    public void jumpGameThree(int[] nums, int startIndex) {
        //https://leetcode.com/problems/jump-game-iii/
        int n = nums.length;
        boolean[] visited = new boolean[n];
        Queue<Integer> queue = new LinkedList<>();
        queue.add(startIndex);

        while (!queue.isEmpty()) {
            int currIndex = queue.poll();

            if (visited[currIndex] == true) {
                continue;
            }
            visited[currIndex] = true;
            if (nums[currIndex] == 0) {
                System.out.println("Can we reach where nums value is 0 from startIndex index: YES");
                return;
            }

            int leftIndex = currIndex - nums[currIndex];
            int rightIndex = currIndex + nums[currIndex];

            if (leftIndex >= 0) {
                queue.add(leftIndex);
            }
            if (rightIndex < n) {
                queue.add(rightIndex);
            }
        }
        System.out.println("Can we reach where nums value is 0 from startIndex index: NO");
    }

    public int jumpGameFour(int[] nums) {
        //https://leetcode.com/problems/jump-game-iv/
        //exlanation: https://youtu.be/XgP3w7Txvlc
        //BFS approach
        int n = nums.length;

        Map<Integer, List<Integer>> valIndexMap = new HashMap<>();
        for (int i = 0; i < n; i++) {
            valIndexMap.putIfAbsent(nums[i], new ArrayList<>());
            valIndexMap.get(nums[i]).add(i);
        }

        int src = 0;
        Set<Integer> visited = new HashSet<>();
        Queue<Integer> queue = new LinkedList<>();
        queue.add(src);

        int steps = 0;

        while (!queue.isEmpty()) {

            int size = queue.size();

            for (int i = 0; i < size; i++) {

                int currIndex = queue.poll();

                visited.add(currIndex);

                if (currIndex == n - 1) {
                    return steps;
                }

                //i + 1 where: i + 1 < arr.length.
                //i - 1 where: i - 1 >= 0.
                int forward = currIndex + 1;
                int backward = currIndex - 1;

                //j where: arr[i] == arr[j] and i != j.
                List<Integer> sameValIndexJumps = valIndexMap.get(nums[currIndex]);

                //adding in sameValIndexJumps because they will eventually be checked in
                //below for loop if we need to add it in queue
                sameValIndexJumps.add(forward);
                sameValIndexJumps.add(backward);

                for (int jIndex : sameValIndexJumps) {
                    //jIndex || forward || backward indexes is not in arr range
                    // OR these indexes are already visited then skip
                    if (!(jIndex >= 0 && jIndex < n) || visited.contains(jIndex)) {
                        continue;
                    }
                    queue.add(jIndex);
                }
                //clearing because we have already added all the jIndex in queue
                //if at any other point we see arr[i] == arr[j] and i != j.
                //we dont need to redundently add them again and again.
                sameValIndexJumps.clear();
            }
            steps++;
        }
        return -1;
    }

    public boolean jumpGameSeven(String str, int minJump, int maxJump) {
        //https://leetcode.com/problems/jump-game-vii/
        //explanation: https://youtu.be/v1HpZUnQ4Yo
        //BFS approach
        int n = str.length();
        char[] arr = str.toCharArray();

        int src = 0;
        Queue<Integer> queue = new LinkedList<>();
        queue.add(src);

        int farthtestReached = 0;

        while (!queue.isEmpty()) {

            int currIndex = queue.poll();
            int start = Math.max(currIndex + minJump, farthtestReached + 1);
            //range: from currIndex we can move to maxJump + 1 (+1 is to include maxJump also)
            //it can be possible that currIndex + maxJump + 1 is beyond arr[] length
            //so choosing the min of two ranges
            int range = Math.min(currIndex + maxJump + 1, n);
            for (int next = start; next < range; next++) {
                if (arr[next] == '0') {
                    if (next == n - 1) {
                        return true;
                    }
                    queue.add(next);
                }
            }
            farthtestReached = currIndex + maxJump;
        }
        return false;
    }

    public boolean frogJump(int[] stones) {
        //https://leetcode.com/problems/frog-jump/
        class Jump {

            int currStone;
            int jump;

            public Jump(int currStone, int jump) {
                this.currStone = currStone;
                this.jump = jump;
            }

        }

        int n = stones.length;
        int lastStone = stones[n - 1];
        Queue<Jump> queue = new LinkedList<>();
        Set<String> visited = new HashSet<>();
        Set<Integer> stonesSet = new HashSet<>();
        for (int stone : stones) {
            stonesSet.add(stone);
        }

        int currStone = 0;
        int jump = 0;
        queue.add(new Jump(currStone, jump));

        while (!queue.isEmpty()) {
            Jump curr = queue.poll();
            currStone = curr.currStone;
            jump = curr.jump;

            //if we have reached the last stone
            if (currStone == lastStone) {
                return true;
            }

            //if any currStone doesn't exist in the stone list,
            //we can't move there, so continue
            if (!stonesSet.contains(currStone)) {
                continue;
            }
            //from curr jump we can furtehr move to k, k - 1, k + 1 move
            for (int moveK = -1; moveK <= 1; moveK++) {
                //if any currStone and jump previously visited, continue
                if (visited.contains(currStone + "," + (jump + moveK))) {
                    continue;
                }
                queue.add(new Jump(currStone + jump + moveK, jump + moveK));
                visited.add(currStone + "," + (jump + moveK));
            }
        }
        return false;
    }

    public void nextGreaterElement2_CyclicArray(int[] arr) {

        //explanation: https://leetcode.com/problems/next-greater-element-ii/solution/
        //array to be considered as cyclic
        int n = arr.length;
        int[] output = new int[n];
        Stack<Integer> stack = new Stack<>();
        for (int i = 2 * n - 1; i >= 0; i--) {

            while (!stack.isEmpty() && arr[stack.peek()] <= arr[i % n]) {
                stack.pop();
            }

            if (stack.isEmpty()) {
                output[i % n] = -1;
            } else {
                output[i % n] = arr[stack.peek()];
            }

            stack.push(i % n);
        }

        //output:
        for (int x : output) {
            System.out.print(x + " ");
        }

        System.out.println();
    }

    public void findMedianInDataStream(int[] stream) {

        //explanantion: https://leetcode.com/problems/find-median-from-data-stream/solution/
        //[HEAP BASED]
        PriorityQueue<Integer> minHeap = new PriorityQueue<>((a, b) -> a.compareTo(b));
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a, b) -> b.compareTo(a));

        for (int num : stream) {

            maxHeap.add(num);
            minHeap.add(maxHeap.poll());

            if (maxHeap.size() < minHeap.size()) {
                maxHeap.add(minHeap.poll());
            }

            double median = maxHeap.size() > minHeap.size()
                    ? maxHeap.peek()
                    : (double) (maxHeap.peek() + minHeap.peek()) * 0.5;

            System.out.println("Median: " + median);
        }
    }

    public void numPairsDivisibleBy60(int[] times) {

        //https://leetcode.com/problems/pairs-of-songs-with-total-durations-divisible-by-60/
        int[] seen = new int[60];
        int result = 0;

        for (int time : times) {
            int hashedTime = time % 60;
            int complement = 60 - hashedTime == 60 || 60 - hashedTime < 0
                    ? 0 : 60 - hashedTime;
            result += seen[complement];
            seen[hashedTime]++;
        }

        //output:
        System.out.println("Total pair: " + result);
    }

    private int mergeSort(int[] arr, int start, int mid, int end) {

        int[] left = Arrays.copyOfRange(arr, start, mid + 1); //start to mid
        int[] right = Arrays.copyOfRange(arr, mid + 1, end + 1); //mid + 1 to end
//        IntStream.of(left).boxed().forEach(x -> System.out.println(x));
//        System.out.println();
//        IntStream.of(right).boxed().forEach(x -> System.out.println(x));
        int i = 0;
        int j = 0;
        int k = start;
        int swaps = 0;
        while (i < left.length && j < right.length) {

            if (left[i] <= right[j]) {
                arr[k++] = left[i++];
            } else {
                arr[k++] = right[j++];
                swaps += (mid + 1) - (start + i);
            }
        }

        while (i < left.length) {
            arr[k++] = left[i++];
        }

        while (j < right.length) {
            arr[k++] = right[j++];
        }

        return swaps;
    }

    private int divideAndMerge(int[] arr, int start, int end) {
        int count = 0;
        if (end > start) {

            int mid = start + (end - start) / 2;
            count += divideAndMerge(arr, start, mid);
            count += divideAndMerge(arr, mid + 1, end);
            count += mergeSort(arr, start, mid, end);
        }
        return count;
    }

    public void countInversion(int[] arr) {

        //..........................T: O(n log n)
        //..........................S: O(N) temp(left, right arrays)
        //https://www.geeksforgeeks.org/counting-inversions/
        int n = arr.length;
        int countInversion = divideAndMerge(arr, 0, n - 1);
        System.out.println("Count inversion: " + countInversion);
        //array is also got sorted
//        for(int x: arr){
//            System.out.print(x+" ");
//        }
    }

    public void minimumWindowSubarrayForTargetSumK(int[] arr, int K) {

        //SLIDING WINDOW BASIC
        //explanation: https://youtu.be/jKF9AcyBZ6E
        int start = 0;
        int end = 0;
        int sum = 0;
        int win = Integer.MAX_VALUE;
        int index = 0;
        while (end < arr.length) {

            sum += arr[end];

            while (sum >= K) {
                win = Math.min(win, end - start + 1);
                index = start;
                sum -= arr[start];
                start++;
            }
            end++;
        }

        //output:
        System.out.println("Minimum length of subarrays whose sum (>= K): " + (win >= Integer.MAX_VALUE ? 0 : win));
        System.out.println("array element:");
        if (win != Integer.MAX_VALUE) {
            for (int i = 0; i < win; i++) {
                System.out.print(arr[index + i] + " ");
            }
        }
        System.out.println();
    }

    public void flipMZerosFindMaxLengthOfConsecutiveOnes(int[] arr, int M) {
        //https://leetcode.com/problems/max-consecutive-ones-iii/
        int start = 0;
        int end = 0;
        int zeroCount = 0;
        int bestWin = 0;
        int bestStart = 0;

        while (end < arr.length) {

            if (zeroCount <= M) {
                if (arr[end] == 0) {
                    zeroCount++;
                }
                end++;
            }

            if (zeroCount > M) {
                if (arr[start] == 0) {
                    zeroCount--;
                }
                start++;
            }

            if (end - start > bestWin && zeroCount <= M) {
                bestWin = end - start;
                bestStart = start;
            }
        }

        //output
        System.out.println("Length of consecutive ones after flipping M zeros: " + bestWin);
        System.out.println("Indexs of zeros to flip");
        for (int i = 0; i < bestWin; i++) {
            if (arr[bestStart + i] == 0) {
                System.out.print((bestStart + i) + " ");
            }
        }
        System.out.println();
    }

    public void firstNegativeNumberInWindowKFromArray(int[] arr, int K) {

        //SLIDING WINDOW
        //MODIFICATION of maximum number in window of K size (maximumOfAllSubArrayOfSizeK())
        Deque<Integer> q = new LinkedList<>();
        List<Integer> result = new ArrayList<>();
        int i = 0;
        for (; i < K; i++) {
            if (arr[i] < 0) {
                q.addLast(i);
            }
        }

        while (i < arr.length) {

            if (!q.isEmpty()) {
                result.add(arr[q.peekFirst()]);
            } else {
                result.add(0);
            }

            while (!q.isEmpty() && q.peekFirst() <= i - K) {
                q.removeFirst();
            }

            if (arr[i] < 0) {
                q.addLast(i);
            }
            i++;
        }

        if (!q.isEmpty()) {
            result.add(arr[q.peekFirst()]);
        } else {
            result.add(0);
        }

        //output
        System.out.println("First negative number in window of K: " + result);
    }

    public boolean handOfStraight(int[] arr, int W) {

        //https://leetcode.com/problems/hand-of-straights/
        //explanation: https://leetcode.com/problems/hand-of-straights/solution/
        int n = arr.length;
        //if we can not make a group of size (W) out of n length arr return false
        if (n % W != 0) {
            return false;
        }

        TreeMap<Integer, Integer> map = new TreeMap<>(); //TreeMap is important
        for (int c : arr) {
            map.put(c, map.getOrDefault(c, 0) + 1);
        }

        while (map.size() > 0) {
            //treemap.firstKey() returns first lowest key frm the treemap
            int firstLowestKey = map.firstKey();
            for (int c = firstLowestKey; c < firstLowestKey + W; c++) {
                if (!map.containsKey(c)) {
                    return false;
                }

                map.put(c, map.get(c) - 1);
                if (map.get(c) <= 0) {
                    map.remove(c);
                }
            }
        }

        return true;
    }

    public void sortedSquaresOfSortedArray_1(int[] arr) {

        //..........................T: O(N)
        //..........................S: O(N)
        //https://leetcode.com/problems/squares-of-a-sorted-array/
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for (int x : arr) {
            minHeap.add(x * x);
        }

        int[] result = new int[arr.length];
        int index = 0;
        while (!minHeap.isEmpty()) {
            result[index++] = minHeap.poll();
        }

        //output:
        for (int x : result) {
            System.out.print(x + " ");
        }
        System.out.println();
    }

    public void sortedSquaresOfSortedArray_2(int[] arr) {

        //OPTIMISED
        //..........................T: O(N)
        //..........................S: O(1)
        //https://leetcode.com/problems/squares-of-a-sorted-array/
        int n = arr.length;

        int firstPositiveIndex = 0;
        while (firstPositiveIndex < n && arr[firstPositiveIndex] < 0) {
            firstPositiveIndex++;
        }

        int lastNegativeIndex = firstPositiveIndex - 1;

        int[] result = new int[n];
        int index = 0;
        while (lastNegativeIndex >= 0 && firstPositiveIndex < n) {

            int negativeSqr = arr[lastNegativeIndex] * arr[lastNegativeIndex];
            int positiveSqr = arr[firstPositiveIndex] * arr[firstPositiveIndex];

            if (negativeSqr < positiveSqr) {
                result[index++] = negativeSqr;
                lastNegativeIndex--;
            } else {
                result[index++] = positiveSqr;
                firstPositiveIndex++;
            }
        }

        while (lastNegativeIndex >= 0) {

            int negativeSqr = arr[lastNegativeIndex] * arr[lastNegativeIndex];
            result[index++] = negativeSqr;
            lastNegativeIndex--;
        }

        while (firstPositiveIndex < n) {

            int positiveSqr = arr[firstPositiveIndex] * arr[firstPositiveIndex];
            result[index++] = positiveSqr;
            firstPositiveIndex++;
        }

        //output:
        for (int x : result) {
            System.out.print(x + " ");
        }
        System.out.println();
    }

    public void moveZeroesToEnd(int[] arr) {

        //https://leetcode.com/problems/move-zeroes/
        int index = 0;
        //move all non-zero elements to index location and update index ptr
        for (int x : arr) {
            if (x != 0) {
                arr[index++] = x;
            }
        }

        //from curr index location fill array with 0
        for (; index < arr.length; index++) {
            arr[index] = 0;
        }

        //output:
        for (int x : arr) {
            System.out.print(x + " ");
        }
        System.out.println();
    }

    public void containerWithMostWater(int[] height) {

        //.......................T: O(N)
        //https://leetcode.com/problems/container-with-most-water/
        //explanation: https://youtu.be/6PrIRPpTI9Q
        //maxArea b/w two maxima
        //ex: [1,8,6,2,5,4,8,3,7]
        //min of two maxima(height[]) * dist b/w two maxima(index dist)
        //first 2 maxima = 8(ind 1) & 8(ind 6) area = min(8, 8) * (6 - 1) = 8 * 5 = 40
        //other 2 maxima = 8(ind 1) & 7(ind 8) area = min(8, 7) * (8 - 1) = 7 * 7 = 49 (MAX_AREA)
        int n = height.length;
        int start = 0;
        int end = n - 1;
        int maxArea = 0;
        int area = 0;

        while (end > start) {
            //water can be contained within two heights where one of height is
            //lesser than the other one and max height upto water can be filled without
            //overflowing is min(height[start], height[end])
            if (height[start] < height[end]) {
                //calculations: area = length * breadth
                area = height[start] * (end - start);
                maxArea = Math.max(maxArea, area);
                start++;
            } else {
                area = height[end] * (end - start);
                maxArea = Math.max(maxArea, area);
                end--;
            }
        }

        //output:
        System.out.println("Most water can be contained with area of: " + maxArea);
    }

    public int smallestMissingPositiveNumber(int[] arr) {

        //........................T: O(N)
        //explanation: https://www.youtube.com/watch?v=-lfHWWMmXXM
        if (arr == null || arr.length == 0) {
            return 0;
        }

        int n = arr.length;
        int i = 0;

        for (; i < n; i++) {
            int index = arr[i] - 1;
            while (arr[i] > 0 && arr[i] < n
                    && arr[i] != arr[index]) {

                //swap
                int temp = arr[i];
                arr[i] = arr[index];
                arr[index] = temp;
                index = arr[i] - 1; //update
            }
        }

        for (i = 0; i < n; i++) {
            if (arr[i] != i + 1) {
                return i + 1;
            }
        }

        return n + 1;
    }

    public void subarraySumEqualsK(int[] arr, int K) {
        //https://leetcode.com/problems/subarray-sum-equals-k/
        //explanation: https://youtu.be/BrWp4gf10fs
        int result = 0;
        int prefixSum = 0;
        Map<Integer, Integer> prefixSumCounter = new HashMap<>();
        prefixSumCounter.put(0, 1); //default sum -> occurence

        for (int element : arr) {
            prefixSum += element;
            result += prefixSumCounter.getOrDefault(prefixSum - K, 0);
            prefixSumCounter.put(prefixSum, prefixSumCounter.getOrDefault(prefixSum, 0) + 1);
        }

        //output:
        System.out.println("Subarrays whose sum equals to K: " + result);
    }

    public void subarrayProductLessThanK(int[] arr, int K) {

        //SLIDING WINDOW
        //https://leetcode.com/problems/subarray-product-less-than-k/
        //explanation: https://youtu.be/SxtxCSfSGlo
        /*
        
         ex: [10, 5, 2, 6]
         The 8 subarrays that have product less than 100 are: 
         [10], [5], [2], [6], [10, 5], [5, 2], [2, 6], [5, 2, 6].
         Note that [10, 5, 2] is not included as the product of 100 
         is not strictly less than k.
        
         */
        int result = 0;
        int prod = 1;
        int n = arr.length;
        int start = 0;
        int end = 0;

        while (end < n) {

            prod *= arr[end];

            while (prod >= K) {
                prod /= arr[start];
                start++;
            }

            if (prod < K) {
                result += end - start + 1;
            }
            end++;
        }

        //output:
        System.out.println("Total subarrays with product less than K: " + result);
    }

    public boolean globalAndLocalInversionCountAreEqual(int[] arr) {

        //THIS QUESTION IS NOT SIMILAR TO COUNT INVERSION, follow link for question desc
        //https://leetcode.com/problems/global-and-local-inversions
        //explanantion: https://youtu.be/vFH3zrUbvD4
        /*
        
         The number of (global) inversions is the number of i < j with 0 <= i < j < N and A[i] > A[j].

         The number of local inversions is the number of i with 0 <= i < N and A[i] > A[i+1].
        
         ex: arr = 1, 0, 2
         global = 1, 0 as index 0 < 1 and arr[0] > arr[1]
         local = 1, 0 as index 0 < N and arr[0] > arr[0 + 1]
         so total global inversion  == total local inversion
        
         */
        int n = arr.length;
        int max = -1;
        for (int i = 0; i < n - 2; i++) {
            max = Math.max(max, arr[i]);
            if (max > arr[i + 2]) {
                return false;
            }
        }
        return true;
    }

    public int longestConsecutiveSequence(int[] arr) {
        //https://leetcode.com/problems/longest-consecutive-sequence/
        int N = arr.length;

        if (N == 1) {
            return 1;
        }

        Set<Integer> set = new HashSet<>();
        for (int val : arr) {
            set.add(val);
        }

        int maxLen = 0;
        int currLen;
        for (int val : arr) {
            if (set.contains(val - 1)) {
                continue;
            }
            currLen = 0;
            int nextNum = val;
            while (set.contains(nextNum)) {
                set.remove(nextNum);
                nextNum++;
                currLen++;
            }
            maxLen = Math.max(maxLen, currLen);
        }
        return maxLen;
    }

    public void maxDifferenceOfIndexes(int[] arr) {

        //........................T: O(N)
        //........................S: O(N)
        //OPTIMISED
        //https://www.geeksforgeeks.org/given-an-array-arr-find-the-maximum-j-i-such-that-arrj-arri/
        /*
         The task is to find the maximum of j - i subjected to the constraint of A[i] <= A[j].
         */
        int n = arr.length;
        int i;
        int j;
        int maxDiff = -1;
        int[] leftMin = new int[n];
        int[] rightMax = new int[n];

        leftMin[0] = arr[0];
        for (i = 1; i < n; i++) {
            leftMin[i] = Math.min(arr[i], leftMin[i - 1]);
        }

        rightMax[n - 1] = arr[n - 1];
        for (i = n - 2; i >= 0; i--) {
            rightMax[i] = Math.max(arr[i], rightMax[i + 1]);
        }

        i = 0;
        j = 0;

        while (i < n && j < n) {
            if (leftMin[i] <= rightMax[j]) {
                maxDiff = Math.max(maxDiff, j - i);
                j++;
            } else {
                i++;
            }
        }

        //output
        System.out.println("Mmax diff: " + maxDiff);
    }

    public void rearrangeArrayElements(int[] arr) {

        //OPTIMISED
        //.................................T: O(N)
        //.................................S: O(1)
        //https://www.geeksforgeeks.org/rearrange-given-array-place/
        /*
         Given an array arr[] of size N where every element is in the range 
         from 0 to n-1. 
         Rearrange the given array so that arr[i] becomes arr[arr[i]].
         */
        int n = arr.length;

        for (int i = 0; i < n; i++) {
            arr[i] += (arr[arr[i]] % n) * n;
        }

        for (int i = 0; i < n; i++) {
            arr[i] /= n;
        }

        //output
        for (int x : arr) {
            System.out.print(x + " ");
        }
        System.out.println();
    }

    public void minimumRangeContainingAtleastOneElementFromKSortedList(int[][] kSortedList) {

        ////............................T: O(N*K*LogK)
        //............................S: O(K)
        //https://www.geeksforgeeks.org/find-smallest-range-containing-elements-from-k-lists/
        //APPROACH SIMILAR TO k sorted list/array
        class Data {

            final int row;
            final int len;
            int col;

            public Data(int row, int col, int len) {
                this.row = row;
                this.len = len;
                this.col = col;
            }
        }

        int K = kSortedList.length;

        int max = Integer.MIN_VALUE;
        int minRange = Integer.MAX_VALUE;
        int minElementRange = -1;
        int maxElementRange = -1;

        PriorityQueue<Data> minHeap = new PriorityQueue<>((a, b) -> {
            return kSortedList[a.row][a.col] - kSortedList[b.row][b.col];
        });

        for (int r = 0; r < K; r++) {
            minHeap.add(new Data(r, 0, kSortedList[r].length));
            max = Math.max(max, kSortedList[r][0]); //max element from all the first element of K sorted list([r][0])
        }

        while (!minHeap.isEmpty()) {

            Data curr = minHeap.poll(); //its minHeap, root is always min

            int min = kSortedList[curr.row][curr.col];

            //find range
            if ((max - min) < minRange) {
                minRange = max - min;
                minElementRange = min;
                maxElementRange = max;
            }

            if (curr.col + 1 < curr.len) {
                curr.col++;
                max = Math.max(max, kSortedList[curr.row][curr.col]); //max(currMax, updated element from the kSortedList)
                minHeap.add(curr);
            } else {
                //if some kTh row reached its max len, break (col == KSortedList[row].length)
                //because we have find range from [0, K]
                break;
            }
        }

        //output
        System.out.println("Min range of elements that is present in all K sorted list: "
                + "[" + minElementRange + ", " + maxElementRange + "]");

    }

    public void tupleWithSameProduct(int[] arr) {

        //https://practice.geeksforgeeks.org/problems/sum-equals-to-sum4006/1 (SAME APPROACH)
        //https://leetcode.com/problems/tuple-with-same-product/
        Map<Integer, Integer> map = new HashMap<>();
        int count = 0;

        for (int i = 0; i < arr.length; i++) {
            for (int j = i + 1; j < arr.length; j++) {
                int mul = arr[i] * arr[j];
                if (map.containsKey(mul)) {
                    count += map.get(mul);
                }
                map.put(mul, map.getOrDefault(mul, 0) + 1);
            }
        }

        //output
        //explanation of multiply with 8 = https://www.tutorialspoint.com/tuple-with-the-same-product-in-cplusplus
        System.out.println("Possible tuple counts: " + (count * 8));
    }

    public void kDiffPairsInArray(int[] arr, int k) {
        //https://leetcode.com/problems/k-diff-pairs-in-an-array/
        //https://leetcode.com/problems/count-number-of-pairs-with-absolute-difference-k/
        int pair = 0;
        Map<Integer, Long> map = IntStream.of(arr).boxed()
                .collect(Collectors.groupingBy(
                        Function.identity(),
                        Collectors.counting()
                ));

        for (int key : map.keySet()) {

            long freq = map.get(key);
            //k > 0 && map.contains(a[x] + k) ex: k = 2 then 1 + k == 3, 3 + k == 5
            //should exists as 2 pairs: (1,3) & (3,5) 
            //k == 0 means a[x] - a[y] = k ==> a[x] = a[y] inorder to make a pair, a[x] or a[y] have freq > 1
            //as pair needs 2 element
            if ((k > 0 && map.containsKey(key + k)) || (k == 0 && freq > 1)) {
                pair++;
            }
        }

        //output
        System.out.println("Count of pair of array element with diff equal to k: " + pair);
    }

    public void leastNumberOfUniqueIntegersLeftAfterKRemoval(int[] arr, int K) {

        //https://leetcode.com/problems/least-number-of-unique-integers-after-k-removals/
        //can't remove K elements if K is more than array's length
        if (K > arr.length) {
            return;
        }
        //get the frequency count of all the array element
        Map<Integer, Long> map = IntStream.of(arr).boxed()
                .collect(Collectors.groupingBy(
                        Function.identity(),
                        Collectors.counting()
                ));

        //arrange all the array element at top having less frequency
        PriorityQueue<Integer> minFreq = new PriorityQueue<>(
                (a, b) -> (int) (map.get(a) - map.get(b))
        );

        //put all the unique integers in the minFreq queue
        //minFreq.addAll(IntStream.of(arr).boxed().collect(Collectors.toSet()));
        minFreq.addAll(map.keySet());

        while (K > 0 && !minFreq.isEmpty()) {

            int curr = minFreq.poll();
            //reduce the freq of curr array element 
            //which is least freq untill K becomes 0
            map.put(curr, map.get(curr) - 1);
            //if freq of curr array element is 0 remove it from map and dont add it in minFreq
            if (map.get(curr) <= 0) {
                map.remove(curr);
            } else {
                //if freq of array element is greater than 0, put it back in minFreq
                minFreq.add(curr);
            }
            K--;
        }

        //output
        System.out.println("Total unique integer after K removals: " + minFreq.size() + " K integers left: " + minFreq);
    }

    private void printAllPermutationOfDistinctIntegerArray_Helper(int[] arr,
            int index,
            List<List<Integer>> res) {

        if (index == arr.length) {
            res.add(Arrays.stream(arr).boxed().collect(Collectors.toList()));
        }

        for (int i = index; i < arr.length; i++) {

            //swap
            int temp = arr[index];
            arr[index] = arr[i];
            arr[i] = temp;

            printAllPermutationOfDistinctIntegerArray_Helper(arr, index + 1, res);

            //swapping back the integers at their orignal places
            temp = arr[index];
            arr[index] = arr[i];
            arr[i] = temp;
        }
    }

    public void printAllPermutationOfDistinctIntegerArray(int[] arr) {

        //https://leetcode.com/problems/permutations/
        //https://leetcode.com/problems/permutations-ii/
        List<List<Integer>> res = new ArrayList<>();
        printAllPermutationOfDistinctIntegerArray_Helper(arr, 0, res);

        //output
        System.out.println("All permutations of distinct integer: " + res);
    }

    private void combinationSum_1_Helper(int[] arr, int currIndex, int target, int currSum,
            List<Integer> currCombination, List<List<Integer>> result) {

        if (currSum == target) {
            result.add(currCombination);
            return;
        }

        if (currIndex >= arr.length || currSum > target) {
            return;
        }

        //2 choices
        //1.) add the curr val and allow the same val to be used again
        //currSum will also include this val
        currCombination.add(arr[currIndex]);
        combinationSum_1_Helper(arr, currIndex, target,
                currSum + arr[currIndex], new ArrayList<>(currCombination), result);

        //2.) skip the curr val and move to new val then currSum will not include val
        currCombination.remove(currCombination.size() - 1);
        combinationSum_1_Helper(arr, currIndex + 1, target, currSum, currCombination, result);

    }

    public void combinationSum_1(int[] arr, int target) {

        //..............................T: O(2^N)
        //APPROACH SIMILAR TO SUBSET SUM EQUAL TO TARGET
        //https://leetcode.com/problems/combination-sum
        //https://leetcode.com/problems/combinations/
        List<List<Integer>> result = new ArrayList<>();
        combinationSum_1_Helper(arr, 0, target, 0, new ArrayList<>(), result);

        //output
        System.out.println("All combinations whose sum is equal to target: " + result);
    }

    private void combinationSum_3_Helper(
            int[] arr, int targetSum, int kNums, int currIndex, int currSum,
            List<Integer> currCombination, List<List<Integer>> result) {

        if (currSum == targetSum && currCombination.size() == kNums) {
            result.add(currCombination);
            return;
        }

        if (currIndex >= arr.length || currSum > targetSum || currCombination.size() > kNums) {
            return;
        }

        //2 choices
        //1.) skip the curr val and move to new val then currSum will not include val
        combinationSum_3_Helper(
                arr, targetSum, kNums, currIndex + 1, currSum,
                new ArrayList<>(currCombination), result);

        //2.) add the curr val and allow the same val to be used again
        //currSum will also include this val
        currCombination.add(arr[currIndex]);
        combinationSum_3_Helper(
                arr, targetSum, kNums, currIndex + 1, currSum + arr[currIndex],
                new ArrayList<>(currCombination), result);

    }

    public void combinationSum_3(int kNums, int targetSum) {
        //..............................T: O(2^N)
        //APPROACH SIMILAR TO SUBSET SUM EQUAL TO TARGET
        //https://leetcode.com/problems/combination-sum-iii/
        List<List<Integer>> result = new ArrayList<>();
        //nums can only have [1 to 9] and combinantions will not contain duplication
        //prepare dummy muns[]
        int[] nums = new int[9];
        for (int i = 0; i < 9; i++) {
            nums[i] = i + 1;
        }
        combinationSum_3_Helper(nums, targetSum, kNums, 0, 0, new ArrayList<>(), result);

        //output
        System.out.println("All combinations whose sum is targetSum and have Knums: " + result);
    }

    private void combinationSum_2_Helper(int[] arr, int index, int target,
            List<Integer> curr, List<List<Integer>> res) {

        if (target == 0) {
            res.add(new ArrayList<>(curr));
            return;
        }
        for (int i = index; i < arr.length; i++) {
            //skip all the duplicate ones
            if (i > index && arr[i] == arr[i - 1]) {
                continue;
            }

            if (arr[i] > target) {
                break;
            }
            //take only those arr elements that are smaller or equal to target
            curr.add(arr[i]);
            combinationSum_2_Helper(arr, i + 1, target - arr[i], curr, res);
            curr.remove(curr.size() - 1);
        }
    }

    public void combinationSum_2(int[] arr, int target) {

        //https://leetcode.com/problems/combination-sum-ii
        List<List<Integer>> res = new ArrayList<>();
        //resulting combination is req in sorted order
        //and by sorting we prevent duplicated combinations
        Arrays.sort(arr);
        combinationSum_2_Helper(arr, 0, target, new ArrayList<>(), res);

        //output
        System.out.println("All combinations whose sum equal to target: ");
        res.stream().forEach(l -> System.out.print(l));
        System.out.println();
    }

    public int shortestUnsortedContigousSubarray(int[] arr) {

        //https://leetcode.com/problems/shortest-unsorted-continuous-subarray
        /*
         arr[] = [2,6,4,8,10,9,15], unsorted contigous subarray = [6,4,8,10,9]
         if [6,4,8,10,9] this subarray is sorted = [4,6,8,9,10] 
         then whole arr is sorted = [2,4,6,8,9,10,15] 
         find the shortest such kind of this subarray...
        
         explanation: clone the arr and sort the cloned arr(sortedClone)
         arr[] = [2,6,4,8,10,9,15]
         sortedClone[] = [2,4,6,8,9,10,15]
         start iterating on both at same time, 
         if arr[i] != sortedClone[i] 
         (index i of (arr = 6 & sortedClone = 4) where subarray should start)
         (index i of (arr = 9 & sortedClone = 10) where subarray should end)
         start is min(i) and end is max(i)
         */
        int n = arr.length;
        int start = Integer.MAX_VALUE;
        int end = Integer.MIN_VALUE;

        int[] sortedClone = arr.clone();
        Arrays.sort(sortedClone);

        for (int i = 0; i < n; i++) {

            if (arr[i] != sortedClone[i]) {
                //min starting index from where sorting should begin
                start = Math.min(start, i);
                //max ending index till there sorting should end
                end = Math.max(end, i);
            }
        }

        if (start == Integer.MAX_VALUE && end == Integer.MIN_VALUE) {
            return 0;
        }
        return end - start + 1;
    }

    public void minimumOperationsToMakeArrayStrictlyIncr(int[] arr) {

        //https://leetcode.com/problems/minimum-operations-to-make-the-array-increasing/
        /*
         arr[] = [1,1,1]
         start from index i = 1 nextMax possible (prevElement + 1, currElement)
         = (arr[i - 1] + 1, arr[i])
         opr += diff to make currElement increasing = nextMax - currElement
         currElement = nextMax
         i = 1
         nextMax = max(1 + 1, 1)
         opr += 0 + (2 - 1) = 1
         arr[1] = nextMax = 2 => [1,2,1]
         i = 2
         nextMax = max(2 + 1, 1) = 3
         opr += 1 + (3 - 1) = 1 + 2 => 3
         arr[2] = nextMax = 3
         [1,2,3]
         */
        int n = arr.length;
        int opr = 0;
        for (int i = 1; i < n; i++) {
            int nextMax = Math.max(arr[i - 1] + 1, arr[i]);
            opr += nextMax - arr[i];
            arr[i] = nextMax;
        }

        //output
        System.out.println("Minimum operations to make array strictly increasing: " + opr);
    }

    public int countNumberOfJumpsToReachTheEnd(int[] arr) {

        //https://www.geeksforgeeks.org/minimum-number-jumps-reach-endset-2on-solution/
        if (arr.length <= 1) {
            return 0;
        }

        if (arr[0] == 0) {
            return -1;
        }

        int n = arr.length;
        int maxReach = arr[0];
        int steps = arr[0];
        int i = 1;
        int count = 1;

        while (i < n) {

            if (i == n - 1) {
                return count;
            }

            maxReach = Math.max(maxReach, i + arr[i]);

            steps--;

            if (steps == 0) {
                count++;
                if (i >= maxReach) {
                    return -1;
                }
                steps = maxReach - i;
            }
            i++;
        }

        return -1;
    }

    public void longestSubarrayOfConsecutiveOnesAfterDeletingOneElement(int[] arr) {

        //https://leetcode.com/problems/longest-subarray-of-1s-after-deleting-one-element/
        //APPROACH is similar to longest consecutive one after M zero flips
        //flipMZerosFindMaxLengthOfConsecutiveOnes()
        /*logic is try to flip only 1 zero and get the max len consecutive ones, 
         now we have to delete one element so just do maxLen - 1 this will be like
         delete that one element to get rest of consecutive ones.
         edge case: when all the elements are 1 and there are no 0s to flip, 
         then maxLen will remain 0 only, if we do maxLen - 1 it will be -ve that 
         means all elements were 1 in arr[] so just do n - 1
         */
        int n = arr.length;
        int m = 1;
        int zeroCount = 0;
        int maxLen = 0;
        int start = 0;
        int end = 0;

        while (end < n) {

            if (zeroCount <= m) {
                if (arr[end] == 0) {
                    zeroCount++;
                }
                end++;
            }

            if (zeroCount > m) {
                if (arr[start] == 0) {
                    zeroCount--;
                }
                start++;
            }

            if (end - start > maxLen && zeroCount == m) {
                maxLen = end - start;
            }
        }

        //output
        System.out.println("Longest consecutive ones after deleting one element: "
                + (maxLen - 1 < 0 ? n - 1 : maxLen - 1));
    }

    public void countSubarrayWithOddSum(int[] arr) {

        //https://leetcode.com/problems/number-of-sub-arrays-with-odd-sum
        //explanation: https://leetcode.com/problems/number-of-sub-arrays-with-odd-sum/discuss/1199776/81-Faster-C%2B%2B-Solution
        /*
         ex: [1,3,5] = [[1],[3],[5],[1,3,5]] == 4 subarray with sum is odd
         */
        int n = arr.length;
        int evenSumPrefix = 1; //default 0 which is even
        int oddSumPrefix = 0;
        int sum = 0;
        int res = 0;
        int mod = 1000000007;

        for (int val : arr) {
            sum += val;
            if (sum % 2 == 0) {
                res = (res + oddSumPrefix) % mod;
                evenSumPrefix++;
            } else {
                res = (res + evenSumPrefix) % mod;
                oddSumPrefix++;
            }
        }
        //output
        System.out.println("Count subarray with odd sum: " + res);
    }

    public void mergeNewInterval(int[][] intervals, int[] newInterval) {

        if (intervals.length == 0) {
//            return new int[][]{newInterval};
            return;
        }

        /*
         1.
         ..........a-----b
         .....................p-----q
         2.
         ................a-----b
         ......p-----q
        
         3.
         ..........a------b
         ......p------q
         min(p,a) & max(q, b)
        
         .........a------b
         ............p-------q
         min(p,a) & max(q, b)
        
         .....a----b
         p--------------q
         min(p,a) & max(q, b)
        
         a--------------b
         ....p-----q
         min(p,a) & max(q, b)
         */
        List<int[]> left = new ArrayList<>();
        List<int[]> right = new ArrayList<>();

        for (int[] interval : intervals) {

            if (interval[1] < newInterval[0]) {
                left.add(interval);
            } else if (interval[0] > newInterval[1]) {
                right.add(interval);
            } else {
                newInterval[0] = Math.min(interval[0], newInterval[0]);
                newInterval[1] = Math.max(interval[1], newInterval[1]);
            }
        }

        int[][] res = new int[left.size() + 1 + right.size()][2];
        int index = 0;
        for (int[] l : left) {
            res[index++] = l;
        }

        res[index++] = newInterval;

        for (int[] r : right) {
            res[index++] = r;
        }

        //output
        for (int[] e : res) {
            System.out.print(e[0] + "-" + e[1] + "\n");
        }
        System.out.println();
    }

    public void duplicateZeroInArray(int[] arr) {

        //..............T: O(N) if all e in arr[] != 0 AND O(N^2) if some e in arr[] == 0 (AMORTIZED)
        //https://leetcode.com/problems/duplicate-zeros/
        //actual
        System.out.println("Actual");
        for (int e : arr) {
            System.out.print(e + " ");
        }
        System.out.println();

        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == 0) {
                for (int j = arr.length - 2; j >= i; j--) {
                    arr[j + 1] = arr[j];
                }
                i++;
            }
        }

        //output
        for (int e : arr) {
            System.out.print(e + " ");
        }
        System.out.println();
    }

    public void arrayNesting(int[] arr) {

        //https://leetcode.com/problems/array-nesting/
        /*
         You are given an integer array nums of length n where nums is a 
         permutation of the numbers in the range [0, n - 1].

         You should build a set s[k] = {nums[k], nums[nums[k]], nums[nums[nums[k]]], ... } 
         subjected to the following rule:

         The first element in s[k] starts with the selection of the element nums[k] of index = k.
         The next element in s[k] should be nums[nums[k]], and then nums[nums[nums[k]]], and so on.
         We stop adding right before a duplicate element occurs in s[k].
         Return the longest length of a set s[k].
         */
        int res = 0;
        for (int i = 0; i < arr.length; i++) {

            if (arr[i] != Integer.MAX_VALUE) {
                int start = arr[i];
                int len = 0;
                while (arr[start] != Integer.MAX_VALUE) {
                    int temp = start;
                    start = arr[start];
                    len++;
                    arr[temp] = Integer.MAX_VALUE;
                }
                res = Math.max(res, len);
            }
        }
        //output
        System.out.println("Longest length: " + res);
    }

    public boolean escapingGhost(int[][] ghosts, int[] target) {

        //https://leetcode.com/problems/escape-the-ghosts/
        //user init coord x, y = 0, 0
        //dist b/w user from target = abs(x - tX) + abs(y - tY)
        int userDist = Math.abs(0 - target[0]) + Math.abs(0 - target[1]);
        for (int[] ghost : ghosts) {
            //dist of ghost from target
            int ghostDistFromTarget = Math.abs(ghost[0] - target[0]) + Math.abs(ghost[1] - target[1]);
            if (ghostDistFromTarget <= userDist) {
                //if the ghost reaches the target before user or at the same time as user reached,
                //ghost will catch you so false
                return false;
            }
        }
        return true;
    }

    public void removeDuplicateInSortedArray2WhereElementCanHaveAtMostTwiceOccur(int[] arr) {

        //https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/
        //actual
        for (int i = 0; i < arr.length; i++) {
            System.out.print(arr[i] + " ");
        }
        System.out.println();

        int n = arr.length;
        int start = 0;
        int end = 0;
        int currVal = arr[end];
        int atMostCounter = 0;
        while (end < n) {
            //count the occurences of the consecutive same value as that of currVal
            if (arr[end] == currVal) {
                atMostCounter++;
            } else {
                //if the arr[end] didn't match the currVal
                //we will update the currVal and start counting occurences of this value
                currVal = arr[end];
                //reseting counter back to 1 as atleast this curr arr[end] occuring 1 time
                atMostCounter = 1;
            }

            //we only need atmost 2 occurences of the value (arr[end])
            if (atMostCounter <= 2) {
                //put value at start index if it is less that or equal to 2
                //all greater occurences will not be taken into account
                arr[start] = arr[end];
                start++;
            }
            end++;
        }

        //output
        for (int i = 0; i < start; i++) {
            System.out.print(arr[i] + " ");
        }
        System.out.println();
    }

    private double nPowerOfX_Helper(double x, int n) {
        if (n == 0) { //pow(x, 0) = 1
            return 1;
        }

        if (n == 1) { // pow(x, 1) = x
            return x;
        }

        if (n % 2 == 0) {
            return nPowerOfX_Helper(x * x, n / 2);
        }
        return x * nPowerOfX_Helper(x * x, n / 2);
    }

    public void nPowerOfX(double x, int n) {

        //..................T: O(N * LogN)
        //https://leetcode.com/problems/powx-n/
        /* n can be +ve or -ve value*/
        double output = n >= 0
                ? nPowerOfX_Helper(x, n)
                : 1.0 / nPowerOfX_Helper(x, Math.abs(n));
        System.out.println("pow(x, n): "
                + output);
    }

    public void subarraySumDivisibleByK(int[] arr, int K) {

        //https://leetcode.com/problems/subarray-sums-divisible-by-k/
        //https://leetcode.com/problems/subarray-sums-divisible-by-k/discuss/1227888/Java-Map-soln.
        //explanation: https://youtu.be/QM0klnvTQzk
        //APPROACH similar to subarraySumEqualsK()
        /*
         ex: arr = [4,5,0,-2,-3,1], k = 5
         There are 7 subarrays with a sum divisible by k = 5:
         [4, 5, 0, -2, -3, 1], [5], [5, 0], [5, 0, -2, -3], [0], [0, -2, -3], [-2, -3]
         */
        Map<Integer, Integer> prefixSumRemainderCounter = new HashMap<>();
        prefixSumRemainderCounter.put(0, 1); // Default: remainder = 0, freq = 1
        int prefixSum = 0;
        int count = 0;

        for (int val : arr) {
            prefixSum += val;
            int remainder = prefixSum % K;
            //handle -ve remainder
            remainder = remainder < 0 ? remainder + K : remainder;
            count += prefixSumRemainderCounter.getOrDefault(remainder, 0);
            prefixSumRemainderCounter.put(remainder,
                    prefixSumRemainderCounter.getOrDefault(remainder, 0) + 1);
        }
        //output
        System.out.println("Total subarrays divisible by K: " + count);
    }

    public void gasStationCompleteCircuit(int[] gas, int[] cost) {

        //https://leetcode.com/problems/gas-station/
        int n = gas.length;
        int maxVal = Integer.MIN_VALUE;
        int maxIndex = n - 1;
        int sum = 0;

        for (int i = n - 1; i >= 0; i--) {

            sum += (gas[i] - cost[i]);
            if (sum > maxVal) {
                maxVal = sum;
                maxIndex = i;
            }
        }

        //output;
        System.out.println("Starting index of gas station to complete the circuit(approach 1): "
                + (sum < 0 ? -1 : maxIndex));
    }

    private void rangeUpdateAndPointQueries_Update(int[] arr, int left, int right, int val) {
        arr[left] += val;
        if (right + 1 < arr.length) {
            arr[right + 1] -= val;
        }
    }

    private int rangeUpdateAndPointQueries_GetQueries(int[] arr, int i) {
        int prefixSum = 0;
        for (int j = 0; j <= i; j++) {
            prefixSum += arr[j];
        }
        return prefixSum;
    }

    public void rangeUpdateAndPointQueries(int[] arr) {

        //https://www.geeksforgeeks.org/binary-indexed-tree-range-updates-point-queries/
        int left = 2;
        int right = 4;
        int val = 2;
        rangeUpdateAndPointQueries_Update(arr, left, right, val);

        //Find the element at Index 4 
        int index = 4;
        System.out.println("Element at index " + index + " is " + rangeUpdateAndPointQueries_GetQueries(arr, index));

        left = 0;
        right = 3;
        val = 4;
        rangeUpdateAndPointQueries_Update(arr, left, right, val);

        //Find the element at Index 3 
        index = 3;
        System.out.println("Element at index " + index + " is " + rangeUpdateAndPointQueries_GetQueries(arr, index));
    }

    private boolean isOutOfBound(int x, int xLen, int y, int yLen, int z, int zLen) {
        return x >= xLen || y >= yLen || z >= zLen;
    }

    public void allCommonElementIn3SortedArray(int[] a, int[] b, int[] c) {
        //...........................T: O(N), where N = MIN(xLen, yLen, zLen)
        /*
         If arrays was given unsorted, then one way to solve this problem is to sort the arrays first
         using optimised sort(ex merge sort) then time complexity will be
         T: O(xLen.Log(xLen)) + O(yLen.Log(yLen)) + O(zLen.Log(zLen)) + O(N), N = MIN(xLen, yLen, zLen)
         total sorting time will be time taken to sort the longest array = MAX(xLen, yLen, zLen)
         so overall order = T: O(N.LogN) = MAX(xLen, yLen, zLen)
         */
        //https://practice.geeksforgeeks.org/problems/common-elements1132/1
        int x = 0;
        int y = 0;
        int z = 0;
        int xLen = a.length;
        int yLen = b.length;
        int zLen = c.length;
        int prev = Integer.MIN_VALUE;
        List<Integer> result = new ArrayList<>();

        while (!isOutOfBound(x, xLen, y, yLen, z, zLen)) {
            if (a[x] == b[y] && b[y] == c[z]) {
                //If duplicate values are not required
                if (prev != a[x]) {
                    prev = a[x];
                    result.add(a[x]);
                }
                //if duplicate values are req
                //result.add(a[x]);
                x++;
                y++;
                z++;
            } else if (a[x] < b[y]) {
                x++;
            } else if (b[y] < c[z]) {
                y++;
            } else {
                z++;
            }
        }
        //output
        System.out.println("All common elements in 3 sorted arrays: " + result);
    }

    public void randomlyRearrangeElementsOfArray(int[] arr) {
        //.............................T: O(N)
        //.............................S: O(1), in-place
        // random{} generates random num between 0 & 1
        // (i + 1) * Math.random(), (i + 1) is length of arr at ith iteration 
        // so if N = 5, ex: 1)iteration: i = N - 1 => 4 => i => 4 => 4
        // (i) * random ranges => 4 * 0 & 4 * 1 => 0 & 4
        // floor will keep the ranges to lower bounds that also matches with array indexes
        int N = arr.length;
        for (int i = N - 1; i > 0; i--) {
            int randomIndex = (int) Math.floor(i * Math.random());
            //swap
            int temp = arr[i];
            arr[i] = arr[randomIndex];
            arr[randomIndex] = temp;
        }
        //output
        for (int element : arr) {
            System.out.print(element + " ");
        }
        System.out.println();
    }

    public int teemoAttackingAshee(int[] timeSeries, int duration) {
        //..............................T: O(N)
        //..............................S: O(1)
        //https://leetcode.com/problems/teemo-attacking
        //https://leetcode.com/problems/teemo-attacking/solution/
        int n = timeSeries.length;
        if (n == 0) {
            return 0;
        }
        int totalTime = 0;
        for (int i = 0; i < n - 1; i++) {
            totalTime += Math.min(timeSeries[i + 1] - timeSeries[i], duration);
        }
        return totalTime + duration;
    }

    public void minOperationToMakeArrayOfSizeNEqual(int n) {
        //https://leetcode.com/problems/minimum-operations-to-make-array-equal/
        //https://leetcode.com/problems/minimum-operations-to-make-array-equal/discuss/2075557/Java-1-Liner-solution
        int operations = n % 2 == 0 ? (n / 2) * (n / 2) : (n / 2) * (n / 2 + 1);
        System.out.println("Min operation to make array equal: " + operations);
    }

    public int findPivotIndex(int[] nums) {
        //https://leetcode.com/problems/find-pivot-index/
        int arrSum = 0;
        int leftPrefixSum = 0;
        for (int val : nums) {
            arrSum += val;
        }
        //ex: [1,7,3,6,5,6] at index  = 3
        //arrSum = 1 + 7 + 3 + 6 + 5 + 6 = 28
        //leftPrefixSum till this is 1 + 7 + 3 = 11
        //leftPrefixSum == arrSum - leftPrefixSum - arr[i]
        //11 == 28 - 11 - 6 ==> 11 == 11 pivot found
        for (int i = 0; i < nums.length; i++) {
            if (leftPrefixSum == arrSum - leftPrefixSum - nums[i]) {
                return i;
            }
            //prefixSum
            leftPrefixSum += nums[i];
        }
        return -1;
    }

    public void maximumSubarraySumWithUniqueElements(int[] arr) {

        //https://leetcode.com/problems/maximum-erasure-value/
        //same as longest substring without repeating character, longestSubstringWithoutRepeatingChar()
        //SLIDING WIINDOW approach
        int n = arr.length;
        Map<Integer, Integer> freq = new HashMap<>();
        int start = 0;
        int end = 0;
        int maxSum = 0;
        int currSum = 0;
        int maxLenSubarray = 0;
        while (end < n) {
            int val = arr[end];
            freq.putIfAbsent(val, 0);
            //in order to have unique elements, its atmost freq can only be 1
            //if a val is coming more than 1 then we will have to slide the window
            if (freq.get(val) < 1) {
                freq.put(val, freq.get(val) + 1);
                currSum += val;
                maxSum = Math.max(maxSum, currSum);
                maxLenSubarray = Math.max(maxLenSubarray, (end - start + 1));
                end++;
            } else {
                int startVal = arr[start++];
                freq.put(startVal, freq.get(startVal) - 1);
                currSum -= startVal;
            }
        }
        //output
        System.out.println("Maximum subarray sum with unique elements: " + maxSum + " subarray length: " + maxLenSubarray);
    }

    public boolean partitionArrayIntoThreePartsWithEqualSum(int[] arr) {
        //https://leetcode.com/problems/partition-array-into-three-parts-with-equal-sum/
        //ex arr[a,b,c,l,m,n,p,q,r,s]
        //let say partition be sum1 = [a,b,c], sum2 = [l,m,n], sum3 = [p,q,r,s]
        //we have to prove sum1 == sum2 == sum3
        //total arrSum = sum
        //sum1 + sum2 + sum3 = arrSum
        //if sum1 == sum2 == sum3 then 3 * sum1 = arrSum ==> sum1 = arrSum / 3
        //out total arrSum should be div by 3
        int sumArr = 0;
        for (int val : arr) {
            sumArr += val;
        }

        if (sumArr % 3 != 0) {
            return false;
        }

        int sumPerPartition = sumArr / 3;
        int times = 0;
        int sum = 0;
        for (int val : arr) {
            sum += val;
            if (sum == sumPerPartition * (times + 1)) {
                System.out.println(sum);
                times++;
                if (times == 3) {
                    return true;
                }
            }
        }
        return false;
    }

    public void topKFrequentElements(int[] arr, int k) {
        //https://leetcode.com/problems/top-k-frequent-elements/
        Map<Integer, Long> freq = Arrays.stream(arr).boxed()
                .collect(Collectors.groupingBy(val -> val, Collectors.counting()));
//                or 
//                .collect(Collectors.groupingBy(
//                        Function.identity(),
//                        Collectors.counting())
//                );

        PriorityQueue<Integer> minHeap = new PriorityQueue<>(
                (val1, val2) -> (int) (freq.get(val1) - freq.get(val2))
        );

        for (int key : freq.keySet()) {
            minHeap.add(key);
            if (minHeap.size() > k) {
                minHeap.poll();
            }
        }
        //output:
        System.out.println("Frequent elements: ");
        //to convert result in minHeap to int[]
        int[] result = minHeap.stream().mapToInt(val -> val).toArray();
        for (int val : result) {
            System.out.print(val + " ");
        }
        System.out.println();
    }

    public void maximumSumCircularSubarray(int[] arr) {
        //https://leetcode.com/problems/maximum-sum-circular-subarray
        //explanation: https://youtu.be/uHsPcy3xUT0
        //ex: [-1,5,5,-2] = maxSumSubArr = [5,5] = 10 by kadens algo
        //ex: [5,-1,-2,5] = maxSumSubArr(circular) = 5]..[5 = 10
        //intutions: 
        //if sum are to be max in circular array i.e around the ends of array in that
        //case there should exists some min sum in between the max sum end and then
        //totalArrSum - -minSumSubArr = maxSumSubArr(circular)
        //ex: [5,-1,-2,5] = maxSumSubArr(circular) = 5]..[5 = 10
        //totalArrSum = 5 + -1 + -2 + 5 = 7
        //minSumSubArr = ]...minSumSubArr...[ => reverseAllSigns(arr[]) =>
        // [-5,1,2,-5] = 1 + 2 = 3 OR - (1 + 2) = -3
        //maxSumSubArr(circular) = 5]..[5 = 10 => totalArrSum - -minSumSubArr = 7 - -3 = 10
        int totalArrSum = Arrays.stream(arr).sum();
        int maxSumSubArr = kadaneAlgorithm(arr);
        //reverseAllSign(arr)
        int minSumSubArr = kadaneAlgorithm(Arrays.stream(arr).map(val -> -val).toArray());
        int maxSumSubArrCircular = maxSumSubArr > 0
                ? Math.max(maxSumSubArr, (totalArrSum - -minSumSubArr))
                //case where arr[] is all negative then smallest element in arr is maxSumSubArr
                : maxSumSubArr;
        //output
        System.out.println("Max sum circular subarray: " + maxSumSubArrCircular);
    }

    public boolean nonDecreasingArrayWithAtmostOneChange(int[] nums) {
        //https://leetcode.com/problems/non-decreasing-array/
        //explanation: https://youtu.be/RegQckCegDk
        int n = nums.length;
        boolean atMostOneChangeHasDone = false;

        for (int i = 0; i < n - 1; i++) {

            //already increasing ex: 1 2 Or 1 1
            if (nums[i] <= nums[i + 1]) {
                continue;
            }

            //we can modify the array with atmost one element
            //if we have previouly done it, we can't do it second time
            //hence false
            if (atMostOneChangeHasDone) {
                return false;
            }

            //if curr ith value and its next value >= its prev value
            //then convert that ith value to next value nums[i + 1]
            //ex: i == 0 cases like [4, 2], other cond cases like [3,2,5]
            if (i == 0 || nums[i - 1] <= nums[i + 1]) {
                //update nums[i] = 2 with nums[i + 1] = 5
                nums[i] = nums[i + 1];
            } else {
                nums[i + 1] = nums[i];
            }
            atMostOneChangeHasDone = true;
        }
        return true;
    }

    public void twoSum_UnsortedArray(int[] nums, int target) {
        //.........................T: O(N)
        //.........................S: O(N)
        //https://leetcode.com/problems/two-sum/
        Map<Integer, Integer> map = new HashMap<>();
        int n = nums.length;
        int[] result = null;
        for (int i = 0; i < n; i++) {

            int diff = target - nums[i];
            if (map.containsKey(diff)) {
                result = new int[]{map.get(diff), i};
                break;
            }
            map.put(nums[i], i);
        }
        //output
        if (result == null) {
            System.out.println("No Two element found that sums equal to target");
        } else {
            System.out.println("Indexes(0-based index) of two element that sums equal to target: "
                    + result[0] + " " + result[1]);
        }
    }

    public void twoSum2_SortedArray(int[] nums, int target) {
        //.........................T: O(N)
        //.........................S: O(1)
        //https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/
        //explanation: https://youtu.be/cQ1Oz4ckceM
        //this problem can also be solved with above twoSum_UnsortedArray()
        int n = nums.length;
        int[] result = null;

        int start = 0;
        int end = n - 1;

        while (end > start) {

            int sum = nums[start] + nums[end];
            if (sum == target) {
                result = new int[]{start + 1, end + 1};
                break;
            } else if (sum > target) {
                //curr sum > target because nums[end] > nums[start] as
                //array is sorted so try to reduce larger num from end
                end--;
            } else {
                //curr sum < target because nums[end] > nums[start] as
                //array is sorted so try to increase larger num from start
                start++;
            }
        }

        //output
        if (result == null) {
            System.out.println("No Two element found that sums equal to target");
        } else {
            System.out.println("Indexes(1-based index) of two element that sums equal to target: "
                    + result[0] + " " + result[1]);
        }
    }

    private void fourSum_twoSumSortedApproach(
            int[] nums, long target, int index, List<Integer> currList, List<List<Integer>> result) {

        int n = nums.length;
        int start = index;
        int end = n - 1;
        while (end > start) {
            long currSum = nums[start] + nums[end];
            if (currSum == target) {
                List<Integer> kSumList = new ArrayList<>(currList);
                kSumList.add(nums[start]);
                kSumList.add(nums[end]);
                result.add(kSumList);
                start++;
                while (start < end && nums[start] == nums[start - 1]) {
                    start++;
                }
            } else if (currSum > target) {
                end--;
            } else {
                start++;
            }
        }
    }

    private void fourSum_CombinationSum2Approach(
            int[] nums, int target, int index, int kSum, List<Integer> currList, List<List<Integer>> result) {
        if (kSum == 2) {
            fourSum_twoSumSortedApproach(nums, target, index, currList, result);
            return;
        }

        for (int i = index; i < nums.length; i++) {
            if (i > index && nums[i] == nums[i - 1]) {
                continue;
            }
            currList.add(nums[i]);
            fourSum_CombinationSum2Approach(
                    nums, target - nums[i], i + 1, kSum - 1, new ArrayList<>(currList), result);
            currList.remove(currList.size() - 1);
        }
    }

    public void fourSum(int[] nums, int target) {
        //https://leetcode.com/problems/4sum/
        //explanation: https://youtu.be/EYeR-_1NRlQ
        //This approach is generic approach for K-sum (4Sum, 5sum...)
        //based on combination_2 & twoSum-Sorted
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        int kSum = 4;
        fourSum_CombinationSum2Approach(nums, target, 0, kSum, new ArrayList<>(), result);
        //output
        System.out.println("Four sum/ Generic KSum: " + result);
    }

    public void maximumLengthOfSubarrayWithPositiveProduct(int[] nums) {
        //https://leetcode.com/problems/maximum-length-of-subarray-with-positive-product/
        int n = nums.length;
        int positiveCount = 0;
        int negativeCount = 0;
        int temp;
        int length = 0;

        for (int val : nums) {

            int anyCase = val < 0 ? -1 : val == 0 ? 0 : 1;
            switch (anyCase) {
                case 0:
                    //reset
                    positiveCount = 0;
                    negativeCount = 0;
                    break;
                case 1:
                    positiveCount++;
                    if (negativeCount != 0) {
                        negativeCount++;
                    }
                    break;
                case -1:
                    temp = positiveCount;
                    positiveCount = negativeCount > 0 ? negativeCount + 1 : 0;
                    negativeCount = temp + 1;
                    break;
            }
            length = Math.max(length, positiveCount);
        }
        //output
        System.out.println("max length of subarray with positive product: " + length);
    }

    public void minimumIntervalToIncludeEachQuery(int[][] intervals, int[] queries) {
        //https://leetcode.com/problems/minimum-interval-to-include-each-query/
        //explanation: https://youtu.be/5hQ5WWW5awQ
        class Node {

            int dist;
            int rightEnd;

            public Node(int dist, int rightEnd) {
                this.dist = dist;
                this.rightEnd = rightEnd;
            }
        }

        int intervalLen = intervals.length;
        int queriesLen = queries.length;
        int[] result = new int[queriesLen];
        //<queries[i], List(i)>
        Map<Integer, List<Integer>> queriesIndex = new HashMap<>();
        for (int i = 0; i < queriesLen; i++) {
            queriesIndex.putIfAbsent(queries[i], new ArrayList<>());
            queriesIndex.get(queries[i]).add(i);
        }

        Arrays.sort(queries);
        Arrays.sort(intervals, (a, b) -> a[0] - b[0]);

        PriorityQueue<Node> minHeapDist = new PriorityQueue<>((n1, n2) -> n1.dist - n2.dist);
        int intervalIndex = 0;

        for (int query : queries) {

            int queryIndex = queriesIndex.get(query).remove(0);
            if (queriesIndex.get(query).size() == 0) {
                queriesIndex.remove(query);
            }

            while (intervalIndex < intervalLen && intervals[intervalIndex][0] <= query) {
                int left = intervals[intervalIndex][0];
                int right = intervals[intervalIndex][1];

                int dist = right - left + 1;
                minHeapDist.add(new Node(dist, right));
                intervalIndex++;
            }

            while (!minHeapDist.isEmpty() && minHeapDist.peek().rightEnd < query) {
                minHeapDist.poll();
            }

            result[queryIndex] = minHeapDist.isEmpty() ? -1 : minHeapDist.peek().dist;
        }
        //output
        for (int i = 0; i < queriesLen; i++) {
            System.out.println("For query : " + queries[i] + " dist : " + result[i]);
        }
    }

    public void candyDistributionToNStudent(int[] ratings) {
        //.........................T: O(N)
        //.........................S: O(N + N), left & right neighbour
        //https://leetcode.com/problems/candy/
        //explanation: https://youtu.be/h6_lIwZYHQw
        /*
         rule:
         You are giving candies to these children subjected to the following requirements:
         1. Each child must have at least one candy.
         2. Children with a higher rating get more candies than their neighbors.
         */
        int n = ratings.length;
        int[] leftNeighbour = new int[n];
        int[] rightNeighbour = new int[n];
        int candiesNeeded = 0;
        //atleast 1 candy is given to each student
        leftNeighbour[0] = 1;
        for (int i = 1; i < n; i++) {
            //if rating of curr ith student is greater than its prev left student
            //that curr student should have 1 candy extra than its prev left student
            //otherwise atleast 1 candy should be given
            leftNeighbour[i] = ratings[i] > ratings[i - 1]
                    ? leftNeighbour[i - 1] + 1
                    : 1;
        }
        //atleast 1 candy is given to each student
        rightNeighbour[n - 1] = 1;
        for (int i = n - 2; i >= 0; i--) {
            //if rating of curr ith student is greater than its next right student
            //that curr student should have 1 candy extra than its next right student
            //otherwise atleast 1 candy should be given
            rightNeighbour[i] = ratings[i] > ratings[i + 1]
                    ? rightNeighbour[i + 1] + 1
                    : 1;
        }

        for (int i = 0; i < n; i++) {
            //add all the max candies req by each ith student
            candiesNeeded += Math.max(leftNeighbour[i], rightNeighbour[i]);
        }
        //output
        System.out.println("Candies required to be distributed: " + candiesNeeded);
    }

    public void candyDistributionToNStudent2(int[] ratings) {
        //.........................T: O(N)
        //.........................S: O(N), candies
        //OPTIMISED
        //https://leetcode.com/problems/candy/
        //https://leetcode.com/problems/candy/solution/
        //explanation: https://youtu.be/h6_lIwZYHQw
        /*
         rule:
         You are giving candies to these children subjected to the following requirements:
         1. Each child must have at least one candy.
         2. Children with a higher rating get more candies than their neighbors.
         */
        int n = ratings.length;
        int[] candies = new int[n];
        //atleast candy is given to each student
        Arrays.fill(candies, 1);
        int candiesNeeded = 0;
        for (int i = 1; i < n; i++) {
            //if rating of curr ith student is greater than its prev left student
            //that curr student should have 1 candy extra than its prev left student
            //otherwise atleast 1 candy should be given
            candies[i] = ratings[i] > ratings[i - 1]
                    ? candies[i - 1] + 1
                    : 1;
        }

        candiesNeeded = candies[n - 1];
        for (int i = n - 2; i >= 0; i--) {
            //if rating of curr ith student is greater than its next right student
            //that curr student should have 1 candy extra than its next right student
            //otherwise atleast 1 candy should be given
            if (ratings[i] > ratings[i + 1]) {
                candies[i] = Math.max(candies[i], candies[i + 1] + 1);
            }
            candiesNeeded += candies[i];
        }
        //output
        System.out.println("Candies required to be distributed: " + candiesNeeded);
    }

    public void findTriangularSumOfArray(int[] nums) {
        //https://leetcode.com/problems/find-triangular-sum-of-an-array/
        int n = nums.length;
        int index = 1;
        while (n > 1) {
            while (index < n) {
                int sum = nums[index - 1] + nums[index];
                nums[index - 1] = sum % 10;
                index++;
            }
            index = 1;
            n--;
        }
        //output
        System.out.println("Triangular sum of array: " + nums[0]);
    }

    public boolean has132Pattern(int[] nums) {
        //https://leetcode.com/problems/132-pattern/
        //explanation: https://youtu.be/q5ANAl8Z458
        /*
         a 132 pattern is a subsequence of three integers 
         nums[i], nums[j] and nums[k] 
         such that i < j < k and nums[i] < nums[k] < nums[j].
         */
        class Pair {

            int numJ;
            int numI;

            public Pair(int numJ, int numI) {
                this.numJ = numJ;
                this.numI = numI;
            }

        }

        int n = nums.length;
        int currMin = nums[0]; //some prev nums[i] pattern
        Stack<Pair> stack = new Stack();

        for (int k = 1; k < n; k++) {

            int numK = nums[k];
            //when the loop break at stack.peek().val greater than currVal that 
            //stack.peek().val = nums[j] as nums[k](here currVal) < nums[j]
            while (!stack.isEmpty() && stack.peek().numJ <= numK) {
                stack.pop();
            }
            //here, stack.peek().prevMin = nums[i]
            //currVal = nums[k]
            //stack.peek().val = nums[j]
            if (!stack.isEmpty() && numK > stack.peek().numI) {
                return true;
            }
            stack.push(new Pair(numK, currMin));
            currMin = Math.min(currMin, numK);
        }
        return false;
    }

    public void amountToPaintTheArea(int[][] areaPoints) {
        //https://leetcode.com/problems/amount-of-new-area-painted-each-day/
        //https://leetcode.com/discuss/interview-question/2072036/Google-or-Onsite-or-banglore-or-May-2022-or-Paint-a-line
        //<start, end>
        Map<Integer, Integer> paintMap = new HashMap<>();
        List<Integer> result = new ArrayList<>();
        for (int[] point : areaPoints) {

            int start = point[0];
            int end = point[1];

            int cost = 0;
            //for each start to end value, paint each unit sequentially
            //and calculate the cost per unit
            //ex: [4, 10] = seq unit cost = start = 4, cost = 1
            //seq unit cost = start = 5, cost = 2
            //seq unit cost = start = 6, cost = 3 until seq unit = start = 9
            //at the same time add these seq unit and given end in map
            //because in future any other point[] comes that tries to paint the same area
            //we can easily find that and map.value will give the end value which is previous
            //area painted
            while (start < end) {
                if (paintMap.containsKey(start)) {
                    //if curr start is already pained previously
                    //it value = end will tell us upto which point it was covered that time
                    //ex: [4, 10] in between seq unit 7 was also covered map[7 = 10]
                    //now when another point[7, 13] will come it will first check 7 is already painted or not
                    //here it was painted when point[4, 10] so will get the end value
                    //it painted which was map[7] = 10 that means from this 10 we might need to paint
                    start = paintMap.get(start);
                    continue;
                }
                //cost of painting per unit area (start to seq)
                cost++;
                paintMap.put(start, end);
                start++;
            }
            result.add(cost);
        }
        //output
        System.out.println("Total cost to paint the given area points effectively : " + result);
    }

    public void mThElementAfterKArrayRotation(int[] arr, int k, int m) {
        //https://www.geeksforgeeks.org/cpp-program-to-find-the-mth-element-of-the-array-after-k-left-rotations/
        int n = arr.length;
        k %= n;
        int mthIndex = (k + m - 1) % n;
        //output
        System.out.println("Mth element after k Array rtation : " + arr[mthIndex]);
    }

    public void maximizeSumAfterRemovingValleys(int[] mountains) {
        //https://www.geeksforgeeks.org/maximize-sum-of-given-array-after-removing-valleys/
        int n = mountains.length;
        int[] smallerInLeft = new int[n];
        int[] smallerInRight = new int[n];

        Stack<Integer> stack = new Stack<>();

        //smller to left
        for (int i = 0; i < n; i++) {
            int val = mountains[i];
            while (!stack.isEmpty()
                    && mountains[stack.peek()] >= val) {
                stack.pop();
            }

            if (stack.isEmpty()) {
                //i + 1 == len upto ith element 
                //like if arr supposed to [1,1,1] len = 3 and val = 1 then len * val = 3
                smallerInLeft[i] = (i + 1) * val;
            } else {
                int smallIndex = stack.peek();
                smallerInLeft[i] = smallerInLeft[smallIndex] + (i - smallIndex) * val;
            }
            stack.push(i);
        }

        stack.clear();

        //smller in right
        for (int i = n - 1; i >= 0; i--) {
            int val = mountains[i];
            while (!stack.isEmpty()
                    && mountains[stack.peek()] >= val) {
                stack.pop();
            }

            if (stack.isEmpty()) {
                //n - i is same as len upto ith element
                smallerInRight[i] = (n - i) * val;
            } else {
                int smallIndex = stack.peek();
                smallerInRight[i] = smallerInRight[smallIndex] + (smallIndex - i) * val;
            }
            stack.push(i);
        }
        int maxSum = 0;
        for (int i = 0; i < n; i++) {
            int currSum = smallerInLeft[i] + smallerInRight[i] - mountains[i];
            maxSum = Math.max(maxSum, currSum);
        }
        //output:
        System.out.println("Max sum after removing valleys: " + maxSum);
    }

    public void numberOfVisiblePeopleInQueue(int[] heights) {
        //https://leetcode.com/problems/number-of-visible-people-in-a-queue/
        //based on nextGreaterElementInRight
        int n = heights.length;
        Stack<Integer> stack = new Stack<>();
        int[] personVisible = new int[n];

        for (int i = n - 1; i >= 0; i--) {
            //this while cond allow heights[i] person to check all those 
            //person who are smaller to itself and count them
            while (!stack.isEmpty() && stack.peek() < heights[i]) {
                personVisible[i]++;
                stack.pop();
            }

            //this cond is for the case where height[i] < height[j] where  i < j
            //greater height[j] person is still visible to height[i] person
            if (!stack.isEmpty()) {
                personVisible[i]++;
            }
            if (stack.isEmpty() || heights[i] != stack.peek()) {
                stack.push(heights[i]);
            }
        }
        //output
        for (int val : personVisible) {
            System.out.print(val + " ");
        }
        System.out.println();
    }

    private boolean splitArrayInLargestSum_CanSplit(int[] nums, int largestSum, int m) {
        int subarrays = 1;
        int currSum = 0;
        for (int val : nums) {
            currSum += val;
            if (currSum > largestSum) {
                currSum = val;
                subarrays++;
            }
        }
        return subarrays <= m;
    }

    public void splitArrayInLargestSum(int[] nums, int m) {
        //https://leetcode.com/problems/split-array-largest-sum/
        //explanation: https://youtu.be/YUF3_eBdzsk | https://youtu.be/bcAwHkL7A3A
        /*
         intution:
         1. if m = 1 that means, a single subarray which means the whole array
         is that subarray where the largest sum possible would be sum of arr
         2. if m = arr.length that means, each element of arr is itself a subarray
         (and no subarray is possible m > arr.length) in this case the largest
         sum of sub array would be the max num of arr
         Now if we need to split the arr in m subarrays with largest sum
         that could be formed within range of m >= 1 && m <= n where largest sum would
         lie in range of [maxNum, arrSum]
         ex: arr [7,2,5,10,8] m = 2, maxNum = 10, arrSum = 32
         1. partition (m = 2 subarrays)= [7] | [2,5,10,8] sum = 7 | 25 largest = 25
         2. partition (m = 2 subarrays)= [7,2] | [5,10,8] sum = 9 | 23 largest = 23
         3. partition (m = 2 subarrays)= [7,2,5] | [10,8] sum = 14 | 18 largest = 18
         3. partition (m = 2 subarrays)= [7,2,5,10] | [8] sum = 24 | 8 largest = 24
         these are the possible largest sums we will get if we divide the arr
         into m (non-empty subarrys). From these we need the min largest sum which is 18
         see all these largest sums are in the range of [maxNum, arrSum]
         */
        int maxNum = nums[0];
        int arrSum = 0;
        for (int val : nums) {
            maxNum = Math.max(maxNum, val);
            arrSum += val;
        }

        //minimizing the largest sum would lie between [maxNum, arrSum]
        //Binary search
        int start = maxNum;
        int end = arrSum;
        int minimizedSum = 0;
        while (end >= start) {
            int midSum = start + (end - start) / 2;
            if (splitArrayInLargestSum_CanSplit(nums, midSum, m)) {
                minimizedSum = midSum;
                //since we are trying to minimize the sum, we will reduce end
                end = midSum - 1;
            } else {
                start = midSum + 1;
            }
        }
        //output
        System.out.println("Minimized the largest sum among m subarrays: " + minimizedSum);
    }

    private boolean shipWeightsWithinGivenDays_CheckDays(int[] weights, int maxWeight, int days) {
        int day = 1;
        int currWeightSum = 0;
        for (int weight : weights) {
            currWeightSum += weight;
            if (currWeightSum > maxWeight) {
                day++;
                currWeightSum = weight;
            }
        }
        return day <= days;
    }

    private int shipWeightsWithinGivenDays_BinarySearch(
            int[] weights, int days, int startWeight, int endWeight) {
        int capacity = -1;
        while (endWeight >= startWeight) {
            int midWeight = startWeight + (endWeight - startWeight) / 2;

            if (shipWeightsWithinGivenDays_CheckDays(weights, midWeight, days)) {
                capacity = midWeight;
                endWeight = midWeight - 1;
            } else {
                startWeight = midWeight + 1;
            }
        }
        return capacity;
    }

    public void shipWeightsWithinGivenDays(int[] weights, int days) {
        //https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/
        //explanation: https://youtu.be/YUF3_eBdzsk | https://youtu.be/bcAwHkL7A3A
        //based on splitArrayInLargestSum()
        int totalWeight = 0;
        int maxShipmentWeight = 0;
        for (int weight : weights) {
            totalWeight += weight;
            maxShipmentWeight = Math.max(maxShipmentWeight, weight);
        }

        int capacityOfShipNeeded = shipWeightsWithinGivenDays_BinarySearch(
                weights, days, maxShipmentWeight, totalWeight);
        //output:
        System.out.println("Capacity of ship required to ship all weights in given days: " + capacityOfShipNeeded);
    }

    private boolean minimizePageAllocationsToStudents_CanAllocate(
            int[] pages, int maxPageAllocation, int students) {

        int studentsReq = 1;
        int currPagesSum = 0;
        for (int page : pages) {
            currPagesSum += page;
            if (currPagesSum > maxPageAllocation) {
                currPagesSum = page;
                studentsReq++;
            }
        }
        return studentsReq <= students;
    }

    public void minimizePageAllocationsToStudents(int[] pages, int students) {
        //............................T: O(N + N.Log(RANGE)), N = length of pages array
        //RANGE is [min(pages[i]) to sum(pages[i])]
        //https://www.interviewbit.com/problems/allocate-books/
        //explanation: https://youtu.be/gYmWHvRHu-s
        //based on splitArrayInLargestSum()
        int minPage = pages[0];
        int totalPages = 0;
        for (int page : pages) {
            minPage = Math.min(minPage, page);
            totalPages += page;
        }

        //minimize the page allocations to student will lie in range
        //[minPage to totalPages]
        //overall T: O(N * Log(RANGE))
        //Binary search: T; O(Log(RANGE)) 
        int startPage = minPage;
        int endPage = totalPages;
        int minimizesPageAllocation = totalPages;
        while (endPage > startPage) {
            int midPage = startPage + (endPage - startPage) / 2;
            //T: O(N), at worst the below method will check for N pages in pages[]
            //whether we can allocate all of the pages to given student or not
            if (minimizePageAllocationsToStudents_CanAllocate(pages, midPage, students)) {
                minimizesPageAllocation = midPage;
                endPage = midPage - 1;
            } else {
                startPage = midPage + 1;
            }
        }
        //output
        System.out.println("Minimized pages allocation to students: " + minimizesPageAllocation);
    }

    private boolean kokoEatingBananas_CanEat(int[] piles, int maxPile, int hour) {
        int currHr = 0;
        for (int pile : piles) {
            currHr += pile / maxPile;
            if (pile % maxPile != 0) {
                currHr++;
            }
        }
        return currHr <= hour;
    }

    public void kokoEatingBananas(int[] piles, int hour) {
        //https://leetcode.com/problems/koko-eating-bananas/
        //https://leetcode.com/problems/koko-eating-bananas/discuss/1691271/similar-to-allocate-pages-of-books
        int maxPile = Integer.MIN_VALUE;
        for (int pile : piles) {
            maxPile = Math.max(maxPile, pile);
        }

        int start = 1;
        int end = maxPile;

        while (end >= start) {
            int mid = start + (end - start) / 2;
            if (kokoEatingBananas_CanEat(piles, mid, hour)) {
                end = mid - 1;
            } else {
                start = mid + 1;
            }
        }
        //output
        System.out.println("Minimized speed in koko can eat all the bananas within given hours: " + start);
    }

    public void subseqOfLengthKWithLargestSum(int[] nums, int k) {
        //......................T: O(N.LogK) + O(K.LogK), minHeap + List<Pair> sorting
        //https://leetcode.com/problems/find-subsequence-of-length-k-with-the-largest-sum/
        class Pair {

            int val;
            int index;

            public Pair(int val, int index) {
                this.val = val;
                this.index = index;
            }
        }

        //first find the K largest elements, so sort by value
        //by finding K largest elements, we will have largest sum
        PriorityQueue<Pair> minHeap = new PriorityQueue<>((a, b) -> a.val - b.val);

        //K largest elements
        for (int i = 0; i < nums.length; i++) {
            minHeap.add(new Pair(nums[i], i));
            if (minHeap.size() > k) {
                minHeap.poll();
            }
        }

        //after above loop, we will left with K largest elements in the heap
        //now sort the values by min index to maintain subseq order
        List<Pair> values = new ArrayList<>(minHeap);
        Collections.sort(values, (a, b) -> a.index - b.index);

        int[] subseq = new int[k]; // values.size == k
        for (int i = 0; i < values.size(); i++) {
            subseq[i] = values.get(i).val;
        }
        //output
        for (int val : subseq) {
            System.out.print(val + " ");
        }
        System.out.println();
    }

    public void rangeSumQueries_BruteForce(int[] nums, int[][] queries) {
        //.........................T: O(M*N), for processing M queries it will
        //take in worst case O(N) time (if for mth query left = 0 and right = N - 1)
        //https://leetcode.com/problems/range-sum-query-immutable/
        List<Integer> sumsInQueryRange = new ArrayList<>();
        //T: O(M)
        for (int[] query : queries) {

            int left = query[0];
            int right = query[1];
            int sum = 0;
            //T: O(N) in worst case
            for (int i = left; i <= right; i++) {
                sum += nums[i];
            }
            sumsInQueryRange.add(sum);
        }
        //output
        System.out.println("Sums from all the queries (Brute Force): " + sumsInQueryRange);
    }

    public void rangeSumQueries(int[] nums, int[][] queries) {
        //.........................T: O(N + M), for finding prefix sum,
        //for processing M queries it will take O(1)
        //https://leetcode.com/problems/range-sum-query-immutable/
        //explanation: https://youtu.be/k5Im14rNUFA
        //OPTIMISED
        //based on PREFIX SUM
        //convert nums array to its prefix sum array
        int prefix = nums[0];
        int n = nums.length;
        for (int i = 1; i < n; i++) {
            prefix += nums[i];
            nums[i] = prefix;
        }
        List<Integer> sumsInQueryRange = new ArrayList<>();
        for (int[] query : queries) {

            int left = query[0];
            int right = query[1];

            int sum = left - 1 < 0
                    ? nums[right]
                    : nums[right] - nums[left - 1];

            sumsInQueryRange.add(sum);
        }
        //output
        System.out.println("Sums from all the queries: " + sumsInQueryRange);
    }

    public void partitionArrayOnGivenPivot(int[] nums, int pivot) {
        //https://leetcode.com/problems/partition-array-according-to-given-pivot/
        //https://leetcode.com/problems/rearrange-array-elements-by-sign
        int n = nums.length;
        List<Integer> smaller = new ArrayList<>();
        List<Integer> same = new ArrayList<>();
        List<Integer> larger = new ArrayList<>();
        List<Integer> result = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            if (nums[i] < pivot) {
                smaller.add(nums[i]);
            } else if (nums[i] == pivot) {
                same.add(nums[i]);
            } else {
                larger.add(nums[i]);
            }
        }

        result.addAll(smaller);
        result.addAll(same);
        result.addAll(larger);

        //output
        //int[] output = result.stream().mapToInt(val -> val).toArray();
        System.out.println("Partition array on given pivot with relative order maintained: " + result);
    }

    public void distinctBarcodes(int[] barcodes) {
        //https://leetcode.com/problems/distant-barcodes/
        //based on reorganizeString()
        //Acc to quest, rearrange barcodes so that no two adjacent barcodes are equal.
        int n = barcodes.length;
        int[] result = new int[n];
        int index = 0;

        Map<Integer, Integer> freq = new HashMap<>();
        for (int val : barcodes) {
            freq.put(val, freq.getOrDefault(val, 0) + 1);
        }

        PriorityQueue<Integer> maxHeapFreq = new PriorityQueue<>(
                (a, b) -> freq.get(b) - freq.get(a));

        for (int val : freq.keySet()) {
            maxHeapFreq.add(val);
        }

        //until there are 2 or more elements to pick
        while (maxHeapFreq.size() > 1) {

            int firstMostCommonNum = maxHeapFreq.poll();
            int secondMostCommonNum = maxHeapFreq.poll();

            result[index++] = firstMostCommonNum;
            result[index++] = secondMostCommonNum;

            //since we have taken 1 unit of firstMostCommonNum we must reduce its freq
            freq.put(firstMostCommonNum, freq.getOrDefault(firstMostCommonNum, 0) - 1);
            //if there are still some freq for curr firstMostCommonNum is left
            //we will add it back to maxHeap
            if (freq.get(firstMostCommonNum) > 0) {
                maxHeapFreq.add(firstMostCommonNum);
            }

            //since we have taken 1 unit of secondMostCommonNum we must reduce its freq
            freq.put(secondMostCommonNum, freq.getOrDefault(secondMostCommonNum, 0) - 1);
            //if there are still some freq for curr secondMostCommonNum is left
            //we will add it back to maxHeap
            if (freq.get(secondMostCommonNum) > 0) {
                maxHeapFreq.add(secondMostCommonNum);
            }
        }

        //there are chances, that there will remain 1 element in the maxHeap
        //we simply add it to our result 
        if (!maxHeapFreq.isEmpty()) {
            result[index] = maxHeapFreq.poll();
        }
        //output
        System.out.println(Arrays.toString(result));
    }

    public void rotateMatrixClockWise90Deg(int[][] mat) {
        //https://leetcode.com/problems/rotate-image
        int col = mat[0].length;

        int left = 0;
        int right = col - 1;

        while (right > left) {
            //(right - left is amount of element need to take)
            for (int i = 0; i < (right - left); i++) {
                int top = left;
                int bottom = right;

                /*
                
                 top-[left + i]  ---> [top + i]-right
                 /\                             |
                 |   Clock dir                  |
                 |                             \/
                 [bottom - i]-left <---- bottom-[right - i]
                
                 */
                //save top-left corner value
                int topLeftCornerValue = mat[top][left + i];
                //in top-left put bottom-left value
                mat[top][left + i] = mat[bottom - i][left];
                //in bottom-left put bottom-right value
                mat[bottom - i][left] = mat[bottom][right - i];
                //in bottom-right put top-right value
                mat[bottom][right - i] = mat[top + i][right];
                //in top-right put top-left value
                mat[top + i][right] = topLeftCornerValue;
            }
            left++;
            right--;
        }

        //output
        System.out.println("N * N matrix 90deg clockwise rotation: ");
        for (int[] r : mat) {
            for (int c : r) {
                System.out.print(c + "\t");
            }
            System.out.println();
        }
    }

    public void rotateMatrixAntiClockWise90Deg(int[][] mat) {

        int row = mat.length;
        int col = mat[0].length;

        int left = 0;
        int right = col - 1;

        while (right > left) {

            for (int i = 0; i < (right - left); i++) {
                int top = left;
                int bottom = right;

                /*
                
                 [top + i]-left  <--- top-[right - i]
                 |                              /\
                 |   AntiClock dir              |
                 \/                             |
                 bottom-[left + i] ----> [bottom - i]-right
                
                 */
                //save top-left corner value
                int topLeftCornerValue = mat[top + i][left];
                //in top-right put top-right value
                mat[top + i][left] = mat[top][right - i];
                //in top-right put bottom-right value
                mat[top][right - i] = mat[bottom - i][right];
                //in bottom-right put bottom-left value
                mat[bottom - i][right] = mat[bottom][left + i];
                //in bottom-left put top-left value
                mat[bottom][left + i] = topLeftCornerValue;
            }
            left++;
            right--;
        }

        //output
        System.out.println("N * N matrix 90deg anticlockwise rotation: ");
        for (int[] r : mat) {
            for (int c : r) {
                System.out.print(c + "\t");
            }
            System.out.println();
        }
    }

    private int areaPerRow(int[] hist) {

        //same as laregstAreaHistogram method
        Stack<Integer> stack = new Stack<>();
        int n = hist.length;
        int maxArea = 0;
        int top;
        int areaWithTop;
        int i = 0;
        while (i < n) {

            if (stack.isEmpty() || hist[stack.peek()] <= hist[i]) {
                stack.push(i++);
            } else {
                top = stack.pop();
                areaWithTop = hist[top] * (stack.isEmpty() ? i : i - stack.peek() - 1);
                maxArea = Math.max(maxArea, areaWithTop);
            }
        }

        while (!stack.isEmpty()) {
            top = stack.pop();
            areaWithTop = hist[top] * (stack.isEmpty() ? i : i - stack.peek() - 1);
            maxArea = Math.max(maxArea, areaWithTop);
        }

        return maxArea;
    }

    public void maxAreaOfRectangleInBinaryMatrix(int[][] mat) {
        //https://leetcode.com/problems/maximal-rectangle
        //problem statment & sol: https://www.geeksforgeeks.org/maximum-size-rectangle-binary-sub-matrix-1s/
        //explanation: https://youtu.be/dAVF2NpC3j4
        //find max area of per row int the matrix
        //each row in the matrix is histogram
        //use max area histogram
        int R = mat.length;
        int C = mat[0].length;

        int maxArea = areaPerRow(mat[0]);

        for (int r = 1; r < R; r++) {
            for (int c = 0; c < C; c++) {
                if (mat[r][c] == 1) {
                    mat[r][c] += mat[r - 1][c];
                }
            }
            maxArea = Math.max(maxArea, areaPerRow(mat[r]));
        }

        //output:
        System.out.println("Max area in binary matrix: " + maxArea);
    }

    public void maximumOnesInRowOfABinarySortedMatrix_1(int[][] mat) {

        //.....................................T; O(M*N)
        //problem statement: https://www.geeksforgeeks.org/find-the-row-with-maximum-number-1s/
        int maxOnes = 0;
        int index = 0;
        for (int i = 0; i < mat.length; i++) {

            int onePerRow = areaPerRow(mat[i]);
            if (maxOnes < onePerRow) {
                maxOnes = onePerRow;
                index = i;
            }
        }
        //output;
        System.out.println("Max 1(s) found at index: " + (maxOnes == 0 ? -1 : index) + " counts of is are: " + maxOnes);
    }

    public void maximumOnesInRowOfABinarySortedMatrix_2(int[][] mat) {

        //............................T: O(M.LogN)
        //OPTIMISED
        int maxOnes = 0;
        int index = 0;
        for (int r = 0; r < mat.length; r++) {
            int C = mat[r].length;
            int firstIndexOfOne = findFirstOccurenceKInSortedArray(mat[r], 1, 0, C - 1, C);

            //if no index is found
            if (firstIndexOfOne == -1) {
                continue;
            }

            int onePerRow = C - firstIndexOfOne;
            if (maxOnes < onePerRow) {
                maxOnes = onePerRow;
                index = r;
            }
        }

        //output;
        System.out.println("Max 1(s) found at index: " + (maxOnes == 0 ? -1 : index) + " counts of is are: " + maxOnes);
    }

    public void findAValueInRowWiseSortedMatrix(int[][] mat, int K) {
        //https://leetcode.com/problems/search-a-2d-matrix-ii/
        int M = mat.length;
        int N = mat[0].length;

        int i = 0;
        int j = N - 1;

        //search starts from top right corner
        while (i < M && j >= 0) {

            if (mat[i][j] == K) {
                System.out.println("Found at: " + i + ", " + j);
                return;
            } else if (K < mat[i][j]) {
                j--;
            } else {
                i++;
            }
        }

        //K is not there in the matrix at all
        System.out.println("Not found");
    }

    public void spiralMatrixTraversal(int[][] mat) {

        List<Integer> result = new ArrayList<>();

        int R = mat.length;
        int C = mat[0].length;

        int top = 0; // top row
        int bottom = R - 1; //bottom row
        int left = 0; //left col
        int right = C - 1; //right col
        int totalElement = R * C;

        //with each row and col processing we will shrink the matrix bounds (top++, bottom--, left++, right--)
        //result list is going to hold all the elements in the matrix
        while (result.size() < totalElement) {

            //get the top row from left to right
            for (int i = left; i <= right && result.size() < totalElement; i++) {
                result.add(mat[top][i]); //top row and ith col left to right
            }
            top++; //since we traversed top row now to next row next time.

            //top to bottom but right col
            for (int i = top; i <= bottom && result.size() < totalElement; i++) {
                result.add(mat[i][right]); //top to bottom but right col
            }
            right--; //now till here we have traversed the very right col of mat. for next time right col is prev

            //right to left but bottom row only
            for (int i = right; i >= left && result.size() < totalElement; i--) {
                result.add(mat[bottom][i]); //bottom row from right to left
            }
            bottom--; //completed bottom row also next time bottom to be prev one

            //bottom to top but only left col
            for (int i = bottom; i >= top && result.size() < totalElement; i--) {
                result.add(mat[i][left]); //left col
            }
            left++; //for next itr left will be moved ahead
        }

        //output:
        System.out.println("Spiral matrix: " + result);
    }

    public void diagonalMatrixTraversal(int[][] mat) {

        int R = mat.length;
        int C = mat[0].length;
        int x = 0;
        int y = 0;
        boolean isGoingUp = true;
        int totalElements = R * C;
        int element = 0;
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> diagonal = new ArrayList<>();
        while (element < totalElements) {

            if (isGoingUp) {

                while (x >= 0 && y < C) {
                    diagonal.add(mat[x][y]);
                    x--;
                    y++;
                    element++;
                }

                if (x < 0 && y <= C - 1) {
                    x = 0;
                }

                if (y == C) {
                    x += 2;
                    y--;
                }
            } else {

                while (x < R && y >= 0) {
                    diagonal.add(mat[x][y]);
                    x++;
                    y--;
                    element++;
                }

                if (x <= R - 1 && y < 0) {
                    y = 0;
                }

                if (x == R) {
                    y += 2;
                    x--;
                }
            }

            isGoingUp = !isGoingUp;
            result.add(diagonal);
            diagonal = new ArrayList<>();
        }

        //output:
        System.out.println("Diagonal matrix: " + result);
    }

    public void sumOfElementsInMatrixExceptGivenRowAndCol(int[][] matrix, int[][] rowAndCol) {

        //OPTIMISED
        //..........................T: O((R * C) + n) where n = rowAndCol.length
        //..........................S: O(R + C) because rowSum and colSum array of size R & C respec
        //https://www.geeksforgeeks.org/find-sum-of-all-elements-in-a-matrix-except-the-elements-in-given-row-andor-column-2/
        int R = matrix.length;
        int C = matrix[0].length;

        int[] rowSum = new int[R];
        int[] colSum = new int[C];

        int sumOfElements = 0;

        for (int x = 0; x < R; x++) {
            for (int y = 0; y < C; y++) {
                sumOfElements += matrix[x][y]; //total sum of elements of matrix
                rowSum[x] += matrix[x][y]; //all the sum of elements in current x row
                colSum[y] += matrix[x][y]; //all the sum of elements in current y col
            }
        }

        for (int[] except : rowAndCol) {

            int row = except[0];
            int col = except[1];
            int sumWithoutRowAndCol = sumOfElements - rowSum[row] - colSum[col] + matrix[row][col];
            System.out.println("Sum of elements except row: " + row + " & col: " + col + " sum: " + sumWithoutRowAndCol);
        }
    }

    public void sortTheMatrixDiagonally(int[][] mat) {

        //.................................T: O(R * C) 
        //DATA STRUCTURE BASED SORTING
        //https://leetcode.com/problems/sort-the-matrix-diagonally/
        Map<Integer, PriorityQueue<Integer>> map = new HashMap<>();
        int R = mat.length;
        int C = mat[0].length;

        //actual:
        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {
                System.out.print(mat[r][c] + "\t");
            }
            System.out.println();
        }

        //All the elements in the diagonal lie at same location of r - c
        //map key will have all r - c locations of diagonal and inside a priority queue(minHeap)
        //which will keep elements of that diagonal in sorted order as we add it in.
        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {

                map.putIfAbsent(r - c, new PriorityQueue<>());
                map.get(r - c).add(mat[r][c]);
            }
        }
        //generate the matrix from the data structure
        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {
                mat[r][c] = map.get(r - c).poll();
            }
        }

        //output:
        System.out.println("Diagonally sorted matrix");
        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {
                System.out.print(mat[r][c] + "\t");
            }
            System.out.println();
        }
    }

    public void minimumPathSumInGrid(int[][] grid) {
        //https://leetcode.com/problems/minimum-path-sum/
        int R = grid.length;
        int C = grid[0].length;

        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {
                if (r == 0 && c == 0) {
                    //skip the top-left corner
                    continue;
                } else if (r == 0) {
                    //curr row,col to be added with same-row, prev-col
                    grid[r][c] += grid[r][c - 1];
                } else if (c == 0) {
                    //curr row,col to be added with prev-row, same-col
                    grid[r][c] += grid[r - 1][c];
                } else {
                    //curr row,col to be added with min((prev-row, same-col) OR (same-row, prev-sol)) 
                    grid[r][c] += Math.min(grid[r - 1][c], grid[r][c - 1]);
                }
            }
        }
        //output
        System.out.println("Min path sum in grid from top-left to bottom-right: " + grid[R - 1][C - 1]);
    }

    public int triangleMinPathSumTopToBottom(List<List<Integer>> triangle) {
        //https://leetcode.com/problems/triangle/
        //explanation: https://youtu.be/OM1MTokvxs4
        int size = triangle.size();
        if (size == 1) {
            return triangle.get(0).get(0);
        }
        int secondLastRow = size - 2;
        //find the min values from the end rows and proceed from bottom to top
        //add each min value to curr row value
        for (int row = secondLastRow; row >= 0; row--) {

            int rowBelowCurr = row + 1;
            List<Integer> rowBelowCurrList = triangle.get(rowBelowCurr);
            List<Integer> currRowList = triangle.get(row);
            int currRowSize = currRowList.size();

            for (int c = 0; c < currRowSize; c++) {

                int currRowValue = currRowList.get(c);

                int firstChild = rowBelowCurrList.get(c);
                int secondChild = rowBelowCurrList.get(c + 1);

                int minOfTwoChild = Math.min(firstChild, secondChild);
                currRowList.set(c, currRowValue + minOfTwoChild);
            }
        }
        return triangle.get(0).get(0);
    }

    public void smallestRectangleEnclosingBlackPixels(int[][] image) {
        //https://leetcode.com/problems/smallest-rectangle-enclosing-black-pixels/
        //https://www.lintcode.com/problem/600/

        /*
         ....-----
         [[0, 0, 1, 0]]
         [[0, 1, 1, 0]]
         [[0, 1, 0, 0]]
         ....-----
         All black pixel region is covered under ---- which is this
         [[0,1]]
         [[1,1]]
         [[1,0]]
         //across the row
         topRow = 0 as first '1' is observed at image(0, 2)
         bottomRow = 2 as last '1' is observed at image(2, 1)
         //accross the col
         leftCol = 1 as first '1' is observed at image(1, 1)
         rightCol = 2 as last '1' is observed at image(1, 2)
         */
        int ROW = image.length;
        int COL = image[0].length;

        //topRow and leftCol are the initial row and col
        //point where we found the 1(black pixel)
        //that topRow and leftCol will enclose all these black pixels
        int topRow = ROW - 1;
        int leftCol = COL - 1;
        //bottomRow and rightCol are the last row and col
        //point where we found the 1(black pixel)
        //that bottomRow and rightCol will enclose all these black pixels
        int bottomRow = 0;
        int rightCol = 0;

        for (int row = 0; row < ROW; row++) {
            for (int col = 0; col < COL; col++) {
                if (image[row][col] == 1) {
                    topRow = Math.min(topRow, row);
                    bottomRow = Math.max(bottomRow, row);

                    leftCol = Math.min(leftCol, col);
                    rightCol = Math.max(rightCol, col);
                }
            }
        }

        //simialr like lengths be like end - start + 1
        int length = bottomRow - topRow + 1;
        int breadth = rightCol - leftCol + 1;

        int areaThatEncloseAllBlackPixels = length * breadth;
        //output
        System.out.println("Area that enclose all balck pixels: " + areaThatEncloseAllBlackPixels);
    }

    private boolean checkIfMoveIsLegal_IsOutOfBound(int row, int col, int ROW, int COL) {
        return row < 0 || row >= ROW || col < 0 || col >= COL;
    }

    private boolean checkIfMoveIsLegal_IsLegal(
            String[][] board, int row, int col,
            String color, int[] dir) {

        int ROW = board.length;
        int COL = board[0].length;
        //passed in row and col already have start point with
        //given color so need now to find a end point with same color
        //so we can simply start from next point in same dir
        row += dir[0];
        col += dir[1];

        int length = 1;

        while (!checkIfMoveIsLegal_IsOutOfBound(row, col, ROW, COL)) {
            //incr length for good line
            length++;
            String currCell = board[row][col];
            //if in the curr path in given dir[] from given row and col
            //we see any empty space(.) then that path is not valid
            if (currCell.equals(".")) {
                return false;
            }
            //if in the curr path in given dir[] from given row and col
            //we see same color in the endpoint then that path is a good line
            //like start point = [B W...W B] OR [W B...B W] == end point same as start
            if (currCell.equals(color)) {
                //but that good line should have a lenght of atleast 3
                return length >= 3;
            }
            //update row and col, straight in given dir[]
            row += dir[0];
            col += dir[1];
        }
        return false;
    }

    public void checkIfMoveIsLegal(String[][] board, int row, int col, String color) {
        //https://leetcode.com/problems/check-if-move-is-legal/
        //explanation: https://youtu.be/KxK33AcQZpQ
        board[row][col] = color;
        //need to check in all 8 dirs
        int[][] dirs = {
            {-1, 0},
            {1, 0},
            {0, -1},
            {0, 1},
            {-1, -1},
            {-1, 1},
            {1, -1},
            {1, 1}
        };

        for (int[] dir : dirs) {
            //move curr row and col in same curr dir[]
            //and check if in that particular dir move is legal or not
            //if from any one dir we get legal move == true, answer is legal
            //for each dir we will be starting from same row and col point
            if (checkIfMoveIsLegal_IsLegal(board, row, col, color, dir)) {
                System.out.println("Move is legal");
                return;
            }
        }
        System.out.println("Move is not legal");
    }

    public void islandPerimeter(int[][] grid) {
        //https://leetcode.com/problems/island-perimeter/
        /*
         4-edged perimeter
         -----
         | 1 |
         -----
        
         if curr cell also have cell at upper row
         total perimeter for both cell are 8 but we see
         there is common edge in between that should be removed (i.e -2)
         -----
         | 1 |
         -----
         -----
         | 1 |
         -----
         like this and now the actual perimeter will be 6
         -----
         | 1 |
         | 1 |
         -----
        
         if curr cell also have cell at left col
         total perimeter for both cell are 8 but we see
         there is common edge in between that should be removed (i.e -2)
         ----- -----
         | 1 | | 1 |
         ----- -----
         like this and now the actual perimeter will be 6
         ----- ----
         | 1    1 |
         ----- ----
         */
        int ROW = grid.length;
        int COL = grid[0].length;

        int perimeter = 0;

        for (int r = 0; r < ROW; r++) {
            for (int c = 0; c < COL; c++) {
                //a single cell of 1 is 4-edged island
                if (grid[r][c] == 1) {
                    perimeter += 4;

                    //if upper cell also has 1 that means the curr cell and
                    //cell above will form 6-edged perimeter and both of them sharing
                    //one edge in common so removing the common edge from both cell
                    if (r > 0 && grid[r - 1][c] == 1) {
                        perimeter -= 2;
                    }
                    //if left cell also has 1 that means the curr cell and
                    //cell at left will form 6-edged perimeter and both of them sharing
                    //one edge in common so removing the common edge from both cell
                    if (c > 0 && grid[r][c - 1] == 1) {
                        perimeter -= 2;
                    }
                }
            }
        }
        //output:
        System.out.println("Island perimeter: " + perimeter);
    }

    public void rangeSumQuery2D(int[][] matrix, int[][] queries) {
        //https://leetcode.com/problems/range-sum-query-2d-immutable/
        //explanation: https://youtu.be/KE8MQuwE2yA
        int ROW = matrix.length;
        int COL = matrix[0].length;

        //convert matrix to prefix sum matrix
        for (int r = 0; r < ROW; r++) {
            int prefix = 0;
            for (int c = 0; c < COL; c++) {
                prefix += matrix[r][c];
                int aboveRow = r - 1 < 0 ? 0 : matrix[r - 1][c];
                matrix[r][c] = prefix + aboveRow;
            }
        }

        List<Integer> queriesSum = new ArrayList<>();
        for (int[] query : queries) {

            int topLeftX = query[0];
            int topLeftY = query[1];
            int bottomRightX = query[2];
            int bottomRightY = query[3];

            int bottomRightSum = matrix[bottomRightX][bottomRightY];
            int aboveRowSum = topLeftX - 1 < 0 ? 0 : matrix[topLeftX - 1][bottomRightY];
            int leftColSum = topLeftY - 1 < 0 ? 0 : matrix[bottomRightX][topLeftY - 1];
            int cornerValAboveTopLeft = (topLeftX - 1 < 0 || topLeftY - 1 < 0)
                    ? 0
                    : matrix[topLeftX - 1][topLeftY - 1];
            int sum = bottomRightSum - aboveRowSum - leftColSum + cornerValAboveTopLeft;
            queriesSum.add(sum);
        }
        //output
        System.out.println("Range sum queries in 2D: " + queriesSum);
    }

    public String reverseString(String str) {

        int len = str.length();
        char[] ch = str.toCharArray();

        //.....................reverse by length
        //.....................O(N)
        for (int i = 0; i < len / 2; i++) {
            char temp = ch[i];
            ch[i] = ch[len - i - 1];
            ch[len - i - 1] = temp;
        }

        //output
        System.out.println("output reverse by length: " + String.valueOf(ch));

        //....................reverse by two pointer
        //....................O(N)
        int f = 0;
        int l = len - 1;
        ch = str.toCharArray();

        while (f < l) {

            char temp = ch[f];
            ch[f] = ch[l];
            ch[l] = temp;
            f++;
            l--;

        }

        //output
        System.out.println("output reverse by two pointer: " + String.valueOf(ch));

        //............................reverse by STL
        String output = new StringBuilder(str)
                .reverse()
                .toString();
        System.out.println("output reverse by STL: " + output);

        return output;

    }

    public boolean isStringPallindrome(String str) {

        return str.equals(reverseString(str));

    }

    public void printDuplicatesCharInString(String str) {
        System.out.println("For: " + str);

        Map<Character, Integer> countMap = new HashMap<>();
        for (char c : str.toCharArray()) {
            countMap.put(c, countMap.getOrDefault(c, 0) + 1);
        }

        countMap.entrySet().stream()
                .filter(e -> e.getValue() > 1)
                .forEach(e -> System.out.println(e.getKey() + " " + e.getValue()));

    }

    public void romanStringToDecimal(String str) {

        //actual
        System.out.println("roman: " + str);

        Map<Character, Integer> roman = new HashMap<>();
        roman.put('I', 1);
        roman.put('V', 5);
        roman.put('X', 10);
        roman.put('L', 50);
        roman.put('C', 100);
        roman.put('D', 500);
        roman.put('M', 1000);

        int decimal = 0;
        for (int i = 0; i < str.length(); i++) {

            char c = str.charAt(i);
            if (i > 0 && roman.get(str.charAt(i - 1)) < roman.get(c)) {
                decimal += roman.get(c) - 2 * roman.get(str.charAt(i - 1));
            } else {
                decimal += roman.get(c);
            }
        }

        //output
        System.out.println("Decimal: " + decimal);
    }

    public void integerToRomanString(int num) {

        Map<Integer, String> map = new HashMap<>();
        map.put(null, "");
        map.put(0, "");
        map.put(1, "I");
        map.put(2, "II");
        map.put(3, "III");
        map.put(4, "IV");
        map.put(5, "V");
        map.put(6, "VI");
        map.put(7, "VII");
        map.put(8, "VIII");
        map.put(9, "IX");
        map.put(10, "X");
        map.put(20, "XX");
        map.put(30, "XXX");
        map.put(40, "XL");
        map.put(50, "L");
        map.put(60, "LX");
        map.put(70, "LXX");
        map.put(80, "LXXX");
        map.put(90, "XC");
        map.put(100, "C");
        map.put(200, "CC");
        map.put(300, "CCC");
        map.put(400, "CD");
        map.put(500, "D");
        map.put(600, "DC");
        map.put(700, "DCC");
        map.put(800, "DCCC");
        map.put(900, "CM");
        map.put(1000, "M");
        map.put(2000, "MM");
        map.put(3000, "MMM");
        int actualNum = num;
        int pow = 0;
        StringBuilder sb = new StringBuilder();
        while (num != 0) {

            // System.out.println(num +" -- " +(Math.pow(10, mul) * (num%10))+" -- "+map.get((int)(Math.pow(10, mul++) * (num%10))));
            int remainder = num % 10;
            int tens = (int) Math.pow(10, pow++);
            sb.insert(0, map.get((int) (tens * remainder)));
            num /= 10;
        }
        //output
        System.out.println("Given " + actualNum + " as roman string: " + sb.toString());
    }

    public void longestCommonSubsequence(String a, String b) {

        //memoization
        int[][] memo = new int[a.length() + 1][b.length() + 1];
        //base cond
        for (int[] x : memo) {
            Arrays.fill(x, 0);
        }

        for (int x = 1; x < a.length() + 1; x++) {
            for (int y = 1; y < b.length() + 1; y++) {
                if (a.charAt(x - 1) == b.charAt(y - 1)) {
                    memo[x][y] = memo[x - 1][y - 1] + 1;
                } else {
                    memo[x][y] = Math.max(memo[x - 1][y], memo[x][y - 1]);
                }
            }
        }

        int l = a.length();
        int m = b.length();
        StringBuilder sb = new StringBuilder();
        while (l > 0 && m > 0) {

            if (a.charAt(l - 1) == b.charAt(m - 1)) {
                sb.insert(0, a.charAt(l - 1));
                l--;
                m--;
            } else {

                if (memo[l - 1][m] > memo[l][m - 1]) {
                    l--;
                } else {
                    m--;
                }
            }
        }

        //output
        System.out.println("Longest common subseq: " + sb.toString());
    }

    public String countAndSay_Helper(int n) {

        //https://leetcode.com/problems/count-and-say/
        //base cond
        if (n == 1) {
            return "1";
        }

        String ans = countAndSay_Helper(n - 1);

        StringBuilder sb = new StringBuilder();
        char ch = ans.charAt(0);
        int counter = 1;
        //for i==ans.length() i.e very last itr of loop
        //this itr will only invoke else cond below
        for (int i = 1; i <= ans.length(); i++) {
            //i<ans.length() bound the calculations upto string length
            if (i < ans.length() && ans.charAt(i) == ch) {
                counter++;
            } else {
                sb.append(counter).append(ch);
                //i<ans.length() bound the calculations upto string length
                if (i < ans.length()) {
                    ch = ans.charAt(i);
                }
                counter = 1;
            }
        }
        return sb.toString();
    }

    public void countAndSay(int n) {
        System.out.println("Count and say: " + countAndSay_Helper(n));
    }

    public void removeConsecutiveDuplicateInString(String str) {

        //https://www.geeksforgeeks.org/remove-consecutive-duplicates-string/
        char[] ch = str.toCharArray();
        int start = 0;
        int end = 1;

        while (end < ch.length) {

            if (ch[start] != ch[end]) {
                ch[start + 1] = ch[end];
                start++;
            }
            end++;
        }

        System.out.println("output: " + String.valueOf(ch, 0, start + 1));
    }

    private void printSentencesFromCollectionOfWords_Propagte_Recursion(String[][] words,
            int m, int n,
            String[] output) {
        // Add current word to output array
        output[m] = words[m][n];

        // If this is last word of 
        // current output sentence, 
        // then print the output sentence
        if (m == words.length - 1) {
            for (int i = 0; i < words.length; i++) {
                System.out.print(output[i] + " ");
            }
            System.out.println();
            return;
        }

        // Recur for next row
        for (int i = 0; i < words.length; i++) {
            if (!"".equals(words[m + 1][i]) && m < words.length) {
                printSentencesFromCollectionOfWords_Propagte_Recursion(words, m + 1, i, output);
            }
        }
    }

    public void printSentencesFromCollectionOfWords(String[][] words) {

        //https://www.geeksforgeeks.org/recursively-print-all-sentences-that-can-be-formed-from-list-of-word-lists/
        String[] output = new String[words.length];

        // Consider all words for first 
        // row as starting points and
        // print all sentences
        for (int i = 0; i < words.length; i++) {
            if (words[0][i] != "") {
                printSentencesFromCollectionOfWords_Propagte_Recursion(words, 0, i, output);
            }
        }
    }

    public void longestPrefixAlsoSuffixInString_KMPAlgo(String s) {

        int n = s.length();
        int[] lps = new int[n];
        int j = 0, i = 1;
        while (i < n) {
            if (s.charAt(j) == s.charAt(i)) {
                j++;
                lps[i] = j;
                i++;
            } else {
                if (j == 0) {
                    i++;
                } else {
                    j = lps[j - 1];
                }
            }
        }

        System.out.println("Length of longest prefix: " + (j == 0 ? j : j));
        System.out.println("Longest prefix substring: " + (j == 0 ? "" : s.substring(0, j)));
    }

    public String reorganizeString(String S) {
        //https://leetcode.com/problems/reorganize-string/
        int N = S.length();
        Map<Character, Long> map = S.chars().mapToObj(c -> (char) c)
                .collect(Collectors.groupingBy(
                        Function.identity(),
                        Collectors.counting()
                ));

        PriorityQueue<Character> heap = new PriorityQueue<>(
                //if freq are same for two char
                //sort then alphabetically else max freq first
                (e1, e2) -> map.get(e1) == map.get(e2)
                ? e1 - e2
                : (int) (map.get(e2) - map.get(e1))
        );

        //check all unique char in string S and put it in heap
        //alternatively stream api : for(char ch :  S.chars().mapToObj(c -> (char)c).collect(Collectors.toSet()))
        for (char ch : map.keySet()) {
            if (map.get(ch) > (N + 1) / 2) {
                return "";
            }
            heap.add(ch);
        }

        StringBuilder sb = new StringBuilder();
        while (heap.size() >= 2) {

            char chA = heap.poll();
            char chB = heap.poll();

            sb.append(chA);
            sb.append(chB);

            map.put(chA, map.get(chA) - 1);
            if (map.get(chA) > 0) {
                heap.add(chA);
            }

            map.put(chB, map.get(chB) - 1);
            if (map.get(chB) > 0) {
                heap.add(chB);
            }
        }

        if (heap.size() > 0) {
            sb.append(heap.poll());
        }

        return sb.toString();
    }

    public void longestCommonPrefix(String[] strs) {

        //https://leetcode.com/problems/longest-common-prefix/
        if (strs == null || strs.length == 0) {
            return;
        }

        if (strs.length == 1) {
            System.out.println("Longest common prefix in list of strings: " + strs[0]);
            return;
        }

        String str = strs[0]; // first string as starting point
        String result = str;
        int index = 0;
        for (int i = 1; i < strs.length; i++) {
            String s = strs[i];
            int minLen = Math.min(s.length(), str.length());
            while (index < minLen && str.charAt(index) == s.charAt(index)) {
                index++;
            }
            String pref = str.substring(0, index);
            if (pref.length() < result.length()) {
                result = pref;
            }
            index = 0;
        }
        System.out.println("Longest common prefix in list of strings: " + result);
    }

    public void secondMostOccuringWordInStringList(String[] list) {

        Map<String, Integer> map = new HashMap<>();
        for (String s : list) {
            map.put(s, map.getOrDefault(s, 0) + 1);
        }

        PriorityQueue<Map.Entry<String, Integer>> minHeap = new PriorityQueue<>(
                (e1, e2) -> e1.getValue() - e2.getValue()
        );

        for (Map.Entry<String, Integer> e : map.entrySet()) {
            minHeap.add(e);
            if (minHeap.size() > 2) {
                minHeap.poll();
            }
        }

        System.out.println("Second most occuring word: " + minHeap.poll().getKey());
    }

    public boolean checkIsomorphicStrings_1(String s1, String s2) {

        int m = s1.length();
        int n = s2.length();

        if (m != n) {
            System.out.println("Not isomorphic strings");
            return false;
        }

        int SIZE = 256; //to handle numric & alphabetic ascii ranges
        boolean[] marked = new boolean[SIZE];
        int[] map = new int[SIZE];
        Arrays.fill(map, -1);

        for (int i = 0; i < m; i++) {
            if (map[s1.charAt(i)] == -1) {

                if (marked[s2.charAt(i)] == true) {
                    return false;
                }

                marked[s2.charAt(i)] = true;
                map[s1.charAt(i)] = s2.charAt(i);

            } else if (map[s1.charAt(i)] != s2.charAt(i)) {
                return false;
            }
        }

        return true;
    }

    public boolean checkIsomorphicStrings_2(String s1, String s2) {

        //........................T: O(N)
        //EASIER EXPLAINATION
        int m = s1.length();
        int n = s2.length();

        if (m != n) {
            System.out.println("Not isomorphic strings");
            return false;
        }

        Map<Character, Character> map = new HashMap<>();
        for (int i = 0; i < m; i++) {
            char sChar = s1.charAt(i);
            char tChar = s2.charAt(i);

            if (map.containsKey(sChar) && map.get(sChar) != tChar) {
                return false;
            }
            map.put(sChar, tChar);
        }

        map.clear();

        for (int i = 0; i < m; i++) {
            char sChar = s1.charAt(i);
            char tChar = s2.charAt(i);

            if (map.containsKey(tChar) && map.get(tChar) != sChar) {
                return false;
            }
            map.put(tChar, sChar);
        }

        return true;
    }

    public int transformOneStringToAnotherWithMinOprn(String src, String target) {

        //count if two strings are same length or not
        int m = src.length();
        int n = target.length();

        if (m != n) {
            //if length are not same, strings can't be transformed
            return -1;
        }

        //check if the two strings contain same char and their count should also be same
        int[] charCount = new int[256];
        for (int i = 0; i < m; i++) {

            charCount[src.charAt(i)]++;
            charCount[target.charAt(i)]--;
        }

        //if same char are there and count are equal then charCount should have been balanced out to 0
        for (int count : charCount) {
            if (count != 0) {
                return -1;
            }
        }

        int i = m - 1;
        int j = n - 1;
        int result = 0;
        while (i >= 0) {

            if (src.charAt(i) != target.charAt(j)) {
                result++;
            } else {
                j--;
            }
            i--;
        }

        return result;
    }

    public void arrangeAllWordsAsTheirAnagrams(List<String> words) {

        Map<String, List<String>> anagramGroups = new HashMap<>();
        for (String str : words) {

            char[] ch = str.toCharArray();
            Arrays.sort(ch);
            String sortedString = String.valueOf(ch);

            anagramGroups.putIfAbsent(sortedString, new ArrayList<>());
            anagramGroups.get(sortedString).add(str);
        }

        //output:
        System.out.println("Output: " + anagramGroups);
    }

    public void characterAddedAtFrontToMakeStringPallindrome(String str) {

        //https://www.geeksforgeeks.org/minimum-characters-added-front-make-string-palindrome/
        int charCount = 0;
        while (str.length() > 0) {

            if (isStringPallindrome(str)) {
                break;
            } else {
                charCount++;
                //removing 1 char from end until we get a subtring which is pallindrome
                //the no of char removed (charCount) is the number that needs to be added at front
                str = str.substring(0, str.length() - 1);
            }
        }

        //output:
        System.out.println("No. of character to be added at front to make it pallindrome: " + charCount);
    }

    public void shortestPallindrome(String s) {
        //https://leetcode.com/problems/shortest-palindrome
        //another approach to characterAddedAtFrontToMakeStringPallindrome()
        //KMP-LPS approach
        String rev = new StringBuilder(s).reverse().toString();
        String str = s + "#" + rev;
        int N = str.length();
        int n = s.length();

        int[] lps = new int[N];
        int prefixIndex = 0;
        int suffixIndex = 1;

        while (suffixIndex < N) {
            if (str.charAt(prefixIndex) == str.charAt(suffixIndex)) {
                prefixIndex++;
                lps[suffixIndex] = prefixIndex;
                suffixIndex++;
            } else if (prefixIndex == 0) {
                lps[suffixIndex] = prefixIndex;
                suffixIndex++;
            } else {
                prefixIndex = lps[prefixIndex - 1];
            }
        }
        //output
        String shortestPallindrome = rev.substring(0, n - lps[N - 1]) + s;
        System.out.println("Shortest pallindrome required to be formed: " + shortestPallindrome);
    }

    public boolean checkIfOneStringRotationOfOtherString(String str1, String str2) {
        return (str1.length() == str2.length())
                && ((str1 + str1).indexOf(str2) != -1);
    }

    private int countOccurenceOfGivenStringInCharArray_Count = 0;

    private void countOccurenceOfGivenStringInCharArray_Helper(char[][] charArr, int x, int y,
            int startPoint, String str, StringBuilder sb) {

        if (sb.toString().equals(str)) {
            //once set of string is found reset stringbuilder
            sb.setLength(0);
            countOccurenceOfGivenStringInCharArray_Count++;
            return;
        }

        if (x < 0 || x >= charArr.length || y < 0 || y >= charArr[0].length
                || startPoint >= str.length()
                || charArr[x][y] != str.charAt(startPoint)
                || charArr[x][y] == '-') {
            return;
        }

        char original = charArr[x][y];
        sb.append(charArr[x][y]);
        charArr[x][y] = '-';

        //UP
        countOccurenceOfGivenStringInCharArray_Helper(charArr, x - 1, y, startPoint + 1, str, sb);

        //Down
        countOccurenceOfGivenStringInCharArray_Helper(charArr, x + 1, y, startPoint + 1, str, sb);

        //Left
        countOccurenceOfGivenStringInCharArray_Helper(charArr, x, y - 1, startPoint + 1, str, sb);

        //Right
        countOccurenceOfGivenStringInCharArray_Helper(charArr, x, y + 1, startPoint + 1, str, sb);

        charArr[x][y] = original;
    }

    public void countOccurenceOfGivenStringInCharArray(char[][] charArr, String str) {

        countOccurenceOfGivenStringInCharArray_Count = 0; //reset/init
        StringBuilder sb = new StringBuilder();
        int N = charArr.length;
        int startPoint = 0;
        for (int x = 0; x < N; x++) {
            for (int y = 0; y < N; y++) {
                countOccurenceOfGivenStringInCharArray_Helper(charArr, x, y, startPoint, str, sb);
            }
        }

        //output
        System.out.println("Count of the given string is: " + countOccurenceOfGivenStringInCharArray_Count);
    }

    private void printAllSubSequencesOfAString_Helper(String str, int start, int N,
            String current, Set<String> subseq) {

        if (start == N) {
            subseq.add(current);
            return;
        }

        for (int i = start; i < N; i++) {
            printAllSubSequencesOfAString_Helper(str, i + 1, N, current + str.charAt(i), subseq);
            printAllSubSequencesOfAString_Helper(str, i + 1, N, current, subseq);
        }
    }

    public void printAllSubSequencesOfAString(String str) {

        int N = str.length();
        Set<String> subseq = new HashSet<>();
        printAllSubSequencesOfAString_Helper(str, 0, N, "", subseq);

        //output:
        System.out.println("All possible subsequences of string: " + subseq);
    }

    public boolean balancedParenthesisEvaluation(String s) {
        //https://leetcode.com/problems/valid-parentheses/
        Stack<Character> stack = new Stack<>();
        for (char ch : s.toCharArray()) {

            if (ch == '{' || ch == '[' || ch == '(') {
                stack.push(ch);
            } else if (!stack.isEmpty() && stack.peek() == '(' && ch == ')') {
                stack.pop();
            } else if (!stack.isEmpty() && stack.peek() == '{' && ch == '}') {
                stack.pop();
            } else if (!stack.isEmpty() && stack.peek() == '[' && ch == ']') {
                stack.pop();
            } else {
                return false;
            }
        }

        return stack.isEmpty();
    }

    public void firstNonRepeatingCharacterFromStream(String stream) {

        List<Character> list = new ArrayList<>();
        Set<Character> visited = new HashSet<>();
        for (int i = 0; i < stream.length(); i++) {
            char ch = stream.charAt(i);
            if (!visited.contains(ch)) {

                if (list.contains(ch)) {
                    list.remove((Character) ch);
                    visited.add(ch);
                } else {
                    list.add(ch);
                }
            }

            System.out.println("First non repeating character till " + stream.substring(0, i + 1));
            if (!list.isEmpty()) {
                System.out.println(list.get(0) + " ");
            } else {
                //when there are no non repeating char print #
                System.out.println("#");
            }
        }
    }

    public boolean wordBreak_Recursive(String str, Set<String> set) {

        //https://www.geeksforgeeks.org/word-break-problem-dp-32/
        int n = str.length();
        if (n == 0 || set.contains(str)) {
            return true;
        }

        for (int i = 1; i <= n; i++) {
            if (set.contains(str.substring(0, i)) && wordBreak_Recursive(str.substring(i, n), set)) {
                return true;
            }
        }
        return false;
    }

    public boolean wordBreak_DP_Problem(String str, Set<String> set) {
        //https://leetcode.com/problems/word-break/
        //https://leetcode.com/problems/word-break/discuss/1068441/Detailed-Explanation-of-Top-Down-and-Bottom-Up-DP
        //similar to longestIncreasingSubseq()
        boolean[] memo = new boolean[str.length() + 1];
        //base cond
        memo[0] = true; // str with no length is also true

        for (int i = 1; i <= str.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (set.contains(str.substring(j, i)) && memo[j]) {
                    memo[i] = true;
                    break;
                }
            }
        }
        return memo[str.length()];
    }

    public void longestSubstringWithoutRepeatingChar(String str) {

        //SLIDING WINDOW ALGO
        //https://leetcode.com/problems/longest-substring-without-repeating-characters/
        int n = str.length();
        int start = 0;
        int end = 0;
        int maxLen = 0;
        Map<Character, Integer> freq = new HashMap<>();
        String substr = "";
        while (end < n) {

            char chEnd = str.charAt(end);

            freq.putIfAbsent(chEnd, 0);

            if (freq.get(chEnd) < 1) {

                freq.put(chEnd, freq.get(chEnd) + 1);

                if (maxLen < (end - start + 1)) {
                    maxLen = end - start + 1;
                    substr = str.substring(start, start + maxLen);
                }
                end++;
            } else {
                char chStart = str.charAt(start);
                freq.put(chStart, freq.get(chStart) - 1);
                start++;
            }
        }

        //output:
        System.out.println("Longest subtring without repeating char: " + maxLen + " (" + substr + ")");
    }

    public void minimumWindowSubstring(String s, String t) {

        //Explanation: https://www.youtube.com/watch?v=nMaKzLWceFg&feature=youtu.be
        //SLIDING WINDOW ALGO
        //prepare the count map for string t 
        //to know how many char we need to find in string s
        Map<Character, Integer> tMap = new HashMap<>();
        for (char ch : t.toCharArray()) {
            tMap.put(ch, tMap.getOrDefault(ch, 0) + 1);
        }

        int tCount = 0;
        int start = 0;
        int end = 0;
        int minLenWindow = Integer.MAX_VALUE;
        int substrIndex = 0;
        int N = s.length();
        while (end < N) {

            char chEnd = s.charAt(end);
            //each we find the char in string s that is also in string t
            //we decreament the count of that char from map
            //to know how much of that char (chEnd) we needed and how much we have found
            if (tMap.containsKey(chEnd)) {
                //we choose 1 char from string s that is also in string t
                //so we will reduce the freq of this char, freq of this char >= 0
                //shows how much char is req before we found some extra occurences
                //of this char in string s because we don't care about the extra chars
                //only char[freq to 0]
                tMap.put(chEnd, tMap.get(chEnd) - 1);
                //eacg time we are reduciing freq we are taking a char so tCount++
                //below if block won't consider any extra occuerences of this char
                if (tMap.get(chEnd) >= 0) {
                    tCount++;
                }
            }

            //below while loop with cond tCount == t.length()
            //signifies that now we have all char of string t in string s
            //we can calculate window and at same time try to minimize the window size
            while (tCount == t.length()) {

                if (minLenWindow > (end - start + 1)) {
                    minLenWindow = end - start + 1;
                    substrIndex = start;
                }

                //adjust the start pointer now
                char chStart = s.charAt(start);
                if (tMap.containsKey(chStart)) {
                    tMap.put(chStart, tMap.get(chStart) + 1);
                    if (tMap.get(chStart) > 0) {
                        tCount--;
                    }
                }
                start++;
            }
            end++;
        }

        //output:
        String output = minLenWindow > s.length() ? "" : s.substring(substrIndex, substrIndex + minLenWindow);
        System.out.println("Min window substring containg all char of string t in string s: "
                + (minLenWindow > s.length() ? -1 : minLenWindow) + " : "
                + output);
    }

    public void countAllOccurencesOfPatternInGivenString(String txt, String pat) {

        //........................T: O(N)
        //SLIDING WINDOW
        Map<Character, Integer> patMap = new HashMap<>();
        for (char chPat : pat.toCharArray()) {
            patMap.put(chPat, patMap.getOrDefault(chPat, 0) + 1);
        }

        int totalOccurences = 0;
        int patCharCount = 0;
        int patLen = pat.length();
        int txtLen = txt.length();
        int start = 0;
        int end = 0;
        List<Integer> occuerencesIndex = new ArrayList<>();
        List<String> occuerencesSubstring = new ArrayList<>();
        while (end < txtLen) {

            char chEnd = txt.charAt(end);
            if (patMap.containsKey(chEnd)) {
                patMap.put(chEnd, patMap.get(chEnd) - 1);
                if (patMap.get(chEnd) >= 0) {
                    patCharCount++;
                }
            }

            //once window for pattern is reached
            //balance out by moving this window
            //and removing start from patCharCount
            while ((end - start + 1) == patLen) {
                //if the curr window also contains the same char as
                //that of pattern string (patCharCount == patLen)
                if (patCharCount == patLen) {
                    totalOccurences++;
                    occuerencesIndex.add(start);
                    occuerencesSubstring.add(txt.substring(start, start + patLen));
                }

                char chStart = txt.charAt(start);
                if (patMap.containsKey(chStart)) {
                    patMap.put(chStart, patMap.get(chStart) + 1);
                    if (patMap.get(chStart) > 0) {
                        patCharCount--;
                    }
                }
                start++;
            }
            end++;
        }

        //output:
        System.out.println("Total occurences: " + totalOccurences);
        System.out.println("All the indexes: " + occuerencesIndex);
        System.out.println("All the substring of anagrams: " + occuerencesSubstring);
    }

    public void partitionLabels(String str) {

        //problem: https://leetcode.com/problems/partition-labels/
        //explanation: https://youtu.be/5NCjHqx2v-k
        List<Integer> result = new ArrayList<>();
        List<String> substring = new ArrayList<>();
        int n = str.length();
        Map<Character, Integer> lastIndexes = new HashMap<>();
        for (int i = 0; i < n; i++) {
            lastIndexes.put(str.charAt(i), i);
        }

        int start = 0;
        int end = 0;
        int maxEnd = 0;
        while (end < n) {
            maxEnd = Math.max(maxEnd, lastIndexes.get(str.charAt(end)));
            if (end == maxEnd) {
                int partitionLen = end - start + 1;
                result.add(partitionLen);
                substring.add(str.substring(start, start + partitionLen));
                start = end + 1;
            }
            end++;
        }

        //output:
        System.out.println("All the partition lengths: " + result + " \nPartition strings: " + substring);
    }

    public void longestRepeatingCharacterByKReplacement(String str, int K) {
        //https://leetcode.com/problems/longest-repeating-character-replacement/
        //explanation: https://youtu.be/gqXU1UyA8pk
        //SLIDING WINDOW
        int[] charFreq = new int[26];
        int start = 0;
        int end = 0;
        int maxLen = 0;
        int mostFreqCharTill = 0;
        int n = str.length();
        while (end < n) {

            char chEnd = str.charAt(end);
            charFreq[chEnd - 'A']++;

            mostFreqCharTill = Math.max(mostFreqCharTill, charFreq[chEnd - 'A']);
            //let suppose curr win len (end - start + 1) has substr = ..."BABB"...
            //mostFreqCharTill = charFreq[B] = 3
            //now see out these BABB if you leave mostFreqChar(B) you are left with
            //A's like this winLen = 4, B = 3, A = 1
            //then winLen - mostFreqChar(B) ==> 4 - 3 = 1(i.e freq of A)
            //now you just have replace them, least freq chars (which we are allowed replace only K) 
            //if these least freq char are more than K we must minimize our win
            while ((end - start + 1) - mostFreqCharTill > K) {
                charFreq[str.charAt(start) - 'A']--;
                start++;
            }

            maxLen = Math.max(maxLen, end - start + 1);
            end++;
        }

        //output:
        System.out.println("Max length: " + maxLen);
    }

    private void generateBalancedParenthesis_Helper(int n, String curr, int open, int close, List<String> result) {

        //any req parenthesis of size n needs 2 * n parenthesis to balance
        //ex: if n = 1 => "{" to balance it need "}" which "{}" is n * 2
        if (curr.length() == 2 * n) {
            result.add(curr);
            return;
        }

        if (open < n) {
            generateBalancedParenthesis_Helper(n, curr + "{", open + 1, close, result);
        }

        if (close < open) {
            generateBalancedParenthesis_Helper(n, curr + "}", open, close + 1, result);
        }
    }

    public void generateBalancedParenthesis(int n) {
        //https://leetcode.com/problems/generate-parentheses
        //https://www.geeksforgeeks.org/print-all-combinations-of-balanced-parentheses/
        List<String> result = new ArrayList<>();
        generateBalancedParenthesis_Helper(n, "", 0, 0, result);
        //output:
        System.out.println("All balanced parenthesis: " + result);
    }

    public void scoreOfParenthesis(String str) {

        //https://leetcode.com/problems/score-of-parentheses
        //explanatin: https://youtu.be/jfmJusJ0qKM
        int score = 0;
        Stack<Integer> stack = new Stack<>();
        for (char ch : str.toCharArray()) {

            if (ch == '(') {
                stack.push(score);
                score = 0;
            } else {
                score = stack.pop() + Math.max(score * 2, 1);
            }
        }

        //output:
        System.out.println("Score: " + score);
    }

    public void minimumCharRemovalToMakeValidParenthesis(String str) {

        //https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses
        class CharIndex {

            char ch;
            int index;

            public CharIndex(char ch, int index) {
                this.ch = ch;
                this.index = index;
            }
        }

        StringBuilder sb = new StringBuilder();
        Stack<CharIndex> stack = new Stack<>();
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            sb.append(c);
            if (c == '(' || c == ')') {

                if (!stack.isEmpty() && stack.peek().ch == '(' && c == ')') {
                    stack.pop();
                } else {
                    stack.push(new CharIndex(c, i));
                }
            }
        }

        while (!stack.isEmpty()) {
            sb.deleteCharAt(stack.pop().index);
        }

        //output:
        System.out.println("Balanced string: " + sb.toString());
    }

    public boolean repeatedSubstringPattern(String str) {

        //https://leetcode.com/problems/repeated-substring-pattern/
        //explanantion: https://youtu.be/bClIZj66dVE
        int len = str.length();
        for (int i = len / 2; i >= 1; i--) {

            if (len % i == 0) {

                int numRepeats = len / i;
                String sub = str.substring(0, i);
                StringBuilder sb = new StringBuilder();
                for (int j = 0; j < numRepeats; j++) {
                    sb.append(sub);
                }

                if (sb.toString().equals(str)) {
                    return true;
                }
            }
        }

        return false;
    }

    private int longestPallindromicSubstring_ExpandFromMiddle(String str, int left, int right) {

        if (str == null || left > right) {
            return 0;
        }

        while (left >= 0 && right < str.length() && str.charAt(left) == str.charAt(right)) {
            left--;
            right++;
        }

        //total pallindromic char b/w left and right
        return right - left - 1;
    }

    public void longestPallindromicSubstring(String str) {

        //.....................T: O(N^2)
        //https://leetcode.com/problems/longest-palindromic-substring/
        //explanation: https://youtu.be/y2BD4MJqV20
        int start = 0;
        int end = 0;

        for (int i = 0; i < str.length(); i++) {

            //case to handle odd length string 
            //there will be exactly one middle char in that
            //ex: "racecar" middle char is 'e' 
            int len1 = longestPallindromicSubstring_ExpandFromMiddle(str, i, i);
            //case to handle even length string
            //the middle will in b/w the two char of (str.length / 2)  and ((str.length /2) + 1)
            //ex: "aabbaa" middle char will be in b/w b|b
            int len2 = longestPallindromicSubstring_ExpandFromMiddle(str, i, i + 1);

            int len = Math.max(len1, len2);

            if (len > end - start) {
                start = i - (len - 1) / 2;
                end = i + (len / 2);
            }
        }

        //output:
        System.out.println("Longest pallindromic substring: " + str.substring(start, end + 1));
    }

    public boolean stringComparisionAfterProcessingBackspaceChar(String a, String b) {

        //.............................T: O(M+N)
        //.............................S: O(1)
        //https://leetcode.com/problems/backspace-string-compare/solution/
        int i = a.length() - 1;
        int j = b.length() - 1;

        int skipA = 0;
        int skipB = 0;

        while (i >= 0 || j >= 0) {

            while (i >= 0) {
                if (a.charAt(i) == '#') {
                    skipA++;
                    i--;
                } else if (skipA > 0) {
                    skipA--;
                    i--;
                } else {
                    break;
                }
            }

            while (j >= 0) {
                if (b.charAt(j) == '#') {
                    skipB++;
                    j--;
                } else if (skipB > 0) {
                    skipB--;
                    j--;
                } else {
                    break;
                }
            }

            if (i >= 0 && j >= 0 && a.charAt(i) != b.charAt(j)) {
                return false;
            }

            //if one string length is smaller than others or run out of characters than the
            //other string
            if ((i >= 0) != (j >= 0)) {
                return false;
            }

            i--;
            j--;
        }
        return true;
    }

    public void stringZigZag(String str, int row) {

        if (row == 1) {
            //edge case
            System.out.println("String zig zag: " + str);
            return;
        }

        String[] zigZag = new String[row];
        Arrays.fill(zigZag, "");

        int iRow = 0;
        boolean isGoingDown = true;

        for (int i = 0; i < str.length(); i++) {

            zigZag[iRow] += str.charAt(i);

            if (isGoingDown) {
                iRow++;
            } else {
                iRow--;
            }

            if (iRow == row - 1) {
                isGoingDown = false;
            } else if (iRow == 0) {
                isGoingDown = true;
            }
        }

        StringBuilder sb = new StringBuilder();
        for (String s : zigZag) {
            sb.append(s);
        }

        //output
        System.out.println("String zig zag: " + sb.toString());
    }

    private List<String> stepsToOpenTheLock_AllCombinations(String originalLockStr,
            Set<String> deadEnd, Set<String> visitedCombination) {

        List<String> allCombinations = new ArrayList<>();

        for (int wheelNumber = 0; wheelNumber < 4; wheelNumber++) {

            char[] lockStrClone = String.valueOf(originalLockStr).toCharArray();

            //backwards combinations 2->1, 8->7 so on...
            if (lockStrClone[wheelNumber] == '0') {
                lockStrClone[wheelNumber] = '9';
            } else {
                lockStrClone[wheelNumber] = (char) ((int) lockStrClone[wheelNumber] - 1);
            }

            String newLockStr = String.valueOf(lockStrClone);

            //new lock string pattern should not be a dead end pattern
            //and should not be a already-generated-pattern 
            //(Set->add(data) returns true if data is not already present else returns false) 
            if (!deadEnd.contains(newLockStr) && visitedCombination.add(newLockStr)) {
                allCombinations.add(newLockStr);
            }

            //upwards combinatons 1->2, 5->6 so on
            lockStrClone = String.valueOf(originalLockStr).toCharArray();
            if (lockStrClone[wheelNumber] < '9') {

                lockStrClone[wheelNumber] = (char) ((int) lockStrClone[wheelNumber] + 1);
                newLockStr = String.valueOf(lockStrClone);

                //new lock string pattern should not be a dead end pattern
                //and should not be a already-generated-pattern 
                //(Set->add(data) returns true if data is not already present else returns false) 
                if (!deadEnd.contains(newLockStr) && visitedCombination.add(newLockStr)) {
                    allCombinations.add(newLockStr);
                }
            }
        }
        return allCombinations;
    }

    public int stepsToOpenTheLock(String[] deadends, String target) {

        //https://leetcode.com/problems/open-the-lock
        //https://leetcode.com/problems/open-the-lock/discuss/1123396/Java-BFS-with-very-detailed-explanations
        String start = "0000";
        Set<String> deadEnd = new HashSet<>(Arrays.asList(deadends));
        Set<String> visitedCombination = new HashSet<>();
        Queue<String> q = new LinkedList<>();

        int steps = 0;

        if (start.equals(target)) {
            return steps;
        }

        if (deadEnd.contains(start)) {
            return -1;
        }

        visitedCombination.add(start);
        q.add(start);

        while (!q.isEmpty()) {

            steps++;
            int size = q.size();

            for (int i = 0; i < size; i++) {

                String lockStr = q.poll();
                List<String> allCombinations = stepsToOpenTheLock_AllCombinations(lockStr, deadEnd, visitedCombination);
                for (String combination : allCombinations) {
                    if (combination.equals(target)) {
                        return steps;
                    }
                    q.add(combination);
                }
            }
        }

        return -1;
    }

    public void longestSubstringWithKUniqueCharacter(String str, int K) {

        //SLIDING WINDOW
        //https://practice.geeksforgeeks.org/problems/longest-k-unique-characters-substring0853/1#
        int n = str.length();
        int start = 0;
        int end = 0;
        int maxLen = 0;
        int index = 0;
        Map<Character, Integer> map = new HashMap<>();

        while (end < n) {

            char chEnd = str.charAt(end);
            map.put(chEnd, map.getOrDefault(chEnd, 0) + 1);

            if (map.size() == K) {
                if (maxLen < end - start + 1) {
                    maxLen = end - start + 1;
                    index = start;
                }
            }

            while (map.size() > K) {
                char chStart = str.charAt(start);
                map.put(chStart, map.get(chStart) - 1);
                if (map.get(chStart) <= 0) {
                    map.remove(chStart);
                }
                start++;
            }
            end++;
        }

        //output:
        System.out.println("Max length substring with K unique char: " + (maxLen == 0 ? -1 : maxLen));
        if (maxLen != 0) {
            System.out.println("Substring with K unique char: " + str.substring(index, index + maxLen));
        }
    }

    public void smallestSubstringWithKUniqueCharacter(String str, int K) {
        //https://www.codingninjas.com/codestudio/problems/smallest-subarray-with-k-distinct-elements_630523?leftPanelTab=0
        int n = str.length();
        Map<Character, Integer> freq = new HashMap<>();
        int start = 0;
        int end = 0;
        int firstIndex = 0;
        int minLen = n + 1;
        while (end < n) {
            char chEnd = str.charAt(end);
            freq.put(chEnd, freq.getOrDefault(chEnd, 0) + 1);

            //if in freq map we have got all the K unique char
            //we will try to minimize the window to find the smallest substring
            while (freq.size() == K) {
                if (end - start + 1 < minLen) {
                    minLen = end - start + 1;
                    firstIndex = start;
                }

                char chStart = str.charAt(start);
                freq.put(chStart, freq.getOrDefault(chStart, 0) - 1);
                if (freq.get(chStart) <= 0) {
                    freq.remove(chStart);
                }
                start++;
            }
            end++;
        }
        //output
        if (minLen == n + 1) {
            System.out.println("Smallest substring with K unique character not possible: -1");
            return;
        }
        System.out.println("Smallest substring with K unique character: " + minLen);
        System.out.println("Smallest substring with K unique character substring: "
                + str.substring(firstIndex, firstIndex + minLen));
    }

    public void smallestSubarrayWithKDistinctElements(int[] nums, int K) {
        //https://www.codingninjas.com/codestudio/problems/smallest-subarray-with-k-distinct-elements_630523?leftPanelTab=0
        //based on smallestSubstringWithKUniqueCharacter()
        int n = nums.length;
        Map<Integer, Integer> freq = new HashMap<>();
        int start = 0;
        int end = 0;
        int firstIndex = 0;
        int lastIndex = 0;
        int minLen = n + 1;
        while (end < n) {
            int valEnd = nums[end];
            freq.put(valEnd, freq.getOrDefault(valEnd, 0) + 1);

            //if in freq map we have got all the K distinct elements
            //we will try to minimize the window to find the smallest subarray
            while (freq.size() == K) {
                if (end - start + 1 < minLen) {
                    minLen = end - start + 1;
                    firstIndex = start;
                    lastIndex = end;
                }

                int valStart = nums[start];
                freq.put(valStart, freq.getOrDefault(valStart, 0) - 1);
                if (freq.get(valStart) <= 0) {
                    freq.remove(valStart);
                }
                start++;
            }
            end++;
        }
        //output
        if (minLen == n + 1) {
            System.out.println("Smallest subarray with K distinct elements not possible: -1");
            return;
        }
        System.out.println("Smallest subarray with K distinct elements: " + minLen);
        for (int i = firstIndex; i <= lastIndex; i++) {
            System.out.print(nums[i] + " ");
        }
        System.out.println();
    }

    public void largestNumberFromSetOfNumbers(String[] nums) {

        //https://leetcode.com/problems/largest-number/
        //https://practice.geeksforgeeks.org/problems/largest-number-formed-from-an-array1117/1
        //https://www.geeksforgeeks.org/given-an-array-of-numbers-arrange-the-numbers-to-form-the-biggest-number/
        //ex 3, 30, 34, 5, 9
        //int priority queue will have the numbers based on the max(ab, ba) combinations
        //if a  = 3, b = 30 then ab = 330 & ba = 303 max(330, 303) so on
        //a, b are going to store in this camparion only
        PriorityQueue<String> combinations = new PriorityQueue<>((a, b) -> {
            String ab = a + b;
            String ba = b + a;
//            System.out.println(ab + " & " + ba + " compareTo " + ab.compareTo(ba));
            return ab.compareTo(ba) > 0 ? -1 : 1;
        });

        StringBuilder sb = new StringBuilder();

        combinations.addAll(Arrays.asList(nums));

        while (!combinations.isEmpty()) {
            sb.append(combinations.poll());
        }

        //remove any starting zeroes
        while (sb.length() > 1 && sb.charAt(0) == '0') {
            sb.deleteCharAt(0);
        }

        //output
        System.out.println("Largest number formed from the given set of numbers: " + sb.toString());
    }

    public long smallestNumber(long num) {
        //https://leetcode.com/problems/smallest-value-of-the-rearranged-number
        boolean isNegative = num < 0;
        //converting the given num to its positive value
        num = Math.abs(num);
        //convert the given num to its char[] arr form
        char[] numArr = String.valueOf(num).toCharArray();
        int n = numArr.length;
        //sorting the char[] arr so that all the smaller digits
        //comes first, we will just need to pick the one by one and form a string
        Arrays.sort(numArr);

        int index = 0;
        String numStr = "";

        //in case if our num was negative, then picking the digits from the end
        //will give us the smaller number
        //like +5 > +2 but -2 > -5
        //sorted arr[] = [2,5] for positive pick from left to right
        //for negative pick from right to left
        if (isNegative) {
            index = n - 1;
            for (; index >= 0; index--) {
                numStr += numArr[index];
            }
            return Long.parseLong(numStr) * -1;
        }

        //we are working for positive num cases, there is a edge case
        //if our inital num had zero in it
        //like 54321000 ==> sorted arr[] = [0,0,0,1,2,3,4,5]
        //now if we simply pick from left to right our numStr will formed as
        //00012345 which is equivalent to 12345 but without zeroes we can't omit
        //all zeroes and we can't keep zeroes in starting
        //so fetch all the zeroes from starting and form zeroes str = "000"
        //below loop will break at index where numArr[index] != '0' that means '1'
        //we will simply put all the zeroes after first non-zero digit (i.e, '1')
        String zeroes = "";
        while (index < n && numArr[index] == '0') {
            zeroes += numArr[index++];
        }
        //index here is at first non-zero digit in sorted numArr
        numStr = index < n ? numArr[index] + zeroes : zeroes;
        //we have considered the first non-zero digit
        //like this "1" + "000" = "1000"
        //now we can start from next index onwards and starting picking
        //digits one by one
        index++;
        for (; index < n; index++) {
            numStr += numArr[index];
        }
        return Long.parseLong(numStr);
    }

    public void stringCompression(char[] arr) {
        //https://leetcode.com/problems/string-compression/
        //explanation: https://youtu.be/IhJgguNiYYk
        int n = arr.length;
        int start = 0;
        int end = 0;
        int index = 0;
        while (end < n) {
            start = end;
            while (end < n && arr[start] == arr[end]) {
                end++;
            }
            arr[index++] = arr[start];
            int count = end - start;
            if (count <= 1) {
                continue;
            }
            for (char digit : String.valueOf(count).toCharArray()) {
                arr[index++] = digit;
            }
        }

        //output
        for (int i = 0; i < index; i++) {
            System.out.print(arr[i]);
        }
        System.out.println();
    }

    public int closestStringDistance(List<String> strs, String w1, String w2) {

        //https://practice.geeksforgeeks.org/problems/closest-strings0611/1#
        if (w1.equals(w2)) {
            return 0;
        }
        int n = strs.size();
        int dist = n + 1;
        //find first index of any of the word
        int first = 0;
        int i = 0;
        for (; i < n; i++) {
            if (strs.get(i).equals(w1) || strs.get(i).equals(w2)) {
                first = i;
                break;
            }
        }

        //find other index of other word which is not found
        while (i < n) {

            if (strs.get(i).equals(w1) || strs.get(i).equals(w2)) {
                if (!strs.get(first).equals(strs.get(i)) && (i - first) < dist) {
                    dist = i - first;
                }
                first = i;
            }
            i++;
        }

        return dist;
    }

    public boolean longPressedNames(String name, String typed) {

        //https://leetcode.com/problems/long-pressed-name/
        char[] nameCh = name.toCharArray();
        char[] typedCh = typed.toCharArray();

        int i = 0;
        int j = 0;

        while (i < name.length() && j < typed.length()) {

            if (nameCh[i] == typedCh[j]) {
                i++;
                j++;
            } else if (j > 0 && typedCh[j] == typedCh[j - 1]) {
                j++;
            } else {
                return false;
            }
        }

        if (i != name.length()) {
            return false;
        } else {
            while (j < typed.length()) {
                if (typedCh[j] != typedCh[j - 1]) {
                    return false;
                }
                j++;
            }
        }
        return true;
    }

    private void interleavingOfTwoStrings_Helper(String a, String b,
            int m, int n,
            StringBuilder sb, List<String> res) {

        if (sb.length() == a.length() + b.length()) {
            res.add(sb.toString());
            return;
        }

        if (m < a.length()) {
            sb.append(a.charAt(m));
            interleavingOfTwoStrings_Helper(a, b,
                    m + 1, n,
                    sb, res);
            sb.deleteCharAt(sb.length() - 1);
        }

        if (n < b.length()) {
            sb.append(b.charAt(n));
            interleavingOfTwoStrings_Helper(a, b,
                    m, n + 1,
                    sb, res);
            sb.deleteCharAt(sb.length() - 1);
        }
    }

    public void interleavingOfTwoStrings(String a, String b) {

        //.............................T: O(M + N) M = a.length(), N = b.length()
        //https://www.geeksforgeeks.org/print-all-interleavings-of-given-two-strings/
        StringBuilder sb = new StringBuilder();
        List<String> res = new ArrayList<>();
        interleavingOfTwoStrings_Helper(a, b, 0, 0, sb, res);

        //output
        System.out.println("Interleaving of two strings: " + res);
    }

    private void printAllPermutationOfString_Helper(char[] chArr, int index, List<String> res) {

        if (index == chArr.length) {
            res.add(String.valueOf(chArr));
        }

        for (int i = index; i < chArr.length; i++) {

            //swap
            char temp = chArr[index];
            chArr[index] = chArr[i];
            chArr[i] = temp;

            //recursive call
            printAllPermutationOfString_Helper(chArr, index + 1, res);

            //swap back the char to its original place
            //to keep chaArr original before swaping and for next recursive call
            temp = chArr[index];
            chArr[index] = chArr[i];
            chArr[i] = temp;
        }
    }

    public void printAllPermutationOfDistinctCharInString(String str) {
        //.............................T: O(N!) factorial(N), N = str.length()
        //https://leetcode.com/problems/permutations/
        //https://leetcode.com/problems/permutations-ii/
        //explanantion: https://www.youtube.com/watch?v=GuTPwotSdYw
        //we are fixing char at index location perform operations on rest of 
        //remaining chArr so ech time N * N-1 * N-2 ....0 = N!
        char[] chArr = str.toCharArray();
        List<String> res = new ArrayList<>();

        printAllPermutationOfString_Helper(chArr, 0, res);

        //output
        System.out.println("All permutation of given string: " + res);
    }

    public void longestSubstringHavingAllVowelsInOrder(String word) {

        //https://leetcode.com/problems/longest-substring-of-all-vowels-in-order/
        /*
         the string only contains char [a,e,i,o,u]
         */
        int maxLen = 0;
        int currLen = 1;
        int vowelCount = 1;
        int n = word.length();

        for (int i = 1; i < n; i++) {
            if (word.charAt(i - 1) == word.charAt(i)) {
                //if curr and prev char is same/repeating
                currLen++;
            } else if (word.charAt(i - 1) < word.charAt(i)) {
                //this if increment vowelCount because
                //actual vowel order is a, e, i, o, u 
                //where a < e < i < o < u (ASCII values)
                vowelCount++;
                currLen++;
            } else {
                //reset both variables when vowel order doesn't matches
                vowelCount = 1;
                currLen = 1;
            }

            if (vowelCount == 5) {
                maxLen = Math.max(maxLen, currLen);
            }
        }

        //output
        System.out.println("Longest substring having all vowels in order: " + maxLen);
    }

    public void arithematicExpressionEvaluationBasicCalculator(String expr) {

        //https://leetcode.com/problems/basic-calculator-ii/
        int n = expr.length();
        int currNum = 0;
        int lastNum = 0;
        int result = 0;

        char opr = '+';

        for (int i = 0; i < n; i++) {
            char ch = expr.charAt(i);
            if (Character.isDigit(ch)) {
                currNum = (currNum * 10) + (ch - '0');
            }

            if (!Character.isDigit(ch) && ch != ' ' || i == n - 1) {
                if (opr == '+' || opr == '-') {
                    result += lastNum;
                    lastNum = (opr == '+') ? currNum : -currNum;
                } else if (opr == '*') {
                    lastNum = lastNum * currNum;
                } else if (opr == '/') {
                    lastNum = lastNum / currNum;
                }
                opr = ch;
                currNum = 0;
            }
        }
        result += lastNum;

        //output
        System.out.println("Evaluation: " + result);
    }

    public void evaluateBracketPatternAndReplaceWithGivenWord(String s, List<List<String>> replaceWith) {

        //https://leetcode.com/problems/evaluate-the-bracket-pairs-of-a-string
        Map<String, String> map = new HashMap<>();
        for (List<String> list : replaceWith) {
            map.put(list.get(0), list.get(1));
        }

        boolean bracketStart = false;
        String key = "";
        StringBuilder sb = new StringBuilder();
        int i = 0;
        int n = s.length();

        while (i < n) {
            char ch = s.charAt(i++);

            //all such char which are not of pattern : (SomeChar)
            if (Character.isAlphabetic(ch) && bracketStart == false) {
                sb.append(ch);
            } else if (Character.isAlphabetic(ch) && bracketStart == true) {
                //all such char which are part of pattern : (SomeChar)
                //key = SomeChar not including (, )
                key += ch;
            } else if (ch == '(') {
                //if bracket starts
                bracketStart = true;
            } else if (bracketStart == true && ch == ')') {
                //if bracket flag is true and ending bracket has arrived
                //means at this point, key = SomeChar has been formed
                String replace = map.getOrDefault(key, "?"); //if key is not provided, replace with ?
                sb.append(replace);
                key = ""; //reset for next key generation
                bracketStart = false; //reset to find next bracket start
            }
        }

        //output
        System.out.println("Bracket pattern evaluated: " + sb.toString().trim());
    }

    private void allPhoneDigitLetterCombinations_Helper(String digits,
            int digitIndex, String curr,
            Map<Integer, String> map, List<String> combinations) {

        if (digitIndex == digits.length()) {
            combinations.add(curr);
            return;
        }

        //letters in digit of digits in phone keypad
        String keyChars = map.get(digits.charAt(digitIndex) - '0');
        for (char ch : keyChars.toCharArray()) {
            allPhoneDigitLetterCombinations_Helper(digits,
                    digitIndex + 1, curr + ch,
                    map, combinations);
        }
    }

    public void allPhoneDigitLetterCombinations(String digits) {

        //https://leetcode.com/problems/letter-combinations-of-a-phone-number
        /*
         think of keypad based phones, letters are arranged in some number of keypad
         digits provided, form all combinations of letters from each digit in digits
         */
        List<String> combinations = new ArrayList<>();

        if (digits.length() == 0) {
            return;
        }

        //representation of phone keypad
        Map<Integer, String> map = new HashMap<>();
        map.put(2, "abc");
        map.put(3, "def");
        map.put(4, "ghi");
        map.put(5, "jkl");
        map.put(6, "mno");
        map.put(7, "pqrs");
        map.put(8, "tuv");
        map.put(9, "wxyz");

        allPhoneDigitLetterCombinations_Helper(digits, 0, "", map, combinations);

        //output
        System.out.println("All combinations as the digits given: ");
        combinations.stream().forEach(s -> System.out.print(s + " "));
        System.out.println();
    }

    public boolean checkIfOneCharSwapMakeStringEqual(String s1, String s2) {

        //https://leetcode.com/problems/check-if-one-string-swap-can-make-strings-equal/
        if (s1.length() != s2.length()) {
            return false;
        }

        int count = 0;
        char a1 = '.';
        char a2 = '.';
        char b1 = '.';
        char b2 = '.';

        for (int i = 0; i < s1.length(); i++) {

            if (s1.charAt(i) != s2.charAt(i)) {
                count++;
                if (count == 1) {
                    a1 = s1.charAt(i);
                    a2 = s2.charAt(i);
                } else if (count == 2) {
                    b1 = s1.charAt(i);
                    b2 = s2.charAt(i);
                } else {
                    return false;
                }
            }
        }

        return count == 0 || (a1 == b2 && a2 == b1);
    }

    public void largestSubstringBetweenTwoSameChar(String str) {

        //https://leetcode.com/problems/largest-substring-between-two-equal-characters/
        /*
         ex "abca", two same char a & a, set of char between them bc of len = 2
         */
        int[] startingIndexes = new int[26];
        Arrays.fill(startingIndexes, -1);
        int max = -1;
        int end = 0;
        while (end < str.length()) {
            char ch = str.charAt(end);
            if (startingIndexes[ch - 'a'] == -1) {
                //starting index of any char
                startingIndexes[ch - 'a'] = end;
            } else {
                //char ch is seen more than once in else{} 
                //calculate max set of char b/w curr seen char ch and prev
                //index of same char ch
                //end = index of curr ch - indexes[ch - 'a'] index of same char ch - 1
                max = Math.max(max, end - startingIndexes[ch - 'a'] - 1);
            }
            end++;
        }

        /*
         partitionLabel() approach
         int[] lastCharIndexes = new int[26];
         for(int i = 0; i < s.length(); i++){
         lastCharIndexes[s.charAt(i) - 'a'] = i;
         }
        
         int maxLen = -1;
        
         for(int i = 0; i < s.length(); i++){
         char ch = s.charAt(i);
         int end = lastCharIndexes[ch - 'a'];
         maxLen = Math.max(maxLen, end - i - 1);
         }
        
         return maxLen;
         */
        //output
        System.out.println("Largest subtring of chars between two same char: " + max);
    }

    public boolean determineIfTwoStringCanBeMadeClose(String str1, String str2) {

        //https://leetcode.com/problems/determine-if-two-strings-are-close/
        if (str1.length() != str2.length()) {
            return false;
        }

        Map<Character, Integer> str1Map = new HashMap<>();
        for (char c : str1.toCharArray()) {
            str1Map.put(c, str1Map.getOrDefault(c, 0) + 1);
        }

        Map<Character, Integer> str2Map = new HashMap<>();
        for (char c : str2.toCharArray()) {
            str2Map.put(c, str2Map.getOrDefault(c, 0) + 1);
        }

        //same char of str1 should be in str2
        for (char key : str1Map.keySet()) {
            if (!str2Map.containsKey(key)) {
                return false;
            }
        }

        //same char of str2 should be in str1
        for (char key : str2Map.keySet()) {
            if (!str1Map.containsKey(key)) {
                return false;
            }
        }

        //values list of their freq should be equal
        List<Integer> str1Val = str1Map.values().stream().sorted().collect(Collectors.toList());
        List<Integer> str2Val = str2Map.values().stream().sorted().collect(Collectors.toList());

        if (!str1Val.equals(str2Val)) {
            return false;
        }
        return true;
    }

    public void minCharacterRequiredToMakeStringTAnagramOfS(String s, String t) {
        //https://leetcode.com/problems/minimum-number-of-steps-to-make-two-strings-anagram/
        Map<Character, Long> sMap = s.chars().mapToObj(c -> (char) c)
                .collect(Collectors.groupingBy(
                        Function.identity(),
                        Collectors.counting()
                ));

        Map<Character, Long> tMap = t.chars().mapToObj(c -> (char) c)
                .collect(Collectors.groupingBy(
                        Function.identity(),
                        Collectors.counting()
                ));

        //if freq of a char in t is less than the freq of same char in s
        //freq(t.char, defaultIfSameCharNotPresent = 0) < freq(s.char)
        //req chars will be freq(s.char) - freq(t.char, defaultIfSameCharNotPresent = 0)
        long req = 0;
        for (char key : sMap.keySet()) {
            long freqT = tMap.getOrDefault(key, 0l);
            if (freqT < sMap.get(key)) {
                req += sMap.get(key) - freqT;
            }
        }

        //output
        System.out.println("Minimum character required to make string t anagram of s: " + req);
    }

    public void minCharacterRemovedToMakeStringTAndSAnagrams(String s, String t) {

        //https://www.geeksforgeeks.org/remove-minimum-number-characters-two-strings-become-anagram/
        //OPTIMISED from above link
        Map<Character, Long> sMap = s.chars().mapToObj(c -> (char) c)
                .collect(Collectors.groupingBy(
                        Function.identity(),
                        Collectors.counting()
                ));
        //reduce the freq of char of string t from freq Map(sMap) of string s
        //this reduce will balance all the common char of both strings(anagrams)
        //all the char left in map should be removed, where String T & S are anagrams
        for (char ch : t.toCharArray()) {
            sMap.put(ch, sMap.getOrDefault(ch, 0l) - 1l);
        }

        long removed = 0;
        for (char c = 'a'; c <= 'z'; c++) {
            removed += Math.abs(sMap.getOrDefault(c, 0l));
        }

        //output
        System.out.println("Minimum character removed to make string t & s anagrams: " + removed);
    }

    public void convertPostfixToInfixExpression(String postfix) {

        //https://www.geeksforgeeks.org/postfix-to-infix/
        //Approach similar to postfix expression evaluation
        Stack<String> stack = new Stack<>();

        for (char ch : postfix.toCharArray()) {

            if (Character.isLetterOrDigit(ch)) {
                stack.push(ch + "");
            } else {
                String c2 = stack.pop();
                String c1 = stack.pop();

                String str = "(";
                str += c1;
                str += ch;
                str += c2;
                str += ")";
                stack.push(str);
            }
        }

        //output
        System.out.println("Infix expression: " + stack.peek());
    }

    private int convertInfixToPostfixExpression_OperatorPrecedence(char ch) {
        switch (ch) {
            case '+':
            case '-':
                return 1;
            case '*':
            case '/':
                return 2;
            case '^':
                return 3;
            default:
                return -1;
        }
    }

    public void convertInfixToPostfixExpression(String infix) {

        Stack<Character> bracketsAndOperatorStack = new Stack<>();
        StringBuilder sb = new StringBuilder();

        for (char ch : infix.toCharArray()) {

            if (Character.isLetterOrDigit(ch)) {
                sb.append(ch);
            } else if (ch == '(') {
                bracketsAndOperatorStack.push(ch);
            } else if (ch == ')') {
                while (!bracketsAndOperatorStack.isEmpty() && bracketsAndOperatorStack.peek() != '(') {
                    sb.append(bracketsAndOperatorStack.pop());
                }
                //this pop is to remove the '(' that broke the while loop above
                bracketsAndOperatorStack.pop();
            } else {
                while (!bracketsAndOperatorStack.isEmpty()
                        && convertInfixToPostfixExpression_OperatorPrecedence(ch)
                        <= convertInfixToPostfixExpression_OperatorPrecedence(bracketsAndOperatorStack.peek())) {
                    sb.append(bracketsAndOperatorStack.pop());
                }
                bracketsAndOperatorStack.push(ch);
            }
        }

        while (!bracketsAndOperatorStack.isEmpty()) {
            sb.append(bracketsAndOperatorStack.pop());
        }

        //output
        System.out.println("Postfix expression: " + sb.toString());
    }

    public void flipStringToMonotoneIncrease(String str) {
        //https://leetcode.com/problems/flip-string-to-monotone-increasing/
        /*
         A string of '0's and '1's is monotone increasing if it consists of some 
         number of '0's (possibly 0), followed by some number of '1's 
         (also possibly 0.)
         //example of monotones
         ex: 00000
         ex: 11111
         ex: 00011
         */
        int one = 0;
        int flip = 0;
        for (char ch : str.toCharArray()) {
            //count 1's that are coming after the consecutive 0's(possibly none)
            //intutions says that any 0 coming after 1s those will be counted in flip++
            //if no 1 has been counted so far one will remain 0 indicating
            //we don't need to do any flips ==> flips = min(one, flip)
            if (ch == '1') {
                one++;
            } else {
                flip++;
            }
            flip = Math.min(one, flip);
        }

        //output
        System.out.println("Min flip of either 0 or 1 to create monotone string: " + flip);
    }

    public void minDeletionCostToAvoidRepeatingChar(String str, int[] cost) {

        //https://leetcode.com/problems/minimum-deletion-cost-to-avoid-repeating-letters/
        /*
         ex: "abaac", cost[] = {1,2,3,4,5}
         delete char = 'a' from index 2 which cost[2] = 3 and by that str will be
         abac ('a' at index 2 removed) and there are no char repeating after itself
         */
        int n = str.length();
        int res = 0;
        for (int i = 0; i < n - 1; i++) {
            if (str.charAt(i) == str.charAt(i + 1)) {
                if (cost[i] < cost[i + 1]) {
                    res += cost[i]; //min(cost[i], cost[i + 1]) = cost[i]
                } else {
                    res += cost[i + 1]; //min(cost[i], cost[i + 1]) = cost[i + 1]
                    //since here in else block the cost[i + 1] < cost[i]
                    //updating cost[i + 1] with greater cost[i] signifies that we have
                    //deleted a char with min cost (here cost[i + 1]) but we still
                    //have a char with a value cost[i], incase we found a same char at s[(i + 1) + 1]
                    //ex: s = "aaa" cost[4,1,1] first mid 'a' will be deleted
                    //as s[0] == 'a' have cost[0] == 4 && s[0 + 1] == 'a' have cost[0 + 1] == 1
                    //minCost = 1 but at the same time update cost[0 + 1] = cost[0] == 4
                    //meaning s[0 + 1] & cost[0 + 1] is deleted
                    cost[i + 1] = cost[i];
                }
            }
        }

        //output:
        System.out.println("Deletion cost: " + res);
    }

    public void removeAdjacentDuplicateKCharInString(String str, int K) {

        //https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string
        //https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string-ii/
        class Pair {

            char ch;
            int count;

            public Pair(char ch, int count) {
                this.ch = ch;
                this.count = count;
            }
        }

        Stack<Pair> stack = new Stack();
        for (char c : str.toCharArray()) {

            if (!stack.isEmpty() && stack.peek().ch == c) {
                stack.push(new Pair(c, stack.peek().count + 1));

                if (stack.peek().count == K) {
                    while (!stack.isEmpty() && stack.peek().ch == c) {
                        stack.pop();
                    }
                }
            } else {
                stack.push(new Pair(c, 1));
            }
        }

        String res = "";
        while (!stack.isEmpty()) {
            res = stack.pop().ch + res;
        }

        //output
        System.out.println("Remove K adjacent char and print remaining: " + res);
    }

    public int minSwapRequiredToMakeBinaryStringAlternate(String binaryString) {
        //..................................T: O(N), N = length of string
        //..................................S: O(1), countArr[2][2], 4 storage area in arr
        //https://leetcode.com/problems/minimum-number-of-swaps-to-make-the-binary-string-alternating/
        //Sol: https://leetcode.com/problems/minimum-number-of-swaps-to-make-the-binary-string-alternating/discuss/1990131/Java-counting

        /*
         row\col    count 0, count 1
         evenIndex 0[   0        0  ]             
         oddIndex  1[   0        0  ]   
         */
        int evenIndex = 0;
        int oddIndex = 1;
        int[][] countZeroAndOne = new int[2][2];
        for (int i = 0; i < binaryString.length(); i++) {
            char ch = binaryString.charAt(i);
            countZeroAndOne[i % 2][ch - '0']++;
        }

        if ((countZeroAndOne[evenIndex][0] == 0 && countZeroAndOne[oddIndex][1] == 0)
                || (countZeroAndOne[evenIndex][1] == 0 && countZeroAndOne[oddIndex][0] == 0)) {
            return 0;
        }

        if ((countZeroAndOne[evenIndex][0] != countZeroAndOne[oddIndex][1])
                && (countZeroAndOne[evenIndex][1] != countZeroAndOne[oddIndex][0])) {
            return -1;
        }

        int res1 = countZeroAndOne[evenIndex][0] == countZeroAndOne[oddIndex][1]
                ? countZeroAndOne[evenIndex][0]
                : Integer.MAX_VALUE;

        int res2 = countZeroAndOne[evenIndex][1] == countZeroAndOne[oddIndex][0]
                ? countZeroAndOne[evenIndex][1]
                : Integer.MAX_VALUE;

        return Math.min(res1, res2);
    }

    public void minimumIndexSumOfTwoStringArray(String[] list1, String[] list2) {
        //................................T: O(M + N), M = list1.length, N = list2.length
        //................................S: O(M), hashmap for list1
        //https://leetcode.com/problems/minimum-index-sum-of-two-lists/
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i < list1.length; i++) {
            map.put(list1[i], i);
        }

        List<String> strs = new ArrayList<>();
        int minIndexSum = Integer.MAX_VALUE;
        for (int i = 0; i < list2.length && i <= minIndexSum; i++) {
            if (map.containsKey(list2[i])) {
                int currIndexSum = map.get(list2[i]) + i;
                if (currIndexSum < minIndexSum) {
                    minIndexSum = currIndexSum;
                    strs.clear();
                    strs.add(list2[i]);
                } else if (currIndexSum == minIndexSum) {
                    strs.add(list2[i]);
                }
            }
        }
        //Output
        //return  strs.toArray(new String[strs.size()]);
        System.out.println("Min index sum: " + strs);
    }

    public int firstUniqueCharacterInString(String str) {
        //https://leetcode.com/problems/first-unique-character-in-a-string/
        Map<Character, Long> freq = str.chars().mapToObj(c -> (char) c)
                .collect(Collectors.groupingBy(
                        Function.identity(),
                        Collectors.counting()
                ));
        for (int i = 0; i < str.length(); i++) {
            if (freq.get(str.charAt(i)) == 1) {
                return i;
            }
        }
        return -1;
    }

    private boolean substringWithConcatenationsOfGivenWords_Check(int index, String str,
            int lengthPerWord,
            int subStringLength,
            int wordsLength,
            Map<String, Integer> wordsCounter) {

        Map<String, Integer> remaining = new HashMap<>(wordsCounter);
        int wordsUsed = 0;

        for (int i = index; i < index + subStringLength; i += lengthPerWord) {
            String sub = str.substring(i, i + lengthPerWord);
            if (remaining.getOrDefault(sub, 0) != 0) {
                remaining.put(sub, remaining.get(sub) - 1);
                wordsUsed++;
            } else {
                return false;
            }
        }
        return wordsUsed == wordsLength;
    }

    public void substringWithConcatenationsOfGivenWords(String str, String[] words) {
        //https://leetcode.com/problems/substring-with-concatenation-of-all-words/
        //https://leetcode.com/problems/substring-with-concatenation-of-all-words/solution/
        //BRUTE FORCE approach
        //make below variable global so that there is no need to pass them as 
        //method paramters
        int N = str.length();
        int wordsLength = words.length;
        //length of each word will be same in given words[]
        int lengthPerWord = words[0].length();
        //total length of the string formed from concatenating the words in words[]
        int subStringLength = wordsLength * lengthPerWord;
        Map<String, Integer> wordsCounter = new HashMap<>();

        List<Integer> indexes = new ArrayList<>();

        for (String word : words) {
            wordsCounter.put(word, wordsCounter.getOrDefault(word, 0) + 1);
        }

        //all substrings of size subStringLength in given str
        for (int i = 0; i < N - subStringLength + 1; i++) {
            if (substringWithConcatenationsOfGivenWords_Check(i, str,
                    lengthPerWord,
                    subStringLength,
                    wordsLength,
                    wordsCounter)) {
                indexes.add(i);
            }
        }
        //output
        System.out.println("Indexes of all the subtrings that contains concatenations of given words[](word can be in jumbled ordered) "
                + indexes);
    }

    public void maximumWordsThatYouCanType(String text, String brokenLetters) {
        //https://leetcode.com/problems/maximum-number-of-words-you-can-type/
        int brokenWordCount = 0;
        String[] splitWords = text.split(" ");
        char[] brokenChars = brokenLetters.toCharArray();
        //using standard lib indexOf()
//        for (String word : splitWords) {
//            for (char brokenChar : brokenChars) {
//                if (word.indexOf(brokenChar) >= 0) {
//                    brokenWordCount++;
//                    break;
//                }
//            }
//        }

        //using data structure
        Set<Character> isBrokenCharPresentInWord = new HashSet<>();
        for (String word : splitWords) {
            //to check if brokenChar is there in  given word in O(1) via contains()
            for (char charInWord : word.toCharArray()) {
                isBrokenCharPresentInWord.add(charInWord);
            }
            for (char brokenChar : brokenChars) {
                //a single brokenChar is going to make the whole word broken
                //so count that word and break
                if (isBrokenCharPresentInWord.contains(brokenChar)) {
                    brokenWordCount++;
                    break;
                }
            }
            isBrokenCharPresentInWord.clear();
        }

        //output
        //Maximum Number of Words You Can Type = Total Words You Have - All Broken Words
        int wordsCanBeTyped = splitWords.length - brokenWordCount;
        System.out.println("Maxmum words that you can type : " + wordsCanBeTyped);
    }

    public boolean checkIfParenthesisStringCanBeValid(String s, String locked) {
        //https://leetcode.com/problems/check-if-a-parentheses-string-can-be-valid/
        //https://leetcode.com/problems/check-if-a-parentheses-string-can-be-valid/discuss/1905993/Keep-a-Range-1-Pass-O(n)-Java
        int n = s.length();
        if (n % 2 == 1) {
            return false;
        }

        int max = 0;
        int min = 0;

        for (int i = 0; i < n; i++) {
            max += locked.charAt(i) == '0' || s.charAt(i) == '(' ? 1 : -1;
            min += locked.charAt(i) == '0' || s.charAt(i) == ')' ? -1 : 1;
            if (max < 0) {
                return false;
            }
            min = Math.max(0, min);
        }
        return min == 0;
    }

    public void largestOddNumInGivenNumString(String numString) {
        //https://leetcode.com/problems/largest-odd-number-in-string/
        int n = numString.length();
        int lastChar = numString.charAt(n - 1);
        int lastDigit = lastChar - '0';

        //if last digit of a num is odd then the whole num is odd itself
        //ex: 7, 27, 427,...so on
        if (lastDigit % 2 == 1) {
            System.out.println("Largest odd number in given num string: " + numString);
            return;
        }

        //we already checked lastDigit above, it was not odd
        //so start end from second last char index
        int end = n - 2;
        //loop until we reach a odd char/digit from end
        while (end >= 0 && (numString.charAt(end) - '0') % 2 != 1) {
            end--;
        }
        //output
        //end-index at which we will find the first odd chr/digit from end
        //0 to that end-index will be our largest odd string
        System.out.println("Largest odd number in given num string: " + numString.substring(0, end + 1));
    }

    public void smallestStringWithGivenLengthNAndCharSumValueK(int n, int k) {
        //https://leetcode.com/problems/smallest-string-with-a-given-numeric-value/
        char[] smallestString = new char[n];
        //lexicographically smallest string of length n can be full of 'a'
        Arrays.fill(smallestString, 'a');
        //we have taken n length string full of 'a'
        k = k - n;
        int index = n - 1;
        while (index >= 0 && k > 0) {
            int minAscii = Math.min(25, k); // at max we can choose 25 == z
            smallestString[index--] = (char) (minAscii + 'a');
            k -= minAscii;
        }
        //output
        System.out.println("Lexicographically smallest string of length n and char sum k: "
                + String.valueOf(smallestString));
    }

    public void replaceAllQuestionMarksWithACharAndNoConsecutiveRepeatingChar(String str) {
        //https://leetcode.com/problems/replace-all-s-to-avoid-consecutive-repeating-characters/
        int n = str.length();

        char[] arr = str.toCharArray();
        for (int i = 0; i < n; i++) {

            if (arr[i] == '?') {

                int left = i - 1;
                int right = i + 1;

                char leftCh = '.';
                char rightCh = '.';

                if (left >= 0) {
                    leftCh = arr[left];
                }

                if (right < n) {
                    rightCh = arr[right];
                }

                //generate a new char from leftCh + 1 (next Acsii value as char)
                //if that is a valid char use that char at ith '?' otherwise default 'a'
                char candidateChar = (char) (leftCh + 1);
                arr[i] = Character.isAlphabetic(candidateChar) ? candidateChar : 'a';

                //if it happens to be that our newly generated char at ith
                //pos is also similar to its rightCh char, we have choose a new 
                //candidate char which will be rightCh + 1 (next Acsii value as char)
                if (arr[i] == rightCh) {
                    candidateChar = (char) (rightCh + 1);
                    arr[i] = Character.isAlphabetic(candidateChar) ? candidateChar : 'a';
                }
            }
        }
        //output
        System.out.println("Removing ? and no consecutive char are repeating: "
                + String.valueOf(arr));
    }

    public String simplifyPath(String path) {
        //https://leetcode.com/problems/simplify-path/
        //explanation: https://youtu.be/qYlHrAKJfyA
        Stack<String> filesOrDirs = new Stack<>();
        StringBuilder canonicalPath = new StringBuilder();
        String currFilesOrDirs = "";
        for (char ch : path.toCharArray()) {

            if (ch == '/') {

                if (currFilesOrDirs.equals("..")) {
                    //".." represent as move to parent dir in filesysytem
                    //so poping means removing curr file or dir and moving to parent file or dir
                    if (!filesOrDirs.isEmpty()) {
                        filesOrDirs.pop();
                    }
                } else if (!currFilesOrDirs.equals("") && !currFilesOrDirs.equals(".")) {
                    //cases when there // there will be currFilesOrDirs == "" and
                    //"." represent as current working dir, we don't to do anything with that
                    //if currFilesOrDirs is none of that, push in stack
                    filesOrDirs.push(currFilesOrDirs);
                }
                //reset
                currFilesOrDirs = "";
            } else {
                currFilesOrDirs += ch;
            }
        }

        if (filesOrDirs.isEmpty()) {
            return "/";
        }

        while (!filesOrDirs.isEmpty()) {
            canonicalPath.insert(0, "/" + filesOrDirs.pop());
        }
        return canonicalPath.toString();
    }

    public void findCommonCharacters(String[] words) {
        //https://leetcode.com/problems/find-common-characters/
        List<Character> result = new ArrayList<>();
        int[] minFreq = new int[26];
        Arrays.fill(minFreq, Integer.MAX_VALUE);
        for (String word : words) {
            //calculate freq of each charater in the curr word
            int[] currFreq = new int[26];
            for (char ch : word.toCharArray()) {
                currFreq[ch - 'a']++;
            }

            //choose min of freq of each char from curr word or any prev words
            for (int i = 0; i < 26; i++) {
                minFreq[i] = Math.min(
                        minFreq[i],
                        currFreq[i]
                );
            }
        }

        for (int i = 0; i < 26; i++) {
            int freq = minFreq[i];
            while (freq != 0) {
                result.add((char) (i + 'a'));
                freq--;
            }
        }
        //output
        System.out.println("All common chars from the given words[]: " + result);
    }

    public void searchSuggestionSystem(String[] words, String search) {
        //https://leetcode.com/problems/search-suggestions-system/
        //explanation: https://youtu.be/D4T2N0yAr20

        int n = words.length;
        //this map is used for output purpose, not req by question
        Map<Character, List<String>> wordsPerSearchedChar = new HashMap<>();
        List<List<String>> result = new ArrayList<>();
        //sort lexicographically order
        Arrays.sort(words);
        int reqAtMost = 3;
        int start = 0;
        int end = n - 1;

        for (int i = 0; i < search.length(); i++) {
            char searchChar = search.charAt(i);
            List<String> curr = new ArrayList<>();

            while (end >= start
                    //cond is to skip all those words from start which are
                    //1. smaller than our search word ex: search = "apple", word[start] = "app"
                    //at i = 3 search[3] = l where i >= "app".length()
                    //2. the curr searchChar at ith pos is not matching with the ith char 
                    //of word[start].charAt(i) ex search = "apple", word[start] = "ape"
                    //at i = 2 searchChar = p !=  word[start].charAt(i) = e
                    && (i >= words[start].length()
                    || words[start].charAt(i) != searchChar)) {
                start++;
            }

            while (end >= start
                    //cond is to skip all those words from end which are
                    //1. smaller than our search word ex: search = "apple", word[end] = "app"
                    //at i = 3 search[3] = l where i >= "app".length()
                    //2. the curr searchChar at ith pos is not matching with the ith char 
                    //of word[end].charAt(i) ex search = "apple", word[end] = "ape"
                    //at i = 2 searchChar = p !=  word[end].charAt(i) = e
                    && (i >= words[end].length()
                    || words[end].charAt(i) != searchChar)) {
                end--;
            }

            //we need atmost 3 words but if we get any less amount then consume that much
            int wordsToConsume = Math.min(reqAtMost, end - start + 1);
            for (int j = 0; j < wordsToConsume; j++) {
                curr.add(words[start + j]);
            }
            result.add(curr);
            //map is used for output purpose only
            wordsPerSearchedChar.put(searchChar, curr);
        }
        //output
        System.out.println("Search suggestions system: " + result);
        System.out.println("Search suggestions system words per char: " + wordsPerSearchedChar);
    }

    public int maximumLengthOfSubstringThatExistsAsSubseqInOtherString(String main, String curr) {
        //https://www.geeksforgeeks.org/maximum-length-prefix-one-string-occurs-subsequence-another/?ref=rp
        int currLenCovered = 0;
        int mainLen = main.length();
        for (int i = 0; i < mainLen; i++) {
            if (currLenCovered == curr.length()) {
                break;
            }
            if (main.charAt(i) == curr.charAt(currLenCovered)) {
                currLenCovered++;
            }
        }
        //output
        String substr = curr.substring(0, currLenCovered);
        System.out.println("Maximum length of substring exists as subseq in main string: "
                + currLenCovered + " substr " + substr);
        return currLenCovered;
    }

    public boolean isSubsequence(String main, String curr) {
        //https://leetcode.com/problems/is-subsequence/
        //https://www.geeksforgeeks.org/maximum-length-prefix-one-string-occurs-subsequence-another/?ref=rp
        int currLenCovered = 0;
        int mainLen = main.length();
        int currLen = curr.length();
        for (int i = 0; i < mainLen; i++) {
            if (currLenCovered == currLen) {
                return true;
            }
            if (main.charAt(i) == curr.charAt(currLenCovered)) {
                currLenCovered++;
            }
        }
        return currLen == currLenCovered;
    }

    public void partitionsInCurrStringWherePrefixExistsAsSubseqInMainString(String main, String curr) {
        //MY GOOGLE INTERVIEW QUESTION
        /*
         main = "aaaabbc"
         curr = "abcbbabc"
         partition in curr string should be 3 as abc | bb | abc
         */
        int partitions = 0;
        String prefixStr = curr;
        int prefixLengthCurrCovered = 0;
        Set<String> cache = new HashSet<>();
        while (prefixStr.length() > 0) {
            //checks if prefixStr is already processes before then we don't 
            //need to call the below fun again for same
            //ex: curr = "abcbbabc" first prefix abc will again going
            //to be checked later after bb
            if (cache.contains(prefixStr)) {
                prefixLengthCurrCovered = prefixStr.length();
                prefixStr = prefixStr.substring(prefixLengthCurrCovered);
                partitions++;
                continue;
            }
            //if prefixStr is not cached previously
            //we must find a prefix in curr that exists as subseq in main
            prefixLengthCurrCovered = maximumLengthOfSubstringThatExistsAsSubseqInOtherString(
                    main, prefixStr);
            //if any prefix is not present as subseq the length will return as 0
            //that means we can't divide curr string
            if (prefixLengthCurrCovered == 0) {
                System.out.println("No partition is possible");
                return;
            }
            //if any prefix of curr found as subseq cache what prefix is that
            //ex: curr = "abcbbabc" first prefix = "abc" exists as subseq in "aaaabbc"
            //save substring(0, prefixLengthCurrCovered) i.e, first "abc" and so on...
            cache.add(prefixStr.substring(0, prefixLengthCurrCovered));
            //remove the prefix that has been found
            //like above first "abc" is found as subseq now reduce our actual prefixStr
            //substring(prefixLengthCurrCovered) ==> "bbabc" and so on...
            prefixStr = prefixStr.substring(prefixLengthCurrCovered);
            //if we can do all this we mean that we have a partition
            partitions++;
        }
        //output
        System.out.println("Partitions of curr string where each substring"
                + " exists as subseq in mains string: " + partitions);
    }

    public void numberOfMatchingSubseq(String main, String[] words) {
        //https://leetcode.com/problems/number-of-matching-subsequences/
        int n = words.length;
        Set<String> alreadyFound = new HashSet<>();
        Set<String> notFound = new HashSet<>();
        int totalMatch = 0;
        for (String word : words) {

            if (notFound.contains(word)) {
                continue;
            } else if (alreadyFound.contains(word)) {
                totalMatch++;
            } else if (isSubsequence(main, word)) {
                totalMatch++;
                alreadyFound.add(word);
            } else {
                notFound.add(word);
            }
        }
        //output
        System.out.println("Total words matches that exists as subseq in main string: " + totalMatch);
    }

    public void minimumSwapsToMakeParenthesisStringBalanced(String str) {
        //https://leetcode.com/problems/minimum-number-of-swaps-to-make-the-string-balanced/
        //explanation: https://youtu.be/3YDBT9ZrfaU
        int closingBrackets = 0;
        int maxClosingBrackets = 0;
        for (char bracket : str.toCharArray()) {
            if (bracket == '[') {
                closingBrackets -= 1;
            } else {
                closingBrackets += 1;
            }
            maxClosingBrackets = Math.max(maxClosingBrackets, closingBrackets);
        }
        //output
        //each time we make a swap, we will be balance two brackets
        int swaps = (maxClosingBrackets + 1) / 2;
        System.out.println("Min swaps to make parenthesis string balanced : " + swaps);
    }

    public void minimumAdditionsToMakeParenthesisStringValid(String str) {
        //https://leetcode.com/problems/minimum-add-to-make-parentheses-valid/
        int balancedBrackets = 0;
        int inserts = 0;
        for (char bracket : str.toCharArray()) {
            balancedBrackets += bracket == '(' ? 1 : -1;
            if (balancedBrackets == -1) {
                inserts++;
                balancedBrackets++;
            }
        }
        //output
        System.out.println("Min additions to make parenthesis string valid: "
                + (balancedBrackets + inserts));
    }

    public boolean areAlienWordsSorted(String[] words, String alienAlphabet) {
        //https://leetcode.com/problems/verifying-an-alien-dictionary/
        //explanation: https://youtu.be/OVgPAJIyX6o

        //save the index of all the char in this alienAlphabet letter set
        int[] alphabetIndex = new int[26];
        for (int i = 0; i < alienAlphabet.length(); i++) {
            char ch = alienAlphabet.charAt(i);
            alphabetIndex[ch - 'a'] = i;
        }

        //iterate over all the adjacent words
        for (int i = 0; i < words.length - 1; i++) {
            //curr and next words are always adjacent
            String currWord = words[i];
            String nextWord = words[i + 1];

            for (int j = 0; j < currWord.length(); j++) {
                //since it is given that the words[] is lexcographically sorted 
                //then in that case these words "app" & "apple" will be sorted like this
                //but edge cases like "apple" & "app" they are not sorted as "app" is small
                //so j loop running on "apple" there will come a iteration
                //where j == 3 >= "app".length() till this point no char mismatch is found
                //it will not be found later on as "app" will have no char to check
                if (j >= nextWord.length()) {
                    return false;
                }

                int currCharIndex = currWord.charAt(j) - 'a';
                int nextCharIndex = nextWord.charAt(j) - 'a';
                if (currCharIndex != nextCharIndex) {
                    //since words are sorted so 
                    //ex curr = "app", next = "cat" they are sorted, first mismatch char is a & c
                    //in normal english index/ascii index value is like a < c 
                    //but suppose curr = "cat", next = "app" was given then c < a is a false condition
                    //similary in given alienAplhabet nextChar should have higer index value than currChar
                    //if this case comes opposite currChar have higer index value that nextChar
                    //then alien word[] dictionary is not sorted.
                    if (alphabetIndex[currCharIndex] > alphabetIndex[nextCharIndex]) {
                        return false;
                    } else {
                        break;
                    }
                }
            }
        }
        return true;
    }

    public void generateNumberFollowingPattern(String pattern) {
        //https://practice.geeksforgeeks.org/problems/number-following-a-pattern3126/1#
        //number pattern should contain digits[1 to 9]
        //ex: "D" output "21" as 1 is dec from 2
        //ex: "IIDDD" output "126543" as 1 < 2 < 6 > 5 > 4 > 3
        int num = 1;
        String numStr = "";
        Stack<Integer> stack = new Stack<>();
        for (char ch : pattern.toCharArray()) {
            if (ch == 'D') {
                //D == decreasing pattern in digits
                stack.push(num);
                num++;
            } else {
                //I == increasing pattern in digits
                stack.push(num);
                num++;
                while (!stack.isEmpty()) {
                    numStr += stack.pop();
                }
            }
        }
        stack.push(num);
        while (!stack.isEmpty()) {
            numStr += stack.pop();
        }
        //output
        System.out.println(pattern + " number following this pattern: " + numStr);
    }

    public void minMovesToMakeStringPallindrome(String str) {
        //https://leetcode.com/problems/minimum-number-of-moves-to-make-palindrome/
        //https://leetcode.com/problems/minimum-number-of-moves-to-make-palindrome/discuss/2191696/USING-TWO-POINTER-oror-SIMPLEST-SOLUTION

        int n = str.length();

        char[] charArr = str.toCharArray();

        int start = 0;
        int end = n - 1;

        int currStart = start;
        int currEnd = end;

        int swaps = 0;

        while (end > start) {
            //greedily compare start and end char update pointer if they are same
            if (charArr[start] == charArr[end]) {
                start++;
                end--;
                continue;
            }

            currStart = start;
            currEnd = end;
            //if the start and end chars are not same, try to find the char
            //eaual to char[start] from the end
            while (charArr[currStart] != charArr[currEnd]) {
                currEnd--;
            }

            //swap
            //currEnd would be the char that match to char[start]
            //adjacentNextIndex = currEnd + 1 == adjacent index
            //swap(currEnd, nextToCurrEnd)
            int adjacentNextIndex = currEnd + 1;
            char temp = charArr[currEnd];
            charArr[currEnd] = charArr[adjacentNextIndex];
            charArr[adjacentNextIndex] = temp;

            swaps++;
        }
        //output
        System.out.println("Min moves(swaps) to make to the string pallindrome: " + swaps);
    }

    public void largestThreeSameDigitNumInString(String num) {
        //https://leetcode.com/problems/largest-3-same-digit-number-in-string/
        //SLIDING WINDOW approach
        int n = num.length();
        int start = 0;
        int end = 0;
        Map<Character, Integer> counter = new HashMap<>();
        int maxEffectiveNum = Integer.MIN_VALUE;

        String result = "";

        while (end < n) {

            char chEnd = num.charAt(end);
            counter.put(chEnd, counter.getOrDefault(chEnd, 0) + 1);

            //same digit and length to be 3
            //so counter size should remain 1 for a single/same digit process
            //like "777" map['7' = 3] size == 1
            if (counter.size() == 1
                    //window here req to be 3
                    && (end - start + 1) == 3) {

                int currEffectiveNum = chEnd - '0';
                //incase we already have processed "777" 
                //so currEffectiveNum = 7, maxEffectiveNum also be currEffectiveNum == 7
                //later on we see "333"
                //then currEffectiveNum = 3
                //but we already have a maxEffectiveNum as 7
                //so our result will not be updated to "333"
                if (currEffectiveNum > maxEffectiveNum) {
                    maxEffectiveNum = currEffectiveNum;
                    //same digits of length 3
                    result = currEffectiveNum + "" + currEffectiveNum + "" + currEffectiveNum;
                }
            }

            //if we need same digit of length 3
            //incase we added 2 different digit then map.size > 1
            //so we will move our window from start
            while (counter.size() > 1) {
                char chStart = num.charAt(start);
                counter.put(chStart, counter.getOrDefault(chStart, 0) - 1);
                if (counter.get(chStart) <= 0) {
                    counter.remove(chStart);
                }
                start++;
            }
            end++;
        }
        //output
        System.out.println("Largest same digit num of length 3: " + result);
    }

    public void minFlipsToMakeBinaryStringAlternating(String str) {
        //https://leetcode.com/problems/minimum-number-of-flips-to-make-the-binary-string-alternating/
        //explanation: https://youtu.be/MOeuK6gaC2A
        //SLIDING WINDOW approach
        /*
         You are given a binary string s. You are allowed to perform two types of 
         operations on the string in any sequence:

         Type-1: Remove the character at the start of the string s and append 
         it to the end of the string.
         Type-2: Pick any character in s and flip its value, i.e., if its value 
         is '0' it becomes '1' and vice-versa.
         Return the minimum number of type-2 operations you need to perform 
         such that s becomes alternating.
         */

        int actualLen = str.length();

        //to achieve Type-1 operation
        str = str + str;

        int appenedStrlen = str.length();

        String targetAltStr1 = "";
        String targetAltStr2 = "";

        //generate the possible alternating binary strings that is of same length
        //as of our actual str (n)
        for (int i = 0; i < appenedStrlen; i++) {
            targetAltStr1 += i % 2 == 0 ? "0" : "1";
            targetAltStr2 += i % 2 == 0 ? "1" : "0";
        }

        int start = 0;
        int end = 0;

        int bitDiffTarget1 = 0;
        int bitDiffTarget2 = 0;

        int minFlips = appenedStrlen;

        while (end < appenedStrlen) {

            if (str.charAt(end) != targetAltStr1.charAt(end)) {
                bitDiffTarget1++;
            }
            if (str.charAt(end) != targetAltStr2.charAt(end)) {
                bitDiffTarget2++;
            }

            while (end - start + 1 > actualLen) {
                if (str.charAt(start) != targetAltStr1.charAt(start)) {
                    bitDiffTarget1--;
                }
                if (str.charAt(start) != targetAltStr2.charAt(start)) {
                    bitDiffTarget2--;
                }
                start++;
            }

            if (end - start + 1 == actualLen) {
                minFlips = Math.min(minFlips, Math.min(bitDiffTarget1, bitDiffTarget2));
            }
            end++;
        }
        //output
        System.out.println("Min flips to make binary string alternating: " + minFlips);
    }

    private boolean decodedString_IsStringDigit(String str) {
        char ch = str.charAt(0);
        return ch >= '0' && ch <= '9';
    }

    public void decodedString(String expression) {
        //https://leetcode.com/problems/decode-string/
        //explanation: https://youtu.be/qB0zZpBJlh8
        int n = expression.length();
        Stack<String> stack = new Stack<>();
        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < n; i++) {
            char ch = expression.charAt(i);

            if (ch == ']') {

                String substr = "";
                //ex stack be like [ '[', 'b', 'c' ] then closing bracket comes ']'
                //then form this inner sub string "bc" until we meet its opening bracket '[' in stack
                while (!stack.isEmpty() && !stack.peek().equals("[")) {
                    substr = stack.pop() + substr;
                }

                //pop the opening bracket '[' from the above loop check
                stack.pop();

                String numStr = "";
                //each decoded string has some num K as "K[substr]", where we need to
                //repeat this substr * K time. In the loop below we will form this num K
                //as numStr
                while (!stack.isEmpty() && decodedString_IsStringDigit(stack.peek())) {
                    numStr = stack.pop() + numStr;
                }

                int num = Integer.parseInt(numStr);
                String repeatSubStrNumTimes = "";
                for (int rep = 0; rep < num; rep++) {
                    repeatSubStrNumTimes += substr;
                }

                //till here we have processed the sub-expression
                //in format "num[substr]", we will push this sub-expression
                stack.push(repeatSubStrNumTimes);

            } else {
                //stringified form of curr char ch
                stack.push(ch + "");
            }
        }

        //forming result from all the processed sub-expressions
        String joined = String.join("", stack);
//        while (!stack.isEmpty()) {
//            sb.insert(0, stack.pop());
//        }

        //output:
//        System.out.println("Decoded string: " + sb.toString());
        System.out.println("Decoded string: " + joined);
    }

    public boolean validPallindromeTwo(String str) {
        //https://leetcode.com/problems/valid-palindrome-ii/
        //explanation: https://youtu.be/JrxRYBwG6EI
        int n = str.length();
        int start = 0;
        int end = n - 1;

        while (end > start) {
            if (str.charAt(start) == str.charAt(end)) {
                start++;
                end--;
                continue;
            }

            //not include char at start-th index ==> start + 1
            //include end char do end + 1
            String skipStart = str.substring(start + 1, end + 1);
            //skip end char
            String skipEnd = str.substring(start, end);

            return isStringPallindrome(skipStart) || isStringPallindrome(skipEnd);
        }
        return true;
    }

    public void textJustification(String[] words, int maxWidth) {
        //https://leetcode.com/problems/text-justification/
        //https://www.interviewbit.com/problems/justified-text/hints/
        int n = words.length;
        int SPACE_CONSTANT = 1;
        List<String> result = new ArrayList<>();
        int index = 0;
        while (index < n) {

            int lineLength = -1;
            int wordIndex = index;
            while (wordIndex < n
                    && words[wordIndex].length() + SPACE_CONSTANT + lineLength <= maxWidth) {
                lineLength += words[wordIndex].length() + SPACE_CONSTANT;
                wordIndex++;
            }

            StringBuilder sb = new StringBuilder(words[index]);
            int spaces = 1;
            int extras = 0;

            if (wordIndex != index + 1 && wordIndex != n) {
                spaces = (maxWidth - lineLength) / (wordIndex - index - 1) + SPACE_CONSTANT;
                extras = (maxWidth - lineLength) % (wordIndex - index - 1);
            }

            for (int j = index + 1; j < wordIndex; j++) {
                //add min spaces req between two words
                for (int space = 0; space < spaces; space++) {
                    sb.append(" ");
                }
                //if any extra spaces are also req, that should also be added
                if (extras-- > 0) {
                    sb.append(" ");
                }
                //after adding all the spaces, we can add the curr jth word
                sb.append(words[j]);
            }

            int remaining = maxWidth - sb.length();
            while (remaining-- > 0) {
                sb.append(" ");
            }
            result.add(sb.toString());
            index = wordIndex;
        }
        //output
        System.out.println("Text justification: " + result);
    }

    public void positionsOfLargeGroups(String str) {
        //https://leetcode.com/problems/positions-of-large-groups/
        //SLIDING WINDOW approach
        List<List<Integer>> positions = new ArrayList<>();
        int n = str.length();
        Map<Character, Integer> freq = new HashMap<>();
        int start = 0;
        int end = 0;
        while (end < n) {

            char chEnd = str.charAt(end);
            freq.put(chEnd, freq.getOrDefault(chEnd, 0) + 1);

            //size == 2 means we now have discontinuity from a potential large group,
            //in that discontinuity we can add previous group if it was >= 3
            //we are not calculating window length as (end - start + 1) because
            //end is already ahead of large group where we found a discontinuity,
            //so length to be considered here till end - 1
            if (freq.size() == 2) {
                if (end - start >= 3) {
                    positions.add(Arrays.asList(start, end - 1));
                }
            }

            //large group will be of same chars whose freq is >= 3
            //if we have more than 1 char in our freq map, we need to adjust our window
            //to find next coming large groups because if previous was a large groups, it
            //would have been calculated in above if block
            while (freq.size() > 1) {
                char chStart = str.charAt(start);
                freq.put(chStart, freq.getOrDefault(chStart, 0) - 1);
                if (freq.get(chStart) <= 0) {
                    freq.remove(chStart);
                }
                start++;
            }
            end++;
        }
        //this if block is handle cases of str = "aaa", "abcaaaa"
        //here after end > n, we are not able to check the very last large groups of str
        //so need this cond here also
        if (end - start >= 3) {
            positions.add(Arrays.asList(start, end - 1));
        }
        //output
        System.out.println("Positions of all the large groups of strings: " + positions);
    }

    public void positionsOfLargeGroups_Optimized(String str) {
        //https://leetcode.com/problems/positions-of-large-groups/
        //TWO POINTER approach
        List<List<Integer>> positions = new ArrayList<>();
        int n = str.length();
        int start = 0;
        int end = 0;
        while (end < n) {
            if (end == n - 1 || str.charAt(end) != str.charAt(end + 1)) {
                if (end - start + 1 >= 3) {
                    positions.add(Arrays.asList(start, end));
                }
                start = end + 1;
            }
            end++;
        }
        //output
        System.out.println("Positions of all the large groups of strings: " + positions);
    }

    public Node<Integer> reverseLinkedList_Iterative(Node<Integer> node) {
        System.out.println("Reverse linked list iterative");
        //actual
        new LinkedListUtil<>(node).print();

        Node<Integer> curr = node;
        Node<Integer> prev = null;
        Node<Integer> next = null;

        while (curr != null) {

            next = curr.getNext();
            curr.setNext(prev);
            prev = curr;
            curr = next;
        }

        //output
        new LinkedListUtil<>(prev).print();

        //to use by other methods when req 
        return prev;
    }

    Node<Integer> reverseLinkedList_Recursive_NewHead;

    private Node<Integer> reverseLinkedList_Recursive_Helper(Node<Integer> node) {

        if (node.getNext() == null) {
            reverseLinkedList_Recursive_NewHead = node;
            return node;
        }

        Node<Integer> revNode = reverseLinkedList_Recursive_Helper(node.getNext());
        revNode.setNext(node);
        node.setNext(null);

        return node;
    }

    public void reverseLinkedList_Recursive(Node<Integer> node) {
        System.out.println("Reverse linked list recursive");
        //actual
        LinkedListUtil<Integer> ll = new LinkedListUtil<>(node);
        ll.print();

        reverseLinkedList_Recursive_Helper(node);

        //output
        LinkedListUtil<Integer> output = new LinkedListUtil<>(reverseLinkedList_Recursive_NewHead);
        output.print();

    }

    private Stack<Integer> sumOfNumbersAsLinkedList_ByStack_ToStack(Node<Integer> node) {

        Stack<Integer> s = new Stack<>();
        Node<Integer> temp = node;
        while (temp != null) {

            s.push(temp.getData());
            temp = temp.getNext();

        }

        return s;

    }

    public void sumOfNumbersAsLinkedList_ByStack(Node<Integer> n1, Node<Integer> n2) {

        //..................................T: O(N1 + N2) where N1 = n1.length, N2 = n2.length
        //..................................S: O(N1 + N2) because of 2 stacks used
        Stack<Integer> nS1 = sumOfNumbersAsLinkedList_ByStack_ToStack(n1);
        Stack<Integer> nS2 = sumOfNumbersAsLinkedList_ByStack_ToStack(n2);

        int carry = 0;
        LinkedListUtil<Integer> ll = new LinkedListUtil<>();
        while (!nS1.isEmpty() || !nS2.isEmpty()) {

            int sum = carry;

            if (!nS1.isEmpty()) {
                sum += nS1.pop();
            }

            if (!nS2.isEmpty()) {
                sum += nS2.pop();
            }

            carry = sum / 10;
            ll.addAtHead(sum % 10);

        }

        if (carry > 0) {
            ll.addAtHead(carry);
        }

        //output
        ll.print();

    }

    public void sumOfNumbersAsLinkedList_ByReversingList(Node<Integer> n1, Node<Integer> n2) {

        //OPTIMISED
        //..................................T: O(N1 + N2) where N1 = n1.length, N2 = n2.length
        //..................................S: O(1)
        Node<Integer> n1Rev = reverseLinkedList_Iterative(n1);
        Node<Integer> n2Rev = reverseLinkedList_Iterative(n2);

        Node<Integer> head = null;

        int carry = 0;
        while (n1Rev != null || n2Rev != null) {

            int sum = carry;

            if (n1Rev != null) {
                sum += n1Rev.getData();
                n1Rev = n1Rev.getNext();
            }

            if (n2Rev != null) {
                sum += n2Rev.getData();
                n2Rev = n2Rev.getNext();
            }

            //adding start ptr as head
            Node<Integer> start = new Node<>(sum % 10);
            start.setNext(head);
            head = start;
            carry = sum / 10;
        }

        if (carry > 0) {
            Node<Integer> start = new Node<>(carry);
            start.setNext(head);
            head = start;
        }

        //output
        new LinkedListUtil<Integer>(head).print();
    }

    public void removeDuplicateFromSortedLinkedList(Node<Integer> head) {

        //actual
        new LinkedListUtil<>(head).print();

        Node<Integer> curr = head;
        Node<Integer> temp = head.getNext();

        while (temp != null) {
            if (curr.getData() != temp.getData()) {
                curr.setNext(temp);
                curr = temp;
            }
            temp = temp.getNext();
        }

        curr.setNext(temp);

        //output
        new LinkedListUtil<>(head).print();
    }

    public void mergeKSortedLinkedList(Node<Integer>[] nodes) {

        //............................T: O(N*K*LogK)
        //............................S: O(K)
        //https://www.geeksforgeeks.org/merge-k-sorted-linked-lists-set-2-using-min-heap/
        //HEAP based method
        PriorityQueue<Node<Integer>> minHeap = new PriorityQueue<>(
                (o1, o2) -> o1.getData().compareTo(o2.getData())
        );

        //at this point we have added K node heads in minHeap
        //so the minheap size after this loop will K
        for (Node<Integer> node : nodes) {
            if (node != null) {
                minHeap.add(node);
            }
        }

        //head to point arbitary infinite value to start with
        Node<Integer> head = new Node<>(Integer.MIN_VALUE);
        //saving the actual head's ref
        Node<Integer> copyHead = head;
        while (!minHeap.isEmpty()) {

            //we poll out one node from heap
            Node<Integer> curr = minHeap.poll();
            //and add in one node 
            //if its next node is not null
            //so the size of the min heap remains atmost K
            if (curr.getNext() != null) {
                minHeap.add(curr.getNext());
            }
            //isolate the current node by saying curr.next=NULL,
            //also curr.next node value is already preserved in
            //minHeap(above if block), if not null already
            curr.setNext(null);
            copyHead.setNext(curr);
            copyHead = copyHead.getNext();
        }

        //actual merged list starts with next of arbitary head pointer
        new LinkedListUtil<>(head.getNext()).print();
    }

    public void kThNodeFromEndOfLinkedList_1(Node node, int K) {

        //1. Approach
        //using additional space (Stack)
        //................O(N)+O(K)
        //time O(N) creating stack of N nodes from linked list + O(K) reaching out to Kth node
        //in the stack.
        //.......................space complexity O(N)
        Stack<Node> stack = new Stack<>();
        Node temp = node;
        //T: O(N)
        //S: O{N}
        while (temp != null) {
            stack.push(temp);
            temp = temp.getNext();
        }

        //T: O(K)
        while (!stack.isEmpty()) {

            K--;
            Object element = stack.pop().getData();
            if (K == 0) {
                System.out.println("Kth node from end is: " + element);
            }
        }
    }

    public void kThNodeFromEndOfLinkedList_2(Node node, int K) {

        //2. Approach
        //using Len - K + 1 formula
        //calculate the full length of the linked list frst 
        //then move the head pointer upto (Len - K + 1) limit which
        // is Kth node from the end
        //.................T: O(N) + O(Len - K + 1)
        //1. calculating Len O(N)
        //2. moving to Len - k + 1 pointer is O(Len - K + 1)
        int len = 0;
        Node temp = node;
        while (temp != null) {
            temp = temp.getNext();
            len++;
        }

        //Kth node from end = len - K + 1
        temp = node;
        //i=1 as we consider the first node from 1 onwards
        for (int i = 1; i < (len - K + 1); i++) {
            temp = temp.getNext();
        }

        //output
        System.out.println("Kth node from end is: " + temp.getData());

    }

    public void kThNodeFromEndOfLinkedList_3(Node node, int K) {

        //3. Approach (OPTIMISED)
        //Two pointer method
        //Theory: 
        //maintain ref pointer, main pointer
        //both start from the head ref
        //move ref pointer to K dist. Once ref pointer reaches the K dist from main pointer
        //start moving the ref and main pointer one by one.
        //at the time ref pointer reaches the end of linked list
        //main pointer will be K dist behind the ref pointer(already at end now)
        //print the main pointer that will be answer
        //............T: O(N) S: O(1)
        Node ref = node;
        Node main = node;

        while (K-- != 0) {
            ref = ref.getNext();
            if (ref == null) {
                ref = node;
            }
        }

        if (ref == main) {
            //output
            System.out.println("Kth node from end is: " + main.getData());
            return;
        }

        //now ref is K dist ahead of main pointer
        //now move both pointer one by one
        //until ref reaches end of linked list
        //bt the time main pointer will be K dist behind the ref pointer
        while (ref != null) {

            main = main.getNext();
            ref = ref.getNext();
        }

        //output
        System.out.println("Kth node from end is: " + main.getData());
    }

    public Node<Integer> swapLinkedListNodesInPair(Node<Integer> node) {

        //https://www.geeksforgeeks.org/reverse-a-list-in-groups-of-given-size/
        //https://leetcode.com/problems/swap-nodes-in-pairs/
        int K = 2; //because swap in pairs;
        Node current = node;
        Node next = null;
        Node prev = null;

        int count = 0;

        /* Reverse first k nodes of linked list */
        while (count < K && current != null) {
            next = current.getNext();
            current.setNext(prev);
            prev = current;
            current = next;
            count++;
        }

        /* next is now a pointer to (k+1)th node  
         Recursively call for the list starting from current. 
         And make rest of the list as next of first node */
        if (next != null) {
            node.setNext(swapLinkedListNodesInPair(next));
        }

        // prev is now head of input list 
        return prev;
    }

    public Node<Integer> reverseLinkedListInKGroups(Node<Integer> head, int K) {
        //https://www.geeksforgeeks.org/reverse-a-list-in-groups-of-given-size/
        //https://leetcode.com/problems/reverse-nodes-in-k-group/
        Node current = head;
        //in case the passed LinkedList length is less than k
        for (int i = 0; i < K; i++) {
            if (current == null) {
                return head;
            }
            current = current.getNext();
        }

        current = head;
        Node next = null;
        Node prev = null;

        int count = 0;

        /* Reverse first k nodes of linked list */
        while (count < K && current != null) {
            next = current.getNext();
            current.setNext(prev);
            prev = current;
            current = next;
            count++;
        }

        /* next is now a pointer to (k+1)th node  
         Recursively call for the list starting from current. 
         And make rest of the list as next of first node */
        if (next != null) {
            head.setNext(reverseLinkedListInKGroups(next, K));
        }

        // prev is now head of input list 
        return prev;
    }

    public Node<Integer> reverseLinkedListInKGroupsAlternatively(Node<Integer> head, int K) {

        //Adaptation of reverseLinkedListInKGroups method
        //reverse first K groups
        Node<Integer> curr = head;
        Node<Integer> prev = null;
        Node<Integer> next = null;

        int count = 0;

        while (count < K && curr != null) {

            next = curr.getNext();
            curr.setNext(prev);
            prev = curr;
            curr = next;

            count++;
        }

        //move straight K nodes without reversing
        count = 0;
        while (count < K && next != null) {

            head.setNext(next);
            head = head.getNext();
            next = next.getNext();
            count++;
        }

        //pass the alternate K group recursively
        if (next != null) {
            head.setNext(reverseLinkedListInKGroupsAlternatively(next, K));
        }

        return prev;
    }

    public boolean detectLoopCycleInLinkedList_HashBased(Node node) {

        //......................T: O(N)
        //......................S: O(N)
        Set<Node> set = new HashSet<>();
        Node temp = node;
        while (temp != null) {

            if (set.contains(temp)) {
                System.out.println("Hash Based Cycle at: " + temp.getData());
                return true;
            }
            set.add(temp);
            temp = temp.getNext();
        }

        System.out.println("Hash Based No cycle found");
        return false;
    }

    public boolean detectLoopCycleInLinkedList_Iterative(Node head) {

        //......................T: O(N)
        //......................S: O(1)
        Node slow = head;
        Node fast = head.getNext();
        while (fast != null && fast.getNext() != null) {

            if (slow == fast) {
                break;
            }
            slow = slow.getNext();
            fast = fast.getNext().getNext();
        }

        if (slow == fast) {
            slow = head;
            while (slow != fast.getNext()) {
                slow = slow.getNext();
                fast = fast.getNext();
            }

            //fast.next is where the loop starts...
            System.out.println("Iterative approach Cycle at: " + fast.getNext().getData());
            return true;
        }

        System.out.println("Iterative approach No cycle found");
        return false;
    }

    public void detectAndRemoveLoopCycleInLinkedList_HashBased(Node node) {

        //......................T: O(N)
        //......................S: O(N)
        Set<Node> set = new HashSet<>();
        Node loopEnd = null;
        Node temp = node;
        while (temp != null) {

            if (set.contains(temp)) {
                loopEnd.setNext(null);
                break;
            }

            set.add(temp);
            loopEnd = temp;
            temp = temp.getNext();
        }

        //output;
        System.out.println("Hash Based approach detect and remove a loop cycle in linked list output:");
        new LinkedListUtil(node).print();
    }

    public void detectAndRemoveLoopCycleInLinkedList_Iterative(Node head) {

        //......................T: O(N)
        //......................S: O(1)
        Node slow = head;
        Node fast = head.getNext();
        while (fast != null && fast.getNext() != null) {

            if (slow == fast) {
                break;
            }
            slow = slow.getNext();
            fast = fast.getNext().getNext();
        }

        //if there is a loop in linked list
        if (slow == fast) {
            slow = head;
            while (slow != fast.getNext()) {
                slow = slow.getNext();
                fast = fast.getNext();
            }

            //fast is the node where it should end the loop
            fast.setNext(null);
        }

        //output
        System.out.println("Iterative approach detect and remove a loop cycle in linked list output:");
        new LinkedListUtil(head).print();
    }

    public void removeDuplicatesFromUnSortedLinkedListOnlyConsecutive(Node<Integer> node) {

        //...............................T: O(N)
        //...............................S: O(1)
        Node<Integer> curr = node;
        Node<Integer> temp = node.getNext();
        while (temp != null) {

            if (curr.getData() != temp.getData()) {
                curr.setNext(temp);
                curr = temp;
            }

            temp = temp.getNext();

        }

        curr.setNext(temp);

        //output
        System.out.println("Remove duplicates that are consecutive in lisnked list output:");
        new LinkedListUtil<>(node).print();

    }

    public void removeDuplicatesFromUnSortedLinkedListAllExtraOccuernce(Node<Integer> head) {

        //...............................T: O(N)
        //...............................S: O(N)
        Set<Integer> visited = new HashSet<>();
        Node<Integer> curr = head;
        Node<Integer> temp = head.getNext();
        visited.add(curr.getData());
        while (temp != null) {

            if (curr.getData() != temp.getData() && !visited.contains(temp.getData())) {
                curr.setNext(temp);
                curr = temp;
            }
            visited.add(temp.getData());
            temp = temp.getNext();
        }

        curr.setNext(temp);

        //output
        System.out.println("Remove duplicates all extra occuernce in lisnked list output:");
        new LinkedListUtil<>(head).print();
    }

    public Node<Integer> findMiddleNodeOfLinkedList(Node<Integer> head) {

        if (head == null || head.getNext() == null) {
            return head;
        }

        Node<Integer> slow = head;
        Node<Integer> fast = head.getNext();
        while (fast != null && fast.getNext() != null) {

            slow = slow.getNext();
            fast = fast.getNext().getNext();
        }

        //middle node = slow
        return slow;
    }

    public Node<Integer> mergeSortInLinkedList_Asc_Recursion(Node<Integer> n1, Node<Integer> n2) {

        if (n1 == null) {
            return n2;
        }

        if (n2 == null) {
            return n1;
        }

        if (n1.getData() <= n2.getData()) {
            Node<Integer> a = mergeSortInLinkedList_Asc_Recursion(n1.getNext(), n2);
            n1.setNext(a);
            return n1;
        } else {
            Node<Integer> b = mergeSortInLinkedList_Asc_Recursion(n1, n2.getNext());
            n2.setNext(b);
            return n2;
        }
    }

    public Node<Integer> mergeSortDivideAndMerge(Node<Integer> node) {

        if (node == null || node.getNext() == null) {
            return node;
        }

        Node<Integer> middle = findMiddleNodeOfLinkedList(node);
        Node<Integer> secondHalf = middle.getNext();
        //from node to middle is first half, so middle.next = null 
        //splites as 1. node -> middle.next->NULL 2. middle.next -> tail.next->NULL
        middle.setNext(null);

        return mergeSortInLinkedList_Asc_Recursion(mergeSortDivideAndMerge(node),
                mergeSortDivideAndMerge(secondHalf));
    }

    public boolean checkIfLinkedListIsCircularLinkedList(Node node) {

        if (node == null || node.getNext() == node) {
            return true;
        }

        Node headRef = node;
        Node temp = node;
        while (temp.getNext() != headRef && temp.getNext() != null) {
            temp = temp.getNext();
        }
        return temp.getNext() == headRef;
    }

    private int quickSortInLinkedList_Partition(List<Integer> arr, int low, int high) {

        int pivot = arr.get(high);
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (arr.get(j) < pivot) {

                i++;
                //swap
                int temp = arr.get(i);
                arr.set(i, arr.get(j));
                arr.set(j, temp);
            }
        }

        int temp = arr.get(i + 1);
        arr.set(i + 1, arr.get(high));
        arr.set(high, temp);

        return i + 1;
    }

    private void quickSortInLinkedList_Helper(List<Integer> arr, int low, int high) {

        if (high >= low) {

            int pivotIndex = quickSortInLinkedList_Partition(arr, low, high);
            quickSortInLinkedList_Helper(arr, low, pivotIndex - 1);
            quickSortInLinkedList_Helper(arr, pivotIndex + 1, high);
        }
    }

    public void quickSortInLinkedList(Node<Integer> node) {

        if (node == null || node.getNext() == null) {
            new LinkedListUtil<Integer>(node).print();
            return;
        }

        List<Integer> intArr = new ArrayList<>();
        Node<Integer> temp = node;
        while (temp != null) {
            intArr.add(temp.getData());
            temp = temp.getNext();
        }

        quickSortInLinkedList_Helper(intArr, 0, intArr.size() - 1);

        // System.out.println(intArr);
        temp = node;
        for (int x : intArr) {

            temp.setData(x);
            temp = temp.getNext();
        }

        //output
        new LinkedListUtil<Integer>(node).print();
    }

    public void moveLastNodeToFrontOfLinkedList(Node<Integer> node) {

        Node curr = node;
        Node prev = null;

        while (curr.getNext() != null) {
            prev = curr;
            curr = curr.getNext();
        }

        prev.setNext(curr.getNext());
        curr.setNext(node);
        node = curr;

        //output:
        new LinkedListUtil<Integer>(node).print();
    }

    private int addOneToLinkedList_Helper(Node<Integer> node) {

        if (node.getNext() == null) {
            //adding 1 to very last node(or last digit of number in linkedlist form)
            int sum = node.getData() + 1;
            node.setData(sum % 10);
            return sum / 10;
        }

        int carry = addOneToLinkedList_Helper(node.getNext());
        int sum = carry + node.getData();
        node.setData(sum % 10);
        return sum / 10;
    }

    public void addOneToLinkedList(Node<Integer> head) {

        if (head == null) {
            return;
        }

        int carry = addOneToLinkedList_Helper(head);
        //edge case for L [9 -> 9 -> 9 -> NULL] + 1 = [1 -> 0 -> 0 -> 0 -> NULL]
        //extra 1 is the newHead in this case...
        if (carry > 0) {
            Node<Integer> newHead = new Node<>(carry);
            newHead.setNext(head);
            head = newHead;
        }

        //output
        new LinkedListUtil<Integer>(head).print();
    }

    public void sortLinkedListOf012_2(Node<Integer> node) {

        //approach 1 is just using merger sort on linked list. 
        //merge sort method is already been implemented
        //approach 2 will be similar to my approach of solving
        //sortArrayOf012_1()
        int[] count = new int[3]; //we just have 3 digits (0, 1, 2)
        Node<Integer> curr = node;
        while (curr != null) {
            count[curr.getData()]++;
            curr = curr.getNext();
        }

        //manipulate the linked list 
        curr = node;
        for (int i = 0; i < 3; i++) { //O(3)

            while (count[i]-- != 0) {
                //O(N) as N = count[0]+count[1]+cout[2] == total no of node already in the linked list
                curr.setData(i);
                curr = curr.getNext();
            }
        }

        //output:
        new LinkedListUtil<Integer>(node).print();
    }

    public void reverseDoublyLinkedList(Node node) {

        //actual
        new LinkedListUtil(node).print();

        Node curr = node;
        Node nextToCurr = null;
        Node prevToCurr = null;
        while (curr != null) {

            nextToCurr = curr.getNext();
            curr.setNext(prevToCurr);
            curr.setPrevious(nextToCurr);
            prevToCurr = curr;
            curr = nextToCurr;
        }

        //output:
        //new head will pre prevToCurr
        new LinkedListUtil(prevToCurr).print();
    }

    public void intersectionOfTwoSortedLinkedList(Node<Integer> node1, Node<Integer> node2) {

        //....................T: O(M+N)
        //....................S: O(M+N)
        Set<Integer> node1Set = new HashSet<>();
        while (node1 != null) {
            node1Set.add(node1.getData());
            node1 = node1.getNext();
        }

        Set<Integer> node2Set = new HashSet<>();
        Node<Integer> newHead = new Node<>(Integer.MIN_VALUE);
        Node<Integer> copy = newHead;
        while (node2 != null) {

            //all the in node2 that is present in node1 set but same node2 should not be repested in node2 set
            if (node1Set.contains(node2.getData()) && !node2Set.contains(node2.getData())) {
                copy.setNext(node2);
                copy = copy.getNext();
            }
            node2Set.add(node2.getData());
            node2 = node2.getNext();
        }
        copy.setNext(null);
        //output:
        new LinkedListUtil<Integer>(newHead.getNext()).print();
    }

    private int lengthOfLinkedList(Node<Integer> node) {
        int len = 0;
        Node<Integer> curr = node;
        while (curr != null) {
            len++;
            curr = curr.getNext();
        }

        return len;
    }

    private Node<Integer> moveLinkedListNodeByDiff(Node<Integer> node, int diff) {

        int index = 0;
        Node<Integer> curr = node;
        while (index++ < diff) { //evaluates as index++ -> 0+1 -> 1 then 1 < diff
            curr = curr.getNext();
        }
        return curr;
    }

    private int intersectionPointOfTwoLinkedListByRef_Helper(Node<Integer> n1, Node<Integer> n2) {

        Node<Integer> currN1 = n1;
        Node<Integer> currN2 = n2;
        while (currN1 != null && currN2 != null) {

            //nodes get common by ref
            if (currN1 == currN2) {
                return currN1.getData();
            }

            currN1 = currN1.getNext();
            currN2 = currN2.getNext();
        }

        return -1;
    }

    public void intersectionPointOfTwoLinkedListByRef(Node<Integer> node1, Node<Integer> node2) {
        //https://leetcode.com/problems/intersection-of-two-linked-lists
        //find length of node1 T: O(M)
        int M = lengthOfLinkedList(node1);
        //find length of node2 T: O(N)
        int N = lengthOfLinkedList(node2);

        //find the absolute diff in both the length
        //diff = abs(M - N)
        int diff = Math.abs(M - N);

        //if M > N move ptr in node1 by diff forward else move ptr in node2
        //once ptr is available move ptr and node1 or node2 till null and find the intersection point
        //by ref
        int intersectedData = -1;
        Node<Integer> curr = null;
        if (M > N) {
            curr = moveLinkedListNodeByDiff(node1, diff);
            intersectedData = intersectionPointOfTwoLinkedListByRef_Helper(curr, node2);
        } else {
            curr = moveLinkedListNodeByDiff(node2, diff);
            intersectedData = intersectionPointOfTwoLinkedListByRef_Helper(curr, node1);
        }

        //output:
        System.out.println("Two linked list are intersected at: " + intersectedData);
    }

    public void intersectionPointOfTwoLinkedListByRef_HashBased(Node<Integer> node1, Node<Integer> node2) {

        //................................T: O(N + M)
        //................................S: O(N)
        //https://leetcode.com/problems/intersection-of-two-linked-lists
        Set<Node<Integer>> set1 = new HashSet<>();
        Node<Integer> curr = node1;
        while (curr != null) {
            set1.add(curr);
            curr = curr.getNext();
        }

        int intersectedData = -1;
        curr = node2;
        while (curr != null) {

            if (set1.contains(curr)) {
                intersectedData = curr.getData();
                break;
            }
            curr = curr.getNext();
        }

        //output:
        System.out.println("Two linked list are intersected at (hashbased): " + intersectedData);
    }

    public void intersectionPointOfTwoLinkedListByRef_Iterative(Node<Integer> headA, Node<Integer> headB) {
        //................................T: O(N + M)
        //................................S: O(1)
        //https://leetcode.com/problems/intersection-of-two-linked-lists
        //explanation: https://youtu.be/D0X0BONOQhI
        Node<Integer> currA = headA;
        Node<Integer> currB = headB;

        while (currA != currB) {
            currA = currA == null ? headB : currA.getNext();
            currB = currB == null ? headA : currB.getNext();
        }

        //output:
        System.out.println("Two linked list are intersected at (iterative): " + currA.getData());
    }

    public boolean checkIfLinkedListPallindrome_1(Node<Integer> node) {

        //.........................T: O(N)
        //.........................S: O(N)
        //https://leetcode.com/problems/palindrome-linked-list
        //empty list or 1 node list is by default true
        if (node == null || node.getNext() == null) {
            return true;
        }

        Node<Integer> curr = node;
        Stack<Node<Integer>> stack = new Stack<>();
        while (curr != null) {
            stack.push(curr);
            curr = curr.getNext();
        }

        //pop stack and start checking from the head of list
        curr = node;
        while (!stack.isEmpty()) {

            Node<Integer> popped = stack.pop();
            if (curr.getData() != popped.getData()) {
                return false;
            }
            curr = curr.getNext();
        }

        //if while loop doesn't prove false
        return true;
    }

    public boolean checkIfLinkedListPallindrome_2(Node<Integer> node) {

        //.........................T: O(N)
        //.........................S: O(1)
        //https://leetcode.com/problems/palindrome-linked-list/
        //empty list or 1 node list is by default true
        if (node == null || node.getNext() == null) {
            return true;
        }

        //ex LL = 1->2->2->1->N
        //find the mid of the linked list
        Node<Integer> slow = node;
        Node<Integer> fast = node;
        while (fast != null && fast.getNext() != null) {
            slow = slow.getNext();
            fast = fast.getNext().getNext();
        }

        //LL head = 1->2->2->1->N
        //LL slow = 2->1->N
        //from slow ptr reverse it
        Node<Integer> curr = slow;
        Node<Integer> prev = null;
        Node<Integer> next = null;
        while (curr != null) {
            next = curr.getNext();
            curr.setNext(prev);
            prev = curr;
            curr = next;
        }
        //after reversing slow ptr
        //LL new Head(prev) = 1->2->N

        //now prev is the new head for the reversed-slow ptr
        curr = node; //actual head
        //now comparing
        //LL head = 1->2->2->1->N
        //prev = 1->2->N
        //if reversed-half of the linked list(prev) is equal to first-half-part of the actual linked list (head)
        //its pallindrme otherwise ret false
        while (prev != null) {
            if (curr.getData() != prev.getData()) {
                return false;
            }

            prev = prev.getNext();
            curr = curr.getNext();
        }

        //if while loop doesn't prove false
        return true;
    }

    public void reorderLinkedList(Node<Integer> head) {

        //First 2 approaches are in SomePracticeQuestion reorderList_1/reorderList_2
        //actual
        new LinkedListUtil<>(head).print();

        //explanantion: https://youtu.be/xRYPjDMSUFw
        //ex: 1-> 2-> 3-> 4-> 5-> NULL
        //slow = 3->..
        //break it into 2 list (2 half)
        //l1 = 1-> 2-> NULL
        //l2 = 3-> 4-> 5-> NULL
        //reverse(l2) = 5-> 4-> 3-> NULL
        //reorderMerger(l1, l2) = 1-> 5-> 2-> 4-> 3-> NULL
        if (head == null || head.getNext() == null) {
            return;
        }

        //find mid of the linked list
        Node<Integer> slow = head;
        Node<Integer> fast = head.getNext();
        while (fast != null && fast.getNext() != null) {
            slow = slow.getNext();
            fast = fast.getNext().getNext();
        }

        Node<Integer> mid = slow;

        Node<Integer> tempSecond = mid.getNext();
        mid.setNext(null);

        //reversing the second half of the list
        Node<Integer> curr = tempSecond;
        Node<Integer> prev = null;
        Node<Integer> next = null;

        while (curr != null) {
            next = curr.getNext();
            curr.setNext(prev);
            prev = curr;
            curr = next;
        }

        Node<Integer> second = prev;
        Node<Integer> first = head;

        while (second != null) {

            Node<Integer> firstNext = first.getNext();
            Node<Integer> secondNext = second.getNext();

            first.setNext(second);

            if (firstNext == null) {
                break;
            }

            second.setNext(firstNext);

            first = firstNext;
            second = secondNext;
        }
        //output
        new LinkedListUtil<>(head).print();
    }

    public void rearrangeLinkedListAsOddIndexFirstAndEvenIndexAtEnd(Node<Integer> head) {

        //rearrange all node in linked list in such a way that nodes at odd index comes first and nodes at even index 
        //comes last and nodex should maintain the order of their occurence in actual list
        //nodes data is not to be consider for even and odd but their INDEX ONLY
        //https://leetcode.com/problems/odd-even-linked-list/
        //explanation: https://youtu.be/C_LA6SOwVTM
        //actual
        Node<Integer> forPrinting = head;
        int index = 0;
        while (forPrinting != null) {
            System.out.print((++index) + ") " + forPrinting.getData() + "\t");
            forPrinting = forPrinting.getNext();
        }
        System.out.println();

        Node<Integer> odd = head; //at index 1)
        Node<Integer> even = head.getNext(); //at index 2)
        Node<Integer> evenHead = even; //saving the starting ref of even pointr i.e, index 2) in evenHead

        while (even != null && even.getNext() != null) {
            odd.setNext(even.getNext()); //next odd index we will get after even index i.e, 3) after 2)
            odd = odd.getNext(); //index 1) now pointing to the update i.e, index 3) and so on...
            even.setNext(odd.getNext()); //similarly next even index we will get after odd index i.e, 4) after 3)
            even = even.getNext(); //index 2) now pointing to the update i.e, index 4) and so on...
        }

        //odd last ref will be having even starting index ref (last nth odd index -> index 2)
        odd.setNext(evenHead);

        //output:
        new LinkedListUtil<Integer>(head).print();
    }

    public void linkedListComponent(Node<Integer> head, int[] subset) {

        //https://leetcode.com/problems/linked-list-components/
        //ex L = 0-> 1-> 2-> 3-> 4
        //subset[] = [0,3,1,4]
        //first consecutive segment = 0-> 1
        //second consecutive segment = 3-> 4
        //also 0, 1, 3, 4 are in subset, so total segment = 2
        //this logic updates count when consecutivity breaks like 0->1->2 (b/w 1 & 2) count++
        //--->curr = 1 if(set(1) == true && (1.next == null => false || !set(1.next.data(=> 2)) == true)) == TRUE
        //and 3->4->Null (b/w 4 & Null) count++
        //--->curr = 4 if(set(4) == true && (4.next == null => true || !set(4.next.data(=> 2)) == false)) == TRUE
        Set<Integer> set = IntStream.of(subset).boxed().collect(Collectors.toSet());

        Node<Integer> curr = head;
        int count = 0;

        while (curr != null) {
            if (set.contains(curr.getData())
                    && (curr.getNext() == null || !set.contains(curr.getNext().getData()))) {
                count++;
            }
            curr = curr.getNext();
        }

        //output
        System.out.println("Total consecutive segment of linkedlist that are also in subset: " + count);
    }

    public void partitionList(Node<Integer> head, int x) {

        //https://leetcode.com/problems/partition-list
        //explanation: https://youtu.be/K5AVJVjdmL0
        //SIMILAR to odd-even linked list rearrangeLinkedListAsOddIndexFirstAndEvenIndexAtEnd()
        //ex: 1-> 4-> 3-> 2-> 5-> 2-> Null, x = 3
        //all values less than x should come in first segment and in same 
        //order of oocurence and values greater than x should come in 
        //second segment and in same order of oocurence
        //value less than x and order oocurence 1, 2, 2
        //after loop:
        //---> beforeHead = Int.MIN, before = 1-> 2-> 2-> so beforeHead becomes Int.MIN-> 1-> 2-> 2->
        //---> afterHead = Int.MIN, after = 4-> 3-> 5-> so afterHead becomes Int.MIN-> 4-> 3-> 5->
        //after to be last segment so after.next = Null afterHead = Int.MIN-> 4-> 3-> 5-> Null
        //next of before to have head of second segment i.e 4, before.next = afterhead.next
        //now beforeHead = Int.MIN-> 1-> 2-> 2-> 4-> 3-> 5-> Null, now beforehead.next is our output
        //print: beforeHead.next = 1-> 2-> 2-> 4-> 3-> 5-> Null
        //actual
        new LinkedListUtil<>(head).print();
        System.out.println();

        if (head == null || head.getNext() == null) {
            return;
        }

        Node<Integer> curr = head;
        Node<Integer> beforeHead = new Node<>(Integer.MIN_VALUE);
        Node<Integer> before = beforeHead;
        Node<Integer> afterHead = new Node<>(Integer.MIN_VALUE);
        Node<Integer> after = afterHead;

        while (curr != null) {

            if (curr.getData() < x) {
                before.setNext(curr);
                before = before.getNext();
            } else {
                after.setNext(curr);
                after = after.getNext();
            }
            curr = curr.getNext();
        }

        after.setNext(null);
        before.setNext(afterHead.getNext());

        //output
        new LinkedListUtil<>(beforeHead.getNext()).print();
        System.out.println();
    }

    public void rotateLinkedListKTimes(Node<Integer> head, int K) {

        //https://leetcode.com/problems/rotate-list/
        //Actual
        new LinkedListUtil<>(head).print();

        Node<Integer> dummy = new Node<>(Integer.MIN_VALUE);
        dummy.setNext(head);
        Node<Integer> fast = dummy;
        Node<Integer> slow = dummy;

        int len = 0;

        while (fast.getNext() != null) {
            len++;
            fast = fast.getNext();
        }

        for (int j = len - K % len; j > 0; j--) {
            slow = slow.getNext();
        }

        fast.setNext(dummy.getNext());
        dummy.setNext(slow.getNext());
        slow.setNext(null);

        //output
        System.out.println("Rotate linked list " + K + " times output approach1: ");
        new LinkedListUtil<>(dummy.getNext()).print();
    }

    public void rotateLinkedListKTimes2(Node<Integer> head, int K) {

        //https://leetcode.com/problems/rotate-list/
        //explanation: https://youtu.be/BHr381Guz3Y
        //approach similar to rotateArrayByK
        //Actual
        new LinkedListUtil<>(head).print();

        //find len of the the linked list
        int len = 0;
        Node<Integer> curr = head;
        while (curr != null) {
            len++;
            curr = curr.getNext();
        }

        //K > len, mod it by len so that K remains under len range
        K = K % len;

        curr = head;
        Node<Integer> prev = null;
        Node<Integer> next = null;

        //reverse linked list normally
        while (curr != null) {
            next = curr.getNext();
            curr.setNext(prev);
            prev = curr;
            curr = next;
        }

        //this is the node where we will append the second part of reversed list
        //ex: LL [1,2,3,4,5]
        //above reverse LL: [5,4,3,2,1]
        //lastNodeInFirstKReverse == prev = 5
        //let say if k = 2, so first k nodes reverse will be [4, 5]
        //second part reversal = [1,2,3] now the 1 should be linked to 5 above
        //lastNodeInFirstKReverse.next = 1
        Node<Integer> lastNodeInFirstKReverse = prev;

        //reverse first K nodes
        int firstK = K;
        curr = prev; // prev is new head after reversing above
        prev = null;
        next = null;
        while (firstK > 0 && curr != null) {
            next = curr.getNext();
            curr.setNext(prev);
            prev = curr;
            curr = next;
            firstK--;
        }

        //after reversing first K nodes its prev will be new head
        Node<Integer> rotatedListHead = prev;

        //reverse remaining nodes of list
        if (next != null) {
            curr = next;
            prev = null;
            next = null;
            while (curr != null) {
                next = curr.getNext();
                curr.setNext(prev);
                prev = curr;
                curr = next;
            }
            //join two parts of the linked list
            lastNodeInFirstKReverse.setNext(prev);
        }

        //output
        System.out.println("Rotate linked list " + K + " times output approach2: ");
        new LinkedListUtil<>(rotatedListHead).print();
    }

    public void sortLinkedListInRelativeOrderOfArr(Node<Integer> head, int[] arr) {

        //https://leetcode.com/problems/relative-sort-array/
        //https://www.geeksforgeeks.org/sort-linked-list-order-elements-appearing-array/
        //get the freq of all the elements in linkedlist
        Map<Integer, Integer> map = new TreeMap<>();
        Node<Integer> curr = head;

        while (curr != null) {
            map.put(curr.getData(), map.getOrDefault(curr.getData(), 0) + 1);
            curr = curr.getNext();
        }

        //regenerate linked list as relative order of arr
        curr = head;
        for (int a : arr) {
            int count = map.get(a);
            while (count-- != 0) {
                curr.setData(a);
                curr = curr.getNext();
            }
            map.remove(a);
        }

        //keep the values which are not in the arr[]
        for (int key : map.keySet()) {
            int count = map.get(key);
            while (count-- != 0) {
                curr.setData(key);
                curr = curr.getNext();
            }
        }

        //output
        new LinkedListUtil<>(head).print();
    }

    public void splitLinkedListInKParts(Node<Integer> head, int k) {

        //https://leetcode.com/problems/split-linked-list-in-parts/
        int len = 0;
        Node<Integer> curr = head;
        while (curr != null) {
            len++;
            curr = curr.getNext();
        }

        int width = len / k;
        int rem = len % k;

        Node<Integer>[] splits = new Node[k];
        int i = 0;
        curr = head;
        while (i < k) {

            Node<Integer> l = curr;
            for (int j = 0; j < width + (i < rem ? 1 : 0) - 1; j++) {
                if (curr != null) {
                    curr = curr.getNext();
                }
            }

            if (curr != null) {
                Node<Integer> prev = curr;
                curr = curr.getNext();
                prev.setNext(null);
            }
            splits[i++] = l;
        }

        //output
        for (int j = 0; j < splits.length; j++) {
            System.out.print((j + 1) + ": ");
            Node<Integer> h = splits[j];
            if (h == null) {
                System.out.println("null");
                continue;
            }

            while (h != null) {
                System.out.print(h.getData() + " ");
                h = h.getNext();
            }
            System.out.println();
        }
    }

    public void trimLinkedListAndRemoveAllOccurencesOfGivenVal(Node<Integer> head, int val) {
        //............................T: O(N)
        //............................S: O(1)
        //https://leetcode.com/problems/remove-linked-list-elements
        //actual
        System.out.println("Actual");
        new LinkedListUtil<>(head).print();

        Node<Integer> prev = head;
        Node<Integer> curr = head.getNext();

        while (curr != null) {

            //we are moving all the val-to-remove curr ptr only once so T: remains O(N)
            //move curr ptr till curr.data value is same as val,
            //it will break when they don't match curr.data != val
            while (curr != null && curr.getData() == val) {
                curr = curr.getNext();
            }
            //since prev ptr was not moving, append this curr next to prev
            //this will break all links between prev and the val that needs to be removed
            prev.setNext(curr);
            //update the prev ptr to its next value
            prev = prev.getNext();
            //if curr is not already null, update curr as well
            if (curr != null) {
                curr = curr.getNext();
            }
        }

        //output
        //this cond will occur when val to remove occur at the starting of linked list
        head = head.getData() == val
                ? head.getNext()
                : head;
        new LinkedListUtil<>(head).print();
    }

    public void removeZeroSumConsecutiveNodesFromLinkedList(Node<Integer> head) {
        //https://leetcode.com/problems/remove-zero-sum-consecutive-nodes-from-linked-list/
        //https://leetcode.com/problems/remove-zero-sum-consecutive-nodes-from-linked-list/discuss/2079419/Java-two-pass-solution-using-prefix-sum
        int prefixSum = 0;
        Node<Integer> dummy = new Node<>(0);
        dummy.setNext(head);

        Map<Integer, Node<Integer>> prefixSumWithNode = new HashMap<>();
        Node<Integer> curr = dummy;
        while (curr != null) {
            prefixSum += curr.getData();
            prefixSumWithNode.put(prefixSum, curr);
            curr = curr.getNext();
        }

        prefixSum = 0;
        curr = dummy;
        while (curr != null) {
            prefixSum += curr.getData();
            curr.setNext(prefixSumWithNode.get(prefixSum).getNext());
            curr = curr.getNext();
        }
        //output
        new LinkedListUtil<>(dummy.getNext()).print();
    }

    public void mergeNodesInBetweenZeros(Node<Integer> head) {
        //https://leetcode.com/problems/merge-nodes-in-between-zeros/ 
        //actual
        new LinkedListUtil<Integer>(head).print();

        Node<Integer> curr = head;
        Node<Integer> next = head.getNext();

        while (next != null) {

            if (next.getData() == 0) {
                curr.setNext(next.getNext());
                curr = curr.getNext();
                next = next.getNext();
            } else {
                curr.setData(curr.getData() + next.getData());
            }
            if (next == null) {
                break;
            }
            next = next.getNext();
        }
        //output
        new LinkedListUtil<Integer>(head).print();
    }

    public void levelOrderTraversal_Iterative(TreeNode root) {

        if (root == null) {
            return;
        }

        //actuals
        BinaryTree bt = new BinaryTree(root);
        bt.treeBFS();

        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        Queue<TreeNode> childQueue = new LinkedList<>();

        List<List> levels = new ArrayList<>();
        List nodes = new ArrayList<>();

        while (!queue.isEmpty()) {

            TreeNode curr = queue.poll();
            nodes.add(curr.getData());

            if (curr.getLeft() != null) {
                childQueue.add(curr.getLeft());
            }
            if (curr.getRight() != null) {
                childQueue.add(curr.getRight());
            }

            if (queue.isEmpty()) {
                levels.add(nodes);
                nodes = new ArrayList<>();
                queue.addAll(childQueue);
                childQueue.clear();
            }
        }

        //output
        System.out.println("Level order iterative (childQueue based approach): ");
        for (List level : levels) {
            System.out.println(level);
        }
    }

    public void levelOrderTraversal_Iterative2(TreeNode root) {

        if (root == null) {
            return;
        }

        //actuals
        BinaryTree bt = new BinaryTree(root);
        bt.treeBFS();

        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);

        List<List> levels = new ArrayList<>();
        List currLevelNodes;

        while (!queue.isEmpty()) {

            int size = queue.size();
            currLevelNodes = new ArrayList<>();

            for (int index = 0; index < size; index++) {
                TreeNode curr = queue.poll();

                currLevelNodes.add(curr.getData());

                if (curr.getLeft() != null) {
                    queue.add(curr.getLeft());
                }
                if (curr.getRight() != null) {
                    queue.add(curr.getRight());
                }
            }
            levels.add(currLevelNodes);
        }

        //output
        System.out.println("Level order iterative (size based approach): ");
        for (List level : levels) {
            System.out.println(level);
        }
    }

    public void levelOrderTraversal_Recursive_Helper(TreeNode<Integer> root, int level,
            Map<Integer, List<Integer>> levelOrder) {

        if (root == null) {
            return;
        }

        levelOrder.putIfAbsent(level, new ArrayList<>());
        levelOrder.get(level).add(root.getData());

        levelOrderTraversal_Recursive_Helper(root.getLeft(), level + 1, levelOrder);
        levelOrderTraversal_Recursive_Helper(root.getRight(), level + 1, levelOrder);
    }

    public void levelOrderTraversal_Recursive(TreeNode<Integer> root) {
        Map<Integer, List<Integer>> levelOrder = new TreeMap<>();
        levelOrderTraversal_Recursive_Helper(root, 0, levelOrder);

        //output:
        System.out.println("Level order recursive: ");
        for (List l : levelOrder.values()) {
            System.out.println(l);
        }
    }

    public void reverseLevelOrderTraversal(TreeNode<Integer> root) {

        //actuals
        BinaryTree bt = new BinaryTree(root);
        bt.treeBFS();

        List<Integer> singleListReverseLevelOrder = new ArrayList<>();

        Queue<TreeNode<Integer>> q = new LinkedList<>();
        q.add(root);
        Queue<TreeNode<Integer>> intQ = new LinkedList<>();

        List<List<Integer>> level = new ArrayList<>();
        List<Integer> nodes = new ArrayList<>();

        while (!q.isEmpty()) {

            TreeNode<Integer> temp = q.poll();
            nodes.add(temp.getData());

            if (temp.getLeft() != null) {
                intQ.add(temp.getLeft());
            }

            if (temp.getRight() != null) {
                intQ.add(temp.getRight());
            }

            if (q.isEmpty()) {
                level.add(nodes);
                nodes = new ArrayList<>();
                q.addAll(intQ);
                intQ.clear();
            }

        }

        //output
        System.out.println();
        Collections.reverse(level);
        System.out.println("Level wise: " + level);

        for (List l : level) {
            singleListReverseLevelOrder.addAll(l);
        }
        System.out.println("Single node list: " + singleListReverseLevelOrder);
    }

    public void inOrderTraversal_Iterative(TreeNode root) {

        if (root == null) {
            return;
        }

        Stack<Pair<TreeNode, Integer>> stack = new Stack<>();
        stack.push(new Pair<>(root, 0));

        while (!stack.isEmpty()) {

            Pair<TreeNode, Integer> currPair = stack.pop();
            TreeNode currNode = currPair.getKey();
            int status = currPair.getValue();

            if (currNode == null || status == 3) {
                continue;
            }

            stack.push(new Pair<>(currNode, status + 1));

            if (status == 0) {
                stack.push(new Pair<>(currNode.getLeft(), 0));
            }

            if (status == 1) {
                System.out.print(currNode.getData() + " ");
            }

            if (status == 2) {
                stack.push(new Pair<>(currNode.getRight(), 0));
            }
        }

        System.out.println();
    }

    public void inOrderTraversal_Recursive(TreeNode root) {

        if (root == null) {
            return;
        }

        inOrderTraversal_Recursive(root.getLeft());
        System.out.print(root.getData() + " ");
        inOrderTraversal_Recursive(root.getRight());
    }

    public void preOrderTraversal_Iterative(TreeNode root) {

        if (root == null) {
            return;
        }

        Stack<Pair<TreeNode, Integer>> stack = new Stack<>();
        stack.push(new Pair<>(root, 0));

        while (!stack.isEmpty()) {

            Pair<TreeNode, Integer> currPair = stack.pop();
            TreeNode currNode = currPair.getKey();
            int status = currPair.getValue();

            if (currNode == null || status == 3) {
                continue;
            }

            stack.push(new Pair<>(currNode, status + 1));

            if (status == 0) {
                System.out.print(currNode.getData() + " ");
            }

            if (status == 1) {
                stack.push(new Pair<>(currNode.getLeft(), 0));
            }

            if (status == 2) {
                stack.push(new Pair<>(currNode.getRight(), 0));
            }
        }

        System.out.println();
    }

    public void preOrderTraversal_Recursive(TreeNode root) {

        if (root == null) {
            return;
        }

        System.out.print(root.getData() + " ");
        preOrderTraversal_Recursive(root.getLeft());
        preOrderTraversal_Recursive(root.getRight());

    }

    public void postOrderTraversal_Iterative(TreeNode root) {

        if (root == null) {
            return;
        }

        Stack<Pair<TreeNode, Integer>> stack = new Stack<>();
        stack.push(new Pair<>(root, 0));

        while (!stack.isEmpty()) {

            Pair<TreeNode, Integer> currPair = stack.pop();
            TreeNode currNode = currPair.getKey();
            int status = currPair.getValue();

            if (currNode == null || status == 3) {
                continue;
            }

            stack.push(new Pair<>(currNode, status + 1));

            if (status == 0) {
                stack.push(new Pair<>(currNode.getLeft(), 0));
            }

            if (status == 1) {
                stack.push(new Pair<>(currNode.getRight(), 0));
            }

            if (status == 2) {
                System.out.print(currNode.getData() + " ");
            }
        }

        System.out.println();
    }

    public void postOrderTraversal_recursive(TreeNode root) {

        if (root == null) {
            return;
        }

        postOrderTraversal_recursive(root.getLeft());
        postOrderTraversal_recursive(root.getRight());
        System.out.print(root.getData() + " ");

    }

    public int heightOfTree(TreeNode root) {

        if (root == null) {
            return -1;
        }

        return Math.max(heightOfTree(root.getLeft()),
                heightOfTree(root.getRight())) + 1;
    }

    public TreeNode mirrorOfTree(TreeNode root) {

        if (root == null) {
            return null;
        }

        TreeNode left = mirrorOfTree(root.getLeft());
        TreeNode right = mirrorOfTree(root.getRight());
        root.setLeft(right);
        root.setRight(left);

        return root;
    }

    private void leftViewOfTree_Helper(TreeNode<Integer> root, int level, Map<Integer, Integer> result) {
        if (root == null) {
            return;
        }

        if (!result.containsKey(level)) {
            result.put(level, root.getData());
        }

        //for left view
        leftViewOfTree_Helper(root.getLeft(), level + 1, result);
        leftViewOfTree_Helper(root.getRight(), level + 1, result);
    }

    public void leftViewOfTree(TreeNode<Integer> root) {
        //...................T: O(N)
        //...................S: O(N), worst case of left-skewed tree
        Map<Integer, Integer> result = new TreeMap<>();
        leftViewOfTree_Helper(root, 0, result);

        result.entrySet().stream().forEach(e -> {
            System.out.print(e.getValue() + " ");
        });

        System.out.println();
    }

    public int leftViewOfTreeWithoutExtraSpace_MaxLevel;

    private void leftViewOfTreeWithoutExtraSpace_Helper(TreeNode<Integer> root,
            int level, List<Integer> result) {
        if (root == null) {
            return;
        }

        if (level > leftViewOfTreeWithoutExtraSpace_MaxLevel) {
            leftViewOfTreeWithoutExtraSpace_MaxLevel = level;
            result.add(root.getData());
        }

        //for left view
        leftViewOfTreeWithoutExtraSpace_Helper(root.getLeft(), level + 1, result);
        leftViewOfTreeWithoutExtraSpace_Helper(root.getRight(), level + 1, result);
    }

    public void leftViewOfTreeWithoutExtraSpace(TreeNode<Integer> root) {
        //...................T: O(N)
        //...................S: O(1)
        leftViewOfTreeWithoutExtraSpace_MaxLevel = -1;
        List<Integer> result = new ArrayList<>();
        leftViewOfTreeWithoutExtraSpace_Helper(root, 0, result);

        result.stream().forEach(e -> {
            System.out.print(e + " ");
        });

        System.out.println();
    }

    private void rightViewOfTree_Helper(TreeNode<Integer> root, int level, Map<Integer, Integer> result) {

        if (root == null) {
            return;
        }

        if (!result.containsKey(level)) {
            result.put(level, root.getData());
        }

        //for right view
        rightViewOfTree_Helper(root.getRight(), level + 1, result);
        rightViewOfTree_Helper(root.getLeft(), level + 1, result);
    }

    public void rightViewOfTree(TreeNode<Integer> root) {
        //...................T: O(N)
        //...................S: O(N), worst case of right-skewed tree
        Map<Integer, Integer> result = new TreeMap<>();
        rightViewOfTree_Helper(root, 0, result);

        result.entrySet().stream().forEach(e -> {
            System.out.print(e.getValue() + " ");
        });

        System.out.println();
    }

    public int rightViewOfTreeWithoutExtraSpace_MaxLevel;

    private void rightViewOfTreeWithoutExtraSpace_Helper(TreeNode<Integer> root,
            int level, List<Integer> result) {

        if (root == null) {
            return;
        }

        if (level > rightViewOfTreeWithoutExtraSpace_MaxLevel) {
            rightViewOfTreeWithoutExtraSpace_MaxLevel = level;
            result.add(root.getData());
        }

        //for right view
        rightViewOfTreeWithoutExtraSpace_Helper(root.getRight(), level + 1, result);
        rightViewOfTreeWithoutExtraSpace_Helper(root.getLeft(), level + 1, result);
    }

    public void rightViewOfTreeWithoutExtraSpace(TreeNode<Integer> root) {
        //...................T: O(N)
        //...................S: O(1)
        rightViewOfTreeWithoutExtraSpace_MaxLevel = -1;
        List<Integer> result = new ArrayList<>();
        rightViewOfTreeWithoutExtraSpace_Helper(root, 0, result);

        result.stream().forEach(e -> {
            System.out.print(e + " ");
        });

        System.out.println();
    }

    public void topViewOfTree(TreeNode<Integer> root) {

        Queue<Pair<TreeNode<Integer>, Integer>> queue = new LinkedList<>();
        queue.add(new Pair<>(root, 0));

        Map<Integer, Integer> result = new TreeMap<>();

        while (!queue.isEmpty()) {

            Pair<TreeNode<Integer>, Integer> currPair = queue.poll();
            TreeNode<Integer> currNode = currPair.getKey();
            int vLevel = currPair.getValue();

            if (!result.containsKey(vLevel)) {
                result.put(vLevel, currNode.getData());
            }

            if (currNode.getLeft() != null) {
                queue.add(new Pair<>(currNode.getLeft(), vLevel - 1));
            }
            if (currNode.getRight() != null) {
                queue.add(new Pair<>(currNode.getRight(), vLevel + 1));
            }
        }

        result.entrySet().stream().forEach(e -> {
            System.out.print(e.getValue() + " ");
        });

        System.out.println();
    }

    public void bottomViewOfTree(TreeNode<Integer> root) {

        //pair: node,vlevels
        Queue<Pair<TreeNode<Integer>, Integer>> q = new LinkedList<>();
        q.add(new Pair<>(root, 0));

        Map<Integer, Integer> bottomView = new TreeMap<>();

        while (!q.isEmpty()) {

            Pair<TreeNode<Integer>, Integer> p = q.poll();
            TreeNode<Integer> n = p.getKey();
            int vLevel = p.getValue();

            //updates the vlevel with new node data, as we go down the tree in level order wise
            bottomView.put(vLevel, n.getData());

            if (n.getLeft() != null) {
                q.add(new Pair<>(n.getLeft(), vLevel - 1));
            }

            if (n.getRight() != null) {
                q.add(new Pair<>(n.getRight(), vLevel + 1));
            }

        }

        bottomView.entrySet().stream().forEach(e -> {
            System.out.print(e.getValue() + " ");
        });

        System.out.println();
    }

    public void zigZagTreeTraversal(TreeNode<Integer> root, boolean isLeftToRight) {

        Stack<TreeNode<Integer>> stack = new Stack<>();
        stack.push(root);
        Stack<TreeNode<Integer>> childStack = new Stack<>();

        List<List<Integer>> level = new ArrayList<>();
        List<Integer> zigZagNodes = new ArrayList<>();

        while (!stack.isEmpty()) {

            TreeNode<Integer> curr = stack.pop();
            zigZagNodes.add(curr.getData());

            if (isLeftToRight) {

                if (curr.getRight() != null) {
                    childStack.push(curr.getRight());
                }

                if (curr.getLeft() != null) {
                    childStack.push(curr.getLeft());
                }
            } else {

                if (curr.getLeft() != null) {
                    childStack.push(curr.getLeft());
                }

                if (curr.getRight() != null) {
                    childStack.push(curr.getRight());
                }
            }

            if (stack.isEmpty()) {

                isLeftToRight = !isLeftToRight;
                level.add(zigZagNodes);
                zigZagNodes = new ArrayList<>();
                stack.addAll(childStack);
                childStack.clear();
            }
        }

        //output
        System.out.println("Output: " + level);
    }

    private void minAndMaxInBST_Helper(TreeNode<Integer> root, List<Integer> l) {

        if (root == null) {
            return;
        }

        //inorder traversal
        minAndMaxInBST_Helper(root.getLeft(), l);
        if (root != null) {
            l.add(root.getData());
        }
        minAndMaxInBST_Helper(root.getRight(), l);
    }

    public void minAndMaxInBST(TreeNode<Integer> root) {
        List<Integer> inOrder = new ArrayList<>();
        minAndMaxInBST_Helper(root, inOrder);

        System.out.println("Min & Max in BST: " + inOrder.get(0) + " " + inOrder.get(inOrder.size() - 1));

    }

    TreeNode treeToDoublyLinkedList_Prev;
    TreeNode treeToDoublyLinkedList_HeadOfDLL;

    private void treeToDoublyLinkedList_Helper(TreeNode root) {
        if (root == null) {
            return;
        }

        treeToDoublyLinkedList_Helper(root.getLeft());

        if (treeToDoublyLinkedList_HeadOfDLL == null) {
            treeToDoublyLinkedList_HeadOfDLL = root;
        }

        if (treeToDoublyLinkedList_Prev != null) {
            root.setLeft(treeToDoublyLinkedList_Prev);
            treeToDoublyLinkedList_Prev.setRight(root);
        }

        treeToDoublyLinkedList_Prev = root;

        treeToDoublyLinkedList_Helper(root.getRight());
    }

    private void treeToDoublyLinkedList_Print(TreeNode head) {

        while (head != null) {

            System.out.print(head.getData() + " ");
            head = head.getRight();
        }
        System.out.println();
    }

    public void treeToDoublyLinkedList(TreeNode root) {

        //just resetting
        treeToDoublyLinkedList_Prev = null;
        treeToDoublyLinkedList_HeadOfDLL = null;

        treeToDoublyLinkedList_Helper(root);
        treeToDoublyLinkedList_Print(treeToDoublyLinkedList_HeadOfDLL);
    }

    private void checkIfAllLeafNodeOfTreeAtSameLevel_Helper(TreeNode root, int level, Set<Integer> levels) {

        if (root == null) {
            return;
        }

        //leaf
        if (root.getLeft() == null && root.getRight() == null) {
            levels.add(level);
        }

        checkIfAllLeafNodeOfTreeAtSameLevel_Helper(root.getLeft(), level + 1, levels);
        checkIfAllLeafNodeOfTreeAtSameLevel_Helper(root.getRight(), level + 1, levels);

    }

    public void checkIfAllLeafNodeOfTreeAtSameLevel(TreeNode root) {
        //..........................T: O(N), traversing all the nodes
        //..........................S: O(Leaf Nodes(M)), in worst case if all lead nodes are at different levels
        Set<Integer> levels = new HashSet<>();
        checkIfAllLeafNodeOfTreeAtSameLevel_Helper(root, 0, levels);

        System.out.println("Leaf at same level: " + (levels.size() == 1));
    }

    TreeNode<Integer> isTreeBST_Prev;

    private boolean isTreeBST_Helper(TreeNode<Integer> root) {

        if (root == null) {
            return true;
        }

        boolean leftBst = isTreeBST_Helper(root.getLeft());

        if (isTreeBST_Prev != null && isTreeBST_Prev.getData() > root.getData()) {
            return false;
        }

        isTreeBST_Prev = root;

        boolean rightBst = isTreeBST_Helper(root.getRight());

        return leftBst && rightBst;
    }

    public void isTreeBST(TreeNode<Integer> root) {

        //just resetting
        isTreeBST_Prev = null;

        System.out.println("Tree is BST: " + isTreeBST_Helper(root));
    }

    private void kThLargestNodeInBST_Helper(TreeNode<Integer> root, int K, PriorityQueue<Integer> minHeap) {

        if (root == null) {
            return;
        }

        minHeap.add(root.getData());
        if (minHeap.size() > K) {
            minHeap.poll();
        }

        kThLargestNodeInBST_Helper(root.getLeft(), K, minHeap);
        kThLargestNodeInBST_Helper(root.getRight(), K, minHeap);
    }

    public void kTHLargestNodeInBST(TreeNode<Integer> root, int K) {
        //actual
        //inorder of BST is sorted nodes list
        inOrderTraversal_Iterative(root);

        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        kThLargestNodeInBST_Helper(root, K, minHeap);

        System.out.println(K + " largest node from BST: " + minHeap.poll());
    }

    private int kTHLargestNodeInBSTWithoutHeap_Value;
    private int kTHLargestNodeInBSTWithoutHeap_CurrK;

    public void kTHLargestNodeInBSTWithoutHeap_Helper(TreeNode<Integer> root, int K) {
        if (root == null) {
            return;
        }
        //since it a BST the larger nodes are in right sub tree so 
        //traversing the right subtree first to find the kth node
        kTHLargestNodeInBSTWithoutHeap_Helper(root.getRight(), K);

        kTHLargestNodeInBSTWithoutHeap_CurrK++;

        if (kTHLargestNodeInBSTWithoutHeap_CurrK == K) {
            kTHLargestNodeInBSTWithoutHeap_Value = root.getData();
            return;
        }
        kTHLargestNodeInBSTWithoutHeap_Helper(root.getLeft(), K);
    }

    public void kTHLargestNodeInBSTWithoutHeap(TreeNode<Integer> root, int K) {

        kTHLargestNodeInBSTWithoutHeap_CurrK = 0;
        kTHLargestNodeInBSTWithoutHeap_Value = Integer.MIN_VALUE;
        kTHLargestNodeInBSTWithoutHeap_Helper(root, K);
        System.out.println(K + " largest node from BST without heap: " + kTHLargestNodeInBSTWithoutHeap_Value);
    }

    private void kThSmallestNodeInBST_Helper(TreeNode<Integer> root, int K, PriorityQueue<Integer> maxHeap) {

        if (root == null) {
            return;
        }

        maxHeap.add(root.getData());
        if (maxHeap.size() > K) {
            maxHeap.poll();
        }

        kThSmallestNodeInBST_Helper(root.getLeft(), K, maxHeap);
        kThSmallestNodeInBST_Helper(root.getRight(), K, maxHeap);
    }

    public void kTHSmallestNodeInBST(TreeNode<Integer> root, int K) {
        //actual
        //inorder of BST is sorted nodes list
        inOrderTraversal_Iterative(root);

        //maxHeap
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(
                (o1, o2) -> o2.compareTo(o1)
        );
        kThSmallestNodeInBST_Helper(root, K, maxHeap);

        System.out.println(K + " smallest node from BST: " + maxHeap.poll());
    }

    class Height {

        int height = 0;
    }

    private boolean isTreeHeightBalanced_Helper(TreeNode root, Height currHeight) {

        //this approach calculates height and check height balanced at the same time
        if (root == null) {
            currHeight.height = -1;
            return true;
        }

        Height leftTreeHeight = new Height();
        Height rightTreeHeight = new Height();

        boolean isLeftBal = isTreeHeightBalanced_Helper(root.getLeft(), leftTreeHeight);
        boolean isRightBal = isTreeHeightBalanced_Helper(root.getRight(), rightTreeHeight);

        //calculate the height for the current node
        currHeight.height = Math.max(leftTreeHeight.height, rightTreeHeight.height) + 1;

        //checking the cond if height balanced
        //if diff b/w left subtree or right sub tree is greater than 1 it's
        //not balanced
        if (Math.abs(leftTreeHeight.height - rightTreeHeight.height) > 1) {
            return false;
        }

        //if the above cond doesn't fulfil
        //it should check if both of the left or right sub tree are balanced or not
        return isLeftBal && isRightBal;
    }

    public void isTreeHeightBalanced(TreeNode root) {
        Height h = new Height();
        System.out.println("Is tree heght  balanced: " + isTreeHeightBalanced_Helper(root, h));
    }

    public boolean checkTwoTreeAreMirror(TreeNode<Integer> root1, TreeNode<Integer> root2) {
        if (root1 == null && root2 == null) {
            return true;
        }

        if (root1 == null || root2 == null) {
            return false;
        }

        return root1.getData() == root2.getData()
                && checkTwoTreeAreMirror(root1.getLeft(), root2.getRight())
                && checkTwoTreeAreMirror(root1.getRight(), root2.getLeft());
    }

    private int convertTreeToSumTree_Sum(TreeNode<Integer> root) {

        if (root == null) {
            return 0;
        }

        int lSum = convertTreeToSumTree_Sum(root.getLeft());
        int rSum = convertTreeToSumTree_Sum(root.getRight());

        return lSum + rSum + root.getData();

    }

    public void convertTreeToSumTree(TreeNode<Integer> root) {

        //actual
        BinaryTree<Integer> bt = new BinaryTree<>(root);
        bt.treeBFS();

        Queue<TreeNode<Integer>> q = new LinkedList<>();
        q.add(root);

        while (!q.isEmpty()) {

            TreeNode<Integer> t = q.poll();

            if (t.getLeft() != null) {
                q.add(t.getLeft());
            }

            if (t.getRight() != null) {
                q.add(t.getRight());
            }

            //leaf
            if (t.getLeft() == null && t.getRight() == null) {
                t.setData(0);
                continue;
            }

            // - t.getData() just don't include the value of that node itself
            t.setData(convertTreeToSumTree_Sum(t) - t.getData());

        }

        //output
        System.out.println();
        bt = new BinaryTree<>(root);
        bt.treeBFS();
        System.out.println();

    }

    private int convertTreeToSumTree_Recursion_Helper(TreeNode<Integer> root) {

        if (root == null) {
            return 0;
        }

        int data = root.getData();

        int lSum = convertTreeToSumTree_Recursion_Helper(root.getLeft());
        int rSum = convertTreeToSumTree_Recursion_Helper(root.getRight());

        //leaf
        if (root.getLeft() == null && root.getRight() == null) {
            root.setData(0);
            return data;
        } else {
            root.setData(lSum + rSum);
            return lSum + rSum + data;
        }
    }

    public void convertTreeToSumTree_Recursion(TreeNode<Integer> root) {

        //OPTIMISED
        //actual
        new BinaryTree<>(root).treeBFS();
        System.out.println();

        convertTreeToSumTree_Recursion_Helper(root);

        //output
        new BinaryTree<>(root).treeBFS();
        System.out.println();

    }

    List<Integer> printKSumPathAnyNodeTopToDown_PathList;

    private void printKSumPathAnyNodeTopToDown_Helper(TreeNode<Integer> root, int K) {

        if (root == null) {
            return;
        }

        printKSumPathAnyNodeTopToDown_PathList.add(root.getData());

        printKSumPathAnyNodeTopToDown_Helper(root.getLeft(), K);
        printKSumPathAnyNodeTopToDown_Helper(root.getRight(), K);

        int pathSum = 0;
        for (int i = printKSumPathAnyNodeTopToDown_PathList.size() - 1; i >= 0; i--) {

            pathSum += printKSumPathAnyNodeTopToDown_PathList.get(i);
            if (pathSum == K) {
                //print actual nodes data
                for (int j = i; j < printKSumPathAnyNodeTopToDown_PathList.size(); j++) {
                    System.out.print(printKSumPathAnyNodeTopToDown_PathList.get(j) + " ");
                }
                System.out.println();
            }
        }
        //remove current node
        printKSumPathAnyNodeTopToDown_PathList.remove(printKSumPathAnyNodeTopToDown_PathList.size() - 1);
    }

    public void printKSumPathAnyNodeTopToDown(TreeNode<Integer> root, int K) {
        printKSumPathAnyNodeTopToDown_PathList = new ArrayList<>();
        printKSumPathAnyNodeTopToDown_Helper(root, K);
    }

    private TreeNode<Integer> lowestCommonAncestorOfTree_Helper(TreeNode<Integer> root, int N1, int N2) {

        if (root == null) {
            return null;
        }

        if (N1 == root.getData() || N2 == root.getData()) {
            return root;
        }

        TreeNode<Integer> leftNode = lowestCommonAncestorOfTree_Helper(root.getLeft(), N1, N2);
        TreeNode<Integer> rightNode = lowestCommonAncestorOfTree_Helper(root.getRight(), N1, N2);

        if (leftNode != null && rightNode != null) {
            return root;
        }

        return leftNode == null ? rightNode : leftNode;

    }

    public void lowestCommonAncestorOfTree(TreeNode<Integer> root, int N1, int N2) {
        System.out.println("Lowest common ancestor of " + N1 + " " + N2 + ": " + lowestCommonAncestorOfTree_Helper(root, N1, N2));
    }

    class CheckTreeIsSumTree {

        /*Helper class for checkTreeIsSumTree_Helper method*/ int data = 0;
    }

    public boolean checkTreeIsSumTree_Helper(TreeNode<Integer> root, CheckTreeIsSumTree obj) {

        if (root == null) {
            obj.data = 0;
            return true;
        }

        CheckTreeIsSumTree leftSubTreeSum = new CheckTreeIsSumTree();
        CheckTreeIsSumTree rightSubTreeSum = new CheckTreeIsSumTree();

        boolean isLeftSubTreeSumTree = checkTreeIsSumTree_Helper(root.getLeft(), leftSubTreeSum);
        boolean isRightSubTreeSumTree = checkTreeIsSumTree_Helper(root.getRight(), rightSubTreeSum);

        //calculating data for the current root node itself
        obj.data = root.getData() + leftSubTreeSum.data + rightSubTreeSum.data;

        //current root node should not be be leaf
        if (!(root.getLeft() == null && root.getRight() == null)
                && //current root is not equal to the sum of left and rigth sub tree 
                (root.getData() != leftSubTreeSum.data + rightSubTreeSum.data)) {
            return false;
        }

        return isLeftSubTreeSumTree && isRightSubTreeSumTree;
    }

    public void checkTreeIsSumTree(TreeNode<Integer> root) {
        System.out.println("Check if a tree is sum tree: " + checkTreeIsSumTree_Helper(root, new CheckTreeIsSumTree()));
    }

    class TreeLongestPathNodeSum {

        /*Helper class for longestPathNodeSum method*/
        List<Integer> path = new ArrayList<>();
        List<Integer> nodeInLongestPath = new ArrayList<>();
        int maxLevel = 0;
        int longestPathSum = 0;
        int maxSumOfAnyPath = 0;
    }

    private void longestPathNodeSum_Helper(TreeNode<Integer> root,
            TreeLongestPathNodeSum obj, int level) {

        if (root == null) {
            return;
        }
        obj.path.add(root.getData());
        longestPathNodeSum_Helper(root.getLeft(), obj, level + 1);
        longestPathNodeSum_Helper(root.getRight(), obj, level + 1);

        int currPathSum = 0;
        //to find the max sum of any path this for() is outside of below if() block
        //otherwise only to find sum of longest path move this for() inside the if()
        //it will be optimised then
        for (int nodes : obj.path) {
            currPathSum += nodes;
        }

        if (level > obj.maxLevel) {
            obj.maxLevel = level;
            obj.longestPathSum = currPathSum;
            obj.nodeInLongestPath.clear();
            obj.nodeInLongestPath.addAll(obj.path);
        }

        obj.maxSumOfAnyPath = Math.max(obj.maxSumOfAnyPath, currPathSum);

        //remove the last added node
        obj.path.remove(obj.path.size() - 1);
    }

    public void longestPathNodeSum(TreeNode<Integer> root) {
        TreeLongestPathNodeSum obj = new TreeLongestPathNodeSum();
        longestPathNodeSum_Helper(root, obj, 0);
        System.out.println("The sum of nodes of longest path of tree: " + obj.longestPathSum);
        System.out.println("The length of longest path in the tree: " + obj.maxLevel);
        System.out.println("The nodes of longest path of tree: " + obj.nodeInLongestPath);
        System.out.println("The max sum of nodes that may occur on any path of tree: " + obj.maxSumOfAnyPath);
    }

    private void findPredecessorAndSuccessorInBST_Helper(TreeNode<Integer> root, int key, TreeNode<Integer>[] result) {

        if (root == null) {
            return;
        }

        if (root.getData() == key) {

            if (root.getLeft() != null) {
                //predecessor : rightmost node in the left subtree
                TreeNode<Integer> pred = root.getLeft();
                while (pred.getRight() != null) {
                    pred = pred.getRight();
                }
                result[0] = pred;
            }

            if (root.getRight() != null) {
                //successor : leftmost node in the right subtree
                TreeNode<Integer> succ = root.getRight();
                while (succ.getLeft() != null) {
                    succ = succ.getLeft();
                }
                result[1] = succ;
            }
            return;
        }

        //key is less than root data so move to whole left sub tree
        //ex: root = 2, key = 4
        //2 > 4 -> else
        if (key < root.getData()) {
            result[1] = root; //succ
            findPredecessorAndSuccessorInBST_Helper(root.getLeft(), key, result);
        } else {
            //else move to whole right sub tree
            //in else because 2 is not greater 4 
            //inorder of bst is sorted list [2, 4] and 2 will be pred of 4
            result[0] = root; //pred
            findPredecessorAndSuccessorInBST_Helper(root.getRight(), key, result);
        }
    }

    public void findPredecessorAndSuccessorInBST(TreeNode<Integer> root, int key) {
        //..................................T: O(H), worst case: key can be the leaf node, H = height of BST
        //..................................S: O(H), function call stack 
        //can use list also
        //[0] : predecessor, [1] : successor
        TreeNode<Integer>[] result = new TreeNode[2];
        findPredecessorAndSuccessorInBST_Helper(root, key, result);
        System.out.println("Predecessor and successor of BST: "
                + (result[0] != null ? result[0].getData() : "null") + " "
                + (result[1] != null ? result[1].getData() : "null"));

    }

    private int countNodesThatLieInGivenRange_Count = 0;

    private void countNodesThatLieInGivenRange_Helper(TreeNode<Integer> root, int low, int high) {

        if (root == null) {
            return;
        }

        if (root.getData() >= low && root.getData() <= high) {
            countNodesThatLieInGivenRange_Count++;
        }

        countNodesThatLieInGivenRange_Helper(root.getLeft(), low, high);
        countNodesThatLieInGivenRange_Helper(root.getRight(), low, high);
    }

    public void countNodesThatLieInGivenRange(TreeNode<Integer> root, int low, int high) {
        countNodesThatLieInGivenRange_Count = 0;
        countNodesThatLieInGivenRange_Helper(root, low, high);
        System.out.println("No. of nodes that lie in given range: " + countNodesThatLieInGivenRange_Count);
    }

    public void flattenBSTToLinkedList(TreeNode root) {

        //...........................T: O(N)
        //...........................S: O(N)
        if (root == null) {
            return;
        }

        /*Deque<TreeNode> dQueue = new ArrayDeque<>();
         dQueue.add(root);

         while (!dQueue.isEmpty()) {

         TreeNode curr = dQueue.removeFirst();

         if (curr.getRight() != null) {
         dQueue.addFirst(curr.getRight());
         }

         if (curr.getLeft() != null) {
         dQueue.addFirst(curr.getLeft());
         }

         if (!dQueue.isEmpty()) {
         curr.setRight(dQueue.peek());
         curr.setLeft(null);
         }

         }*/
 /*List<TreeNode> q = new ArrayList<>();
         q.add(root);
         while (!q.isEmpty()) {

         TreeNode curr = q.remove(0);

         if (curr.getRight() != null) {
         q.add(0, curr.getRight());
         }

         if (curr.getLeft() != null) {
         q.add(0, curr.getLeft());
         }

         if (!q.isEmpty()) {
         curr.setRight(q.get(0));
         curr.setLeft(null);
         }

         }*/
        //using LIFO stack
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()) {

            TreeNode curr = stack.pop();

            //we need left at peek of stack so pushing right first
            //and then left so that left can be at peek
            if (curr.getRight() != null) {
                stack.push(curr.getRight());
            }

            if (curr.getLeft() != null) {
                stack.push(curr.getLeft());
            }

            if (!stack.isEmpty()) {

                curr.setRight(stack.peek());
                curr.setLeft(null);
            }
        }

        //output:
        new BinaryTree(root).treeBFS();
        System.out.println();
    }

    private TreeNode flattenBSTToLinkedList_Recursion_Current;

    public void flattenBSTToLinkedList_Recursion_Helper(TreeNode root) {
        if (root == null) {
            return;
        }

        TreeNode left = root.getLeft();
        TreeNode right = root.getRight();

        if (flattenBSTToLinkedList_Recursion_Current != null) {
            flattenBSTToLinkedList_Recursion_Current.setLeft(null);
            flattenBSTToLinkedList_Recursion_Current.setRight(root);
        }

        flattenBSTToLinkedList_Recursion_Current = root;

        flattenBSTToLinkedList_Recursion_Helper(left);
        flattenBSTToLinkedList_Recursion_Helper(right);
    }

    public void flattenBSTToLinkedList_Recursion(TreeNode root) {
        //...........................T: O(N)
        //...........................S: O(1), As we are not taking extra stack other than function call stack
        flattenBSTToLinkedList_Recursion_Current = null;
        flattenBSTToLinkedList_Recursion_Helper(root);
        //output:
        new BinaryTree(root).treeBFS();
        System.out.println();
    }

    private void diagonalTraversalOfTree_Helper(TreeNode<Integer> root, int level, Map<Integer, List<Integer>> result) {

        if (root == null) {
            return;
        }

        result.putIfAbsent(level, new ArrayList<>());
        result.get(level).add(root.getData());

        diagonalTraversalOfTree_Helper(root.getLeft(), level + 1, result);
        diagonalTraversalOfTree_Helper(root.getRight(), level, result);
    }

    public void diagonalTraversalOfTree(TreeNode<Integer> root) {

        Map<Integer, List<Integer>> result = new TreeMap<>();
        diagonalTraversalOfTree_Helper(root, 0, result);
        System.out.println("Diagonal traversal of tree");
        for (Map.Entry<Integer, List<Integer>> e : result.entrySet()) {
            System.out.println(e.getValue());
        }
    }

    private int diameterOfTree_Helper(TreeNode<Integer> root, Height height) {

        if (root == null) {
            height.height = 0;
            return 0;
        }

        Height leftSubTreeHeight = new Height();
        Height rightSubTreeHeight = new Height();

        int leftTreeDiameter = diameterOfTree_Helper(root.getLeft(), leftSubTreeHeight);
        int rightTreeDiameter = diameterOfTree_Helper(root.getRight(), rightSubTreeHeight);

        //current node height
        height.height = Math.max(leftSubTreeHeight.height, rightSubTreeHeight.height) + 1;

        return Math.max(
                Math.max(leftTreeDiameter, rightTreeDiameter),
                leftSubTreeHeight.height + rightSubTreeHeight.height + 1
        );
    }

    public void diameterOfTree(TreeNode<Integer> root) {
        //https://leetcode.com/problems/diameter-of-binary-tree/
        //gives the total no of nodes forming that diameter
        System.out.println("Diameter of tree: " + diameterOfTree_Helper(root, new Height()));
        //gives the no. of edges between these nodes
        System.out.println("Diameter of tree: " + (diameterOfTree_Helper(root, new Height()) - 1));
    }

    class CheckIfBinaryTreeIsMaxHeapClass {

        /*Helper class for checkIfBinaryTreeIsMaxHeap method*/
        int data;
    }

    private boolean checkIfBinaryTreeIsMaxHeap_Helper(TreeNode<Integer> root, CheckIfBinaryTreeIsMaxHeapClass obj) {

        if (root == null) {
            obj.data = Integer.MIN_VALUE;
            return true;
        }

        CheckIfBinaryTreeIsMaxHeapClass leftSubTree = new CheckIfBinaryTreeIsMaxHeapClass();
        CheckIfBinaryTreeIsMaxHeapClass rightSubTree = new CheckIfBinaryTreeIsMaxHeapClass();

        boolean isLeftMaxHeap = checkIfBinaryTreeIsMaxHeap_Helper(root.getLeft(), leftSubTree);
        boolean isRightMaxHeap = checkIfBinaryTreeIsMaxHeap_Helper(root.getRight(), rightSubTree);

        //calculating current node's object
        obj.data = root.getData();

        //if root's data is less than its immediate left OR right child return false
        if (root.getData() < leftSubTree.data || root.getData() < rightSubTree.data) {
            return false;
        }

        return isLeftMaxHeap && isRightMaxHeap;
    }

    public void checkIfBinaryTreeIsMaxHeap(TreeNode<Integer> root) {
        System.out.println("Given binary tree is max heap: "
                + checkIfBinaryTreeIsMaxHeap_Helper(root, new CheckIfBinaryTreeIsMaxHeapClass()));
    }

    public boolean checkIfAllLevelsOfTwoTreesAreAnagrams_1(TreeNode<Integer> root1, TreeNode<Integer> root2) {

        //this approach performs level order traversal first and then anagrams checking
        if (root1 == null && root2 == null) {
            return true;
        }

        if (root1 == null || root2 == null) {
            return false;
        }

        Map<Integer, List<Integer>> levelOrder1 = new TreeMap<>();
        levelOrderTraversal_Recursive_Helper(root1, 0, levelOrder1); //T: O(N)

        Map<Integer, List<Integer>> levelOrder2 = new TreeMap<>();
        levelOrderTraversal_Recursive_Helper(root2, 0, levelOrder2); //T: O(N)

        //if both tree are of different levels then two trees acn't be anagrams
        if (levelOrder1.size() != levelOrder2.size()) {
            return false;
        }

        //T: O(H) H = height of tree
        for (int level = 0; level < levelOrder1.size(); level++) {

            List<Integer> l1 = levelOrder1.get(level);
            List<Integer> l2 = levelOrder2.get(level);

            //sort: T: O(Logl1) + O(Logl2)
            Collections.sort(l1);
            Collections.sort(l2);

            //if levels of two trees after sorting are not equal then they are not anagram
            //ex l1.sort: [2,3], l2.sort: [3,4] then l1 != l2
            if (!l1.equals(l2)) {
                return false;
            }
        }

        return true;
    }

    public boolean checkIfAllLevelsOfTwoTreesAreAnagrams_2(TreeNode<Integer> root1, TreeNode<Integer> root2) {

        //this approach performs level order traversal and anagrams checking at the same time
        if (root1 == null && root2 == null) {
            return true;
        }

        if (root1 == null || root2 == null) {
            return false;
        }

        Queue<TreeNode<Integer>> q1 = new LinkedList<>();
        Queue<TreeNode<Integer>> q2 = new LinkedList<>();
        q1.add(root1);
        q2.add(root2);

        Queue<TreeNode<Integer>> intQ1 = new LinkedList<>();
        Queue<TreeNode<Integer>> intQ2 = new LinkedList<>();

        List<Integer> l1 = new ArrayList<>();
        List<Integer> l2 = new ArrayList<>();

        while (!q1.isEmpty() && !q2.isEmpty()) {

            TreeNode<Integer> curr1 = q1.poll();
            TreeNode<Integer> curr2 = q2.poll();

            l1.add(curr1.getData());
            l2.add(curr2.getData());

            if (curr1.getLeft() != null) {
                intQ1.add(curr1.getLeft());
            }

            if (curr1.getRight() != null) {
                intQ1.add(curr1.getRight());
            }

            if (curr2.getLeft() != null) {
                intQ2.add(curr2.getLeft());
            }

            if (curr2.getRight() != null) {
                intQ2.add(curr2.getRight());
            }

            if (q1.isEmpty() && q2.isEmpty()) {

                Collections.sort(l1);
                Collections.sort(l2);

                //if after sorting the nodes at a paticular level from both
                //the tree are not equal
                //ex l1.sort: [2,3], l2.sort: [3,4] then l1 != l2
                if (!l1.equals(l2)) {
                    return false;
                }

                l1.clear();
                l2.clear();

                //intQ holds the immediate child nodes of a parent node
                //if the no. of immediate child nodes are different then further 
                //checking for anagrams are not req.
                //ex T1: 1.left = 2, 1.right = 3
                //T2: 1.left = 2
                //at parent node = 1 intQ will hold immediate childs
                //intQ1 = [2,3], intQ2 = [2] here intQ1.size != intQ2.size
                if (intQ1.size() != intQ2.size()) {
                    return false;
                }

                q1.addAll(intQ1);
                q2.addAll(intQ2);

                intQ1.clear();
                intQ2.clear();
            }
        }

        //if none of the cond in while is false then all the levels in both tree are anagrams
        return true;
    }

    private boolean areTwoTreeIsoMorphic_Helper(TreeNode<Integer> root1, TreeNode<Integer> root2) {

        if (root1 == null && root2 == null) {
            return true;
        }

        if (root1 == null || root2 == null) {
            return false;
        }

        return root1.getData() == root2.getData()
                && ((areTwoTreeIsoMorphic_Helper(root1.getLeft(), root2.getRight()) && areTwoTreeIsoMorphic_Helper(root1.getRight(), root2.getLeft()))
                || (areTwoTreeIsoMorphic_Helper(root1.getLeft(), root2.getLeft()) && areTwoTreeIsoMorphic_Helper(root1.getRight(), root2.getRight())));
    }

    public boolean areTwoTreeIsoMorphic(TreeNode<Integer> root1, TreeNode<Integer> root2) {
        return areTwoTreeIsoMorphic_Helper(root1, root2);
    }

    private String findDuplicateSubtreeInAGivenTree_Inorder(TreeNode<Integer> root,
            Map<String, Integer> map, List<TreeNode<Integer>> subtrees) {

        if (root == null) {
            return "";
        }

        String str = "(";
        str += findDuplicateSubtreeInAGivenTree_Inorder(root.getLeft(), map, subtrees);
        str += String.valueOf(root.getData());
        str += findDuplicateSubtreeInAGivenTree_Inorder(root.getRight(), map, subtrees);
        str += ")";

//        System.out.println(str);
        if (map.containsKey(str) && map.get(str) == 1) {
            //System.out.println(root.getData()+ " "); //print the starting node of suplicate subtree
            subtrees.add(root);
        }

        map.put(str, map.getOrDefault(str, 0) + 1);

        return str;
    }

    public void findDuplicateSubtreeInAGivenTree(TreeNode<Integer> root) {
        Map<String, Integer> map = new HashMap<>();
        List<TreeNode<Integer>> subtrees = new ArrayList<>();
        findDuplicateSubtreeInAGivenTree_Inorder(root, map, subtrees);

        //output:
        //print level order of found subtrees
        for (TreeNode<Integer> tree : subtrees) {
            levelOrderTraversal_Recursive(tree);
        }
    }

    private void allNodesAtKDistanceFromRoot(TreeNode<Integer> root, int level,
            int K, List<Integer> result) {

        if (root == null) {
            return;
        }

        if (level == K) {
            result.add(root.getData());
        }

        allNodesAtKDistanceFromRoot(root.getLeft(), level + 1, K, result);
        allNodesAtKDistanceFromRoot(root.getRight(), level + 1, K, result);
    }

    private int printAllTheNodesAtKDistanceFromTargetNode_DFS(TreeNode<Integer> root, int target,
            int K, List<Integer> result) {

        if (root == null) {
            return -1;
        }

        if (root.getData() == target) {
            //search all the nodes at K dist below the target node
            allNodesAtKDistanceFromRoot(root, 0, K, result);
            return 1;
        }

        int left = printAllTheNodesAtKDistanceFromTargetNode_DFS(root.getLeft(), target, K, result);

        if (left != -1) {
            if (left == K) {
                result.add(root.getData());
                return -1;
            }
            allNodesAtKDistanceFromRoot(root.getRight(), left + 1, K, result);
            return left + 1;
        }

        int right = printAllTheNodesAtKDistanceFromTargetNode_DFS(root.getRight(), target, K, result);

        if (right != -1) {
            if (right == K) {
                result.add(root.getData());
                return -1;
            }
            allNodesAtKDistanceFromRoot(root.getLeft(), right + 1, K, result);
            return right + 1;
        }

        return -1;
    }

    public void printAllTheNodesAtKDistanceFromTargetNode(TreeNode<Integer> root, int target, int K) {

        List<Integer> result = new ArrayList<>();
        printAllTheNodesAtKDistanceFromTargetNode_DFS(root, target, K, result);
        //output:
        System.out.println("All nodes at K distance from target node: " + result);
    }

    private boolean deleteTreeNodesAndReturnForest_Helper(TreeNode<Integer> root,
            Set<Integer> deleteSet, List<TreeNode<Integer>> result) {

        if (root == null) {
            return false;
        }

        boolean deleteLeft = deleteTreeNodesAndReturnForest_Helper(root.getLeft(), deleteSet, result);
        boolean deleteRight = deleteTreeNodesAndReturnForest_Helper(root.getRight(), deleteSet, result);

        if (deleteLeft) {
            root.setLeft(null);
        }

        if (deleteRight) {
            root.setRight(null);
        }

        if (deleteSet.contains(root.getData())) {

            if (root.getLeft() != null) {
                result.add(root.getLeft());
            }

            if (root.getRight() != null) {
                result.add(root.getRight());
            }
            return true;
        }

        return deleteLeft && deleteRight && deleteSet.contains(root.getData());

    }

    private TreeNode<Integer> deleteTreeNodesAndReturnForest_Helper2(TreeNode<Integer> root,
            Set<Integer> deleteSet, List<TreeNode<Integer>> result) {

        //Easier explanation
        if (root == null) {
            return null;
        }

        root.setLeft(deleteTreeNodesAndReturnForest_Helper2(root.getLeft(), deleteSet, result));
        root.setRight(deleteTreeNodesAndReturnForest_Helper2(root.getRight(), deleteSet, result));

        if (deleteSet.contains(root.getData())) {

            if (root.getLeft() != null) {
                result.add(root.getLeft());
            }

            if (root.getRight() != null) {
                result.add(root.getRight());
            }

            return null;
        }

        return root;
    }

    public void deleteTreeNodesAndReturnForest(TreeNode<Integer> root, int[] toDelete) {

        List<TreeNode<Integer>> result = new ArrayList<>();

        if (root == null) {
            return;
        }

        Set<Integer> deleteSet = new HashSet<>();
        for (int x : toDelete) {
            deleteSet.add(x);
        }

//        boolean res = deleteTreeNodesAndReturnForest_Helper(root, deleteSet, result);
//
//        if (res == false || (res && !deleteSet.contains(root.getData()))) {
//            result.add(root);
//        }
        //Easier explanation
        //if curr root is not in delete set then root node is also a forest
        if (!deleteSet.contains(root.getData())) {
            result.add(root);
        }
        root = deleteTreeNodesAndReturnForest_Helper2(root, deleteSet, result);

        //output:
        for (TreeNode<Integer> curr : result) {
            levelOrderTraversal_Iterative(curr);
            System.out.println();
        }
    }

    private TreeNode<Integer> constructBinaryTreeFromInorderPreorderArray_Helper(
            int preIndex, int inStart, int inEnd, Map<Integer, Integer> inorderMap, int[] preorder) {

        if (preIndex >= preorder.length || inStart > inEnd) {
            return null;
        }

        TreeNode<Integer> root = new TreeNode<>(preorder[preIndex]);

        int index = inorderMap.get(preorder[preIndex]);

        root.setLeft(constructBinaryTreeFromInorderPreorderArray_Helper(
                preIndex + 1, inStart, index - 1, inorderMap, preorder));
        root.setRight(constructBinaryTreeFromInorderPreorderArray_Helper(
                preIndex + 1 + (index - inStart), index + 1, inEnd, inorderMap, preorder));

        return root;
    }

    public void constructBinaryTreeFromInorderPreorderArray(int[] inorder, int[] preorder) {

        System.out.println("Inorder & Preorder");
        if (inorder.length != preorder.length) {
            return;
        }

        int n = inorder.length;

        Map<Integer, Integer> inorderMap = new HashMap<>();
        for (int i = 0; i < n; i++) {
            inorderMap.put(inorder[i], i);
        }

        TreeNode<Integer> root = constructBinaryTreeFromInorderPreorderArray_Helper(
                0, 0, n - 1, inorderMap, preorder);

        //output
        new BinaryTree<>(root).treeBFS();
        System.out.println();
    }

    private TreeNode<Integer> constructBinaryTreeFromInorderPostorderArray_Helper(
            int[] postorder, int postIndex, int inStart, int inEnd,
            Map<Integer, Integer> inorderMap) {
        if (inStart > inEnd || postIndex < 0 || postIndex > postorder.length) {
            return null;
        }

        TreeNode<Integer> root = new TreeNode<>(postorder[postIndex]);
        int index = inorderMap.get(postorder[postIndex]);

        root.setLeft(constructBinaryTreeFromInorderPostorderArray_Helper(
                postorder,
                // current root minus what's on the right side on the inorder array minus 1
                postIndex - (inEnd - index) - 1,
                inStart,
                index - 1,
                inorderMap
        ));

        root.setRight(constructBinaryTreeFromInorderPostorderArray_Helper(
                postorder,
                // the next right side node will be the next one (backwards) on the postorder list
                postIndex - 1,
                index + 1,
                inEnd,
                inorderMap
        ));

        return root;
    }

    public void constructBinaryTreeFromInorderPostorderArray(int[] inorder, int[] postorder) {

        System.out.println("Inorder & Postorder");
        if (inorder.length != postorder.length) {
            return;
        }

        int n = inorder.length;
        Map<Integer, Integer> inorderMap = new HashMap<>();
        for (int i = 0; i < n; i++) {
            inorderMap.put(inorder[i], i);
        }

        TreeNode<Integer> root = constructBinaryTreeFromInorderPostorderArray_Helper(
                postorder, n - 1, 0, n - 1, inorderMap);

        //output
        new BinaryTree<>(root).treeBFS();
        System.out.println();
    }

    private TreeNode<Integer> constructBinarySearchTreeFromPreorderArray_Helper(int[] preorder, int preStart, int preEnd) {

        if (preStart > preEnd) {
            return null;
        }

        TreeNode<Integer> root = new TreeNode<>(preorder[preStart]);

        int index = preStart + 1;
        while (index <= preEnd && preorder[index] < preorder[preStart]) {
            index++;
        }

        root.setLeft(constructBinarySearchTreeFromPreorderArray_Helper(preorder, preStart + 1, index - 1));
        root.setRight(constructBinarySearchTreeFromPreorderArray_Helper(preorder, index, preEnd));

        return root;
    }

    public void constructBinarySearchTreeFromPreorderArray(int[] preorder) {

        int n = preorder.length;
        TreeNode<Integer> root = constructBinarySearchTreeFromPreorderArray_Helper(preorder, 0, n - 1);

        //output:
        new BinaryTree<Integer>(root).treeInorder();
        System.out.println();
    }

    private TreeNode<Integer> constructBinarySearchTreeFromPostorderArray_Helper(
            int[] postorder, int postStart, int postEnd) {

        if (postEnd > postStart) {
            return null;
        }

        TreeNode<Integer> root = new TreeNode<>(postorder[postStart]);

        int index = postStart - 1;
        while (index >= postEnd && postorder[index] > postorder[postStart]) {
            index--;
        }

        root.setLeft(constructBinarySearchTreeFromPostorderArray_Helper(postorder, index, postEnd));
        root.setRight(constructBinarySearchTreeFromPostorderArray_Helper(postorder, postStart - 1, index + 1));

        return root;
    }

    public void constructBinarySearchTreeFromPostorderArray(int[] postorder) {

        int n = postorder.length;
        TreeNode<Integer> root = constructBinarySearchTreeFromPostorderArray_Helper(postorder, n - 1, 0);

        //output:
        new BinaryTree<Integer>(root).treeInorder();
        System.out.println();
    }

    private TreeNode<Integer> leavesOfTreeToDoublyLinkedListAndRemoveLeaves_DLL;
    private TreeNode<Integer> leavesOfTreeToDoublyLinkedListAndRemoveLeaves_Prev;

    private TreeNode<Integer> leavesOfTreeToDoublyLinkedListAndRemoveLeaves_Helper(TreeNode<Integer> root) {

        if (root == null) {
            return null;
        }

        TreeNode<Integer> leftChild = leavesOfTreeToDoublyLinkedListAndRemoveLeaves_Helper(root.getLeft());
        TreeNode<Integer> rightChild = leavesOfTreeToDoublyLinkedListAndRemoveLeaves_Helper(root.getRight());

        //isLeaf condition
        if (root.getLeft() == null && root.getRight() == null) {

            //left most leaf node
            if (leavesOfTreeToDoublyLinkedListAndRemoveLeaves_DLL == null) {
                leavesOfTreeToDoublyLinkedListAndRemoveLeaves_DLL = root;
            }

            if (leavesOfTreeToDoublyLinkedListAndRemoveLeaves_Prev != null) {
                leavesOfTreeToDoublyLinkedListAndRemoveLeaves_Prev.setRight(root);
                root.setLeft(leavesOfTreeToDoublyLinkedListAndRemoveLeaves_Prev);
            }

            leavesOfTreeToDoublyLinkedListAndRemoveLeaves_Prev = root;

            return null; //to delete leaf, return null to its parent
        }

        //setting current root's left and right child below the isLeaf condition
        //because if we add these child before that condition, current root can become
        //a leaf node itself just before the isLeaf check and in that condition
        //that root will be considered as leaf
        /*
         treeBFS = 2,3,4; 
         3 & 4 are leaf; 
         inside isLeaf cond it will return null to its parent (i.e, 2);
         now if 2.left = null & 2.right = null is before isLeaf the 2 will be
         a leaf itself, which should never be the checked;
         thats why root.left = leftChild & root.right = rightChild should be below
         isLeaf cond to save the parent/intermediate node;
         */
        root.setLeft(leftChild);
        root.setRight(rightChild);

        return root;
    }

    public void leavesOfTreeToDoublyLinkedListAndRemoveLeaves(TreeNode<Integer> root) {

        //actual
        System.out.println("Actual tree");
        new BinaryTree<Integer>(root).treeBFS();
        System.out.println();

        leavesOfTreeToDoublyLinkedListAndRemoveLeaves_DLL = null;
        leavesOfTreeToDoublyLinkedListAndRemoveLeaves_Prev = null;

        //in case if root node is single node,that must be head of DLL and should be deleted(set to null)
        //otherwise it will just return root itselt
        root = leavesOfTreeToDoublyLinkedListAndRemoveLeaves_Helper(root);

        //output;
        System.out.println("Leaf of tree are now DLL");
        treeToDoublyLinkedList_Print(leavesOfTreeToDoublyLinkedListAndRemoveLeaves_DLL);
        System.out.println("Tree after deleting leaf");
        new BinaryTree<Integer>(root).treeBFS();
        System.out.println();
    }

    private void checkIfTwoNAryTreeAreMirror_PushStack(List<List<Integer>> tree1, Stack<Integer> stack, int root) {

        if (tree1.size() == 0) {
            return;
        }

        stack.push(root);
        List<Integer> childs = tree1.get(root);
        for (int childNodes : childs) {
            checkIfTwoNAryTreeAreMirror_PushStack(tree1, stack, childNodes);
        }
    }

    private void checkIfTwoNAryTreeAreMirror_PushQueue(List<List<Integer>> tree2, Queue<Integer> queue, int root) {

        if (tree2.size() == 0) {
            return;
        }

        List<Integer> childs = tree2.get(root);
        for (int childNodes : childs) {
            checkIfTwoNAryTreeAreMirror_PushQueue(tree2, queue, childNodes);
        }
        queue.add(root);
    }

    public boolean checkIfTwoNAryTreeAreMirror(List<List<Integer>> tree1, List<List<Integer>> tree2) {

        //Explanation: https://youtu.be/UGzXSDZv-SY
        Stack<Integer> stack = new Stack<>();
        checkIfTwoNAryTreeAreMirror_PushStack(tree1, stack, 0);

        Queue<Integer> queue = new LinkedList<>();
        checkIfTwoNAryTreeAreMirror_PushQueue(tree2, queue, 0);

//        System.out.println(stack+"---"+queue);
        if (stack.size() != queue.size()) {
            System.out.println("Not a mirror images");
            return false;
        }

        while (!stack.isEmpty() && !queue.isEmpty()) {
            if (stack.pop() != queue.poll()) {
                return false;
            }
        }
        return true;
    }

    public void deepestLeavesSumOfTree_Iterative(TreeNode<Integer> root) {

//        base edge case
//        if(root == null){
//            return 0;
//        }
        //do level order traversal
        Queue<TreeNode<Integer>> q = new LinkedList<>();
        q.add(root);

        int sum = 0;
        while (!q.isEmpty()) {

            sum = 0;

            //size of queue at a particular level is required beforehand
            //because we putting the child of curr node in the queue
            //we don't want those nodes to be consumed in the for() loop
            int size = q.size();
            for (int i = 0; i < size; i++) {

                //add up all the nodes at a particular level in the sum variable
                //and at the same time add the chiild nodes of curr node into queue
                TreeNode<Integer> curr = q.poll();

                sum += curr.getData();
                if (curr.getLeft() != null) {
                    q.add(curr.getLeft());
                }

                if (curr.getRight() != null) {
                    q.add(curr.getRight());
                }
            }
        }

        //output:
        System.out.println("Deepest leaves sum: " + sum);
    }

    private void smallestStringInTreeFromLeafToRoot_Helper(TreeNode<Integer> root, StringBuilder sb,
            PriorityQueue<String> strs) {

        if (root == null) {
            return;
        }

        sb.insert(0, (char) (root.getData() + 'a'));

        smallestStringInTreeFromLeafToRoot_Helper(root.getLeft(), sb, strs);
        smallestStringInTreeFromLeafToRoot_Helper(root.getRight(), sb, strs);

        if (root.getLeft() == null && root.getRight() == null) {
            strs.add(sb.toString());
            //we only need 1 string as answer
            if (strs.size() > 1) {
                strs.poll();
            }
        }

        sb.deleteCharAt(0);
    }

    public void smallestStringInTreeFromLeafToRoot(TreeNode<Integer> root) {

        //each nodes contains value 0 - 25 representing alphabets a - z
        //find the lexicographically smallest string (leaf to root) in the tree
        StringBuilder sb = new StringBuilder();
        //making maxHeap to store smallest string in the last of heap
        PriorityQueue<String> strs = new PriorityQueue<>((a, b) -> b.compareTo(a));
        smallestStringInTreeFromLeafToRoot_Helper(root, sb, strs);

        //output
        System.out.println("Smallest string: " + strs.peek());
    }

    private void printSumWhereRootToLeafPathIsANumber_createNumberHelper(TreeNode<Integer> root, int num, List<Integer> nums) {

        if (root == null) {
            return;
        }

        printSumWhereRootToLeafPathIsANumber_createNumberHelper(root.getLeft(), num * 10 + root.getData(), nums);
        printSumWhereRootToLeafPathIsANumber_createNumberHelper(root.getRight(), num * 10 + root.getData(), nums);

        if (root.getLeft() == null && root.getRight() == null) {
            nums.add(num * 10 + root.getData());
        }
    }

    public void printSumWhereRootToLeafPathIsANumber(TreeNode<Integer> root) {

        //ex: 
        /* root:
         1
         2       3
         */
        //num1 = 12 (as 1->2 is a path)
        //num2 = 13 (as 1->3 is a path)
        //each path from root to leaf is a separate num
        //print sum of all such number
        List<Integer> nums = new ArrayList<>();
        printSumWhereRootToLeafPathIsANumber_createNumberHelper(root, 0, nums);

        int sum = 0;
        for (int num : nums) {
            sum += num;
        }

        //output:
        System.out.println("Sum of path of tree representing as a number: " + sum);
    }

    private void fixTwoSwappedNodesInBST_Helper_IsBST(TreeNode<Integer> root) {
        if (root == null) {
            return;
        }

        fixTwoSwappedNodesInBST_Helper_IsBST(root.getLeft());

        if (fixTwoSwappedNodesInBST_Prev != null && fixTwoSwappedNodesInBST_Prev.getData() > root.getData()) {

            if (fixTwoSwappedNodesInBST_First == null) {
                fixTwoSwappedNodesInBST_First = fixTwoSwappedNodesInBST_Prev;
                fixTwoSwappedNodesInBST_Middle = root;
            } else {
                fixTwoSwappedNodesInBST_Last = root;
            }
        }

        fixTwoSwappedNodesInBST_Prev = root;

        fixTwoSwappedNodesInBST_Helper_IsBST(root.getRight());
    }

    private void fixTwoSwappedNodesInBST_Helper(TreeNode<Integer> root) {

        fixTwoSwappedNodesInBST_First = null;
        fixTwoSwappedNodesInBST_Middle = null;
        fixTwoSwappedNodesInBST_Last = null;
        fixTwoSwappedNodesInBST_Prev = null;

        fixTwoSwappedNodesInBST_Helper_IsBST(root);

        if (fixTwoSwappedNodesInBST_First != null && fixTwoSwappedNodesInBST_Last != null) {
            int temp = fixTwoSwappedNodesInBST_First.getData();
            fixTwoSwappedNodesInBST_First.setData(fixTwoSwappedNodesInBST_Last.getData());
            fixTwoSwappedNodesInBST_Last.setData(temp);
        } else if (fixTwoSwappedNodesInBST_First != null && fixTwoSwappedNodesInBST_Middle != null) {
            int temp = fixTwoSwappedNodesInBST_First.getData();
            fixTwoSwappedNodesInBST_First.setData(fixTwoSwappedNodesInBST_Middle.getData());
            fixTwoSwappedNodesInBST_Middle.setData(temp);
        }
    }

    TreeNode<Integer> fixTwoSwappedNodesInBST_First;
    TreeNode<Integer> fixTwoSwappedNodesInBST_Middle;
    TreeNode<Integer> fixTwoSwappedNodesInBST_Last;
    TreeNode<Integer> fixTwoSwappedNodesInBST_Prev;

    public void fixTwoSwappedNodesInBST(TreeNode<Integer> root) {

        //https://www.geeksforgeeks.org/fix-two-swapped-nodes-of-bst/
        //actual
        new BinaryTree<Integer>(root).treeInorder();
        System.out.println();

        fixTwoSwappedNodesInBST_Helper(root);

        //output
        new BinaryTree<Integer>(root).treeInorder();
        System.out.println();
    }

    private TreeNode<Integer> mergeTwoBinaryTree_Heleper(TreeNode<Integer> root1, TreeNode<Integer> root2) {

        if (root1 == null) {
            return root2;
        }

        if (root2 == null) {
            return root1;
        }

        root1.setData(root1.getData() + root2.getData());

        root1.setLeft(mergeTwoBinaryTree_Heleper(root1.getLeft(), root2.getLeft()));
        root1.setRight(mergeTwoBinaryTree_Heleper(root1.getRight(), root2.getRight()));

        return root1;
    }

    public void mergeTwoBinaryTree(TreeNode<Integer> root1, TreeNode<Integer> root2) {
        //actual
        System.out.println("Actual trees");
        new BinaryTree<Integer>(root1).treeBFS();
        System.out.println();
        new BinaryTree<Integer>(root2).treeBFS();
        System.out.println();

        mergeTwoBinaryTree_Heleper(root1, root2);

        //output
        System.out.println("Merged both trees in tree1");
        new BinaryTree<Integer>(root1).treeBFS();
    }

    private long numberOfWaysToCreateBSTAndBTWithGivenN_Factorial(long N) {
        long result = 1;
        for (long i = 1; i <= N; i++) {
            result *= i;
        }

        return result;
    }

    private long numberOfWaysToCreateBSTAndBTWithGivenN_BinomialCoeff(long N, long K) {
        long result = 1;

        if (K > N - K) {
            K = N - K;
        }

        for (long i = 0; i < K; i++) {
            result *= (N - i);
            result /= (i + 1);
        }

        return result;
    }

    private long numberOfWaysToCreateBSTAndBTWithGivenN_CatalanNumberOfGivenNthNumber(long N) {

        //Catalan number series:
        //https://www.youtube.com/watch?v=CMaZ69P1bAc
        //https://www.geeksforgeeks.org/program-nth-catalan-number/
        //https://www.geeksforgeeks.org/total-number-of-possible-binary-search-trees-with-n-keys/#
        long cat = numberOfWaysToCreateBSTAndBTWithGivenN_BinomialCoeff(2 * N, N);
        return cat / (N + 1);
    }

    public void numberOfWaysToCreateBSTAndBTWithGivenN(long N) {

        //problem: https://leetcode.com/problems/unique-binary-search-trees
        //Explanation :
        //https://www.geeksforgeeks.org/total-number-of-possible-binary-search-trees-with-n-keys/#
        //ways to create BST
        //find the catalan number of given Nth number 
        System.out.println("Number of ways to create a binary search tree with given N nodes: "
                + numberOfWaysToCreateBSTAndBTWithGivenN_CatalanNumberOfGivenNthNumber(N));

        //ways to create BT
        //find the catalan number of given Nth number * factorial(N)
        System.out.println("Number of ways to create a binary tree with given N nodes: "
                + (numberOfWaysToCreateBSTAndBTWithGivenN_CatalanNumberOfGivenNthNumber(N)
                * numberOfWaysToCreateBSTAndBTWithGivenN_Factorial(N)));
    }

    public boolean checkIfBinaryTreeIsCompleteOrNot(TreeNode<Integer> root) {

        //explanation: https://youtu.be/j16cwbLEf9w
        //complete binary tree:
        /*
         In a complete binary tree, every level, except possibly the last, 
         is completely filled, and all nodes in the last level are as far left 
         as possible. It can have between 1 and 2^h nodes inclusive at the 
         last level h.
         */
        //all the nodes at last level should be left-most alinged
        //if any null is present before the very last node at the level
        //that means tree is not complete binary tree
        boolean isNullBeforeLastNode = false;

        Queue<TreeNode<Integer>> q = new LinkedList<>();
        q.add(root);

        while (!q.isEmpty()) {

            TreeNode<Integer> curr = q.poll();

            if (curr == null) {
                isNullBeforeLastNode = true;
            } else {
                if (isNullBeforeLastNode) {
                    return false;
                }
                //put all the left and right child nodes as it is
                //without checking for null
                q.add(curr.getLeft());
                q.add(curr.getRight());
            }
        }
        return true;
    }

    private void maximumWidthOfBinaryTree_Helper(TreeNode<Integer> root, int level,
            int position, Map<Integer, Integer> map) {
        if (root == null) {
            return;
        }

        map.putIfAbsent(level, position);
        maximumWidthOfBinaryTree_MaxWidth = Math.max(maximumWidthOfBinaryTree_MaxWidth,
                position - map.get(level) + 1);

        maximumWidthOfBinaryTree_Helper(root.getLeft(), level + 1, 2 * position, map);
        maximumWidthOfBinaryTree_Helper(root.getRight(), level + 1, 2 * position + 1, map);
    }

    int maximumWidthOfBinaryTree_MaxWidth;

    public void maximumWidthOfBinaryTree(TreeNode<Integer> root) {

        //explanation: https://youtu.be/sm4UdCO2868
        maximumWidthOfBinaryTree_MaxWidth = 0;
        Map<Integer, Integer> map = new HashMap<>();
        maximumWidthOfBinaryTree_Helper(root, 0, 0, map);

        //output
        System.out.println("Max Width of binary tree: " + maximumWidthOfBinaryTree_MaxWidth);
    }

    private int distributeCoinsInBinaryTree_Helper(TreeNode<Integer> root) {
        if (root == null) {
            return 0;
        }

        int leftDistributionMove = distributeCoinsInBinaryTree_Helper(root.getLeft());
        int rightDistributionMove = distributeCoinsInBinaryTree_Helper(root.getRight());

        distributeCoinsInBinaryTree_Moves += (Math.abs(leftDistributionMove) + Math.abs(rightDistributionMove));

        return root.getData() + leftDistributionMove + rightDistributionMove - 1;
    }

    int distributeCoinsInBinaryTree_Moves;

    public void distributeCoinsInBinaryTree(TreeNode<Integer> root) {

        //https://leetcode.com/problems/distribute-coins-in-binary-tree/
        //explanation: https://youtu.be/MfXxic8IhkI
        //coins = root.data, distribution rule: parent to child OR child to parent
        distributeCoinsInBinaryTree_Moves = 0;
        distributeCoinsInBinaryTree_Helper(root);

        //output:
        System.out.println("Number of moves to distribute coins equally in whole tree: " + distributeCoinsInBinaryTree_Moves);
    }

    public boolean checkIfBinaryTreeIsOddEvenTree(TreeNode<Integer> root) {

        //https://leetcode.com/problems/even-odd-tree
        int level = 0;
        TreeNode<Integer> prev = null;
        Queue<TreeNode<Integer>> q = new LinkedList<>();
        q.add(root);

        while (!q.isEmpty()) {

            prev = null; //reset
            int size = q.size();

            for (int i = 0; i < size; i++) {
                TreeNode<Integer> curr = q.poll();
                if (level % 2 == 0) {
                    //level = EVEN then nodes at this level should have 
                    //all odd data and nodes at this level should be in strictly incr order
                    if (curr.getData() % 2 == 0 || (prev != null && prev.getData() >= curr.getData())) {
                        return false;
                    }
                } else {
                    //level = ODD then nodes at this level should have 
                    //all even data and nodes at this level should be in strictly decr order
                    if (curr.getData() % 2 == 1 || (prev != null && prev.getData() <= curr.getData())) {
                        return false;
                    }
                }

                prev = curr;

                if (curr.getLeft() != null) {
                    q.add(curr.getLeft());
                }

                if (curr.getRight() != null) {
                    q.add(curr.getRight());
                }
            }

            level++;
        }
        return true;
    }

    private int maxSumInAnyPathOfTree_Helper(TreeNode<Integer> root) {

        if (root == null) {
            return 0;
        }

        int leftSum = Math.max(0, maxSumInAnyPathOfTree_Helper(root.getLeft()));
        int rightSum = Math.max(0, maxSumInAnyPathOfTree_Helper(root.getRight()));

        maxSumInAnyPathOfTree_MaxPathSum = Math.max(maxSumInAnyPathOfTree_MaxPathSum,
                leftSum + rightSum + root.getData());

        return Math.max(leftSum, rightSum) + root.getData();
    }

    int maxSumInAnyPathOfTree_MaxPathSum;

    public void maxSumInAnyPathOfTree(TreeNode<Integer> root) {

        //SIMILAR to diameter of tree approach
        //https://leetcode.com/problems/binary-tree-maximum-path-sum/
        //explanation: https://youtu.be/mOdetMWwtoI
        maxSumInAnyPathOfTree_MaxPathSum = Integer.MIN_VALUE;
        maxSumInAnyPathOfTree_Helper(root);

        //output
        System.out.println("Max path sum in any path of tree: " + maxSumInAnyPathOfTree_MaxPathSum);
    }

    private int longestEdgeLengthBetweenTreeNodesWithSameValue_Helper(TreeNode<Integer> root) {
        if (root == null) {
            return 0;
        }

        int left = longestEdgeLengthBetweenTreeNodesWithSameValue_Helper(root.getLeft());
        int right = longestEdgeLengthBetweenTreeNodesWithSameValue_Helper(root.getRight());

        int leftEdge = 0;
        if (root.getLeft() != null && root.getLeft().getData() == root.getData()) {
            leftEdge += left + 1;
        }

        int rightEdge = 0;
        if (root.getRight() != null && root.getRight().getData() == root.getData()) {
            rightEdge += right + 1;
        }

        longestEdgeLengthBetweenTreeNodesWithSameValue_LongestEdge = Math.max(longestEdgeLengthBetweenTreeNodesWithSameValue_LongestEdge,
                leftEdge + rightEdge);

        return Math.max(leftEdge, rightEdge);
    }

    int longestEdgeLengthBetweenTreeNodesWithSameValue_LongestEdge;

    public void longestEdgeLengthBetweenTreeNodesWithSameValue(TreeNode<Integer> root) {

        //https://leetcode.com/problems/longest-univalue-path/
        longestEdgeLengthBetweenTreeNodesWithSameValue_LongestEdge = 0;
        longestEdgeLengthBetweenTreeNodesWithSameValue_Helper(root);

        //output
        System.out.println("Longest edge between the nodes having same values: " + longestEdgeLengthBetweenTreeNodesWithSameValue_LongestEdge);
    }

    private int countNodesInCompleteBinaryTree_LeftTreeHeight(TreeNode<Integer> root) {
        int height = 0;
        while (root != null) {
            height++;
            root = root.getLeft();
        }
        return height;
    }

    private int countNodesInCompleteBinaryTree_RightTreeHeight(TreeNode<Integer> root) {
        int height = 0;
        while (root != null) {
            height++;
            root = root.getRight();
        }
        return height;
    }

    public int countNodesInCompleteBinaryTree(TreeNode<Integer> root) {

        //brute force do levelorder and count++ for all the nodes T: O(N)
        //OPTIMISED
        //https://leetcode.com/problems/count-complete-tree-nodes/
        if (root == null) {
            return 0;
        }

        int leftSubtreeHeight = countNodesInCompleteBinaryTree_LeftTreeHeight(root);
        int rightSubtreeHeight = countNodesInCompleteBinaryTree_RightTreeHeight(root);

        if (leftSubtreeHeight == rightSubtreeHeight) {
            //number of node in complete binary tree = (2^H) - 1
            return (int) Math.pow(2, leftSubtreeHeight) - 1;
        }

        return countNodesInCompleteBinaryTree(root.getLeft())
                + countNodesInCompleteBinaryTree(root.getRight()) + 1;
    }

    public int countGoodNodesInBinaryTree_1(TreeNode<Integer> root, int max) {

        //https://leetcode.com/problems/count-good-nodes-in-binary-tree/
        /*
         Given a binary tree root, a node X in the tree is named good if in the 
         path from root to X there are no nodes with a value greater than X.
         */
        if (root == null) {
            return 0;
        }

        int left = countGoodNodesInBinaryTree_1(root.getLeft(), Math.max(max, root.getData()));
        int right = countGoodNodesInBinaryTree_1(root.getRight(), Math.max(max, root.getData()));

        if (root.getData() < max) {
            //curr node is not a goodNode but it can have 
            //max (curr goodNode(i.e 0), left sub tree gootNodes + right sub tree gooNodes)
            return Math.max(0, left + right);
        }

        //left sub tree goodNodes + right sub tree goodNodes + curr root goodNode(i.e 1) 
        return left + right + 1;
    }

    private void countGoodNodesInBinaryTree_2_Helper(TreeNode<Integer> root, int max) {

        if (root == null) {
            return;
        }

        countGoodNodesInBinaryTree_2_Helper(root.getLeft(), Math.max(max, root.getData()));
        countGoodNodesInBinaryTree_2_Helper(root.getRight(), Math.max(max, root.getData()));

        if (root.getData() >= max) {
            countGoodNodesInBinaryTree_2_GoodNodes++;
        }
    }

    int countGoodNodesInBinaryTree_2_GoodNodes = 0;

    public void countGoodNodesInBinaryTree_2(TreeNode<Integer> root) {

        //https://leetcode.com/problems/count-good-nodes-in-binary-tree/
        /*
         Given a binary tree root, a node X in the tree is named good if in the 
         path from root to X there are no nodes with a value greater than X.
         */
        //EASIER APPROACH
        countGoodNodesInBinaryTree_2_GoodNodes = 0;
        countGoodNodesInBinaryTree_2_Helper(root, Integer.MIN_VALUE);

        //output
        System.out.println("Good nodes counts in tree approach 2: " + countGoodNodesInBinaryTree_2_GoodNodes);
    }

    private int minDistanceBetweenGivenTwoNodesInBinaryTree_FindLevelFromLCA(TreeNode<Integer> root, int N, int level) {

        if (root == null) {
            return -1;
        }

        Queue<TreeNode<Integer>> q = new LinkedList<>();
        q.add(root);

        while (!q.isEmpty()) {

            int size = q.size();
            for (int i = 0; i < size; i++) {
                TreeNode<Integer> curr = q.poll();

                if (curr.getData() == N) {
                    return level;
                }

                if (curr.getLeft() != null) {
                    q.add(curr.getLeft());
                }

                if (curr.getRight() != null) {
                    q.add(curr.getRight());
                }
            }
            level++;
        }
        return -1;
    }

    public void minDistanceBetweenGivenTwoNodesInBinaryTree(TreeNode<Integer> root, int N1, int N2) {

        //https://www.geeksforgeeks.org/find-distance-between-two-nodes-of-a-binary-tree/
        TreeNode<Integer> lca = lowestCommonAncestorOfTree_Helper(root, N1, N2);

        int levelN1 = minDistanceBetweenGivenTwoNodesInBinaryTree_FindLevelFromLCA(lca, N1, 0);
        int levelN2 = minDistanceBetweenGivenTwoNodesInBinaryTree_FindLevelFromLCA(lca, N2, 0);

        System.out.println("Min dstance between two given nodes: " + (levelN1 + levelN2));
    }

    private void maxProductIfBinaryTreeIsSplitIntoTwo_FindTreeSum(TreeNode<Integer> root) {
        if (root == null) {
            return;
        }

        maxProductIfBinaryTreeIsSplitIntoTwo_TotalTreeSum += root.getData();

        maxProductIfBinaryTreeIsSplitIntoTwo_FindTreeSum(root.getLeft());
        maxProductIfBinaryTreeIsSplitIntoTwo_FindTreeSum(root.getRight());
    }

    private int maxProductIfBinaryTreeIsSplitIntoTwo_FindMaxProduct(TreeNode<Integer> root) {

        if (root == null) {
            return 0;
        }

        int leftSubTreeSum = maxProductIfBinaryTreeIsSplitIntoTwo_FindMaxProduct(root.getLeft());
        int rightSubTreeSum = maxProductIfBinaryTreeIsSplitIntoTwo_FindMaxProduct(root.getRight());

        int leftSplitProduct = (maxProductIfBinaryTreeIsSplitIntoTwo_TotalTreeSum - leftSubTreeSum) * leftSubTreeSum;
        int rightSplitProduct = (maxProductIfBinaryTreeIsSplitIntoTwo_TotalTreeSum - rightSubTreeSum) * rightSubTreeSum;

        maxProductIfBinaryTreeIsSplitIntoTwo_Product = Math.max(maxProductIfBinaryTreeIsSplitIntoTwo_Product,
                Math.max(leftSplitProduct, rightSplitProduct));

        return leftSubTreeSum + rightSubTreeSum + root.getData(); //sub-tree sum from bottom
    }

    int maxProductIfBinaryTreeIsSplitIntoTwo_TotalTreeSum = 0;
    int maxProductIfBinaryTreeIsSplitIntoTwo_Product = 0;

    public void maxProductIfBinaryTreeIsSplitIntoTwo(TreeNode<Integer> root) {

        //https://leetcode.com/problems/maximum-product-of-splitted-binary-tree/
        /*
        
         treeBFS = [1,2,3,4,5,6,NULL]
         split tree as t1 = [2,4,5], t2 = [1,3,6] link-break-between 1.left = 2 break this
         product = sum{t1} * sum(t2) should be maximum
         trace all such break-point where, product of two subtrees max (prod1, prod2)
         */
        maxProductIfBinaryTreeIsSplitIntoTwo_TotalTreeSum = 0;
        maxProductIfBinaryTreeIsSplitIntoTwo_Product = 0;

        //find the total sum of all tree nodes
        maxProductIfBinaryTreeIsSplitIntoTwo_FindTreeSum(root);

        //find max product by imitating splits
        maxProductIfBinaryTreeIsSplitIntoTwo_FindMaxProduct(root);

        //output
        System.out.println("Max product of sum of trees by splitting into two: "
                + maxProductIfBinaryTreeIsSplitIntoTwo_Product);

    }

    private int maximumDifferenceBetweenNodeAndItsAncestor_Helper(TreeNode<Integer> root) {

        if (root == null) {
            return Integer.MAX_VALUE;
        }

        int left = maximumDifferenceBetweenNodeAndItsAncestor_Helper(root.getLeft());
        int right = maximumDifferenceBetweenNodeAndItsAncestor_Helper(root.getRight());

        int minVal = Math.min(left, right);

        maximumDifferenceBetweenNodeAndItsAncestor_MaxDiff = Math.max(maximumDifferenceBetweenNodeAndItsAncestor_MaxDiff,
                root.getData() - minVal);

        return Math.min(minVal, root.getData());
    }

    int maximumDifferenceBetweenNodeAndItsAncestor_MaxDiff;

    public void maximumDifferenceBetweenNodeAndItsAncestor(TreeNode<Integer> root) {
        maximumDifferenceBetweenNodeAndItsAncestor_MaxDiff = Integer.MIN_VALUE;
        maximumDifferenceBetweenNodeAndItsAncestor_Helper(root);
        //output
        System.out.println("Max difference between a node and its ancestor: "
                + maximumDifferenceBetweenNodeAndItsAncestor_MaxDiff);
    }

    private List<TreeNode<Integer>> deleteTreeNodeFromBinarySearchTree_FindInorderSuccessor(
            TreeNode<Integer> root) {

        if (root == null) {
            return Collections.emptyList();
        }

        TreeNode<Integer> succ = null;
        TreeNode<Integer> succPrev = null;
        if (root.getRight() != null) {
            succPrev = root;
            succ = root.getRight(); //succ is left most node in the right sub tree
            while (succ.getLeft() != null) {
                succPrev = succ;
                succ = succ.getLeft();
            }
        }
        return Arrays.asList(succPrev, succ);
    }

    private TreeNode<Integer> deleteTreeNodeFromBinarySearchTree_Delete(TreeNode<Integer> rootToDelete) {

        //https://www.geeksforgeeks.org/binary-search-tree-set-2-delete/
        if (rootToDelete == null) {
            return null;
        }

        if (rootToDelete.getLeft() == null && rootToDelete.getRight() == null) { // node is leaf
            return null;
        } else if (rootToDelete.getLeft() == null) { //node has one child
            return rootToDelete.getRight();
        } else if (rootToDelete.getRight() == null) { //node has one child
            return rootToDelete.getLeft();
        } else { //node has two child
            List<TreeNode<Integer>> succList = deleteTreeNodeFromBinarySearchTree_FindInorderSuccessor(rootToDelete);

            if (!succList.isEmpty()) {
                TreeNode<Integer> succPrev = succList.get(0);
                TreeNode<Integer> succ = succList.get(1);
                //replace root's data with inorder successor (succ)
                rootToDelete.setData(succ.getData());
                //delete the inorder successor node from its actual place
                if (succPrev != rootToDelete) {
                    succPrev.setLeft(deleteTreeNodeFromBinarySearchTree_Delete(succ));
                } else {
                    succPrev.setRight(deleteTreeNodeFromBinarySearchTree_Delete(succ));
                }
            }

            return rootToDelete;
        }
    }

    public TreeNode<Integer> deleteTreeNodeFromBinarySearchTree(TreeNode<Integer> root, int findToDelete) {

        //https://www.geeksforgeeks.org/binary-search-tree-set-2-delete/
        if (root == null) {
            return null;
        }

        if (root.getData() == findToDelete) {
            return deleteTreeNodeFromBinarySearchTree_Delete(root);
        } else if (findToDelete < root.getData()) {
            root.setLeft(deleteTreeNodeFromBinarySearchTree(root.getLeft(), findToDelete));
        } else {
            root.setRight(deleteTreeNodeFromBinarySearchTree(root.getRight(), findToDelete));
        }
        return root;
    }

    private TreeNode<Integer> deleteTreeNodeFromBinarySearchTreeNotInRange_Helper(TreeNode<Integer> root,
            int low, int high) {

        if (root == null) {
            return null;
        }

        root.setLeft(deleteTreeNodeFromBinarySearchTreeNotInRange_Helper(root.getLeft(),
                low, high));
        root.setRight(deleteTreeNodeFromBinarySearchTreeNotInRange_Helper(root.getRight(),
                low, high));

        //not in range
        if (!(root.getData() >= low && root.getData() <= high)) {
            return deleteTreeNodeFromBinarySearchTree_Delete(root);
        }

        return root;
    }

    public void deleteTreeNodeFromBinarySearchTreeNotInRange(TreeNode<Integer> root,
            int low, int high) {
        TreeNode<Integer> node = deleteTreeNodeFromBinarySearchTreeNotInRange_Helper(root, low, high);
        //output
        new BinaryTree<Integer>(root).treeBFS();
        System.out.println();
    }

    public void checkIfTwoTreeNodesAreCousin(TreeNode<Integer> root, int x, int y) {

        //https://leetcode.com/problems/cousins-in-binary-tree/
        /*
         Two nodes are cousins when they both lie on same level (levelX == levelY) but also 
         their parent should not be same (parentX != parentY)
         */
        int levelX = -1;
        int levelY = -1;

        TreeNode<Integer> parentX = null;
        TreeNode<Integer> parentY = null;

        int level = 0;
        Queue<TreeNode<Integer>> q = new LinkedList<>();
        q.add(root);

        while (!q.isEmpty()) {

            int size = q.size();
            for (int i = 0; i < size; i++) {
                TreeNode<Integer> curr = q.poll();

                if (curr.getLeft() != null) {
                    q.add(curr.getLeft());

                    //x or y can be in left sub tree
                    if ((int) curr.getLeft().getData() == x) {
                        levelX = level;
                        parentX = curr;
                    }

                    if ((int) curr.getLeft().getData() == y) {
                        levelY = level;
                        parentY = curr;
                    }
                }

                if (curr.getRight() != null) {
                    q.add(curr.getRight());

                    //x or y can be in right sub tree
                    if ((int) curr.getRight().getData() == x) {
                        levelX = level;
                        parentX = curr;
                    }

                    if ((int) curr.getRight().getData() == y) {
                        levelY = level;
                        parentY = curr;
                    }
                }
            }

            //if both nodes are found, their parents will not be null
            //and we will not be required to traverse further
            if (parentX != null && parentY != null) {
                break;
            }
            level++;
        }

        if (levelX == levelY && parentX != parentY) {
            System.out.println("Cousins");
        } else {
            System.out.println("Not cousins");
        }
    }

    private void printConnectedNodesAtSameLevelByRandomPointer(TreeNode<Integer> root) {

        int level = 0;
        Queue<TreeNode<Integer>> q = new LinkedList<>();
        q.add(root);

        while (!q.isEmpty()) {

            int size = q.size();
            for (int i = 0; i < size; i++) {

                TreeNode<Integer> curr = q.poll();
                TreeNode<Integer> ptrRandom = curr;
                System.out.print(level + ": ");
                while (ptrRandom != null) {
                    //print the nodes connected by random pointers
                    System.out.print(ptrRandom.getData() + " ");
                    ptrRandom = ptrRandom.getRandom();
                }

                if (curr.getLeft() != null) {
                    q.add(curr.getLeft());
                }

                if (curr.getRight() != null) {
                    q.add(curr.getRight());
                }

                System.out.println();
            }
            level++;
        }
    }

    private void connectTreeNodesAtSameLevel_Recursive_Helper(TreeNode<Integer> root,
            int level, Map<Integer, TreeNode<Integer>> map) {

        if (root == null) {
            return;
        }

        if (map.containsKey(level)) {
            //this will have a previous node to current root
            map.get(level).setRandom(root);
        }

        //update previous node with current root in that level
        map.put(level, root);
        root.setRandom(null); //default random value if there is no next node in that level

        connectTreeNodesAtSameLevel_Recursive_Helper(root.getLeft(), level + 1, map);
        connectTreeNodesAtSameLevel_Recursive_Helper(root.getRight(), level + 1, map);
    }

    public void connectTreeNodesAtSameLevel_Recursive(TreeNode<Integer> root) {
        //https://leetcode.com/problems/populating-next-right-pointers-in-each-node/
        //https://practice.geeksforgeeks.org/problems/connect-nodes-at-same-level/1#
        Map<Integer, TreeNode<Integer>> map = new HashMap<>();
        connectTreeNodesAtSameLevel_Recursive_Helper(root, 0, map);

        //output
        printConnectedNodesAtSameLevelByRandomPointer(root);
    }

    public void connectTreeNodesAtSameLevel_Iterative(TreeNode<Integer> root) {
        //https://leetcode.com/problems/populating-next-right-pointers-in-each-node/
        //https://practice.geeksforgeeks.org/problems/connect-nodes-at-same-level/1#
        Queue<TreeNode<Integer>> queue = new LinkedList<>();
        queue.add(root);
        TreeNode<Integer> prev;
        while (!queue.isEmpty()) {

            int size = queue.size();
            prev = null;

            for (int i = 0; i < size; i++) {

                TreeNode<Integer> curr = queue.poll();
                if (i > 0) {
                    prev.setRandom(curr);
                }
                curr.setRandom(null);
                prev = curr;

                if (curr.getLeft() != null) {
                    queue.add(curr.getLeft());
                }

                if (curr.getRight() != null) {
                    queue.add(curr.getRight());
                }
            }
        }

        //output
        printConnectedNodesAtSameLevelByRandomPointer(root);
    }

    private void binarySearchTreeToGreaterSumTree_Helper(TreeNode<Integer> root) {

        if (root == null) {
            return;
        }

        //go to right most leaf node in right sub tree
        //output starts from there only (otherwise you could choose as per the output demands)
        binarySearchTreeToGreaterSumTree_Helper(root.getRight());

        binarySearchTreeToGreaterSumTree_Sum += root.getData();

        root.setData(binarySearchTreeToGreaterSumTree_Sum);

        //now go to left sub tree as per output
        binarySearchTreeToGreaterSumTree_Helper(root.getLeft());
    }

    int binarySearchTreeToGreaterSumTree_Sum;

    public void binarySearchTreeToGreaterSumTree(TreeNode<Integer> root) {

        //https://leetcode.com/problems/convert-bst-to-greater-tree/
        //actual
        new BinarySearchTree<Integer>(root).treeBFS();
        System.out.println();

        binarySearchTreeToGreaterSumTree_Sum = 0;
        binarySearchTreeToGreaterSumTree_Helper(root);

        //output
        new BinarySearchTree<Integer>(root).treeBFS();
        System.out.println();
    }

    public void subtreeWithAllDeepestNodes(TreeNode<Integer> root) {

        //https://leetcode.com/problems/lowest-common-ancestor-of-deepest-leaves/
        //https://leetcode.com/problems/smallest-subtree-with-all-the-deepest-nodes/
        //Actual
        new BinaryTree<>(root).treeBFS();
        System.out.println();

        //levelorder to find the deepest leaf nodes
        List<Integer> deepLeaf = new ArrayList<>();

        Queue<TreeNode<Integer>> q = new LinkedList<>();
        q.add(root);

        while (!q.isEmpty()) {
            //clear all previous levels, in the end, we will have last level, 
            //which will never be cleared
            deepLeaf.clear();
            int size = q.size();
            for (int i = 0; i < size; i++) {

                TreeNode<Integer> curr = q.poll();
                deepLeaf.add(curr.getData());

                if (curr.getLeft() != null) {
                    q.add(curr.getLeft());
                }

                if (curr.getRight() != null) {
                    q.add(curr.getRight());
                }
            }
        }

        //get leftMost & rightMost leaf nodes in the deepLeaf level
        int n1 = -1;
        int n2 = -1;
        if (!deepLeaf.isEmpty()) {
            n1 = deepLeaf.get(0);
            n2 = deepLeaf.get(deepLeaf.size() - 1);
        }

        //find lca for n1 & n2
        //lca for two deepest leaf nodes will hold subtree, that conatins the deepest leaf nodes
        TreeNode<Integer> lcaForDeepLeafNodes = null;
        if (n1 != -1 && n2 != -1) {
            lcaForDeepLeafNodes = lowestCommonAncestorOfTree_Helper(root, n1, n2);
        }

        //output
        new BinaryTree<>(lcaForDeepLeafNodes).treeBFS();
        System.out.println();
    }

    private void pseudoPallindromicPathInBinaryTree_Helper(TreeNode<Integer> root,
            Map<Integer, Integer> map) {

        if (root == null) {
            return;
        }

        //freq of nodes in the path 
        map.put(root.getData(), map.getOrDefault(root.getData(), 0) + 1);

        pseudoPallindromicPathInBinaryTree_Helper(root.getLeft(), map);
        pseudoPallindromicPathInBinaryTree_Helper(root.getRight(), map);

        if (root.getLeft() == null && root.getRight() == null) {
            //for a path to be pseudo pallindromic, the freq of nodes in the paths can have 
            //either even freq(nodeHasOddFreq == 0) OR atmost 1 node can have odd freq(nodeHasOddFreq == 1)
            int nodeHasOddFreq = 0;
            for (int key : map.keySet()) {
                if (map.get(key) % 2 == 1) {
                    nodeHasOddFreq++;
                }
            }

            //atmost 1 node should have odd freq
            if (nodeHasOddFreq <= 1) {
                pseudoPallindromicPathInBinaryTree_Count++;
            }
        }

        //remove the freq of the last node added once they are processed
        map.put(root.getData(), map.get(root.getData()) - 1);
        if (map.get(root.getData()) <= 0) {
            map.remove(root.getData());
        }
    }

    int pseudoPallindromicPathInBinaryTree_Count;

    public void pseudoPallindromicPathInBinaryTree(TreeNode<Integer> root) {

        //https://leetcode.com/problems/pseudo-palindromic-paths-in-a-binary-tree/
        /*paths are [2,3,3] & [2,1,1] these paths are 
         pseudo pallindromic as they can be represented as pallindrome [3,2,3] & [1,2,1]*/
        pseudoPallindromicPathInBinaryTree_Count = 0;
        Map<Integer, Integer> map = new HashMap<>();
        pseudoPallindromicPathInBinaryTree_Helper(root, map);

        //output
        System.out.println("Pseudo pallindromic paths count: " + pseudoPallindromicPathInBinaryTree_Count);
    }

    int countNumberOfTurnsBetweenRootToGivenKey_CountTurns = 0;

    public boolean countNumberOfTurnsBetweenRootToGivenKey(TreeNode<Integer> root,
            int key, boolean goingLeft) {

        if (root == null) {
            return false;
        }

        if (root.getData() == key) {
            return true;
        }

        if (goingLeft) {
            //as we are goingLeft == true so turn will not be counted 
            //going deeper to left subtrees
            if (countNumberOfTurnsBetweenRootToGivenKey(root.getLeft(),
                    key, goingLeft)) {
                return true;
            }
            //each time we make a move to right subtree that means we are making
            //a turn to find the key
            if (countNumberOfTurnsBetweenRootToGivenKey(root.getRight(), key, !goingLeft)) {
                countNumberOfTurnsBetweenRootToGivenKey_CountTurns += 1;
                return true;
            }
        } else { //goingLeft == false
            //as we are goingLeft == false(going to right) so turn will not be counted 
            //going deeper to right subtrees
            if (countNumberOfTurnsBetweenRootToGivenKey(root.getRight(), key, goingLeft)) {
                return true;
            }
            //each time we make a move to left subtree that means we are making
            //a turn to find the key
            if (countNumberOfTurnsBetweenRootToGivenKey(root.getLeft(), key, !goingLeft)) {
                countNumberOfTurnsBetweenRootToGivenKey_CountTurns += 1;
                return true;
            }
        }
        return false;
    }

    public int countNumberOfTurnsBetweenTwoNodesOfTree(TreeNode<Integer> root, int n1, int n2) {

        //https://www.geeksforgeeks.org/number-turns-reach-one-node-binary-tree/
        //find lca of two nodes
        TreeNode<Integer> lca = lowestCommonAncestorOfTree_Helper(root, n1, n2);

        if (lca == null) {
            return -1;
        }

        countNumberOfTurnsBetweenRootToGivenKey_CountTurns = 0;

        boolean n1FoundInLeftSubtree = false;
        boolean n2FoundInLeftSubtree = false;

        //if lca is not any of the nodes, neither n1 AND n2
        if (lca.getData() != n1 && lca.getData() != n2) {

            //count turns  in finding n1 in both left OR right sub tree of lca 
            n1FoundInLeftSubtree = countNumberOfTurnsBetweenRootToGivenKey(lca.getLeft(), n1, true); //goingLeft == true
            if (!n1FoundInLeftSubtree) {
                //if n1 is not found in left subtree then only go to search in right sub tree
                countNumberOfTurnsBetweenRootToGivenKey(lca.getRight(), n1, false); //goingLeft == false
            }

            //count turns  in finding n2 in both left OR right sub tree of lca 
            n2FoundInLeftSubtree = countNumberOfTurnsBetweenRootToGivenKey(lca.getLeft(), n2, true); //goingLeft == true
            if (!n2FoundInLeftSubtree) {
                //if n2 is not found in left subtree then only go to search in right sub tree
                countNumberOfTurnsBetweenRootToGivenKey(lca.getRight(), n2, false); //goingLeft == false
            }
            return countNumberOfTurnsBetweenRootToGivenKey_CountTurns + 1; //1 turn also made at lca node
        } else if (lca.getData() == n1) { //lca node is one of the given node

            //count turns of the other node from the lca
            n2FoundInLeftSubtree = countNumberOfTurnsBetweenRootToGivenKey(lca.getLeft(), n2, true); //goingLeft == true
            if (!n2FoundInLeftSubtree) {
                //if n2 is not found in left subtree then only go to search in right sub tree
                countNumberOfTurnsBetweenRootToGivenKey(lca.getRight(), n2, false); //goingLeft == false
            }
            return countNumberOfTurnsBetweenRootToGivenKey_CountTurns;
        } else { //lca.getData() == n2
            //count turns of the other node from the lca
            n1FoundInLeftSubtree = countNumberOfTurnsBetweenRootToGivenKey(lca.getLeft(), n1, true); //goingLeft == true
            if (!n1FoundInLeftSubtree) {
                //if n1 is not found in left subtree then only go to search in right sub tree
                countNumberOfTurnsBetweenRootToGivenKey(lca.getRight(), n1, false); //goingLeft == false
            }
            return countNumberOfTurnsBetweenRootToGivenKey_CountTurns;
        }
    }

    class LongestZigZagPathInTreePair {

        int leftZigZagPath;
        int rightZigZagPath;

        public LongestZigZagPathInTreePair(int leftZigZagPath, int rightZigZagPath) {
            this.leftZigZagPath = leftZigZagPath;
            this.rightZigZagPath = rightZigZagPath;
        }
    }

    private void longestZigZagPathInTree_Helper(TreeNode<Integer> root,
            Map<TreeNode<Integer>, LongestZigZagPathInTreePair> map) {

        if (root == null) {
            return;
        }

        map.putIfAbsent(root, new LongestZigZagPathInTreePair(0, 0));

        longestZigZagPathInTree_Helper(root.getLeft(), map);
        longestZigZagPathInTree_Helper(root.getRight(), map);

        //calculate zig zag path for curr root for its both
        //leftZigZagPath i.e, path from root.left and its rightZigZagPath
        //rightZigZagPath i.e, path from root.right and its leftZigZagPath
        map.get(root).leftZigZagPath = 1 + map.get(root.getLeft()).rightZigZagPath;
        map.get(root).rightZigZagPath = 1 + map.get(root.getRight()).leftZigZagPath;

        longestZigZagPathInTree_Length = Math.max(
                longestZigZagPathInTree_Length,
                Math.max(map.get(root).leftZigZagPath,
                        map.get(root).rightZigZagPath));
    }

    int longestZigZagPathInTree_Length;

    public void longestZigZagPathInTree(TreeNode<Integer> root) {

        //https://leetcode.com/problems/longest-zigzag-path-in-a-binary-tree/
        longestZigZagPathInTree_Length = 0;
        Map<TreeNode<Integer>, LongestZigZagPathInTreePair> map = new HashMap<>();
        map.put(null, new LongestZigZagPathInTreePair(0, 0));
        longestZigZagPathInTree_Helper(root, map);

        //output
        System.out.println("Longest zig zag path in tree approach1: " + (longestZigZagPathInTree_Length - 1));
    }

    private int longestZigZagPathInTree2_Helper(
            TreeNode<Integer> root, boolean isComingFromLeft, boolean isComingFromRight, int pathLength) {
        if (root == null) {
            return pathLength;
        }
        int leftPathLength = longestZigZagPathInTree2_Helper(root.getLeft(),
                true, //from curr root going to its left
                false,
                //if previously coming from right
                //and now going to left that means its a zig zag path
                //so pathLen + 1
                //otherwise not coming from right and going to left 
                //consider it starting of some new zig zag path
                isComingFromRight ? pathLength + 1 : 1);
        int rightPathLength = longestZigZagPathInTree2_Helper(root.getRight(),
                false,
                true, //from curr root going to its right
                //if previously coming from left
                //and now going to right that means its a zig zag path
                //so pathLen + 1
                //otherwise not coming from left and going to right 
                //consider it starting of some new zig zag path
                isComingFromLeft ? pathLength + 1 : 1);
        return Math.max(leftPathLength, rightPathLength);
    }

    public void longestZigZagPathInTree2(TreeNode<Integer> root) {
        //FASTER than above approach 1
        //https://leetcode.com/problems/longest-zigzag-path-in-a-binary-tree/
        //https://leetcode.com/problems/longest-zigzag-path-in-a-binary-tree/discuss/2225080/Java-Solution
        int longestZigZagPath = longestZigZagPathInTree2_Helper(root, false, false, 1) - 1;
        //output
        System.out.println("Longest zig zag path in tree approach2: " + longestZigZagPath);
    }

    private int longestPathArithemeticProgressionInBinaryTree_MaxLength;

    private void longestPathArithemeticProgressionInBinaryTree_DFS(
            TreeNode<Integer> root, int difference, int currMaxLengthAP) {

        if (root.getLeft() != null) {
            int currDifference = root.getLeft().getData() - root.getData();
            if (currDifference == difference) {
                longestPathArithemeticProgressionInBinaryTree_DFS(
                        root.getLeft(), currDifference, currMaxLengthAP + 1);

                longestPathArithemeticProgressionInBinaryTree_MaxLength = Math.max(
                        longestPathArithemeticProgressionInBinaryTree_MaxLength,
                        currMaxLengthAP + 1);
            } else {
                longestPathArithemeticProgressionInBinaryTree_DFS(
                        root.getLeft(), currDifference, currMaxLengthAP);
            }
        }

        if (root.getRight() != null) {
            int currDifference = root.getRight().getData() - root.getData();
            if (currDifference == difference) {
                longestPathArithemeticProgressionInBinaryTree_DFS(
                        root.getRight(), currDifference, currMaxLengthAP + 1);

                longestPathArithemeticProgressionInBinaryTree_MaxLength = Math.max(
                        longestPathArithemeticProgressionInBinaryTree_MaxLength,
                        currMaxLengthAP + 1);
            } else {
                longestPathArithemeticProgressionInBinaryTree_DFS(
                        root.getRight(), currDifference, currMaxLengthAP);
            }
        }
    }

    public void longestPathArithemeticProgressionInBinaryTree(TreeNode<Integer> root) {

        //https://www.geeksforgeeks.org/arithmetic-progression/
        //https://www.geeksforgeeks.org/longest-path-to-the-bottom-of-a-binary-tree-forming-an-arithmetic-progression/
        longestPathArithemeticProgressionInBinaryTree_MaxLength = 2;

        if (root == null) {
            longestPathArithemeticProgressionInBinaryTree_MaxLength = 0;
        }

        if (root.getLeft() == null && root.getRight() == null) {
            longestPathArithemeticProgressionInBinaryTree_MaxLength = 1;
        }

        //root & root.left OR root & root.right
        int currMaxLengthAP = 2;
        if (root.getLeft() != null) {
            int difference = root.getLeft().getData() - root.getData();
            longestPathArithemeticProgressionInBinaryTree_DFS(root.getLeft(), difference, currMaxLengthAP);
        }

        if (root.getRight() != null) {
            int difference = root.getRight().getData() - root.getData();
            longestPathArithemeticProgressionInBinaryTree_DFS(root.getRight(), difference, currMaxLengthAP);
        }
        //output
        System.out.println("Max length AP in binary tree: "
                + longestPathArithemeticProgressionInBinaryTree_MaxLength);
    }

    public void timeNeededToInformAllEmployee_NAryTreeDFS(int managerNode,
            Map<Integer, List<Integer>> managerToEmployees,
            int[] informTime,
            PriorityQueue<Integer> minHeap,
            int currTime) {

        if (!managerToEmployees.containsKey(managerNode)) {
            minHeap.add(currTime);
            if (minHeap.size() > 1) {
                minHeap.poll();
            }
        }

        List<Integer> directReportee = managerToEmployees.getOrDefault(managerNode, new ArrayList<>());
        for (int employee : directReportee) {
            timeNeededToInformAllEmployee_NAryTreeDFS(employee,
                    managerToEmployees,
                    informTime,
                    minHeap,
                    currTime + informTime[managerNode]);
        }
    }

    public void timeNeededToInformAllEmployee_NAryTree(int employees, int headManagerID,
            int[] manager, int[] informTime) {
        //Working but little time taking
        //https://leetcode.com/problems/time-needed-to-inform-all-employees/
        //minHeap to store max time taken from headManager to last subordinate 
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        //manager[i] => [i]
        Map<Integer, List<Integer>> managerToEmployees = new HashMap<>();
        for (int i = 0; i < employees; i++) {
            managerToEmployees.putIfAbsent(manager[i], new ArrayList<>());
            managerToEmployees.get(manager[i]).add(i);
        }
        timeNeededToInformAllEmployee_NAryTreeDFS(headManagerID, managerToEmployees, informTime, minHeap, 0);
        System.out.println("Max time required to inform all employee from head manager to subordinate(N-Ary tree DFS): "
                + (minHeap.isEmpty() ? 0 : minHeap.peek()));
    }

    private int timeNeededToInformAllEmployee_DFS_Helper(int employeeNode, int[] manager, int[] informTime) {
        if (manager[employeeNode] == -1) {
            return informTime[employeeNode];
        }

        informTime[employeeNode] += timeNeededToInformAllEmployee_DFS_Helper(manager[employeeNode], manager, informTime);
        manager[employeeNode] = -1;

        return informTime[employeeNode];
    }

    public void timeNeededToInformAllEmployee_DFS(int employees, int headManagerID,
            int[] manager, int[] informTime) {
        //Little more optimized than above solution
        //https://leetcode.com/problems/time-needed-to-inform-all-employees/
        //https://www.geeksforgeeks.org/google-interview-experience-sde-1-off-campus-2022/
        //minHeap to store max time taken from headManager to last subordinate 
        int maxTime = 0;
        //Loop trying to check the max time taken from any node, headMaangerID is not directly used here
        for (int employee = 0; employee < employees; employee++) {
            maxTime = Math.max(maxTime, timeNeededToInformAllEmployee_DFS_Helper(employee, manager, informTime));
        }
        System.out.println("Max time required to inform all employee from head manager to subordinate(Direct DFS): "
                + maxTime);
    }

    private enum BinaryTreeCameraState {

        hasCamera, needCamera, covered
    }
    int binaryTreeCameras_ReqCamera;

    private BinaryTreeCameraState binaryTreeCameras_Helper(TreeNode<Integer> root) {
        if (root == null) {
            return BinaryTreeCameraState.covered;
        }

        BinaryTreeCameraState leftState = binaryTreeCameras_Helper(root.getLeft());
        BinaryTreeCameraState rightState = binaryTreeCameras_Helper(root.getRight());

        if (leftState == BinaryTreeCameraState.needCamera
                || rightState == BinaryTreeCameraState.needCamera) {
            binaryTreeCameras_ReqCamera++;
            return BinaryTreeCameraState.hasCamera;
        }

        if (leftState == BinaryTreeCameraState.hasCamera
                || rightState == BinaryTreeCameraState.hasCamera) {
            return BinaryTreeCameraState.covered;
        }
        return BinaryTreeCameraState.needCamera;
    }

    public void binaryTreeCameras(TreeNode<Integer> root) {
        //https://leetcode.com/problems/binary-tree-cameras/
        //https://leetcode.com/problems/binary-tree-cameras/solution/
        binaryTreeCameras_ReqCamera = 0;
        int output = binaryTreeCameras_Helper(root) == BinaryTreeCameraState.needCamera
                ? binaryTreeCameras_ReqCamera++
                : binaryTreeCameras_ReqCamera;
        System.out.println("Binary tree cameras : " + output);
    }

    private TreeNode<Integer> convertSortedArrayToHeightBalancedBinarySearchTree_Helper(
            int[] arr, int start, int end) {
        if (start > end) {
            return null;
        }

        int mid = start + (end - start) / 2;
        TreeNode<Integer> root = new TreeNode<>(arr[mid]);
        root.setLeft(
                convertSortedArrayToHeightBalancedBinarySearchTree_Helper(arr, start, mid - 1)
        );
        root.setRight(
                convertSortedArrayToHeightBalancedBinarySearchTree_Helper(arr, mid + 1, end)
        );

        return root;
    }

    public void convertSortedArrayToHeightBalancedBinarySearchTree(int[] arr) {
        //https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/
        //explanation: https://youtu.be/0K0uCMYq5ng
        TreeNode<Integer> root = convertSortedArrayToHeightBalancedBinarySearchTree_Helper(
                arr, 0, arr.length - 1);
        //output
        new BinarySearchTree<>(root).treeInorder();
        System.out.println();
    }

    public void binarySearchTreeIterator(TreeNode<Integer> root) {
        //..............T: O(H), at most using addNodes() we are just travelling
        //tree upto the height of tree and not the complete N nodes of tree
        //https://leetcode.com/problems/binary-search-tree-iterator
        BinarySearchTreeIterator<Integer> iterator = new BinarySearchTreeIterator<>(root);
        while (iterator.hasNext()) {
            System.out.println(iterator.next());
        }
    }

    private boolean linkedListInBinaryTree_MatchCheck(Node<Integer> head, TreeNode<Integer> root) {
        if (head == null) {
            return true;
        }

        if (root == null || root.getData() != head.getData()) {
            return false;
        }

        return linkedListInBinaryTree_MatchCheck(head.getNext(), root.getLeft())
                || linkedListInBinaryTree_MatchCheck(head.getNext(), root.getRight());

    }

    private boolean linkedListInBinaryTree_DFS(Node<Integer> head, TreeNode<Integer> root) {

        if (root == null) {
            return false;
        }

        if (linkedListInBinaryTree_MatchCheck(head, root)) {
            return true;
        }

        return linkedListInBinaryTree_DFS(head, root.getLeft())
                || linkedListInBinaryTree_DFS(head, root.getRight());

    }

    public void linkedListInBinaryTree(Node<Integer> head, TreeNode<Integer> root) {
        //https://leetcode.com/problems/linked-list-in-binary-tree/
        boolean output = linkedListInBinaryTree_DFS(head, root);
        //output
        System.out.println("Is linked list present in binary tree: " + output);
    }

    private void printBinaryTreeInStringMatrixFormat_Helper(
            TreeNode<Integer> root, int currRow, int currCol, int height, List<List<String>> treeMatrix) {

        if (root == null) {
            return;
        }

        treeMatrix.get(currRow).set(currCol, root.getData() + "");

        printBinaryTreeInStringMatrixFormat_Helper(root.getLeft(),
                currRow + 1,
                currCol - (int) Math.pow(2, height - currRow - 1),
                height,
                treeMatrix);

        printBinaryTreeInStringMatrixFormat_Helper(root.getRight(),
                currRow + 1,
                currCol + (int) Math.pow(2, height - currRow - 1),
                height,
                treeMatrix);
    }

    public void printBinaryTreeInStringMatrixFormat(TreeNode<Integer> root) {
        //https://leetcode.com/problems/print-binary-tree/
        /*
            1. The height of the tree is height and the number of rows m should be equal to height + 1.
            2. The number of columns n should be equal to 2height+1 - 1.
            3. Place the root node in the middle of the top row (more formally, at location res[0][(n-1)/2]).
            4. For each node that has been placed in the matrix at position res[r][c],
                place its left child at res[r+1][c-2height-r-1] and its right child at res[r+1][c+2height-r-1].
            5. Continue this process until all the nodes in the tree have been placed.
            6. Any empty cells should contain the empty string "".
         */
        int height = heightOfTree(root);
        int m = height + 1;
        int n = (int) Math.pow(2, height + 1) - 1;

        //preparing empty matrix
        String EMPTY_SPOT = ".";
        List<List<String>> treeMatrix = new ArrayList<>();
        for (int r = 0; r < m; r++) {
            treeMatrix.add(new ArrayList<>());
            for (int c = 0; c < n; c++) {
                treeMatrix.get(r).add(EMPTY_SPOT);
            }
        }

        int currRow = 0;
        int currCol = (n - 1) / 2;

        treeMatrix.get(currRow).set(currCol, root.getData() + "");

        printBinaryTreeInStringMatrixFormat_Helper(root.getLeft(),
                currRow + 1,
                currCol - (int) Math.pow(2, height - currRow - 1),
                height,
                treeMatrix);

        printBinaryTreeInStringMatrixFormat_Helper(root.getRight(),
                currRow + 1,
                currCol + (int) Math.pow(2, height - currRow - 1),
                height,
                treeMatrix);
        //output
        System.out.println("Print binary tree in string matrix format: ");
        for (List<String> level : treeMatrix) {
            System.out.println(level);
        }
    }

    // STACK
    int middleElementInStack_Element = Integer.MIN_VALUE;

    private void middleElementInStack_Helper(Stack<Integer> s, int n, int index) {

        if (n == index || s.isEmpty()) {
            return;
        }

        int ele = s.pop();
        middleElementInStack_Helper(s, n, index + 1);
        if (index == n / 2) {
            middleElementInStack_Element = ele;
        }
        s.push(ele);
    }

    public void middleElementInStack(Stack<Integer> stack) {
        int n = stack.size();
        int index = 0;
        //just reseting
        middleElementInStack_Element = Integer.MIN_VALUE;
        middleElementInStack_Helper(stack, n, index);
        //outputs
        System.out.println("Middle eleement of the stack: " + middleElementInStack_Element);
    }

    public void nextSmallerElementInRightInArray(int[] arr) {

        int n = arr.length;
        Stack<Integer> stack = new Stack<>();
        int[] result = new int[n];
        for (int i = n - 1; i >= 0; i--) {

            while (!stack.isEmpty() && stack.peek() > arr[i]) {
                stack.pop();
            }

            if (stack.isEmpty()) {
                result[i] = -1;
            } else {
                result[i] = (stack.peek());
            }
            stack.push(arr[i]);
        }
        //output
        for (int val : result) {
            System.out.print(val + " ");
        }
        System.out.println();
    }

    private void reserveStack_Recursion_Insert(Stack<Integer> stack, int element) {

        if (stack.isEmpty()) {
            stack.push(element);
            return;
        }

        int popped = stack.pop();
        reserveStack_Recursion_Insert(stack, element);
        stack.push(popped);
    }

    private void reserveStack_Recursion(Stack<Integer> stack) {

        if (stack.isEmpty()) {
            return;
        }

        int popped = stack.pop();
        reserveStack_Recursion(stack);
        reserveStack_Recursion_Insert(stack, popped);
    }

    public void reverseStack(Stack<Integer> stack) {
        System.out.println("actual: " + stack);
        reserveStack_Recursion(stack);
        System.out.println("output: " + stack);
    }

    public void nextGreaterElementInRightInArray(int[] arr) {

        Stack<Integer> st = new Stack<>();
        int[] result = new int[arr.length];
        int index = arr.length - 1;
        for (int i = arr.length - 1; i >= 0; i--) {

            while (!st.isEmpty() && st.peek() < arr[i]) {
                st.pop();
            }

            if (st.isEmpty()) {
                result[index--] = -1;
            } else {
                result[index--] = st.peek();
            }
            st.push(arr[i]);
        }

        //output
        for (int x : result) {
            System.out.print(x + " ");
        }
        System.out.println();
    }

    public void largestAreaInHistogram(int[] heights) {
        //https://leetcode.com/problems/largest-rectangle-in-histogram/
        // Create an empty stack. The stack holds indexes of hist[] array 
        // The bars stored in stack are always in increasing order of their 
        // heights. 
        Stack<Integer> stack = new Stack<>();
        int n = heights.length;
        int maxArea = 0; // Initialize max area 
        int top;  // To store top of stack 
        int areaWithTop; // To store area with top bar as the smallest bar 

        // Run through all bars of given histogram 
        int index = 0;
        while (index < n) {
            // If this bar is higher than the bar on top stack, push it to stack 
            if (stack.isEmpty() || heights[stack.peek()] <= heights[index]) {
                stack.push(index++);

                // If this bar is lower than top of stack, then calculate area of rectangle  
                // with stack top as the smallest (or minimum height) bar. 'i' is  
                // 'right index' for the top and element before top in stack is 'left index' 
            } else {

                top = stack.pop();  // store the top index 
                // Calculate the area with hist[tp] stack as smallest bar 
                areaWithTop = heights[top] * (stack.isEmpty() ? index : index - stack.peek() - 1);
                // update max area, if needed 
                maxArea = Math.max(maxArea, areaWithTop);
            }
        }

        // Now pop the remaining bars from stack and calculate area with every 
        // popped bar as the smallest bar 
        while (!stack.isEmpty()) {
            top = stack.pop();
            areaWithTop = heights[top] * (stack.isEmpty() ? index : index - stack.peek() - 1);
            maxArea = Math.max(maxArea, areaWithTop);
        }

        //output:
        System.out.println("Max area of histogram: " + maxArea);
    }

    public void postfixExpressionEvaluation_SingleDigit(String expr) {
        //https://leetcode.com/problems/evaluate-reverse-polish-notation/
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < expr.length(); i++) {

            char ch = expr.charAt(i);
            if (Character.isDigit(ch)) {
                stack.push(ch - '0');
            } else {
                int num1 = stack.pop();
                int num2 = stack.pop();

                switch (ch) {
                    case '+':
                        stack.push(num2 + num1);
                        break;
                    case '-':
                        stack.push(num2 - num1);
                        break;
                    case '*':
                        stack.push(num2 * num1);
                        break;
                    case '/':
                        stack.push(num2 / num1);
                        break;
                }
            }
        }

        //output:
        System.out.println("Evaluation single digit expression: " + stack.pop());
    }

    public void postfixExpressionEvaluation_MultipleDigit(String expr) {
        //https://leetcode.com/problems/evaluate-reverse-polish-notation/
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < expr.length(); i++) {

            char ch = expr.charAt(i);

            //space is needed in expr to distinguish b/w 2 different multiple digit
            if (ch == ' ') {
                continue;
            }

            //if we found atleat one digit
            //try to iterate i until we found a char ch which is not a digit
            if (Character.isDigit(ch)) {
                int createNum = 0;
                while (Character.isDigit(expr.charAt(i))) {
                    createNum = createNum * 10 + (expr.charAt(i) - '0');
                    i++; //this to further iterate i and find digit char 
                }
//                i--; //just to balance to one iter back
                stack.push(createNum);
            } else {
                int num1 = stack.pop();
                int num2 = stack.pop();

                switch (ch) {
                    case '+':
                        stack.push(num2 + num1);
                        break;
                    case '-':
                        stack.push(num2 - num1);
                        break;
                    case '*':
                        stack.push(num2 * num1);
                        break;
                    case '/':
                        stack.push(num2 / num1);
                        break;
                }
            }
        }

        //output:
        System.out.println("Evaluation multiple digit expression: " + stack.pop());
    }

    public void removeKDigitsToCreateSmallestNumber(String num, int K) {

        //explanation: https://youtu.be/vbM41Zql228
        if (K == num.length()) {
            System.out.println("Number formed: 0");
            return;
        }

        StringBuilder sb = new StringBuilder();
        Stack<Integer> stack = new Stack<>();

        for (char ch : num.toCharArray()) {
            int digit = ch - '0';
            //Greedily take the smaller digit than stack's peek digit
            //ex: "1432219"
            //stack [1]
            //stack [1, 4] as not 1 > 4
            //stack [1, 3] as 4 > 3 
            //becasue if number has to be formed in that case 14 > 13 smaller num req 13 and so on.
            while (!stack.isEmpty() && stack.peek() > digit && K != 0) {
                stack.pop();
                K--;
            }
            stack.push(digit);
        }

        //case when all the digits in the stack are same
        //ex: 1111
        while (!stack.isEmpty() && K != 0) {
            stack.pop();
            K--;
        }

        //form the number
        //ex: "1432219"
        //stack would be having [1, 2, 1, 9] <- peek
        //pop element and add then at 0th index so sb = "1219"
        while (!stack.isEmpty()) {
            sb.insert(0, stack.pop());
        }

        //case when there are leading zeros in num 
        //ex: 000234 => 234
        while (sb.length() > 1 && sb.charAt(0) == '0') {
            sb.deleteCharAt(0);
        }

        //output
        System.out.println("Number formed: " + sb.toString());
    }

    public void findTheMostCompetetiveSubsequenceOfSizeKFromArray(int[] nums, int K) {

        //problem: https://leetcode.com/problems/find-the-most-competitive-subsequence/
        //explanation: https://leetcode.com/problems/find-the-most-competitive-subsequence/discuss/1113429/Java-Brute-Force-Stack
        int n = nums.length;
        Stack<Integer> stack = new Stack<>();

        for (int i = 0; i < n; i++) {
            int val = nums[i];

            while (stack.size() > 0 && stack.size() + n - (i + 1) >= K
                    && stack.peek() > val) {
                stack.pop();
            }
            stack.push(val);
        }

        while (!stack.isEmpty() && stack.size() > K) {
            stack.pop();
        }
        //output
        int[] result = stack.stream().mapToInt(val -> val).toArray();
        for (int x : result) {
            System.out.print(x + " ");
        }

        System.out.println();
    }

    public void nextWarmerDayInTheGivenWeatherRecordings(int[] recordings) {

        //SIMILAR TO NEXT GREATER ELEMENT TO RIGHT
        //problem statement: https://youtu.be/0mcAy91rPzE
        //https://leetcode.com/problems/daily-temperatures/
        int n = recordings.length;
        List<Integer> daysAfter = new ArrayList<>();
        Stack<Integer> stack = new Stack<>();
        for (int i = n - 1; i >= 0; i--) {

            while (!stack.isEmpty() && recordings[stack.peek()] < recordings[i]) {
                stack.pop();
            }

            if (stack.isEmpty()) {
                daysAfter.add(0);
            } else {
                daysAfter.add(stack.peek() - i);
            }

            stack.push(i);
        }

        //output:
        Collections.reverse(daysAfter);
        System.out.println("For each recording next warm day occur after:");
        for (int i = 0; i < n; i++) {
            if (daysAfter.get(i) == 0) {
                System.out.println(recordings[i] + ": after " + daysAfter.get(i) + " day, there is no warm day after this recrding");
                continue;
            }
            System.out.println(recordings[i] + ": after " + daysAfter.get(i) + " day, there is a warm day i,e: " + recordings[i + daysAfter.get(i)]);
        }
    }

    private void rotAllAdjacent(int[][] basket,
            int x, int y,
            boolean[][] visited,
            int row, int col) {

        //all aadjacent coordinate
        //check new coordinates are in bounds
        //check new coordinates are not previously visited
        //maake the adjacent rot and mark them visited
        int x1 = -1;
        int y1 = -1;

        //left coordinate to x,y = x, y-1
        x1 = x;
        y1 = y - 1;
        if ((x1 >= 0 && x1 < row) && (y1 >= 0 && y1 < col) && visited[x1][y1] != true && basket[x1][y1] != 0) {
            visited[x1][y1] = true; //maark them visited
            basket[x1][y1] = 2; //make them rot
            rotAllAdjacent(basket, x1, y1, visited, row, col);
        }

        //right coordinate to x,y = x, y+1
        x1 = x;
        y1 = y + 1;
        if ((x1 >= 0 && x1 < row) && (y1 >= 0 && y1 < col) && visited[x1][y1] != true && basket[x1][y1] != 0) {
            visited[x1][y1] = true; //maark them visited
            basket[x1][y1] = 2; //make them rot
            rotAllAdjacent(basket, x1, y1, visited, row, col);
        }

        //top coordinate to x,y = x-1, y
        x1 = x - 1;
        y1 = y;
        if ((x1 >= 0 && x1 < row) && (y1 >= 0 && y1 < col) && visited[x1][y1] != true && basket[x1][y1] != 0) {
            visited[x1][y1] = true; //maark them visited
            basket[x1][y1] = 2; //make them rot
            rotAllAdjacent(basket, x1, y1, visited, row, col);
        }

        //bottom coordinate to x,y = x+1, y
        x1 = x + 1;
        y1 = y;
        if ((x1 >= 0 && x1 < row) && (y1 >= 0 && y1 < col) && visited[x1][y1] != true && basket[x1][y1] != 0) {
            visited[x1][y1] = true; //maark them visited
            basket[x1][y1] = 2; //make them rot
            rotAllAdjacent(basket, x1, y1, visited, row, col);
        }

    }

    public void rottenOranges_DFS(int[][] basket) {

        int rottenTime = 0;
        int row = basket.length;
        int col = basket[0].length;

        boolean[][] visited = new boolean[row][col];

        for (int x = 0; x < row; x++) {
            for (int y = 0; y < col; y++) {
                if (visited[x][y] != true && basket[x][y] == 2) {
                    //rotten oranges == 2
                    visited[x][y] = true;
                    rottenTime++;
                    rotAllAdjacent(basket, x, y, visited, row, col);
                }
            }
        }

        //check if any one is left unrotten(1)
        for (int x = 0; x < row; x++) {
            for (int y = 0; y < col; y++) {
                if (basket[x][y] == 1) {
                    //rotten oranges == 2
                    rottenTime = -1;
                }
            }
        }

        System.out.println("rotten time " + rottenTime);

    }

    public int rottenOranges_HashBased(int[][] grid) {
        //https://leetcode.com/problems/rotting-oranges/
        Set<String> fresh = new HashSet<>();
        Set<String> rotten = new HashSet<>();
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {

                if (grid[i][j] == 1) {
                    fresh.add(i + "" + j);
                }

                if (grid[i][j] == 2) {
                    rotten.add(i + "" + j);
                }
            }
        }

        int minTime = 0;
        int[][] dirs = {
            {-1, 0},
            {1, 0},
            {0, -1},
            {0, 1}
        };
        Set<String> infected = new HashSet<>();
        while (fresh.size() > 0) {
            //loop over all the rotten oranges
            for (String rottenPoint : rotten) {
                //get coord of all the curr rotten orange
                int x = rottenPoint.charAt(0) - '0';
                int y = rottenPoint.charAt(1) - '0';
                //find all the adjacent 4-directions from the 
                //curr rotten orange
                for (int[] dir : dirs) {
                    int newX = x + dir[0];
                    int newY = y + dir[1];
                    //if any adjacent 4-directions contains a fresh orange
                    //that means the curr rotten orange has infected it(in 1 unit time)
                    if (fresh.contains(newX + "" + newY)) {
                        //fresh is now infected by curr rotten orange
                        //so remove from fresh coords
                        fresh.remove(newX + "" + newY);
                        //move this newly infected orange into infected coord
                        infected.add(newX + "" + newY);
                    }
                }
            }

            //if at any point, we are unable to infect any fresh oranges
            //out infected coord will remain empty, so return -1
            if (infected.isEmpty()) {
                return -1;
            }

            //put all the infected oranges into rotten coords and clear infected
            //for next time
            rotten.addAll(infected);
            infected.clear();
            minTime++;
        }

        return minTime;
    }

    public void validSudoku(String[][] grid) {
        //https://leetcode.com/problems/valid-sudoku/
        //https://leetcode.com/problems/check-if-every-row-and-column-contains-all-numbers/
        //Explanantion: https://youtu.be/Pl7mMcBm2b8
        HashSet<String> vis = new HashSet<>();

        for (int x = 0; x < grid.length; x++) {
            for (int y = 0; y < grid[x].length; y++) {

                String curr = grid[x][y];
                if (!curr.equals(".")) {

                    if (!vis.add(curr + " at row: " + x)
                            || !vis.add(curr + " at col: " + y)
                            || !vis.add(curr + " in cell: " + (x / 3) + "-" + (y / 3))) {
                        System.out.println("Invalid sudoku grid");
                        return;
                    }
                }
            }
        }

        System.out.println("Valid sudoku grid");
    }

    public void minCostOfRope(int[] ropes) {

        //GREEDY ALGO
        //HEAP based approach
        PriorityQueue<Integer> minHeapRopes = new PriorityQueue<>();
        for (int rope : ropes) {
            minHeapRopes.add(rope);
        }

        //calculations
        int cost = 0;
        while (minHeapRopes.size() >= 2) {

            int rope1 = minHeapRopes.poll();
            int rope2 = minHeapRopes.poll();

            cost += rope1 + rope2;
            int newRope = rope1 + rope2;
            minHeapRopes.add(newRope);
        }
        //output
        System.out.println("Min cost to combine all rpes into one rope: " + cost);
    }

    public void kLargestElementInArray(int[] arr, int K) {

        PriorityQueue<Integer> minHeap = new PriorityQueue<>((o1, o2) -> o1.compareTo(o2));
        for (int x : arr) {
            minHeap.add(x);
            if (minHeap.size() > K) {
                minHeap.poll();
            }
        }

        int[] result = new int[minHeap.size()];
        int index = minHeap.size() - 1;
        while (!minHeap.isEmpty()) {

            result[index--] = minHeap.poll();
        }

        //output
        for (int x : result) {
            System.out.print(x + " ");
        }
        System.out.println();
    }

    public void mergeKSortedArrays_1(int[][] arr) {
        //......................T: O(M*N*Log(M*N)), where M = row & N = col
        //......................S: O(M*N)
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for (int[] row : arr) {
            for (int cell : row) {
                minHeap.add(cell);
            }
        }

        List<Integer> sortedList = new ArrayList<>();
        while (!minHeap.isEmpty()) {
            sortedList.add(minHeap.poll());
        }

        //output:
        System.out.println("K sorted array into a list: " + sortedList);
    }

    public void mergeKSortedArrays_2(int[][] arr) {

        class Input {

            final int row;
            int col;
            final int colLength;

            public Input(int row, int col, int colLength) {
                this.row = row;
                this.col = col;
                this.colLength = colLength;
            }
        }

        //OPTIMISED
        //........................T: O(N*K*LogK)
        //........................S: O(K)
        PriorityQueue<Input> minHeap = new PriorityQueue<>((a, b) -> arr[a.row][a.col] - arr[b.row][b.col]);
        //minHeap will hold start coordinate(row, col) for all the elements in each row not the total R*C elements directly
        for (int r = 0; r < arr.length; r++) {
            if (arr[r].length > 0) {
                minHeap.add(new Input(r, 0, arr[r].length));
            }
        }
        //after this loop minHeap will have K instance of Input() holding (row, col)  for each row
        //minHeap.size() == K

        List<Integer> sortedList = new ArrayList<>();
        while (!minHeap.isEmpty()) {

            //At any point of time we poll Input from minHeap
            Input in = minHeap.poll();
            //we put the element at that row, col
            sortedList.add(arr[in.row][in.col]);
            //if new col is less than its col length
            //we update col and put that updated input(in) in minHeap back
            if (in.col + 1 < in.colLength) {
                in.col++;
                minHeap.add(in);
            }
        }

        //output:
        System.out.println("K sorted array into a list: " + sortedList);
    }

    public void kThLargestSumFromContigousSubarray(int[] arr, int K) {
        //................................T: O(N^2 * LogK)
        //................................S: O(K), only K elements are stored at a time in heap
        //https://www.geeksforgeeks.org/k-th-largest-sum-contiguous-subarray/
        //arr[]: [20, -5, -1]
        //contSumSubarry: [20, 15, 14, -5, -6, -1]
        //20, 20+(-5), 20+(-5)+(-1), -5, -5+(-1), -1 
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        //generating subarrays
        for (int i = 0; i < arr.length; i++) {
            int contSum = 0;
            for (int j = i; j < arr.length; j++) {
                contSum += arr[j];
                minHeap.add(contSum);
                if (minHeap.size() > K) {
                    minHeap.poll();
                }
            }
        }
        System.out.println("kth largest sum from contigous subarray: " + minHeap.peek());
    }

    public void majorityElement_1(int[] a) {

        //.............T: O(N)
        //.............S: O(Unique ele in a)
        int maj = a.length / 2;

        Map<Integer, Integer> map = new HashMap<>();
        for (int x : a) {
            map.put(x, map.getOrDefault(x, 0) + 1);
        }

        for (Map.Entry<Integer, Integer> e : map.entrySet()) {
            if (e.getValue() > maj) {
                System.out.println("Majority element: " + e.getKey());
                return;
            }
        }

        System.out.println("Majority element: -1");
    }

    public void majorityElement_2(int[] a) {

        //Moore’s Voting Algorithm
        //https://www.geeksforgeeks.org/majority-element/
        //..........T: O(N)
        //..........S: O(1)
        //finding candidate
        int majorIndex = 0;
        int count = 1;
        int i;
        for (i = 1; i < a.length; i++) {
            if (a[majorIndex] == a[i]) {
                count++;
            } else {
                count--;
            }
            if (count == 0) {
                majorIndex = i;
                count = 1;
            }
        }
        int cand = a[majorIndex];

        //validating the cand 
        count = 0;
        for (i = 0; i < a.length; i++) {
            if (a[i] == cand) {
                count++;
            }
        }
        if (count > a.length / 2) {
            System.out.println("Majority element: " + cand);
        } else {
            System.out.println("Majority element: -1");
        }

    }

    public void mergeTwoSortedArraysWithoutExtraSpace(int[] arr1, int[] arr2, int m, int n) {

        //https://www.geeksforgeeks.org/merge-two-sorted-arrays-o1-extra-space/
        // Iterate through all elements of ar2[] starting from 
        // the last element 
        for (int i = n - 1; i >= 0; i--) {
            /* Find the smallest element greater than ar2[i]. Move all 
             elements one position ahead till the smallest greater 
             element is not found */
            int j, last = arr1[m - 1];
            for (j = m - 2; j >= 0 && arr1[j] > arr2[i]; j--) {
                arr1[j + 1] = arr1[j];
            }

            // If there was a greater element 
            if (j != m - 2 || last > arr2[i]) {
                arr1[j + 1] = arr2[i];
                arr2[i] = last;
            }
        }

        //output
        for (int x : arr1) {
            System.out.print(x + " ");
        }
        System.out.println();
        for (int x : arr2) {
            System.out.print(x + " ");
        }

        System.out.println();
    }

    private int findFirstOccurenceKInSortedArray(int[] arr, int K, int start, int end, int N) {
        if (end >= start) {
            int mid = start + (end - start) / 2;
            if ((mid == 0 || K > arr[mid - 1]) && arr[mid] == K) {
                return mid;
            } else if (K > arr[mid]) {
                return findFirstOccurenceKInSortedArray(arr, K, (mid + 1), end, N);
            } else {
                return findFirstOccurenceKInSortedArray(arr, K, start, (mid - 1), N);
            }
        }
        return -1;
    }

    private int findLastOccurenceKInSortedArray(int[] arr, int K, int start, int end, int N) {
        if (end >= start) {
            int mid = start + (end - start) / 2;
            if ((mid == N - 1 || K < arr[mid + 1]) && arr[mid] == K) {
                return mid;
            } else if (K < arr[mid]) {
                return findLastOccurenceKInSortedArray(arr, K, start, (mid - 1), N);
            } else {
                return findLastOccurenceKInSortedArray(arr, K, (mid + 1), end, N);
            }
        }
        return -1;
    }

    public void findFirstAndLastOccurenceOfKInSortedArray(int[] arr, int K) {
        //https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
        //https://leetcode.com/problems/find-target-indices-after-sorting-array/
        int N = arr.length;
        int first = findFirstOccurenceKInSortedArray(arr, K, 0, N - 1, N);
        int last = findLastOccurenceKInSortedArray(arr, K, 0, N - 1, N);

        System.out.println(K + " first and last occurence: " + first + " " + last);
    }

    public int searchInRotatedSortedArray(int[] arr, int K) {
        //https://leetcode.com/problems/search-in-rotated-sorted-array
        //explanation: https://youtu.be/oTfPJKGEHcc
        int start = 0;
        int end = arr.length - 1;
        int N = arr.length;
        int mid = -1;

        while (end >= start) {

            mid = start + (end - start) / 2;
            if (arr[mid] == K) {
                return mid;
            }
            //left sorted section
            if (arr[start] <= arr[mid]) {
                //if target lie in left sorted section
                //then reduce end and search in this particular region
                //else move to right sorted section
                if (K >= arr[start] && K < arr[mid]) {
                    end = mid - 1;
                } else {
                    start = mid + 1;
                }

            } else {
                //right sorted section
                //if target lie in right sorted section 
                //then update start and search in this particular region
                //else move to left sorted section
                if (K > arr[mid] && K <= arr[end]) {
                    start = mid + 1;
                } else {
                    end = mid - 1;
                }
            }
        }

        return -1;
    }

    public int searchInRotatedSortedArrayWithDuplicateArrayElement(int[] arr, int K) {

        int start = 0;
        int end = arr.length - 1;
        int N = arr.length;
        int mid = -1;

        while (end >= start) {

            //shift front till arr[f] is same as arr[f+1]
            //when f and f+1 elements are diff loop will end
            while (start < end && arr[start] == arr[start + 1]) {
                start++;
            }
            //shift last till arr[l] is same as arr[l-1]
            //when l and l-1 elements are diff loop will end
            while (start < end && arr[end] == arr[end - 1]) {
                end--;
            }

            mid = start + (end - start) / 2;
            if (arr[mid] == K) {
                return mid;
            }

            if (arr[start] <= arr[mid]) {

                if (K >= arr[start] && K < arr[mid]) {
                    end = mid - 1;
                } else {
                    start = mid + 1;
                }

            } else {
                if (K > arr[mid] && K <= arr[end]) {
                    start = mid + 1;
                } else {
                    end = mid - 1;
                }
            }
        }

        return -1;
    }

    public void findRepeatingAndMissingInUnsortedArray_1(int[] arr) {

        //problem statement: https://www.geeksforgeeks.org/find-a-repeating-and-a-missing-number/
        //arr: will be of size N and elements in arr[] will be [1..N]
        //.......................T: O(N)
        //.......................S: O(N)
        System.out.println("Approach 1");
        int[] count = new int[arr.length + 1];
        //get the occurence of arr element in count[] where count[i] i: elements in arr
        for (int i = 0; i < arr.length; i++) {
            count[arr[i]]++;
        }

        for (int i = 1; i < count.length; i++) {
            //first ith index that has count[i] = 0 is the element in arr which is supposed to be missing
            //count[i] == 0 => i = element in arr is supposed to be missing
            if (count[i] == 0) {
                System.out.println("Missing: " + i);
                break;
            }
        }

        for (int i = 1; i < count.length; i++) {
            //first ith index which has count[i] > 1 (occuring more that 1)
            //is the element which is repeating
            //count[i] > 1 => i = element in arr which is repeating
            if (count[i] > 1) {
                System.out.println("Repeating: " + i);
                break;
            }
        }
    }

    public void findRepeatingAndMissingInUnsortedArray_2(int[] arr) {

        //problem statement: https://www.geeksforgeeks.org/find-a-repeating-and-a-missing-number/
        //explanation: https://youtu.be/aMsSF1Il3IY
        //OPTIMISED
        //.......................T: O(N)
        //.......................S: O(1)
        System.out.println("Approach 2");
        System.out.println("Repeating element: ");
        for (int val : arr) {
            int absVal = Math.abs(val);
            if (arr[absVal - 1] > 0) {
                arr[absVal - 1] = -arr[absVal - 1];
            } else {
                System.out.println(absVal);
            }
        }

        System.out.println("Missing element: ");
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] > 0) {
                System.out.println(i + 1);
            }
        }
    }

    public boolean checkIfPairPossibleInArrayHavingGivenDiff(int[] arr, int diff) {

        //..................T; O(N)
        //..................S: O(N)
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < arr.length; i++) {
            //arr[x] - arr[y] = diff
            //arr[x] = diff + arr[y]
            //if set.contains(arr[y]) then pair is possible
            if (set.contains(arr[i])) {
                return true;
            }

            //arr[x] = arr[y] +diff
            set.add(arr[i] + diff);
        }

        return false;
    }

    private double squareRootOfANumber_PreciseDoubleValue_BinarySearch(double n, double start, double end) {

        if (end >= start) {

            double mid = start + (end - start) / 2.0;
            double sqr = mid * mid;

            if (sqr == n || Math.abs(n - sqr) < 0.00001) {
                return mid;
            } else if (sqr < n) {
                return squareRootOfANumber_PreciseDoubleValue_BinarySearch(n, mid, end);
            } else {
                return squareRootOfANumber_PreciseDoubleValue_BinarySearch(n, start, mid);
            }
        }

        return 1.0;
    }

    public double squareRootOfANumber_PreciseDoubleValue(double n) {

        //https://leetcode.com/problems/sqrtx/
        if (n == 0.0 || n == 1.0) {
            return n;
        }

        double i = 1;
        while (true) {
            double sqr = i * i;
            if (sqr == n) {
                return i;
            } else if (sqr > n) {
                //at this point where sqr of i is > n then that means sqr root for n lies b/w 
                // i-1 and i
                //ex sqrt(3) == 1.73 (lie b/w 1 and 2)
                // i = 1 sqr = 1*1 = 1
                //i = 2 sqr = 2*2 = 4
                //4 > n i.e 4 > 3 that means sqrt(3) lie in b/w 1 and 2
                //so we will do binary search i-1, i (1, 2)
                return squareRootOfANumber_PreciseDoubleValue_BinarySearch(n, i - 1, i);
            }
            i++;
        }
    }

    public int squareRootOfANumber_RoundedIntValue(int x) {

        //https://leetcode.com/problems/sqrtx/
        long start = 0;
        long end = (x / 2) + 1;

        while (end > start) {

            long mid = start + (end - start) / 2 + 1;
            long sqr = mid * mid;
            if (sqr == x) {
                return (int) mid;
            } else if (x < sqr) {
                end = mid - 1;
            } else {
                start = mid;
            }
        }
        return (int) start;
    }

    public void kThElementInTwoSortedArrays_1(int[] a, int[] b, int K) {

        //HEAP SORTING AND MERGING
        //...........................T: O((N+M).Log(N+M))
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for (int a_ : a) {
            minHeap.add(a_);
        }

        for (int b_ : b) {
            minHeap.add(b_);
        }

        //pop out K element from heap
        int kthElement = -1;
        while (K-- != 0 && !minHeap.isEmpty()) {
            kthElement = minHeap.poll();
        }

        //output:
        System.out.println("Kth element in two sorted arrays: " + kthElement);
    }

    public void kThElementInTwoSortedArrays_2(int[] a, int[] b, int K) {

        //Two arrays are already sorted
        //...........................T: O(K)
        int iK = 0;
        int m = a.length;
        int n = b.length;
        int i = 0;
        int j = 0;
        int kthElement = -1;

        while (i < m && j < n) {

            if (a[i] < b[j]) {
                iK++;
                if (iK == K) {
                    kthElement = a[i];
                    break;
                }
                i++;
            } else {
                iK++;
                if (iK == K) {
                    kthElement = b[j];
                    break;
                }
                j++;
            }
        }

        //if loop ends beacuse we run out of one array element
        while (i < m) {
            iK++;
            if (iK == K) {
                kthElement = a[i];
            }
            i++;
        }

        while (j < n) {
            iK++;
            if (iK == K) {
                kthElement = b[j];
            }
            j++;
        }

        //output:
        System.out.println("Kth element in two sorted arrays: " + kthElement);
    }

    public void findMinimumInRotatedSortedArray(int[] arr) {

        //explanation: https://youtu.be/IzHR_U8Ly6c
        int n = arr.length;
        int start = 0;
        int end = n - 1;
        int minElement = Integer.MIN_VALUE;
        while (end >= start) {
            int mid = start + (end - start) / 2;

            if (mid > 0 && arr[mid] < arr[mid - 1]) {
                minElement = arr[mid];
                break;
            } else if (arr[start] <= arr[mid] && arr[mid] > arr[end]) {
                start = mid + 1;
            } else {
                end = mid - 1;
            }
        }

        System.out.println("Minimum in rotated sorted array: "
                + (minElement == Integer.MIN_VALUE ? arr[start] : minElement));
    }

    public void findLocalMinima(int[] arr) {

        int start = 0;
        int end = arr.length - 1;
        int mid = 0;
        while (end >= start) {

            mid = start + (end - start) / 2;

            if ((mid == 0 || arr[mid - 1] > arr[mid])
                    && (mid == arr.length - 1 || arr[mid + 1] > arr[mid])) {
                break;
            } else if (mid > 0 && arr[mid - 1] < arr[mid]) {
                end = mid - 1;
            } else {
                start = mid + 1;
            }
        }

        //output
        System.out.println("Local minima at index: " + mid + " element: " + arr[mid]);
    }

    public void findLocalMaxima(int[] arr) {

        //LOCAL MAXIMA OR PEAK ELEMENT
        //https://leetcode.com/problems/find-peak-element/
        int start = 0;
        int end = arr.length - 1;
        int mid = 0;
        while (end >= start) {

            mid = start + (end - start) / 2;

            if ((mid == 0 || arr[mid - 1] < arr[mid])
                    && (mid == arr.length - 1 || arr[mid + 1] < arr[mid])) {
                break;
            } else if (mid > 0 && arr[mid - 1] > arr[mid]) {
                end = mid - 1;
            } else {
                start = mid + 1;
            }
        }

        //output
        System.out.println("Local maxima at index: " + mid + " element: " + arr[mid]);
    }

    private int countElementsFromSecondArrayLessOrEqualToElementInFirstArray_FindLastOccurenceOfX(int[] arr,
            int x, int start, int end) {

        //MODIFIED BINARY SERACH FOR THIS QUESTION
        //SIMILAR TO findLastOccurenecOfKInSortedArray()
        if (end >= start) {

            int mid = start + (end - start) / 2;

            if ((mid == arr.length - 1 || x < arr[mid + 1]) && arr[mid] == x) {
                return mid;
            } else if (x < arr[mid]) {
                return countElementsFromSecondArrayLessOrEqualToElementInFirstArray_FindLastOccurenceOfX(
                        arr, x, start, mid - 1);
            } else {
                return countElementsFromSecondArrayLessOrEqualToElementInFirstArray_FindLastOccurenceOfX(
                        arr, x, mid + 1, end);
            }
        }
        return end;
    }

    public void countElementsFromSecondArrayLessOrEqualToElementInFirstArray(int[] first, int[] second) {

        /*
         Brute force : T: O(N^2)
         `1. Use 2 for loop: 
         -> i for first[] 
         ---> count = 0
         ---> j for second[]
         ------> if second[j] <= first[i]: count++
         ---> print: count
         */
        //............................T: O((M + N) * LogN) where M = first.length, N = second.length
        //............................S: O(1)
        //https://www.geeksforgeeks.org/element-1st-array-count-elements-less-equal-2nd-array/
        Arrays.sort(second);
        List<Integer> result = new ArrayList<>();
        for (int x : first) {
            int index = countElementsFromSecondArrayLessOrEqualToElementInFirstArray_FindLastOccurenceOfX(
                    second, x, 0, second.length - 1);
            result.add(index + 1);
        }

        //output
        System.out.println("Count: " + result);
    }

    private int[] KMP_PatternMatching_Algorithm_LPSArray(String pattern, int size) {

        int[] lps = new int[size];
        lps[0] = 0; //always 0th index is 0
        int suffixIndex = 1;
        int prefixIndex = 0;
        while (suffixIndex < size) {

            if (pattern.charAt(prefixIndex) == pattern.charAt(suffixIndex)) {
                prefixIndex++;
                lps[suffixIndex] = prefixIndex;
                suffixIndex++;
            } else if (prefixIndex == 0) {
                lps[suffixIndex] = prefixIndex; // prefixIndex == 0
                suffixIndex++;
            } else {
                prefixIndex = lps[prefixIndex - 1];
            }
        }

        return lps;
    }

    public void KMP_PatternMatching_Algorithm(String text, String pattern) {
        //https://leetcode.com/problems/implement-strstr/
        //explanation: https://youtu.be/JoF0Z7nVSrA
        //explanation: https://youtu.be/ziteu2FpYsA
        int textLen = text.length();
        int patternLen = pattern.length();

        //create LPS array for pattern
        int[] lps = KMP_PatternMatching_Algorithm_LPSArray(pattern, patternLen);

        boolean atleastOneMatchFound = false;
        //text and pattern matching
        int textIndex = 0; // index for text
        int patternIndex = 0; // index for pattern
        while (textIndex < textLen) {

            if (textIndex < textLen && patternIndex < patternLen
                    && text.charAt(textIndex) == pattern.charAt(patternIndex)) {
                textIndex++;
                patternIndex++;
            } else if (patternIndex == 0) {
                textIndex++;
            } else {
                patternIndex = lps[patternIndex - 1];
            }

            if (patternIndex == patternLen) {
                //if atleast one match is found and want to stop matching further then simply do
                //return textIndex - patternLen;

                //this below logic will allow to find all matches in text
                System.out.println("Pattern matched at: " + (textIndex - patternLen));
                atleastOneMatchFound = true;
            }
        }
        if (!atleastOneMatchFound) {
            System.out.println("Pattern not matched");
        }
        //if pattern is not found
        //return -1;
    }

    public int editDistance_Recursion(String s1, String s2, int m, int n) {

        //https://www.geeksforgeeks.org/edit-distance-dp-5/
        //explanation: https://youtu.be/XYi2-LPrwm4
        //https://youtu.be/MiqoA-yF-0M
        //if s1 is empty then whole s2 is to be inserted to convert s1 to s2
        if (m == 0) {
            return n;
        }

        //if s2 is empty then whole s1 is to be deleted to convert s1 to s2
        if (n == 0) {
            return m;
        }

        //if last char of two strings matches then just move ahead one char in both
        if (s1.charAt(m - 1) == s2.charAt(n - 1)) {
            return editDistance_Recursion(s1, s2, m - 1, n - 1);
        }

        //if the char doesn't matches then take the min of below 3
        return Math.min(editDistance_Recursion(s1, s2, m, n - 1), //insert
                Math.min(editDistance_Recursion(s1, s2, m - 1, n), //delete
                        editDistance_Recursion(s1, s2, m - 1, n - 1))) // replace
                + 1;

    }

    public int editDistance_DP_Memoization(String s1, String s2) {

        //https://www.geeksforgeeks.org/edit-distance-dp-5/
        //explanation: https://youtu.be/XYi2-LPrwm4
        //https://youtu.be/MiqoA-yF-0M
        int m = s1.length();
        int n = s2.length();
        int[][] memo = new int[m + 1][n + 1];

        //base cond
        //in order to convert s1 to s2
        //if s1 is "" and s2 is valid then we need to INSERT char in s1 to be s2
        //if s1 is valid but s2 is "" then we need to DELETE char is s1 to be s2
        for (int x = 0; x < m + 1; x++) {
            for (int y = 0; y < n + 1; y++) {
                if (x == 0) {
                    memo[x][y] = y;
                } else if (y == 0) {
                    memo[x][y] = x;
                }
            }
        }

        for (int x = 1; x < m + 1; x++) {
            for (int y = 1; y < n + 1; y++) {
                if (s1.charAt(x - 1) == s2.charAt(y - 1)) {
                    memo[x][y] = memo[x - 1][y - 1];
                } else {
                    memo[x][y] = 1 + Math.min(
                            memo[x][y - 1],
                            Math.min(
                                    memo[x - 1][y],
                                    memo[x - 1][y - 1]));
                }
            }
        }
        return memo[m][n];
    }

    public int coinChange_Recursion(int[] coins, int N, int K) {

        //if no coins available there no way we can make any change
        if (N == 0) {
            return 0;
        }

        //if we are not making any change K despite having N coins, K = 0 can still be made
        if (K == 0) {
            return 1;
        }

        //if a coin value is already greater than the change K we need to make
        //then that coin is not useful, just move ahead of that coin
        if (coins[N - 1] > K) {
            return coinChange_Recursion(coins, N - 1, K);
        }

        //now we have two choices, 
        //1. take the coin and adjust the K value by the coins value
        //2. don't take the coin and don't adjust K just move ahead of that coin
        //total ways we can make change K = 1 + 2 choices
        return (coinChange_Recursion(coins, N, K - coins[N - 1]) + coinChange_Recursion(coins, N - 1, K));
    }

    public void coinChange_DP_Memoization(int[] coins, int amount) {
        //https://leetcode.com/problems/coin-change-2/
        int N = coins.length;
        int[][] memo = new int[N + 1][amount + 1];

        //base
        //if(N == 0)
        for (int col = 0; col < amount + 1; col++) {
            memo[0][col] = 0;
        }
        //if(K == 0)
        for (int row = 0; row < N + 1; row++) {
            memo[row][0] = 1;
        }

        for (int x = 1; x < N + 1; x++) {
            for (int y = 1; y < amount + 1; y++) {

                if (coins[x - 1] > y) {
                    memo[x][y] = memo[x - 1][y];
                } else {
                    memo[x][y] = memo[x][y - coins[x - 1]] + memo[x - 1][y];
                }
            }
        }

        System.out.println("Possible ways to make coin change: " + memo[N][amount]);
    }

    public int knapSack01_Recusrion(int W, int[] weight, int[] value, int N) {

        //if either the value[] is empty(N == 0) we will not be able to make any profit
        //or the knapSack bag don't have the capacity(W == 0) then in that case profit is 0
        if (N == 0 || W == 0) {
            return 0;
        }

        //if the weight of a product is more than the knapSack capacity(W) then
        //in that case we have to just ignore that and move to another product
        if (weight[N - 1] > W) {
            return knapSack01_Recusrion(W, weight, value, N - 1);
        }

        //we now have 2 descision to make, we have to take max of these 2 descision
        //1. we can pick up a product add its value[product] in our profit 
        //and adjust knapSack capacity(W - weight[product]) and move to another product(N-1)
        //2. we can simply ingore this product and just move to another product(N-1)
        return Math.max(
                value[N - 1] + knapSack01_Recusrion(W - weight[N - 1], weight, value, N - 1),
                knapSack01_Recusrion(W, weight, value, N - 1));
    }

    public void knapSack01_DP_Memoization(int W, int[] weight, int[] value, int N) {

        int[][] memo = new int[N + 1][W + 1];

        //base cond
        for (int x = 0; x < N + 1; x++) {
            for (int y = 0; y < W + 1; y++) {
                //No product x == N == 0
                //No knapSack capacity y == W == 0
                if (x == 0 || y == 0) {
                    memo[x][y] = 0;
                }
            }
        }

        for (int x = 1; x < N + 1; x++) {
            for (int y = 1; y < W + 1; y++) {
                if (weight[x - 1] > y) {
                    memo[x][y] = memo[x - 1][y];
                } else {
                    memo[x][y] = Math.max(
                            value[x - 1] + memo[x - 1][y - weight[x - 1]],
                            memo[x - 1][y]);
                }
            }
        }

        System.out.println("The maximum profit with given knap sack: " + memo[N][W]);
    }

    public boolean subsetSum_Recursion(int[] arr, int sum, int N) {

        //if arr is empty and sum to prove is also 0 then in that case sum = 0 possible
        //as empty arr denotes empty sub set {}, {} which default sums up as 0
        if (N == 0 && sum == 0) {
            return true;
        }

        //if arr is empty and sum to prove is a non - zero number( >= 1) then in that case this given sum can't have
        //any sub set from arr as it is already empty
        if (N == 0 && sum != 0) {
            return false;
        }

        //if arr is not empty but the element are greater than sum then that element can't be used as sub set
        //just move to next element
        if (arr[N - 1] > sum) {
            return subsetSum_Recursion(arr, sum, N - 1);
        }

        //we now have 2 descision to make, any of the 2 descision makes the subset then pick that(OR operator)
        //1. we pick an element from arr and assume that it makes the subset then that sum is also to be reduced
        //to sum - arr[N-1]
        //2. we just leave this element from the array and move to next element
        return subsetSum_Recursion(arr, sum - arr[N - 1], N - 1) || subsetSum_Recursion(arr, sum, N - 1);

    }

    public void subsetSum_DP_Memoization(int[] arr, int sum, int N) {

        boolean[][] memo = new boolean[N + 1][sum + 1];

        //base cond
        for (int x = 0; x < N + 1; x++) {
            for (int y = 0; y < sum + 1; y++) {
                //if array is empty then any given sum is not possible (except sum == 0) 
                if (x == 0) {
                    memo[x][y] = false;
                }

                //if the given sum is just 0 then it can be prove even if the arrays is empty or full
                if (y == 0) {
                    memo[x][y] = true;
                }
            }
        }

        for (int x = 1; x < N + 1; x++) {
            for (int y = 1; y < sum + 1; y++) {
                if (arr[x - 1] > y) {
                    memo[x][y] = memo[x - 1][y];
                } else {
                    memo[x][y] = memo[x - 1][y - arr[x - 1]] || memo[x - 1][y];
                }
            }
        }

        System.out.println("The sub set for the given sum is possible: " + memo[N][sum]);
    }

    public void equalsSumPartition_SubsetSum(int[] arr, int N) {

        int arrSum = 0;
        for (int ele : arr) {
            arrSum += ele;
        }

        if (arrSum % 2 == 1) {
            //if odd no equal partition is possble for the given sum
            //arr = {1,5,5,11} arrSum = 22 == even can be divided into 2 half as {11}, {1,5,5}
            //if arrSum = 23 == odd no equal partition possible
            System.out.println("The equal sum partition for the given sum is not possbile as sum of array is odd");
            return;
        }

        System.out.println("The equal sum partition for the given array is possbile: ");
        //if arrSum == even the if we can prove the sum = arrSum/2 is possible
        //then other half of the sub set is by default will be eqaul to arrSum/2
        //arrSum = 22 == sum = arrSum/2 = 11 prove {11} then other half will be {1,5,5}
        subsetSum_DP_Memoization(arr, arrSum / 2, N);
    }

    public int longestCommonSubsequence_Recursion(String s1, String s2, int m, int n) {

        if (m == 0 || n == 0) {
            return 0;
        }

        if (s1.charAt(m - 1) == s2.charAt(n - 1)) {
            return longestCommonSubsequence_Recursion(s1, s2, m - 1, n - 1) + 1;
        }

        return Math.max(longestCommonSubsequence_Recursion(s1, s2, m, n - 1),
                longestCommonSubsequence_Recursion(s1, s2, m - 1, n));

    }

    public int longestCommonSubsequence_DP_Memoization(String s1, String s2, int m, int n) {
        //https://leetcode.com/problems/longest-common-subsequence
        int[][] memo = new int[m + 1][n + 1];

        //base cond
        //if s1 is empty and s2 is non-empty String no subseq length is possible
        //if s2 is empty and s1 is non-empty Strng no subseq length is possible
        for (int[] r : memo) {
            Arrays.fill(r, 0);
        }

        for (int x = 1; x < m + 1; x++) {
            for (int y = 1; y < n + 1; y++) {
                if (s1.charAt(x - 1) == s2.charAt(y - 1)) {
                    memo[x][y] = memo[x - 1][y - 1] + 1;
                } else {
                    memo[x][y] = Math.max(memo[x][y - 1], memo[x - 1][y]);
                }
            }
        }

        System.out.println("The longest common subsequence length for the given two string is: " + memo[m][n]);
        return memo[m][n];
    }

    public void uncrossedLines_DP_Memoization(int[] nums1, int[] nums2) {
        //https://leetcode.com/problems/uncrossed-lines/
        //this questions is also longestCommonSubsequence between two int arrays
        //completely based on longestCommonSubsequence_DP_Memoization
        int n = nums1.length;
        int m = nums2.length;

        int[][] memo = new int[n + 1][m + 1];
        //base cond is if there are no elements in nums1 then
        //there is no lines possible with nums2 similarly if there
        //no elements in nums2 there is no lines possible with nums1
        //so memo[r == 0][c] = 0 and memo[r][c == 0] = 0
        for (int x = 1; x < n + 1; x++) {
            for (int y = 1; y < m + 1; y++) {
                if (nums1[x - 1] == nums2[y - 1]) {
                    //here adding 1 represents that one line between two arrays is made
                    memo[x][y] = memo[x - 1][y - 1] + 1;
                } else {
                    //here we need to choose the max uncrossed lines possible
                    //out of all the combinations
                    memo[x][y] = Math.max(memo[x - 1][y], memo[x][y - 1]);
                }
            }
        }
        //output
        System.out.println("Max uncrossed lines possible with given two num arrays: " + memo[n][m]);
    }

    public void longestPallindromicSubsequence_DP_Memoization(String s) {
        //https://leetcode.com/problems/longest-palindromic-subsequence/
        int len = s.length();
        String rev = new StringBuilder(s).reverse().toString();
        int longestPallindromicSubseq = longestCommonSubsequence_DP_Memoization(s, rev, len, len);
        System.out.println("The longest pallindromic subsequences: "
                + longestPallindromicSubseq);
    }

    public void deleteOperationOfTwoStrings_DP_Memoization(String str1, String str2) {
        //https://leetcode.com/problems/delete-operation-for-two-strings/
        /*
         how many char we need to delete of insert to make str1 transformed to str2
         //ex: str1 = "sea", str2 = "eat"
         longest common subseq = 2 ==> (ea)
         if both strings are combined = str1 + str2 = sea = eat ==> seaeat
         you can see in merges=d form of both strings the lcs come 2 times and if we
         remove these 2 occurences of lcs we will left with those chars
         that we either need to delete or insert
         seaeat - 2 * (ea) ==> st ==> delete(s) and insert(t)
         that's why len1 + len2 - 2 * lcs 
         */
        int len1 = str1.length();
        int len2 = str2.length();
        int longestCommonSubseq = longestCommonSubsequence_DP_Memoization(str1, str2, len1, len2);
        int deleteOprn = len1 + len2 - 2 * longestCommonSubseq;
        System.out.println("Delete operation of two strings: "
                + deleteOprn);
    }

    private int longestRepeatingSubsequence_Recursion_Helper(String a, String b, int m, int n) {

        if (m == 0 || n == 0) {
            return 0;
        } else if (a.charAt(m - 1) == b.charAt(n - 1) && m != n) {
            return longestRepeatingSubsequence_Recursion_Helper(a, b, m - 1, n - 1) + 1;
        }

        return Math.max(longestRepeatingSubsequence_Recursion_Helper(a, b, m, n - 1),
                longestRepeatingSubsequence_Recursion_Helper(a, b, m - 1, n));
    }

    public int longestRepeatingSubsequence_Recursion(String str, int N) {
        return longestRepeatingSubsequence_Recursion_Helper(str, str, N, N);
    }

    public void longestRepeatingSubsequence_DP_Memoization(String str) {

        int N = str.length();
        int[][] memo = new int[N + 1][N + 1];

        //base cond
        //if string length is 0 then no subseq is possible
        //here there is only one string so mem[x][y] where x == 0 OR y == 0 memo[x][y] = 0
        for (int x = 1; x < N + 1; x++) {
            for (int y = 1; y < N + 1; y++) {
                if (str.charAt(x - 1) == str.charAt(y - 1) && x != y) {
                    memo[x][y] = memo[x - 1][y - 1] + 1;
                } else {
                    memo[x][y] = Math.max(memo[x][y - 1], memo[x - 1][y]);
                }
            }
        }

        //output:
        System.out.println("Longest repeating subsequence: " + memo[N][N]);
    }

    public void longestCommonSubstring_DP_Memoization(String a, String b) {
        //https://leetcode.com/problems/maximum-length-of-repeated-subarray/
        int m = a.length();
        int n = b.length();

        int[][] memo = new int[m + 1][n + 1];

        //base cond: if any of the string is empty then common subtring is not possible
        //x == 0 OR y == 0 : memo[0][0] = 0
        int maxLenSubstring = 0;
        for (int x = 1; x < m + 1; x++) {
            for (int y = 1; y < n + 1; y++) {
                if (a.charAt(x - 1) == b.charAt(y - 1)) {
                    memo[x][y] = memo[x - 1][y - 1] + 1;
                    maxLenSubstring = Math.max(maxLenSubstring, memo[x][y]);
                } else {
                    memo[x][y] = 0;
                }
            }
        }

        //output:
        System.out.println("Longest common substring: " + maxLenSubstring);
    }

    public int maximumLengthOfPairChain_DP_Approach(int[][] pairs) {

        //https://leetcode.com/problems/maximum-length-of-pair-chain/solution/
        //.......................T: O(N^2)
        //.......................S: O(N)
        Arrays.sort(pairs, (a, b) -> a[0] - b[0]); //T: O(N.LogN)
        int N = pairs.length;
        int[] dp = new int[N];
        Arrays.fill(dp, 1);

        //T: O(N^2)
        for (int i = 1; i < N; i++) {
            for (int j = 0; j < i; j++) {
                if (pairs[j][1] < pairs[i][0]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
        }

        int ans = 0;
        for (int x : dp) {
            ans = Math.max(ans, x);
        }

        //overall T: O(N^2) as, N^2 > N.LogN
        return ans;
    }

    public int maximumLengthOfPairChain_Greedy_Approach(int[][] pairs) {

        //OPTIMISED
        //https://leetcode.com/problems/maximum-length-of-pair-chain/solution/
        //........................T: O(N.LogN)
        //........................S: O(1)
        Arrays.sort(pairs, (a, b) -> a[1] - b[1]); //T: O(N.LogN)
        int prevEnd = Integer.MIN_VALUE;
        int chain = 0;
        for (int[] currPair : pairs) { //T: O(N)
            int currStart = currPair[0];
            int currEnd = currPair[1];
            if (prevEnd < currStart) {
                prevEnd = currEnd;
                chain++;
            }
        }

        //overall T: O(N.LogN) as, N.LogN > N
        return chain;
    }

    public int findBinomialCoefficient_Recursion(int n, int r) {

        //https://www.geeksforgeeks.org/binomial-coefficient-dp-9/
        //this approach have overlapping subproblems
        //Binomial coefficient : nCr formula = n!/r!(n - r)!
        //if r = 0 OR r = n, ans: 1 as, 
        //r == 0: n!/0!.(n - 0)! => n!/n! => 1
        //r == n: n!/n!.(n - n)! => n!/n! => 1
        //0! = 1
        if (r > n) {
            return 0;
        }

        if (r == 0 || r == n) {
            return 1;
        }

        return findBinomialCoefficient_Recursion(n - 1, r - 1) + findBinomialCoefficient_Recursion(n - 1, r);
    }

    public void findBinomialCoefficient_DP_Memoization(int n, int r) {

        int[][] memo = new int[n + 1][r + 1];
        //base cond
        for (int x = 0; x < n + 1; x++) {
            for (int y = 0; y < r + 1; y++) {

                if (y > x) {
                    memo[x][y] = 0;
                } else if (y == 0 || y == x) {
                    memo[x][y] = 1;
                }
            }
        }

        for (int x = 1; x < n + 1; x++) {
            for (int y = 1; y < r + 1; y++) {
                memo[x][y] = memo[x - 1][y - 1] + memo[x - 1][y];
            }
        }

        //output:
        System.out.println("Binomial coefficient (nCr) DP way: " + memo[n][r]);
    }

    public int friendsPairingProblem_Recursion(int n) {

        //https://www.geeksforgeeks.org/friends-pairing-problem/
        //if no friend is there nothing is possible
        if (n == 0) {
            return 0;
        }

        //if 1 friend is avaialbe he can only remain single
        if (n == 1) {
            return 1;
        }

        //if 2 friends are available there can be two ways
        //friend can remain single: {2} Or can be be paired as {1,2}
        if (n == 2) {
            return 2;
        }

        //if above cond doesn't fulfil we have two choices
        //1. ether we can remain single fun(n-1)
        //2. Or we can keep ourself and check others for pair: (n-1)*fun(n-2)
        return friendsPairingProblem_Recursion(n - 1) + (n - 1) * friendsPairingProblem_Recursion(n - 2);

    }

    public void friendsPairingProblem_DP_Memoization(int n) {

        //.............................T: O(N)
        //.............................S: O(N)
        //https://www.geeksforgeeks.org/friends-pairing-problem/
        int[] memo = new int[n + 1];
        //base cond
        memo[0] = 0;
        memo[1] = 1;
        memo[2] = 2;

        for (int i = 3; i < n + 1; i++) {
            memo[i] = memo[i - 1] + (i - 1) * memo[i - 2];
        }

        //output
        System.out.println("No. ways freinds can be paired: " + memo[n]);
    }

    public void friendsPairingProblem(int n) {

        //.............................T: O(N)
        //.............................S: O(1)
        //OPTIMISED
        //https://www.geeksforgeeks.org/friends-pairing-problem/
        if (n <= 2) {
            System.out.println("No. ways freinds can be paired: " + n);
            return;
        }

        int a = 1;
        int b = 2;
        int c = 0;

        for (int i = 3; i <= n; i++) {

            c = b + (i - 1) * a;
            a = b;
            b = c;
        }

        //output
        System.out.println("No. ways freinds can be paired: " + c);
    }

    public int sticklerThief_Recursion(int[] houses, int n) {

        //if no houses is available
        if (n == 0) {
            return 0;
        }

        //if only one house is available
        if (n == 1) {
            return houses[n - 1];
        }

        //2 choices
        //1. we choose not to pick a house and we simply move to next house
        //2. we choose to pick that house then we have to add the amount 
        //in that house in our result and move to alternate house (which is not adjacent(n-2))
        //just choose the max of these choices
        return Math.max(sticklerThief_Recursion(houses, n - 1),
                houses[n - 1] + sticklerThief_Recursion(houses, n - 2));
    }

    public int sticklerThief_DP_Memoization(int[] houses) {
        //https://leetcode.com/problems/house-robber
        int n = houses.length;
        int[] memo = new int[n + 1];

        //base cond
        memo[0] = 0; //if no house is available
        memo[1] = houses[0]; //if only one house is available

        for (int i = 2; i < memo.length; i++) {
            memo[i] = Math.max(memo[i - 1], houses[i - 1] + memo[i - 2]);
        }

        //output;
        return memo[n];
    }

    public void sticklerThiefTwo_DP_Memoization(int[] houses) {
        //https://leetcode.com/problems/house-robber-ii/
        int n = houses.length;
        int result = Integer.MIN_VALUE;

        int[] currHouses = new int[n - 1];
        //house[0] to house[n - 2] in which last house (n - 1)-th will not be covered
        for (int i = 0; i < n - 1; i++) {
            currHouses[i] = houses[i];
        }

        result = Math.max(result, sticklerThief_DP_Memoization(currHouses));

        currHouses = new int[n];
        //house[1] to house[n - 1] in which first house(0-th) will not be covered
        for (int i = 1; i < n; i++) {
            currHouses[i] = houses[i];
        }

        result = Math.max(result, sticklerThief_DP_Memoization(currHouses));

        //output;
        System.out.println("The maximum amount stickler thief can pick from alternate but circular houses: "
                + result);
    }

    int longestIncreasingSubsequence_LongestSeqLength;

    private int longestIncreasingSubsequence_Recursion_Helper(int[] arr, int n) {

        //if there is one element in arr then the longest seq length is just one
        if (n == 1) {
            return 1;
        }

        int res = 0;
        int maxLengthHere = 1;

        for (int i = 1; i < n; i++) {

            res = longestIncreasingSubsequence_Recursion_Helper(arr, i);
            if (arr[i - 1] < arr[n - 1] && res + 1 > maxLengthHere) {
                maxLengthHere = res + 1;
            }
        }

        longestIncreasingSubsequence_LongestSeqLength = Math.max(
                longestIncreasingSubsequence_LongestSeqLength,
                maxLengthHere);

        return maxLengthHere;
    }

    public int longestIncreasingSubsequence_Recursion(int[] arr, int n) {
        //if array is empty no longest incr seq is possible hence -1,
        //otherwise atleast element will be considered as incr seq hence 1
        longestIncreasingSubsequence_LongestSeqLength = n == 0 ? -1 : 1;
        longestIncreasingSubsequence_Recursion_Helper(arr, n);
        return longestIncreasingSubsequence_LongestSeqLength;
    }

    public void longestIncreasingSubsequence_DP_Memoization(int[] arr, int n) {
        //https://leetcode.com/problems/longest-increasing-subsequence
        //https://leetcode.com/problems/number-of-longest-increasing-subsequence
        //if array is empty, no longest incr seq is possible hence -1,
        //otherwise atleast one element will be considered as incr seq hence 1
        int maxLengthLongestIncSubseq = n == 0 ? -1 : 1;
        //memo[i] will hold the longest incr subseq for ith arr[i] calculated
        int[] memo = new int[n];
        //base cond
        //a single num is also an increasing seq, that's why 1
        Arrays.fill(memo, 1);

        for (int i = 1; i < n; i++) {
            //we iterate over subarray [0 to i]
            for (int j = 0; j < i; j++) {
                //j loop will run over the above subarray
                //while looping in j will check what all 
                //arr[j] are lesser than arr[i] (i.e arr[i] > arr[j])
                //also need to have a check like, at any subarray if memo[i] 
                //already have a longest incr length memo[j] should not modify that
                //only when memo[i] <= memo[j]
                if (arr[i] > arr[j] && memo[i] <= memo[j]) {
                    memo[i] = memo[j] + 1;
                }
            }
            maxLengthLongestIncSubseq = Math.max(maxLengthLongestIncSubseq, memo[i]);
        }

        //output:
        System.out.println("DP Longest inc subseq of the given array is: " + maxLengthLongestIncSubseq);
    }

    public void maxSumIncreasingSubsequence_DP_Memoization(int[] arr) {
        //...............................T: O(N ^ 2), checking all subseq
        //...............................T: O(N), memo[] space
        //https://www.geeksforgeeks.org/maximum-sum-increasing-subsequence-dp-14/
        //https://practice.geeksforgeeks.org/problems/maximum-sum-increasing-subsequence4749/1/
        //approach similar to longestIncreasingSubsequence_DP_Memoization()
        int n = arr.length;
        //if array is empty no longest incr seq is possible hence -1,
        //otherwise atleast one element will be considered as incr seq hence 1
        int maxSumIncSubseq = n == 0 ? -1 : 1;
        //memo[i] will hold the max sum incr subseq for ith arr[i] calculated
        int[] memoSum = new int[n];
        //base cond
        //a single num can be a max sum incr seq, that's why arr[i]
        for (int i = 0; i < n; i++) {
            memoSum[i] = arr[i];
        }

        for (int i = 1; i < n; i++) {
            //we iterate over subarray [0 to i]
            for (int j = 0; j < i; j++) {
                //j loop will run over the above subarray
                //while looping in j will check what all 
                //arr[j] are lesser than arr[i] (i.e arr[i] > arr[j])
                //also need to have a check like, at any subarray if memo[i] 
                //already have a max sum, memo[j] should not modify that
                //only when memo[i] <= memo[j] + arr[i] that means max sum till 
                //memo[j] plus value of arr[i] makes the max sum curr max sum at memo[i]
                if (arr[i] > arr[j] && memoSum[i] <= memoSum[j] + arr[i]) {
                    memoSum[i] = memoSum[j] + arr[i];
                }
            }
            maxSumIncSubseq = Math.max(maxSumIncSubseq, memoSum[i]);
        }

        //output:
        System.out.println("DP Max sum incr subseq of the given array is: " + maxSumIncSubseq);
    }

    public void maximumLengthOfRepeatedSubarray_DP_Memoization(int[] arr1, int[] arr2) {

        //Approach is similar to longest common substring
        int m = arr1.length;
        int n = arr2.length;
        int[][] memo = new int[m + 1][n + 1];

        //base cond:
        //arr1 is empty no repeated values can be checked arr2
        //similarly arr2 is empty no repeated values can be checked with arr1
        //x == 0, y == 0 will 0
        int maxLen = 0;
        for (int x = 1; x < m + 1; x++) {
            for (int y = 1; y < n + 1; y++) {
                if (arr1[x - 1] == arr2[y - 1]) {
                    memo[x][y] = memo[x - 1][y - 1] + 1;
                    maxLen = Math.max(maxLen, memo[x][y]);
                } else {
                    memo[x][y] = 0;
                }
            }
        }

        //output
        System.out.println("Maximum length of repeated subarray: " + maxLen);
    }

    public int minInsertsToMakeStringPallindrome_DP_Memoization(String str) {
        //https://leetcode.com/problems/minimum-insertion-steps-to-make-a-string-palindrome
        //ref: SomePracticeQuestion.minNoOfInsertionInStringToMakeItPallindrome()
        String revString = new StringBuilder(str).reverse().toString();
        int len = str.length();
        int longestCommonSubseq = longestCommonSubsequence_DP_Memoization(str, revString, len, len);
        int inserts = len - longestCommonSubseq;
        //output
        System.out.println("Min inserts to make the string pallindrome: " + inserts);
        return inserts;
    }

    public void minDeletesToMakeStringPallindrome_DP_Memoization(String str) {
        //ref: SomePracticeQuestion.minNoOfDeletionInStringToMakeItPallindrome_LPSBasedApproach()
        int deletes = minInsertsToMakeStringPallindrome_DP_Memoization(str);
        //output
        System.out.println("Min deletes to make the string pallindrome: " + deletes);
    }

    public void paintFence_DP_Memoization(int fence, int paints) {
        //https://www.geeksforgeeks.org/painting-fence-algorithm/
        //explanation: https://youtu.be/ju8vrEAsa3Q
        long ways = paints;
        int mod = 1000000007;

        int paintSame = 0;
        int paintDifferent = paints;

        for (int i = 2; i <= fence; i++) {
            paintSame = paintDifferent;

            paintDifferent = (int) ways * (paints - 1);
            paintDifferent %= mod;

            ways = (paintSame + paintDifferent) % mod;
        }
        //output
        System.out.println("No of ways to paint fences such that adjacent fence are painted with same color: "
                + ways);
    }

    private int decodeWays_Recursive_Memoization_Helper(
            int index, String str, Map<Integer, Integer> cache) {

        int n = str.length();
        //if we have successfully crossed the string
        if (index >= n) {
            return 1;
        }
        //if we already cached the values previously retunn that
        if (cache.containsKey(index)) {
            return cache.get(index);
        }

        int ways = 0;

        int singleDigitNum = str.charAt(index) - '0';
        int twoDigitNum = 0;
        //we can form two digit num only when we are allowed to take (index + 1)th char
        //it should be less than n
        if (index + 1 < n) {
            //this two digit num should be formed with the index-th char which is singleDigitNum
            twoDigitNum = singleDigitNum * 10 + (str.charAt(index + 1) - '0');
        }

        //we have two decision to make to decode our nums in str
        //either 1. we can take the first single digit that is mapped to char[A == 1 to I == 9]
        //Or 2. we can take first two digit num that is mapped to char[J == 10 to Z == 26]
        if (singleDigitNum > 0) {
            //if we are considering just a single digit, then we simply move to next index val(index + 1)
            ways += decodeWays_Recursive_Memoization_Helper(index + 1, str, cache);
        }
        //to handle cases like "06" cond(singleDigitNum > 0 && twoDigitNum > 0)
        if (singleDigitNum > 0 && twoDigitNum > 0 && twoDigitNum <= 26) {
            //if we are considering first two digit, that means we took index-th & (index + 1)-th char
            //then we simply move to (index + 2)
            ways += decodeWays_Recursive_Memoization_Helper(index + 2, str, cache);
        }
        //cache the values
        cache.put(index, ways);
        return ways;
    }

    public void decodeWays_Recursive_Memoization(String str) {
        //https://leetcode.com/problems/decode-ways/
        //explanation: https://youtu.be/N5i7ySYQcgM
        //<index, ways>
        Map<Integer, Integer> cache = new HashMap<>();
        int ways = decodeWays_Recursive_Memoization_Helper(0, str, cache);
        //output
        System.out.println("Ways to decode string into alphabets(Recursive-Memoization): " + ways);
    }

    public void decodeWays_DP_Memoization(String str) {
        //https://leetcode.com/problems/decode-ways/
        //explanation: https://youtu.be/N5i7ySYQcgM
        int n = str.length();
        Map<Integer, Integer> cache = new HashMap<>();
        cache.put(n + 1, 1);
        cache.put(n, 1);
        for (int index = n - 1; index >= 0; index--) {
            int singleDigitNum = str.charAt(index) - '0';
            int twoDigitNum = 0;
            //we can form two digit num only when we are allowed to take (index + 1)th char
            //it should be less than n
            if (index + 1 < n) {
                //this two digit num should be formed with the index-th char which is singleDigitNum
                twoDigitNum = singleDigitNum * 10 + (str.charAt(index + 1) - '0');
            }

            if (singleDigitNum > 0) {
                int ways = cache.getOrDefault(index, 0) + cache.getOrDefault(index + 1, 0);
                cache.put(index, ways);
            }

            if (singleDigitNum > 0 && twoDigitNum > 0 && twoDigitNum <= 26) {
                int ways = cache.getOrDefault(index, 0) + cache.getOrDefault(index + 2, 0);
                cache.put(index, ways);
            }
        }
        //output
        System.out.println("Ways to decode string into alphabets(DP): " + cache.getOrDefault(0, 0));
    }

    private int fillingBooksInShelves_DP_Recusrive_Memoization_Helper(
            int[][] books, int shelfWidth, int index, Map<Integer, Integer> memo) {

        if (index >= books.length) {
            return 0;
        }

        if (memo.containsKey(index)) {
            return memo.get(index);
        }

        int currWidth = 0;
        int maxHeight = 0;

        for (int i = index; i < books.length; i++) {
            currWidth += books[i][0];
            maxHeight = Math.max(maxHeight, books[i][1]);

            if (currWidth > shelfWidth) {
                break;
            }

            int minHeight = Math.min(memo.getOrDefault(index, Integer.MAX_VALUE),
                    maxHeight + fillingBooksInShelves_DP_Recusrive_Memoization_Helper(
                            books, shelfWidth, i + 1, memo));

            memo.put(index, minHeight);
        }
        return memo.get(index);
    }

    public void fillingBooksInShelves_DP_Recusrive_Memoization(int[][] books, int shelfWidth) {
        //https://leetcode.com/problems/filling-bookcase-shelves/
        //https://leetcode.com/problems/filling-bookcase-shelves/discuss/2278530/Split-and-CHILL
        Map<Integer, Integer> memo = new HashMap<>();
        int minHeight = fillingBooksInShelves_DP_Recusrive_Memoization_Helper(books, shelfWidth, 0, memo);
        //output
        System.out.println("Min possible height: " + minHeight);
    }

    private int perfectSquares_DP_Recursive_Memoization_Helper(int n, Map<Integer, Integer> cache) {
        //if n == 0, there are 0 perfect squares that sums upto 0
        if (n == 0) {
            return n; // n == 0
        }

        if (cache.containsKey(n)) {
            return cache.get(n);
        }

        //this is the branching factor.
        //let say n == 12, sqrt(n) = 3.4.. roundOff = 3
        //Now this 3 * 3 == 9 and if we go one more like for 4 ==> 4 * 4 = 16
        //that means this 16 > n, we can't use 4 as perfect sqaures sums because
        //it always be greater than n
        int sqrRootN = (int) Math.sqrt(n);
        int currPerfectSquaresWaysToN = Integer.MAX_VALUE;
        for (int branch = 1; branch <= sqrRootN; branch++) {
            int currPerfectSqr = branch * branch;
            currPerfectSquaresWaysToN = Math.min(currPerfectSquaresWaysToN,
                    1 + perfectSquares_DP_Recursive_Memoization_Helper(n - currPerfectSqr, cache));
        }
        cache.put(n, currPerfectSquaresWaysToN);
        return currPerfectSquaresWaysToN;
    }

    public void perfectSquares_DP_Recursive_Memoization(int n) {
        //........................T: O(n * sqrt(n)), sqrt(n) is the branching factor for
        //each n in the decision tree
        //https://leetcode.com/problems/perfect-squares/submissions/
        //explanation: https://youtu.be/HLZLwjzIVGo
        //<n, currPerfectSquaresWays>
        Map<Integer, Integer> cache = new HashMap<>();
        int perfectSquaresCount = perfectSquares_DP_Recursive_Memoization_Helper(n, cache);
        //output
        System.out.println("Count of perfect squares that will sum up to n: " + perfectSquaresCount);
    }

    public void perfectSquares_DP_Memoization(int n) {
        //........................T: O(n * sqrt(n)), sqrt(n) is the branching factor for
        //each n in the decision tree
        //https://leetcode.com/problems/perfect-squares/submissions/
        //explanation: https://youtu.be/HLZLwjzIVGo
        //<n, currPerfectSquaresWays>
        Map<Integer, Integer> cache = new HashMap<>();
        //if n == 0, 0 are the ways to have sum of perfect squares
        cache.put(0, 0);

        for (int nth = 1; nth <= n; nth++) {
            for (int i = 1; i <= nth; i++) {
                int currPerfectSqr = i * i;
                if (nth - currPerfectSqr < 0) {
                    break;
                }
                int currPerfectSquaresWaysToN = Math.min(
                        cache.getOrDefault(nth, Integer.MAX_VALUE),
                        1 + cache.getOrDefault(nth - currPerfectSqr, Integer.MAX_VALUE));
                cache.put(nth, currPerfectSquaresWaysToN);
            }
        }
        //output
        System.out.println("Count of perfect squares that will sum up to n: " + cache.get(n));
    }

    private int outOfBoundaryPaths_DP_Recursive_Memoization_Helper(
            int m, int n, int maxMove, int row, int col, Map<String, Integer> cache) {
        //out of boundary path found return 1
        if (row < 0 || row >= m || col < 0 || col >= n) {
            return 1;
        }

        //if no moves are left return 0
        if (maxMove == 0) {
            return 0;
        }

        String cacheCoord = row + "," + col + "," + maxMove;

        if (cache.containsKey(cacheCoord)) {
            return cache.get(cacheCoord);
        }
        int mod = 1000000007;
        int currPath = 0;
        currPath += (outOfBoundaryPaths_DP_Recursive_Memoization_Helper(m, n, maxMove - 1, row - 1, col, cache)
                + outOfBoundaryPaths_DP_Recursive_Memoization_Helper(m, n, maxMove - 1, row + 1, col, cache)) % mod;

        currPath += (outOfBoundaryPaths_DP_Recursive_Memoization_Helper(m, n, maxMove - 1, row, col - 1, cache)
                + outOfBoundaryPaths_DP_Recursive_Memoization_Helper(m, n, maxMove - 1, row, col + 1, cache)) % mod;
        cache.put(cacheCoord, currPath % mod);
        return cache.get(cacheCoord);
    }

    public void outOfBoundaryPaths_DP_Recursive_Memoization(int m, int n, int maxMove, int startRow, int startCol) {
        //https://leetcode.com/problems/out-of-boundary-paths/
        Map<String, Integer> cache = new HashMap<>();
        int paths = outOfBoundaryPaths_DP_Recursive_Memoization_Helper(m, n, maxMove, startRow, startCol, cache);
        //output:
        System.out.println("Out of boundary paths:  " + paths);
    }

    private int frogJump_Recursive_Helper(int[] heights, int index) {

        //if no heights is given the energy consumed for the frog
        //from jumping nth height to n-1th or n-2th height becomes 0
        //because there is no such heights
        //also we are coming back from nth step to 0th step
        if (index <= 0) {
            return 0;
        }

        //energy consumed by frog if it is only going 1 step back (i.e, n - 1 steps)
        int oneStepBack = Math.abs(heights[index] - heights[index - 1])
                + frogJump_Recursive_Helper(heights, index - 1);

        //energy consumed by frog if it is only going 2 step back (i.e, n - 2 steps)
        //default value is Int.MAX because this time frog has to jump 2 steps back
        //and possibly n - 2 steps doesn't exist
        int twoStepBack = Integer.MAX_VALUE;
        //if it is possible to go n - 2 steps back then only calculate the energy consumed
        if (index - 2 >= 0) {
            twoStepBack = Math.abs(heights[index] - heights[index - 2])
                    + frogJump_Recursive_Helper(heights, index - 2);
        }

        //return the min energy consumed from both the steps
        return Math.min(oneStepBack, twoStepBack);
    }

    private int frogJump_Recursive_Memoization(int[] heights, int index, Map<Integer, Integer> cache) {

        //if no heights is given the energy consumed for the frog
        //from jumping nth height to n-1th or n-2th height becomes 0
        //because there is no such heights
        //also we are coming back from nth step to 0th step
        if (index <= 0) {
            return 0;
        }

        if (cache.containsKey(index)) {
            return cache.get(index);
        }

        //energy consumed by frog if it is only going 1 step back (i.e, n - 1 steps)
        int oneStepBack = Math.abs(heights[index] - heights[index - 1])
                + frogJump_Recursive_Helper(heights, index - 1);

        //energy consumed by frog if it is only going 2 step back (i.e, n - 2 steps)
        //default value is Int.MAX because this time frog has to jump 2 steps back
        //and possibly n - 2 steps doesn't exist
        int twoStepBack = Integer.MAX_VALUE;
        //if it is possible to go n - 2 steps back then only calculate the energy consumed
        if (index - 2 >= 0) {
            twoStepBack = Math.abs(heights[index] - heights[index - 2])
                    + frogJump_Recursive_Helper(heights, index - 2);
        }

        cache.put(index, Math.min(oneStepBack, twoStepBack));
        //return the min energy consumed from both the steps
        return Math.min(oneStepBack, twoStepBack);
    }

    public void frogJump_Recursive_And_Memoization(int[] heights) {
        //https://www.codingninjas.com/codestudio/problems/frog-jump_3621012?leftPanelTab=0
        //explanation: https://youtu.be/EgG3jsGoPvQ
        int n = heights.length;
        int recursiveFrogJump = frogJump_Recursive_Helper(heights, n - 1);

        Map<Integer, Integer> cache = new HashMap<>();
        int recursiveMemoFrogJump = frogJump_Recursive_Memoization(heights, n - 1, cache);

        System.out.println("Frog jump min energy consumed(Recusrive): " + recursiveFrogJump);
        System.out.println("Frog jump min energy consumed(Recusrive Memoization): " + recursiveMemoFrogJump);
    }

    public void frogJump_DP_Memoization(int[] heights) {
        //https://www.codingninjas.com/codestudio/problems/frog-jump_3621012?leftPanelTab=0
        //explanation: https://youtu.be/EgG3jsGoPvQ
        int n = heights.length;
        int[] memo = new int[n];
        memo[0] = 0;
        for (int i = 1; i < n; i++) {
            //energy consumed by frog if it is only going 1 step back (i.e, n - 1 steps)
            int oneStepBack = Math.abs(heights[i] - heights[i - 1])
                    + memo[i - 1];

            //energy consumed by frog if it is only going 2 step back (i.e, n - 2 steps)
            //default value is Int.MAX because this time frog has to jump 2 steps back
            //and possibly n - 2 steps doesn't exist
            int twoStepBack = Integer.MAX_VALUE;
            //if it is possible to go n - 2 steps back then only calculate the energy consumed
            if (i - 2 >= 0) {
                twoStepBack = Math.abs(heights[i] - heights[i - 2])
                        + memo[i - 2];
            }

            //return the min energy consumed from both the steps
            memo[i] = Math.min(oneStepBack, twoStepBack);
        }
        //output
        System.out.println("Frog jump min energy consumed(DP Memoization): " + memo[n - 1]);
    }

    public void frogJump_DP_Memoization_SpaceOptimization(int[] heights) {
        //https://www.codingninjas.com/codestudio/problems/frog-jump_3621012?leftPanelTab=0
        //explanation: https://youtu.be/EgG3jsGoPvQ
        int n = heights.length;
        int prevStep = 0;
        int secondPrevStep = 0;
        for (int i = 1; i < n; i++) {
            //energy consumed by frog if it is only going 1 step back (i.e, n - 1 steps)
            int oneStepBack = Math.abs(heights[i] - heights[i - 1])
                    + prevStep;

            //energy consumed by frog if it is only going 2 step back (i.e, n - 2 steps)
            //default value is Int.MAX because this time frog has to jump 2 steps back
            //and possibly n - 2 steps doesn't exist
            int twoStepBack = Integer.MAX_VALUE;
            //if it is possible to go n - 2 steps back then only calculate the energy consumed
            if (i - 2 >= 0) {
                twoStepBack = Math.abs(heights[i] - heights[i - 2])
                        + secondPrevStep;
            }
            secondPrevStep = prevStep;
            //return the min energy consumed from both the steps
            prevStep = Math.min(oneStepBack, twoStepBack);
        }
        //output
        System.out.println("Frog jump min energy consumed(DP Memoization Space Optimization): " + prevStep);
    }

    public int ninjaTraining_Recursive_Helper(int[][] points, int day, int skipTask) {

        //base condition
        //we have started from the last day to day 0, at day 0
        //we can only pick a point which is max on day 0 but we will skip
        //that task which was was performed on day0 + 1 == day1
        if (day == 0) {
            int maxPointsAtDay0 = 0;
            for (int task = 0; task < 3; task++) {
                //skiping the task performed before this curr day
                if (skipTask == task) {
                    continue;
                }
                maxPointsAtDay0 = Math.max(maxPointsAtDay0, points[day][task]);
            }
            return maxPointsAtDay0;
        }

        //out of all the given tasks we will try to maximize out points
        //by trying all the task trainings but skiping the task which
        //was performed on its previous day
        int maxPoints = 0;
        for (int task = 0; task < 3; task++) {
            if (skipTask == task) {
                continue;
            }
            //we are perfroming task 'task' on curr day 'day' now we will move to next day
            //day == 'day - 1' but we are also saying that on the next day we must skip this curr 'task'
            int currMax = points[day][task] + ninjaTraining_Recursive_Helper(points, day - 1, task);
            //from the possible attempts to maximizing our points, keep the maxPoints
            maxPoints = Math.max(maxPoints, currMax);
        }
        return maxPoints;
    }

    public int ninjaTraining_Recursive_Memoization_Helper(
            int[][] points, int day, int skipTask, Map<String, Integer> cache) {

        //base condition
        //we have started from the last day to day 0, at day 0
        //we can only pick a point which is max on day 0 but we will skip
        //that task which was was performed on day0 + 1 == day1
        if (day == 0) {
            int maxPointsAtDay0 = 0;
            for (int task = 0; task < 3; task++) {
                //skiping the task performed before this curr day
                if (skipTask == task) {
                    continue;
                }
                maxPointsAtDay0 = Math.max(maxPointsAtDay0, points[day][task]);
            }
            return maxPointsAtDay0;
        }

        String key = day + "," + skipTask;
        if (cache.containsKey(key)) {
            return cache.get(key);
        }

        //out of all the given tasks we will try to maximize out points
        //by trying all the task trainings but skiping the task which
        //was performed on its previous day
        int maxPoints = 0;
        for (int task = 0; task < 3; task++) {
            if (skipTask == task) {
                continue;
            }
            //we are perfroming task 'task' on curr day 'day' now we will move to next day
            //day == 'day - 1' but we are also saying that on the next day we must skip this curr 'task'
            int currMax = points[day][task] + ninjaTraining_Recursive_Memoization_Helper(points, day - 1, task, cache);
            //from the possible attempts to maximizing our points, keep the maxPoints
            maxPoints = Math.max(maxPoints, currMax);
        }

        cache.put(key, maxPoints);
        return maxPoints;
    }

    public void ninjaTraining_Recursive_And_Memoization(int[][] points) {
        //https://www.codingninjas.com/codestudio/problems/ninja-s-training_3621003?leftPanelTab=0
        //explanation: https://youtu.be/AE39gJYuRog
        int n = points.length;
        int skipTask = 3;

        //.............................T: O(~ 2 ^ N), because at each level of descion tree
        //we are left with 2 task to try because one is skipped from previous day, so branching factor is reduced 2
        //..............................S: O(N), function call stack
        int recursiveMaxPoints = ninjaTraining_Recursive_Helper(points, n - 1, skipTask);

        //.............................T: O(N), Overlapping subproblems handled by cache
        //..............................S: O(N + N), function call stack + cache
        //<day+","+skipTaks, maxPoints>
        Map<String, Integer> cache = new HashMap<>();
        int recursiveMemoMaxPoints = ninjaTraining_Recursive_Memoization_Helper(points, n - 1, skipTask, cache);
        //output:
        System.out.println("Ninja Training max points on trainings(Recusrive): " + recursiveMaxPoints);
        System.out.println("Ninja Training max points on trainings(Recusrive Memoization): " + recursiveMemoMaxPoints);
    }

    public void ninjaTraining_DP_Memoization(int[][] points) {
        //...............................T: O(N * 4 * 3), N for all the days,
        //4 = (skipTask (0 = skipTask 0, 1 = skipTask 1, 2 = skipTask 2, 3 = skipTask None means try all tasks))
        //3 = tasks to try per day
        //...............................S: O(N * 4), for all N days we are havings points as per skipTasks
        //https://www.codingninjas.com/codestudio/problems/ninja-s-training_3621003?leftPanelTab=0
        //explanation: https://youtu.be/AE39gJYuRog
        int n = points.length;
        int[][] memo = new int[n][4];
        //base condition
        //on day [0] and by skipping i-th task[i] the max points we can get is
        //the points from other 2 tasks
        //ex skipping task 0 we can pick max points from task 1 or task 2, same for others
        memo[0][0] = Math.max(points[0][1], points[0][2]);
        memo[0][1] = Math.max(points[0][0], points[0][2]);
        memo[0][2] = Math.max(points[0][0], points[0][1]);
        //task 3 actually represents here we didn't skip any task
        //for nth starting day, we won't be haing any skipTask from previous day because
        //that is our starting day, so on that we will try all the task to get our max points
        //from that day to any next day after that, we will some skipTask
        memo[0][3] = Math.max(points[0][0], Math.max(points[0][1], points[0][2]));

        for (int day = 1; day < n; day++) {
            for (int skipTask = 0; skipTask < 4; skipTask++) {

                memo[day][skipTask] = 0;
                for (int task = 0; task < 3; task++) {
                    if (skipTask == task) {
                        continue;
                    }
                    memo[day][skipTask] = Math.max(memo[day][skipTask],
                            points[day][task] + memo[day - 1][task]);
                }
            }
        }
        //output
        int dpMemoMaxPoints = memo[n - 1][3];
        System.out.println("Ninja Training max points on trainings(DP Memoization): " + dpMemoMaxPoints);
    }

    public void ninjaTraining_DP_Memoization_SpaceOptimization(int[][] points) {
        //...............................T: O(N * 4 * 3), N for all the days,
        //4 = (skipTask (0 = skipTask 0, 1 = skipTask 1, 2 = skipTask 2, 3 = skipTask None means try all tasks))
        //3 = tasks to try per day
        //...............................S: O(4), we will need just a prev day points to calculate our curr day points
        //https://www.codingninjas.com/codestudio/problems/ninja-s-training_3621003?leftPanelTab=0
        //explanation: https://youtu.be/AE39gJYuRog
        int n = points.length;
        int[] prevDayPoints = new int[4];
        //base condition
        //on day [0] and by skipping i-th task[i] the max points we can get is
        //the points from other 2 tasks
        //ex skipping task 0 we can pick max points from task 1 or task 2, same for others
        prevDayPoints[0] = Math.max(points[0][1], points[0][2]);
        prevDayPoints[1] = Math.max(points[0][0], points[0][2]);
        prevDayPoints[2] = Math.max(points[0][0], points[0][1]);
        //task 3 actually represents here we didn't skip any task
        //for nth starting day, we won't be haing any skipTask from previous day because
        //that is our starting day, so on that we will try all the task to get our max points
        //from that day to any next day after that, we will some skipTask
        prevDayPoints[3] = Math.max(points[0][0], Math.max(points[0][1], points[0][2]));

        for (int day = 1; day < n; day++) {
            int[] currDayPoints = new int[4];
            for (int skipTask = 0; skipTask < 4; skipTask++) {
                currDayPoints[skipTask] = 0;
                for (int task = 0; task < 3; task++) {
                    if (skipTask == task) {
                        continue;
                    }
                    currDayPoints[skipTask] = Math.max(currDayPoints[skipTask],
                            points[day][task] + prevDayPoints[task]);
                }
            }
            prevDayPoints = currDayPoints;
        }
        //output
        int dpMemoSapceOptMaxPoints = prevDayPoints[3];
        System.out.println("Ninja Training max points on trainings(DP Memoization Space Optimization): " + dpMemoSapceOptMaxPoints);
    }

    public void nMeetingRooms_Greedy(int[] startTime, int[] finishTime) {

        int n = startTime.length;

        int[][] input = new int[n][3];

        for (int i = 0; i < n; i++) {
            input[i][0] = startTime[i];
            input[i][1] = finishTime[i];
            input[i][2] = i;
        }

        //sort the input in incr order of finishTime i.e, input[i][1]
        Arrays.sort(input, (a, b) -> a[1] - b[1]);

        int meetingsCanBeConducted = 1; //at least one meeting can be held
        List<Integer> indexOfMeetingTimings = new ArrayList<>();
        indexOfMeetingTimings.add(input[0][2] + 1); // 1 based index
        int prevEndTime = input[0][1];
        for (int i = 1; i < n; i++) {

            int currStartTime = input[i][0];
            int currEndTime = input[i][1];

            if (prevEndTime < currStartTime) {
                meetingsCanBeConducted++;
                prevEndTime = currEndTime;
                indexOfMeetingTimings.add(input[i][2] + 1);
            }
        }

        System.out.println("No. of meetings can be conducted: " + meetingsCanBeConducted);
        System.out.println("Index of meetings can be conducted: " + indexOfMeetingTimings);
    }

    public void pairsOfMoviesCanBeWatchedDuringFlightDurationK_Greedy(int[] movieLength, int K) {

        //prepare index wise data
        int[][] input = new int[movieLength.length][2];
        for (int i = 0; i < movieLength.length; i++) {
            input[i][0] = movieLength[i];
            input[i][1] = i;
        }

        //sort the input
        Arrays.sort(input, new Comparator<int[]>() {

            @Override
            public int compare(int[] a, int[] b) {
                //0th index holds movieLength and sortng asc on basis of that
                //shortest movie length first
                return a[0] - b[0];
            }
        });

        int n = movieLength.length;
        int start = 0;
        int end = n - 1;
        int i = 0;
        int j = 0;
        int sum = 0;
        while (end > start) {

            int firstMovie = input[start][0];
            int secondMovie = input[end][0];

            if (firstMovie + secondMovie <= K) {

                if (sum < firstMovie + secondMovie) {
                    sum = firstMovie + secondMovie;
                    i = input[start][1];
                    j = input[end][1];
                }
                start++;
            } else if (firstMovie + secondMovie > K) {
                end--;
            }
        }

        //output:
        System.out.println("Pair of movies can be watched during flight duration K: "
                + (i + "," + j) + "-"
                + (movieLength[i] + "," + movieLength[j]));
    }

    public int choclateDistribution_Greedy(int[] choclates, int students) {

        //.........................T: O(N.LogN)
        //https://www.geeksforgeeks.org/chocolate-distribution-problem/
        //if no choclates are there OR there are no student available for distribution
        if (choclates.length == 0 || students == 0) {
            return 0;
        }

        //if the no. of choclates is less than the no. of student
        if (choclates.length < students) {
            return -1;
        }

        //sort the no of choclates
        Arrays.sort(choclates);

        int minDiff = Integer.MAX_VALUE;
        for (int i = 0; i + students - 1 < choclates.length; i++) {
            int currDiff = choclates[i + students - 1] - choclates[i];
            minDiff = Math.min(minDiff, currDiff);
        }

        //output:
        return minDiff;
    }

    public void minimumPlatformNeeded_BruteForce(int[] arr, int[] dep) {

        //.......................T: O(N^2)
        int n = arr.length;

        int maxPlatform = 1;
        for (int i = 0; i < n; i++) {
            int currPlatform = 1;
            for (int j = i + 1; j < n; j++) {
                if ((arr[i] >= arr[j] && arr[i] <= dep[j])
                        || (arr[j] >= arr[i] && arr[j] <= dep[i])) {
                    currPlatform++;
                }
                maxPlatform = Math.max(maxPlatform, currPlatform);
            }
        }

        //output:
        System.out.println("max platfrm needed: " + maxPlatform);
    }

    public void minimumPlatformNeeded_Greedy(int[] arrival, int[] depart) {

        //.......................T: O(N.LogN)
        int n = arrival.length;

        //.................T: O(N.LogN)
        Arrays.sort(arrival);
        Arrays.sort(depart);

        int arrivalIndex = 1;
        int departIndex = 0;
        int maxPlatform = 1;
        int currPlatform = 1;

        while (arrivalIndex < n && departIndex < n) {

            if (arrival[arrivalIndex] <= depart[departIndex]) {
                currPlatform++;
                arrivalIndex++;
            } else if (arrival[arrivalIndex] > depart[departIndex]) {
                currPlatform--;
                departIndex++;
            }

            maxPlatform = Math.max(maxPlatform, currPlatform);
        }

        //output:
        System.out.println("max platform needed: " + maxPlatform);
    }

    public void fractionalKnapsack(int[] weight, int[] value, int W) {

        int n = weight.length;

        //prepare data
        double[][] input = new double[n][3];
        //0: weight, 1: value, 2: costPerWeight
        for (int i = 0; i < n; i++) {

            input[i][0] = weight[i];
            input[i][1] = value[i];
            input[i][2] = (double) value[i] / (double) weight[i];
        }

        //sort the input on basis of costPerWeight desc
        Arrays.sort(input, (a, b) -> (int) (b[2] - a[2])); //2: costPerWeight

        double maxValue = 0;
        for (int i = 0; i < n; i++) {

            int currWeight = (int) input[i][0];
            int currValue = (int) input[i][1];

            if (W - currWeight >= 0) {
                W -= currWeight;
                maxValue += currValue;
            } else {

                //if we can't pick up an item as whole
                //we will calculate the fraction we can pick up
                //i.e W/currWeight
                double frac = (double) W / currWeight;
                //value per frac
                maxValue += (currValue * frac);
                //estimating balanced capacity
                W = (int) (W - (currWeight * frac));
            }
        }

        //output:
        System.out.println("Max value can be picked up by fractional knapsack: " + maxValue);
    }

    public void swapTwoDigitAtMostToFormAGreaterNumber_Greedy(int num) {

        //https://leetcode.com/problems/maximum-swap/
        char[] digit = String.valueOf(num).toCharArray();
        int[] index = new int[10];
        for (int i = 0; i < digit.length; i++) {
            index[digit[i] - '0'] = i;
        }

        for (int i = 0; i < digit.length; i++) {
            for (int d = 9; d > digit[i] - '0'; d--) {
                if (index[d] > i) {
                    char temp = digit[i];
                    digit[i] = digit[index[d]];
                    digit[index[d]] = temp;
                    System.out.println("Greater number after 2 digit swaps atmost: " + Integer.parseInt(String.valueOf(digit)));
                    return;
                }
            }
        }

        //output
        System.out.println("Greater number after 2 digit swaps atmost: " + num);
    }

    public void singleThreadedCPU_Greedy(int[][] tasks) {
        //https://leetcode.com/problems/single-threaded-cpu/
        //explanation: https://youtu.be/RR1n-d4oYqE

        class Task {

            int startTime;
            int processingTime;
            int taskIndex;

            public Task(int startTime, int processingTime, int taskIndex) {
                this.startTime = startTime;
                this.processingTime = processingTime;
                this.taskIndex = taskIndex;
            }
        }

        int n = tasks.length;
        int[] result = new int[n];
        List<Task> tasksList = new ArrayList<>();
        //cpu task waiting queue based on which smallest processing time
        //task can be picked up
        PriorityQueue<Task> minHeap = new PriorityQueue<>(
                (a, b) -> a.processingTime == b.processingTime
                        ? a.taskIndex - b.taskIndex
                        : a.processingTime - b.processingTime);

        for (int i = 0; i < n; i++) {
            int startTime = tasks[i][0];
            int processingTime = tasks[i][1];
            int taskIndex = i;
            tasksList.add(new Task(startTime, processingTime, taskIndex));
        }

        Collections.sort(tasksList, (a, b) -> a.startTime - b.startTime);

        int resIndex = 0;
        int i = 0;
        //since our tasks are sorted by startTime, cpu will pick up the
        //task that is coming first acc to startTime and cpu will be
        //booked for this curr task initially
        int cpuBookedTime = tasksList.get(0).startTime;

        while (resIndex < n) {
            //since my cpu is already booked until cpuBookedTime
            //that means cpu will process the current task until this cpuBookedTime
            //so all the task that cpu will not be able to take because being busy
            //will go to cpu task waiting queue, where it will be sorted acc to
            //processingTime and taskIndex
            while (i < n && tasksList.get(i).startTime <= cpuBookedTime) {
                minHeap.add(tasksList.get(i));
                i++;
            }

            //cpu task waiting queue is empty that means, cpu is idle
            //we can directly assign the curr i-th task to cpu coming at
            //startTime
            if (minHeap.isEmpty()) {
                cpuBookedTime = tasksList.get(i).startTime;
                continue;
            }

            //if the cpu is not idle, we pick the optimal task from waiting queue,
            //simulate, cpuBookedTime by adding currTask processingTime added to it
            //that would mean whatever the task cpu is processing right now, after
            //finishing that assign the currTask to it. In that order we take taskIndex
            Task currTask = minHeap.poll();
            cpuBookedTime += currTask.processingTime;
            result[resIndex++] = currTask.taskIndex;
        }

        //output
        System.out.println("Order of tasks in which tasks can be consumed:");
        Arrays.stream(result).boxed().forEach(x -> System.out.print(x + " "));
        System.out.println();
    }

    public void minimumArrowsToBurstBalloons_Greedy(int[][] balloonPoints) {
        //https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/
        //sort on balloon end point
        Arrays.sort(balloonPoints, (a, b) -> a[1] - b[1]);
        int prevEnd = balloonPoints[0][1];
        int n = balloonPoints.length;
        int minArrow = 1;
        for (int i = 0; i < n; i++) {
            int currStart = balloonPoints[i][0];
            int currEnd = balloonPoints[i][1];
            //curr start should be strictly greater than the prev end
            //that way we know we need extra arrows to burst these curr ballons
            //otherwise all the curr start that lesser than prev end falls under 
            //the range of single arrow shot.
            if (currStart > prevEnd) {
                minArrow++;
                prevEnd = currEnd;
            }
        }
        //output
        System.out.println("Minimum arrows required to burst all balloon: " + minArrow);
    }

    public void partitionArrSuchThatMaxDiffIsK_Greedy(int[] nums, int k) {
        //https://leetcode.com/problems/partition-array-such-that-maximum-difference-is-k/
        int n = nums.length;
        Arrays.sort(nums);
        int partition = 1;
        int prevIndex = 0;

        for (int i = 1; i < n; i++) {
            int currDiffInMaxAndMinVals = nums[i] - nums[prevIndex];
            if (currDiffInMaxAndMinVals > k) {
                partition++;
                prevIndex = i;
            }
        }
        //output
        System.out.println("Partitions : " + partition);
    }

    public void efficientJanitor_Greedy(double[] arr) {
        //https://leetcode.com/problems/boats-to-save-people/
        //https://leetcode.com/discuss/interview-question/490066/Efficient-Janitor-Efficient-Vineet-(Hackerrank-OA)
        //TWO POINTER approach 1
        /*
         ex: {1.01, 1.01, 3.0, 2.7, 1.99, 2.3, 1.7}
         Output: 5 groups: (1.01 , 1.99), (1.01, 1.7), (3.0), (2.7), (2.3)
         */
        int n = arr.length;
        int start = 0;
        int end = n - 1;

        int groups = 0;
        Arrays.sort(arr);

        while (end >= start) {

            if (end == start) {
                groups++;
                break;
            } else if (arr[start] + arr[end] <= 3.0) {
                start++;
                end--;
                groups++;
            } else {
                end--;
                groups++;
            }
        }

        //output
        System.out.println("Groups of item with sum at most 3(approach 1): " + groups);
    }

    public void efficientJanitor2_Greedy(double[] arr) {
        //https://leetcode.com/problems/boats-to-save-people/
        //https://leetcode.com/discuss/interview-question/490066/Efficient-Janitor-Efficient-Vineet-(Hackerrank-OA)
        //TWO POINTER approach 2
        /*
         ex: {1.01, 1.01, 3.0, 2.7, 1.99, 2.3, 1.7}
         Output: 5 groups: (1.01 , 1.99), (1.01, 1.7), (3.0), (2.7), (2.3)
         */
        int n = arr.length;
        int start = 0;
        int end = n - 1;

        int groups = 0;
        Arrays.sort(arr);

        while (end >= start) {
            groups++;
            if (arr[start] + arr[end] <= 3.0) {
                start++;
            }
            end--;
        }

        //output
        System.out.println("Groups of item with sum at most 3(approach 2): " + groups);
    }

    public int farthestBuildingWeCanReachUsingBricksAndLadders_Greedy(
            int[] heights, int bricks, int ladders) {
        //https://leetcode.com/problems/furthest-building-you-can-reach/
        int n = heights.length;
        int bricksUsed = 0;
        int index = 1;

        PriorityQueue<Integer> minHeapHeightDiff = new PriorityQueue<>();

        while (index < n) {

            int buildingHeightDiff = heights[index] - heights[index - 1];

            //if the curr ith building height is greater than its prev building
            //that means we have used either bricks or ladder to reach here
            if (buildingHeightDiff > 0) {
                //add diff of heights in min heap
                minHeapHeightDiff.add(buildingHeightDiff);
                //here we greedily wants to use ladder
                //if all the diff of heights in heap, is more than the ladders
                //we can use, that means we now have to use brick for further move
                if (minHeapHeightDiff.size() > ladders) {
                    //we want to use bricks as min as possible
                    //from min heap we get the min diff of heights
                    //that means we have consumed that much of bricks
                    bricksUsed += minHeapHeightDiff.poll();
                    //if our brickUsed is more than the brick given
                    //that means we could not have reached curr index
                    //so return index - 1;
                    if (bricksUsed > bricks) {
                        return index - 1;
                    }
                }
            }
            index++;
        }
        return n - 1;
    }

    public void taskSchedular_Greedy(char[] tasks, int n) {
        //https://leetcode.com/problems/task-scheduler/
        //https://leetcode.com/problems/task-scheduler/discuss/2314557/Easiest-Solution-or-O(n)-time-O(26)-space-or-Detailed-and-Easiest-explanation
        int len = tasks.length;
        int[] freqMap = new int[26];
        int maxFreq = 0;
        int maxFreqCount = 0;

        for (char task : tasks) {
            int index = task - 'A';
            freqMap[index]++;
            if (freqMap[index] > maxFreq) {
                //reset
                maxFreq = freqMap[index];
                maxFreqCount = 1;
            } else if (freqMap[index] == maxFreq) {
                maxFreqCount++;
            }
        }
        //output
        int output = Math.max(len, (maxFreq - 1) * (n + 1) + maxFreqCount);
        System.out.println("Least unit of time cpu will take to finish all tasks: " + output);
    }

    public void twoCityScheduling_Greedy(int[][] costs) {
        //https://leetcode.com/problems/two-city-scheduling/
        //explanation: https://youtu.be/d-B_gk_gJtQ

        class CostDiff {

            int costDiff;
            int costA;
            int costB;

            public CostDiff(int costDiff, int costA, int costB) {
                this.costDiff = costDiff;
                this.costA = costA;
                this.costB = costB;
            }

        }

        List<CostDiff> diffs = new ArrayList<>();
        //calculating the cost diff with intuition of how musch
        //extra we have to pay if we send ith person to cityB so
        //cityB - cityA = diff
        for (int[] cost : costs) {
            int costA = cost[0];
            int costB = cost[1];
            diffs.add(new CostDiff(costB - costA, costA, costB));
        }

        //sort the list on basis of costDiff, this will sort diffs in incr order
        //that means on the left side of diffs list we will have min cost diff for ith
        //person to send to cityB, Now acc to quest we need to half of the person to cityA and 
        //other half to cityB, we are greedily choosing min cost at which we can send
        //half ith person to cityB first and remaining can go to cityA
        Collections.sort(diffs, (d1, d2) -> d1.costDiff - d2.costDiff);
        int minCost = 0;
        for (int i = 0; i < diffs.size(); i++) {
            if (i < diffs.size() / 2) {
                minCost += diffs.get(i).costB;
            } else {
                minCost += diffs.get(i).costA;
            }
        }
        //output
        System.out.println("Toatal min cost at which half of people can travel to city A and half to cityB: "
                + minCost);
    }

    public void graphBFSAdjList_Graph(int V, List<List<Integer>> adjList) {

        List<Integer> result = new ArrayList<>();
        if (adjList == null || adjList.isEmpty()) {
            return;
        }

        //actual
        for (int i = 0; i < adjList.size(); i++) {
            System.out.print(i + ": ");
            for (int v : adjList.get(i)) {
                System.out.print(v + " ");
            }
            System.out.println();
        }

        int sourceVertex = 0; // source point
        Queue<Integer> queue = new LinkedList<>();
        queue.add(sourceVertex); //source point
        boolean[] visited = new boolean[V];

        while (!queue.isEmpty()) {

            int node = queue.poll();
            visited[node] = true;
            result.add(node);
            List<Integer> childrens = adjList.get(node);
            if (childrens != null && childrens.size() > 0) {
                for (int vertex : childrens) {
                    if (visited[vertex] != true) {
                        queue.add(vertex);
                    }
                }
            }
        }

        //output:
        System.out.println("BFS of graph: " + result);
    }

    public void graphDFSAdjList_Graph(int V, List<List<Integer>> adjList) {

        List<Integer> result = new ArrayList<>();
        if (adjList == null || adjList.isEmpty()) {
            return;
        }

        //actual
        for (int i = 0; i < adjList.size(); i++) {
            System.out.print(i + ": ");
            for (int v : adjList.get(i)) {
                System.out.print(v + " ");
            }
            System.out.println();
        }

        int sourceVertex = 0; //source point
        Stack<Integer> stack = new Stack<>();
        stack.add(sourceVertex); //source point
        boolean[] visited = new boolean[V];

        while (!stack.isEmpty()) {

            int node = stack.pop();
            visited[node] = true;
            result.add(node);
            List<Integer> childrens = adjList.get(node);
            if (childrens != null && childrens.size() > 0) {
                for (int childVertex : childrens) {
                    if (visited[childVertex] != true) {
                        stack.push(childVertex);
                    }
                }
            }
        }

        //output:
        System.out.println("DFS of graph: " + result);
    }

    private void graphDFSAdjList_Recursive_Helper(List<List<Integer>> adjList, int vertex,
            boolean[] visited, List<Integer> result) {

        visited[vertex] = true;
        result.add(vertex);
        List<Integer> childrens = adjList.get(vertex);
        for (int childVertex : childrens) {
            if (visited[childVertex] != true) {
                graphDFSAdjList_Recursive_Helper(adjList, childVertex, visited, result);
            }
        }
    }

    public void graphDFSAdjList_Recursive_Graph(int V, List<List<Integer>> adjList) {
        List<Integer> result = new ArrayList<>();
        boolean[] visited = new boolean[V];
        int sourceVertex = 0; //source point
        graphDFSAdjList_Recursive_Helper(adjList, sourceVertex, visited, result);
        System.out.println("DFS using recursion: " + result);
    }

    private void findPathRatInMaze_Helper(int[][] m, int n, int x, int y,
            StringBuilder sb, ArrayList<String> output) {

        if (x < 0 || x >= n || y < 0 || y >= n || m[x][y] == 0) {
            return;
        }

        if (x == n - 1 && y == n - 1 && m[x][y] == 1) {
            output.add(sb.toString());
            return;
        }

        int original = m[x][y];
        m[x][y] = 0;

        //Down
        sb.append("D");
        findPathRatInMaze_Helper(m, n, x + 1, y, sb, output);
        sb.deleteCharAt(sb.length() - 1);

        //Right
        sb.append("R");
        findPathRatInMaze_Helper(m, n, x, y + 1, sb, output);
        sb.deleteCharAt(sb.length() - 1);

        //Left
        sb.append("L");
        findPathRatInMaze_Helper(m, n, x, y - 1, sb, output);
        sb.deleteCharAt(sb.length() - 1);

        //Up
        sb.append("U");
        findPathRatInMaze_Helper(m, n, x - 1, y, sb, output);
        sb.deleteCharAt(sb.length() - 1);

        m[x][y] = original;
    }

    public void findPathRatInMaze_Graph(int[][] m, int n) {
        ArrayList<String> output = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        findPathRatInMaze_Helper(m, n, 0, 0, sb, output);
        System.out.println("All possible paths: " + output);
    }

    private void numberOfIslands_Helper(int[][] grid, int x, int y,
            int[][] dir, boolean[][] visited) {

        if (x < 0 || x >= grid.length || y < 0 || y >= grid[x].length
                || grid[x][y] == 0
                || visited[x][y] == true) {
            return;
        }

        visited[x][y] = true;
        for (int i = 0; i < dir.length; i++) {

            int x_ = x + dir[i][0];
            int y_ = y + dir[i][1];
            numberOfIslands_Helper(grid, x_, y_, dir, visited);
        }
    }

    public void numberOfIslands_Graph(int[][] grid) {
        //https://leetcode.com/problems/number-of-islands
        int[][] dir = {
            {-1, -1},
            {-1, 0},
            {-1, 1},
            {0, -1},
            {0, 1},
            {1, -1},
            {1, 0},
            {1, 1}
        };
        boolean[][] visited = new boolean[grid.length][grid[0].length];
        int islandCount = 0;
        for (int x = 0; x < grid.length; x++) {
            for (int y = 0; y < grid[x].length; y++) {

                if (grid[x][y] == 1 && visited[x][y] != true) {
                    islandCount++;
                    numberOfIslands_Helper(grid, x, y, dir, visited);
                }
            }
        }

        System.out.println("Number of separated islands: " + islandCount);
    }

    private boolean detectCycleInUndirectedGraphDFS_Helper(
            List<List<Integer>> adjList, int vertex, int parent, boolean[] visited) {

        visited[vertex] = true;
        List<Integer> childrens = adjList.get(vertex);
        for (int childVertex : childrens) {
            if (visited[childVertex] != true) {
                if (detectCycleInUndirectedGraphDFS_Helper(adjList, childVertex, vertex, visited)) {
                    return true;
                }
            } else if (childVertex != parent) {
                return true;
            }
        }
        return false;
    }

    public boolean detectCycleInUndirectedGraphDFS_Graph(int V, List<List<Integer>> adjList) {

        boolean[] visited = new boolean[V];
        for (int u = 0; u < V; u++) {
            if (visited[u] != true) {
                if (detectCycleInUndirectedGraphDFS_Helper(adjList, u, -1, visited)) {
                    return true;
                }
            }
        }
        return false;
    }

    private void topologicalSort_Helper(List<List<Integer>> adjList, int vertex, boolean[] visited, Stack<Integer> resultStack) {

        visited[vertex] = true;
        List<Integer> childrens = adjList.get(vertex);
        for (int childVertex : childrens) {
            if (visited[childVertex] != true) {
                topologicalSort_Helper(adjList, childVertex, visited, resultStack);
            }
        }

        resultStack.push(vertex);
    }

    public void topologicalSort_Graph(int V, List<List<Integer>> adjList) {

        Stack<Integer> resultStack = new Stack<>();
        boolean[] visited = new boolean[V];
        for (int u = 0; u < V; u++) {
            if (visited[u] != true) {
                topologicalSort_Helper(adjList, u, visited, resultStack);
            }
        }

        System.out.println("Topological sort: ");
        while (!resultStack.isEmpty()) {
            System.out.print(resultStack.pop() + " ");
        }
        System.out.println();
    }

    public boolean detectCycleInDirectedGraphDFS_Helper(List<List<Integer>> adjList, int vertex,
            boolean[] visited, boolean[] recurStack) {

        if (recurStack[vertex]) {
            return true;
        }

        if (visited[vertex]) {
            return false;
        }

        recurStack[vertex] = true;
        visited[vertex] = true;

        List<Integer> childrens = adjList.get(vertex);
        if (childrens != null && childrens.size() > 0) {
            for (int childVertex : childrens) {
                if (detectCycleInDirectedGraphDFS_Helper(adjList, childVertex, visited, recurStack)) {
                    return true;
                }
            }
        }
        recurStack[vertex] = false;
        return false;
    }

    public boolean detectCycleInDirectedGraphDFS_Graph(int V, List<List<Integer>> adjList) {
        boolean[] visited = new boolean[V];
        boolean[] recurStack = new boolean[V];
        for (int u = 0; u < V; u++) {
            if (detectCycleInDirectedGraphDFS_Helper(adjList, u, visited, recurStack)) {
                return true;
            }
        }
        return false;
    }

    private void djikstraAlgorithm_Graph_Helper(
            Map<Integer, List<GraphEdge>> graph, int V, int src) {
        //Single source shortest path
        int MAX = Integer.MAX_VALUE;

        //parents arr is optional
        int[] parents = new int[V];
        Arrays.fill(parents, MAX);

        Set<Integer> visited = new HashSet<>();

        int[] minWeights = new int[V];
        Arrays.fill(minWeights, MAX);

        int weightToSrc = 0;
        minWeights[src] = weightToSrc;

        int parentToSrc = -1;
        parents[src] = parentToSrc;

        //minHeap of weights to get the least weighted edge from the breadth first order
        PriorityQueue<GraphEdge> minHeapWeightQueue = new PriorityQueue<>(
                (g1, g2) -> g1.weight - g2.weight);

        minHeapWeightQueue.add(new GraphEdge(src, weightToSrc));

        while (!minHeapWeightQueue.isEmpty()) {

            GraphEdge currEdge = minHeapWeightQueue.poll();
            int currVertex = currEdge.vertex;

            if (visited.contains(currVertex)) {
                continue;
            }

            visited.add(currVertex);

            for (GraphEdge edge : graph.getOrDefault(currVertex, new ArrayList<>())) {

                int childVertex = edge.vertex;
                //cost to reach child vertex
                int childWeight = edge.weight;

                if (minWeights[currVertex] + childWeight < minWeights[childVertex]) {
                    minWeights[childVertex] = minWeights[currVertex] + childWeight;
                    parents[childVertex] = currVertex;
                }
                minHeapWeightQueue.add(edge);
            }
        }
        //output
        for (int u = 0; u < V; u++) {
            System.out.println("From src: "
                    + src
                    + " to vertex: "
                    + u
                    + " min weight is: "
                    + (minWeights[u] == MAX ? " Not Possible " : minWeights[u])
                    + " parent of "
                    + u
                    + " is "
                    + (parents[u] == MAX ? " Not Possible " : parents[u]));
        }
    }

    public void djikstraAlgorithm_Graph() {
        //..........................T: O(V^2)
        //Djikstra algo fails for negatve weight cycle
        //input
        Map<Integer, List<GraphEdge>> graph = new HashMap<>();
        int V = 5;
        for (int u = 0; u < V; u++) {
            graph.put(u, new ArrayList<>());
        }

        graph.get(0).addAll(Arrays.asList(
                new GraphEdge(2, 6),
                new GraphEdge(3, 6)));
        graph.get(1).addAll(Arrays.asList(
                new GraphEdge(0, 3)));
        graph.get(2).addAll(Arrays.asList(
                new GraphEdge(3, 2)));
        graph.get(3).addAll(Arrays.asList(
                new GraphEdge(1, 1),
                new GraphEdge(2, 1)));
        graph.get(4).addAll(Arrays.asList(
                new GraphEdge(1, 4),
                new GraphEdge(3, 2)));

        djikstraAlgorithm_Graph_Helper(graph, V, 4);
        djikstraAlgorithm_Graph_Helper(graph, V, 0);
    }

    public void networkTimeDelay_Graph(int[][] times, int V, int source) {
        //https://leetcode.com/problems/network-delay-time/
        //based on DJIKSTRA ALGO
        Map<Integer, List<GraphEdge>> graph = new HashMap<>();
        for (int[] time : times) {
            int u = time[0];
            int v = time[1];
            int weight = time[2];
            graph.putIfAbsent(u, new ArrayList<>());
            graph.get(u).add(new GraphEdge(v, weight));
        }

        int MAX = Integer.MAX_VALUE;

        int[] delayTimeToNode = new int[V + 1];

        Arrays.fill(delayTimeToNode, MAX);

        int sourceTime = 0;

        delayTimeToNode[source] = sourceTime;

        Queue<GraphEdge> queue = new PriorityQueue<>((a, b) -> a.weight - b.weight);
        queue.add(new GraphEdge(source, sourceTime));

        while (!queue.isEmpty()) {
            GraphEdge currEdge = queue.poll();
            int currU = currEdge.vertex;

            for (GraphEdge childEdge : graph.getOrDefault(currU, new ArrayList<>())) {
                int currV = childEdge.vertex;
                int weight = childEdge.weight;
                if (delayTimeToNode[currV] > delayTimeToNode[currU] + weight) {
                    delayTimeToNode[currV] = delayTimeToNode[currU] + weight;
                    queue.add(childEdge);
                }
            }
        }

        int maxNetworkDelayTime = Integer.MIN_VALUE;
        //considering [1 to V] nodes as given
        for (int i = 1; i < delayTimeToNode.length; i++) {
            int delayTime = delayTimeToNode[i];
            //if from [1 to V] any node is not reachable(disconnected graph) that node's
            //delay time will be left as MAX in that case we have to return -1
            if (delayTime == MAX) {
                maxNetworkDelayTime = -1;
                break;
            }
            maxNetworkDelayTime = Math.max(maxNetworkDelayTime, delayTime);
        }
        //output
        System.out.println("Max network delay time to reach far end node: " + maxNetworkDelayTime);
    }

    public void findEventualSafeNodes_Graph(int[][] edges) {
        //https://leetcode.com/problems/find-eventual-safe-states/
        int V = edges.length;
        //using boolean[] for marking nodes safe beacause this will keep
        //nodes in sorted order, which is req in question
        boolean[] markSafeNodes = new boolean[V];
        List<Integer> safeNodes = new ArrayList<>();
        Queue<Integer> queue = new LinkedList<>();
        //graph or parent = [child] relation
        Map<Integer, Set<Integer>> graph = new HashMap<>();
        //reverse graph or child = [parent] relation
        Map<Integer, Set<Integer>> revGraph = new HashMap<>();

        for (int parent = 0; parent < V; parent++) {
            //if a parent node don't have anu outgoing edges(terminal node)
            if (edges[parent].length == 0) {
                queue.add(parent);
            }
            for (int childVertex : edges[parent]) {
                graph.putIfAbsent(parent, new HashSet<>());
                graph.get(parent).add(childVertex);

                revGraph.putIfAbsent(childVertex, new HashSet<>());
                revGraph.get(childVertex).add(parent);
            }
        }

        //queue will start from terminal nodes where we can't move further
        //and we will fill the parents in queue if they can be made terminal
        //node while going the rev graph
        while (!queue.isEmpty()) {
            int currSrc = queue.poll();
            //if any node is taken out of the queue, that means this curr src node
            //don't have any outgoing edges
            markSafeNodes[currSrc] = true;
            for (int parentVertex : revGraph.getOrDefault(currSrc, new HashSet<>())) {
                //we will remove curr src(terminal node) from its parent graph
                //nodes if that parent can be made terminal. Because if that parent
                //can't be made terminal node that would mean, there will exixst a node from
                //this parent where that whole path will lead to no terminal node
                graph.get(parentVertex).remove(currSrc);
                //parent nodes list is empty means it has become a terminal node now
                //we can add it in the queue
                if (graph.get(parentVertex).isEmpty()) {
                    queue.add(parentVertex);
                }
            }
        }

        for (int src = 0; src < V; src++) {
            if (markSafeNodes[src]) {
                safeNodes.add(src);
            }
        }
        //output
        System.out.println("Safe nodes in sorted fromat: " + safeNodes);
    }

    private void numberOfProvince_Graph_HelperDFS(
            Map<Integer, List<Integer>> graph, int src, Set<Integer> visited) {
        if (visited.contains(src)) {
            return;
        }
        visited.add(src);
        for (int childVertex : graph.getOrDefault(src, new ArrayList<>())) {
            numberOfProvince_Graph_HelperDFS(graph, childVertex, visited);
        }
    }

    public void numberOfProvince_Graph(int[][] isConnected) {
        //https://leetcode.com/problems/number-of-provinces/
        //acc to quest row or col is V
        int ROW = isConnected.length;
        int COL = isConnected[0].length;
        int provinces = 0;
        Map<Integer, List<Integer>> graph = new HashMap<>();
        Set<Integer> visited = new HashSet<>();

        for (int r = 0; r < ROW; r++) {
            graph.putIfAbsent(r + 1, new ArrayList<>());
            for (int c = 0; c < COL; c++) {
                if (r != c && isConnected[r][c] == 1) {
                    graph.get(r + 1).add(c + 1);
                }
            }
        }

        for (int src = 1; src <= ROW; src++) {
            //a single dfs call from curr src will visit all of its
            //connected child vertices. In that way each time we are entring
            //below if cond that means we are going to do a dfs on a disconnected graph
            //ex: 1 = [2,3], 2 = [1], 3 = [1], 4 = []
            //graph : 1 <--> 2 <--> 3     4
            //two component [1,2,3], [4]
            if (!visited.contains(src)) {
                provinces++;
                numberOfProvince_Graph_HelperDFS(graph, src, visited);
            }
        }
        //output
        System.out.println("Number of provinces/ (number of disconnected graphs): " + provinces);
    }

    public void floodFill_Helper(int[][] image, int srcR, int srcC,
            int srcColor, int newColor, boolean[][] visited) {

        //bounds check
        if (srcR < 0 || srcR >= image.length || srcC < 0 || srcC >= image[srcR].length
                || image[srcR][srcC] != srcColor
                || visited[srcR][srcC]) {
            return;
        }

        visited[srcR][srcC] = true;

        //do dfs in 4 adjacent dir
        //UP
        floodFill_Helper(image, srcR - 1, srcC, srcColor, newColor, visited);

        //DOWN
        floodFill_Helper(image, srcR + 1, srcC, srcColor, newColor, visited);

        //LEFT
        floodFill_Helper(image, srcR, srcC - 1, srcColor, newColor, visited);

        //RIGHT
        floodFill_Helper(image, srcR, srcC + 1, srcColor, newColor, visited);

        //at this point, we have the color to change
        //we will update with new color.
        if (image[srcR][srcC] == srcColor) {
            image[srcR][srcC] = newColor;
        }

        visited[srcR][srcC] = false;
    }

    public void floodFill(int[][] image, int srcR, int srcC, int newColor) {
        //https://leetcode.com/problems/flood-fill/
        //actual
        System.out.println();
        for (int[] r : image) {
            for (int c : r) {
                System.out.print(c + "\t");
            }
            System.out.println();
        }

        int srcColor = image[srcR][srcC];
        boolean[][] visited = new boolean[image.length][image[0].length];
        floodFill_Helper(image, srcR, srcC, srcColor, newColor, visited);

        //output
        System.out.println("output: ");
        for (int[] r : image) {
            for (int c : r) {
                System.out.print(c + "\t");
            }
            System.out.println();
        }
    }

    public boolean checkIfGivenUndirectedGraphIsBinaryTree(int V, List<List<Integer>> adjList) {

        //two condition for a undirected graph to be tree
        //1. should not have a cycle
        //2. the graph should be connected
        boolean[] visited = new boolean[V];

        //check undriected cycle
        //if all nodes is reachable via vertex 0 without forming a cycle
        //if cycle is present then its not tree and return false
        //it will also mark visited = true all those nodes that are connected to each other
        if (detectCycleInUndirectedGraphDFS_Helper(adjList, 0, -1, visited)) {
            //if all the vertexes are connected and still there is cycle anywhere in the graph
            //it will be detected here. In case if a part of graph is not connected and that contains
            //cycle(which is not detectable), still below isConnected loop will return false as
            //all nodes are not connected. in visited[]
            return false;
        }

        for (boolean isConnected : visited) {
            if (!isConnected) {
                return false;
            }
        }

        return true;
    }

    private void allPathFromSourceToTargetInDirectedAcyclicGraph_Helper(Map<Integer, List<Integer>> adj, int vertex,
            int target, boolean[] vis, List<Integer> curr, List<List<Integer>> result) {

        vis[vertex] = true;
        curr.add(vertex);

        if (vertex == target) {
            result.add(curr);
        }

        List<Integer> childrens = adj.getOrDefault(vertex, new ArrayList<>());
        for (int childVertex : childrens) {
            if (vis[childVertex] != true) {
                allPathFromSourceToTargetInDirectedAcyclicGraph_Helper(adj, childVertex, target,
                        vis, new ArrayList<>(curr), result);
            }
        }
        vis[vertex] = false;
    }

    public void allPathFromSourceToTargetInDirectedAcyclicGraph(int[][] graph) {

        //https://leetcode.com/problems/all-paths-from-source-to-target/
        List<List<Integer>> result = new ArrayList<>();

        if (graph[0].length == 0 && graph[1].length == 0) {
            return;
        }

        //prepare data
        Map<Integer, List<Integer>> adj = new HashMap<>();
        for (int u = 0; u < graph.length; u++) {
            adj.putIfAbsent(u, new ArrayList<>());
            for (int nodes : graph[u]) {
                adj.get(u).add(nodes);
            }
        }

        int source = 0;
        int target = adj.size() - 1; //target = V - 1
        boolean[] vis = new boolean[graph.length];
        allPathFromSourceToTargetInDirectedAcyclicGraph_Helper(adj, source, target, vis, new ArrayList<>(), result);

        //output:
        System.out.println("All paths: " + result);
    }

    public void allPathFromSourceToTargetInDirectedAcyclicGraph_SameButShort_Helper(
            int[][] graph, int source, int target, List<Integer> curr, List<List<Integer>> result) {
        curr.add(source);
        if (source == target) {
            result.add(curr);
            return;
        }

        for (int childFromSource : graph[source]) {
            allPathFromSourceToTargetInDirectedAcyclicGraph_SameButShort_Helper(graph,
                    childFromSource, target, new ArrayList<>(curr), result);
        }
    }

    public void allPathFromSourceToTargetInDirectedAcyclicGraph_SameButShort(int[][] graph) {

        //https://leetcode.com/problems/all-paths-from-source-to-target/
        //https://leetcode.com/problems/all-paths-from-source-to-target/discuss/2083209/Java-or-DFS-or-Shortest
        List<List<Integer>> result = new ArrayList<>();
        int source = 0;
        int target = graph.length - 1;
        allPathFromSourceToTargetInDirectedAcyclicGraph_SameButShort_Helper(graph, source, target, new ArrayList<>(), result);
        //output:
        System.out.println("All paths short approach: " + result);
    }

    private boolean checkIfPathExistsFromSourceToDestination_DFS(int[][] grid, int x, int y) {

        if (x < 0 || x >= grid.length || y < 0 || y >= grid[x].length
                || grid[x][y] == 0) {
            return false;
        }

        if (grid[x][y] == 2) {
            return true;
        }

        int original = grid[x][y];
        grid[x][y] = 0;

        boolean anyPathPossible
                = checkIfPathExistsFromSourceToDestination_DFS(grid, x - 1, y)
                || checkIfPathExistsFromSourceToDestination_DFS(grid, x + 1, y)
                || checkIfPathExistsFromSourceToDestination_DFS(grid, x, y - 1)
                || checkIfPathExistsFromSourceToDestination_DFS(grid, x, y + 1);

        grid[x][y] = original;

        return anyPathPossible;
    }

    public boolean checkIfPathExistsFromSourceToDestination(int[][] grid) {

        int r = grid.length;
        int c = grid[0].length;

        for (int x = 0; x < r; x++) {
            for (int y = 0; y < c; y++) {
                if (grid[x][y] == 1 && checkIfPathExistsFromSourceToDestination_DFS(grid, x, y)) {
                    return true;
                }
            }
        }

        return false;
    }

    public boolean canWeVisitAllTheRooms(List<List<Integer>> rooms) {

        //problem: https://leetcode.com/problems/keys-and-rooms
        //explanation: https://youtu.be/Rz_-Kx0LN-E
        //what all rooms [based on their index], we can visit or not!
        boolean[] visitRoom = new boolean[rooms.size()];
        //for DFS
        Stack<Integer> stack = new Stack<>();

        //we have to start form 0
        int startingRoom = 0;

        //starting room is visited as we starting from this one
        visitRoom[startingRoom] = true;
        stack.push(startingRoom);
        //do DFS to mark what all rooms we can visit
        while (!stack.isEmpty()) {
            int currRoom = stack.pop();
            List<Integer> roomKeysInCurrRoom = rooms.get(currRoom);
            for (int roomKey : roomKeysInCurrRoom) {
                //if any room wtih roomKey is not already visited put it in DFS stack
                //and mark that roomKey as visited 
                //(because if we have roomKey we can definetly visit that room)
                if (visitRoom[roomKey] != true) {
                    stack.push(roomKey);
                    visitRoom[roomKey] = true;
                }
            }
        }

        //if after traversing all the rooms, if there is any room left 
        //un visited then return false;
        for (boolean roomVisitedStatus : visitRoom) {
            if (roomVisitedStatus == false) {
                return false;
            }
        }

        return true;
    }

    private void vertexWithWeightAllSourceToDestinationPath_Helper(List<List<VertexWithWeight>> adj,
            int mainSource, int source, int target,
            int currWeight, boolean[] vis, Map<String, Integer> result) {

        if (source == target) {
            if (!result.containsKey(target + "-" + mainSource)) {
                result.put(mainSource + "-" + target, currWeight);
            }
        }

        vis[source] = true;
        List<VertexWithWeight> childs = adj.get(source);
        for (VertexWithWeight cv : childs) {

            if (vis[cv.vertex] != true) {
                vertexWithWeightAllSourceToDestinationPath_Helper(adj, mainSource, cv.vertex, target, currWeight + cv.weight, vis, result);
            }
        }
        vis[source] = false;
    }

    public void vertexWithWeightAllSourceToDestinationPath(List<List<VertexWithWeight>> adj) {

        int V = adj.size();
        Map<String, Integer> result = new HashMap<>();
        boolean[] vis = new boolean[V];
        for (int u = 0; u < V; u++) {

            for (int v = 0; v < V; v++) {
//                Arrays.fill(vis, false);
                if (u != v) {
                    vertexWithWeightAllSourceToDestinationPath_Helper(adj, u, u, v, 0, vis, result);
                }
            }
        }

        //output:
        result.entrySet()
                .stream()
                .sorted((a, b) -> a.getKey().compareTo(b.getKey()))
                .forEach(e -> System.out.println(e.getKey() + " = " + e.getValue()));

    }

    public void vertexThroughAllOtherVertexCanBeReachedInDirectedAcyclicGraph_Graph(int V, int[][] edges) {

        //https://leetcode.com/problems/minimum-number-of-vertices-to-reach-all-nodes/
        //explanation: https://leetcode.com/problems/minimum-number-of-vertices-to-reach-all-nodes/discuss/1179313/best-solution
        /*
         all Vertex Through Other Nodes Can Be Reached are those nodes who doesn't have any
         incoming directions to them
         ex: [{0,1},{0,2},{2,5},{3,4},{4,2}]
         0 : [1,2]
         2 : [5]
         3 : [4]
         4 : [2]
         only 0 and 3 are the vertex who doesn't have incoming req
         */
        List<Integer> allVertexThroughOtherNodesCanBeReached = new ArrayList<>();
        int[] incomingRequest = new int[V];
        for (int i = 0; i < edges.length; i++) {
            incomingRequest[edges[i][1]]++;
        }

        for (int i = 0; i < V; i++) {
            if (incomingRequest[i] == 0) {
                allVertexThroughOtherNodesCanBeReached.add(i);
            }
        }

        //output
        System.out.println("Vertexes in DAG through all other vertex can be reached: " + allVertexThroughOtherNodesCanBeReached);
    }

    public boolean checkIfGraphIsBipartite_Graph(int[][] graph) {

        //https://leetcode.com/problems/is-graph-bipartite/
        /*A graph is bipartite if the nodes can be partitioned into two independent 
         sets A and B such that every edge in the graph connects a node 
         in set A and a node in set B.*/
        int V = graph.length;
        int[] color = new int[V];
        Arrays.fill(color, -1);

        Queue<Integer> q = new LinkedList<>();

        for (int u_ = 0; u_ < V; u_++) {
            if (color[u_] != -1) {
                continue;
            }
            color[u_] = 1;
            q.add(u_);
            while (!q.isEmpty()) {

                int u = q.poll();
                for (int cv : graph[u]) {
                    if (color[cv] == -1) {
                        color[cv] = 1 - color[u];
                        q.add(cv);
                    } else if (color[cv] == color[u]) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    private boolean courseSchedule2_IsDirectedGraphCycle(Map<Integer, List<Integer>> graph,
            int vertex, boolean[] visited, boolean[] recurStack) {
        //same as detect cycle in directed graph but with map based graph input
        if (recurStack[vertex]) {
            return true;
        }

        if (visited[vertex]) {
            return false;
        }

        recurStack[vertex] = true;
        visited[vertex] = true;

        List<Integer> childrens = graph.getOrDefault(vertex, new ArrayList<>());
        for (int childVertex : childrens) {
            if (courseSchedule2_IsDirectedGraphCycle(
                    graph, childVertex, visited, recurStack)) {
                return true;
            }
        }

        recurStack[vertex] = false;
        return false;
    }

    private void courseSchedule2_TopoSort(
            Map<Integer, List<Integer>> graph, int vertex,
            boolean[] visited, Stack<Integer> stack) {
        //same as topological sort but with map based graph input
        visited[vertex] = true;
        List<Integer> childrens = graph.getOrDefault(vertex, new ArrayList<>());
        for (int childVertex : childrens) {
            if (visited[childVertex] != true) {
                courseSchedule2_TopoSort(
                        graph, childVertex, visited, stack);
            }
        }

        stack.push(vertex);
    }

    public void courseSchedule2_Graph(int[][] courseSchedule, int courses) {

        //https://leetcode.com/problems/course-schedule-ii/
        //https://leetcode.com/discuss/interview-question/742238/Amazon-or-Student-Order
        //Directed graph
        Map<Integer, List<Integer>> graph = new HashMap<>();
        for (int[] schedule : courseSchedule) {
            int u = schedule[0];
            int v = schedule[1];
            graph.putIfAbsent(u, new ArrayList<>());
            graph.get(u).add(v);
        }

        //graph as input
        System.out.println("Graph: " + graph);

        //check cycle in Directed acyclic graph
        boolean[] visited = new boolean[courses];
        boolean[] recurStack = new boolean[courses];
        for (int u = 0; u < courses; u++) {
            if (courseSchedule2_IsDirectedGraphCycle(graph, u, visited, recurStack)) {
                //if there is cycle in the graph
                //course can't new schedules
                //return new int[]{}; //empty schedule
                System.out.println("Courses can't be sceduled because there is cycle");
                return;
            }
        }

        Arrays.fill(visited, false);
        Stack<Integer> stack = new Stack<>();
        for (int u = 0; u < courses; u++) {
            if (visited[u] != true) {
                courseSchedule2_TopoSort(graph, u, visited, stack);
            }
        }

        //output from topoSort
        int[] result = new int[stack.size()];
        int index = result.length - 1;
        while (!stack.isEmpty()) {
            result[index--] = stack.pop();
        }

        //output
        System.out.println("Order: ");
        for (int e : result) {
            System.out.print(e + " ");
        }
        System.out.println();
    }

    public void alienDictionary_Graph(String[] dict, int alphabets) {

        //.............................T: O(N + alphabets)
        //https://www.geeksforgeeks.org/given-sorted-dictionary-find-precedence-characters/
        //explanation: https://youtu.be/6kTZYvNNyps
        /*
         The first step to create a graph takes O(n + alhpa) time where n is 
         number of given words and alpha is number of characters in given 
         alphabet. The second step is also topological sorting. Note that 
         there would be alpha vertices and at-most (n-1) edges in the graph. 
         The time complexity of topological sorting is O(V+E) which is 
         O(n + aplha) here. So overall time complexity is 
         O(n + aplha) + O(n + aplha) which is O(n + aplha).
         */
        //prepare input
        Map<Integer, List<Integer>> graph = new HashMap<>();
        for (int i = 0; i < dict.length - 1; i++) {
            String wrd1 = dict[i];
            String wrd2 = dict[i + 1];
            int minLenWrd = Math.min(wrd1.length(), wrd2.length());
            for (int j = 0; j < minLenWrd; j++) {
                char chWrd1 = wrd1.charAt(j);
                char chWrd2 = wrd2.charAt(j);
                //this if will find the first non matching char in both the words
                //ex: wrd1 = "wrt" wrd2 = "wrf" first non matching char be t & f
                if (chWrd1 != chWrd2) {
                    graph.putIfAbsent(chWrd1 - 'a', new ArrayList<>());
                    graph.get(chWrd1 - 'a').add(chWrd2 - 'a');
                }
            }
        }

        //do topo sort
        boolean[] visited = new boolean[alphabets];
        Stack<Integer> stack = new Stack<>();

        for (int u = 0; u < alphabets; u++) {
            if (visited[u] != true) {
                //just re-using map based topological sort algo done in course schedule probeln
                courseSchedule2_TopoSort(graph, u, visited, stack);
            }
        }

        String orderOfAlienAlphabet = "";
        while (!stack.isEmpty()) {
            orderOfAlienAlphabet += (char) (stack.pop() + 'a') + ", ";
        }

        //output
        //for removing last from string ", "
        String res = orderOfAlienAlphabet.substring(0, orderOfAlienAlphabet.length() - 2);
        System.out.println("Order of alphabet in alien language: " + res);
    }

    private boolean alienDictionary2_Graph_Cycle_Check_DFS(
            Map<Character, Set<Character>> graph, char vertex,
            Set<Character> visited, Stack<Character> topo) {

        if (visited.contains(vertex)) {
            return true;
        }

        visited.add(vertex);

        for (char childVertex : graph.getOrDefault(vertex, new HashSet<>())) {
            if (alienDictionary2_Graph_Cycle_Check_DFS(graph, childVertex, visited, topo)) {
                return true;
            }
        }
        visited.remove(vertex);
        topo.push(vertex);
        return false;
    }

    public void alienDictionary2_Graph(String[] dict) {

        //.............................T: O(N + alphabets)
        //https://www.geeksforgeeks.org/given-sorted-dictionary-find-precedence-characters/
        //explanation: https://youtu.be/6kTZYvNNyps
        //prepare input
        Map<Character, Set<Character>> graph = new HashMap<>();
        for (int i = 0; i < dict.length - 1; i++) {
            String wrd1 = dict[i];
            String wrd2 = dict[i + 1];
            int minLenWrd = Math.min(wrd1.length(), wrd2.length());
            for (int j = 0; j < minLenWrd; j++) {
                char chWrd1 = wrd1.charAt(j);
                char chWrd2 = wrd2.charAt(j);
                //this if will find the first non matching char in both the words
                //ex: wrd1 = "wrt" wrd2 = "wrf" first non matching char be t & f
                if (chWrd1 != chWrd2) {
                    graph.putIfAbsent(chWrd1, new HashSet<>());
                    graph.get(chWrd1).add(chWrd2);
                    break;
                }
            }
        }

        //do topo sort
        Set<Character> visited = new HashSet<>();
        Stack<Character> topo = new Stack<>();

        for (char key : graph.keySet()) {
            if (alienDictionary2_Graph_Cycle_Check_DFS(graph, key, visited, topo)) {
                System.out.println("Order of alphabet in alien language: NOT POSSIBLE");
                //return "";
            }
        }

        String orderOfAlienAlphabet = "";
        int charReq = graph.size();
        while (!topo.isEmpty() && charReq != 0) {
            orderOfAlienAlphabet += topo.pop() + ", ";
            charReq--;
        }

        //output
        //for removing last from string ", "
        String res = orderOfAlienAlphabet.substring(0, orderOfAlienAlphabet.length() - 2);
        System.out.println("Order of alphabet in alien language: " + res);
    }

    public int findTownJudge_Graph(int n, int[][] trusts) {
        //https://leetcode.com/problems/find-the-town-judge/
        //https://leetcode.com/problems/find-the-town-judge/discuss/2106467/Simple-yet-efficient-java-solution
        int[] inDegree = new int[n];
        for (int[] trust : trusts) {
            //0th person(trust[0]) trust 1st person(trust[1])
            //that way trust[1] got inDegree from trust[0]
            //but if later on it is found that trust[1] is also trusting
            //someone else that means we have to reduce its inDegree
            inDegree[trust[1] - 1]++;
            inDegree[trust[0] - 1]--;
        }
        for (int i = 0; i < n; i++) {
            if (inDegree[i] == n - 1) {
                return i + 1;
            }
        }
        return -1;
    }

    private void surroundedRegions_Graph_DFS(char[][] board, int row, int col) {
        if (row < 0 || row >= board.length
                || col < 0 || col >= board[row].length
                || board[row][col] != 'O') {
            return;
        }

        //replace the O at the border regions with temp char T
        board[row][col] = 'T';
        //convert all the other O that are connected with O at the border regions
        surroundedRegions_Graph_DFS(board, row - 1, col);
        surroundedRegions_Graph_DFS(board, row + 1, col);
        surroundedRegions_Graph_DFS(board, row, col - 1);
        surroundedRegions_Graph_DFS(board, row, col + 1);
    }

    public void surroundedRegions_Graph(char[][] board) {
        //https://leetcode.com/problems/surrounded-regions/
        //explanantion: https://youtu.be/9z2BunfoZ5Y

        int row = board.length;
        int col = board[0].length;

        //actual
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                System.out.print(board[r][c] + " ");
            }
            System.out.println();
        }

        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                //First convert all the O at the border region and all other O 
                //that is connected to these O
                if (board[r][c] == 'O'
                        //border regions ==> row == TOP EDGE || BOTTOM EDGE, col == LEFT EDGE || RIGHT EDGE
                        && ((r == 0 || r == row - 1) || (c == 0 || c == col - 1))) {
                    surroundedRegions_Graph_DFS(board, r, c);
                }
            }
        }

        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                //once all the Os at the border regions are replaced with temp char T
                //we will be left with those O which are surrounded by X only
                //because only those O were not reachable from above dfs
                //we can easily convert these Os to X as per question
                if (board[r][c] == 'O') {
                    board[r][c] = 'X';
                }
            }
        }

        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                //once all the Os that were surrounded by X are replaced with X 
                //in previous loop, we can replace temp char T back with O again
                if (board[r][c] == 'T') {
                    board[r][c] = 'O';
                }
            }
        }

        //output
        System.out.println("Surrounded regions output: ");
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                System.out.print(board[r][c] + " ");
            }
            System.out.println();
        }
        System.out.println();
    }

    private int longestIncreasingPathInMatrixFromAnyPoint_Graph_Memoization_DFS(
            int[][] matrix, int row, int col, int prevVal, Map<String, Integer> memo) {
        if (row < 0 || row >= matrix.length
                || col < 0 || col >= matrix[0].length
                || matrix[row][col] <= prevVal) {
            return 0;
        }

        String key = row + "," + col;
        if (memo.containsKey(key)) {
            return memo.get(key);
        }

        //for each value in matrix that value itself is a longest incr path
        //atleast of length 1
        int currLongestIncrPath = 1;

        //UP
        currLongestIncrPath = Math.max(currLongestIncrPath,
                longestIncreasingPathInMatrixFromAnyPoint_Graph_Memoization_DFS(
                        matrix, row - 1, col, matrix[row][col], memo) + 1);
        //DOWN
        currLongestIncrPath = Math.max(currLongestIncrPath,
                longestIncreasingPathInMatrixFromAnyPoint_Graph_Memoization_DFS(
                        matrix, row + 1, col, matrix[row][col], memo) + 1);
        //LEFT
        currLongestIncrPath = Math.max(currLongestIncrPath,
                longestIncreasingPathInMatrixFromAnyPoint_Graph_Memoization_DFS(
                        matrix, row, col - 1, matrix[row][col], memo) + 1);
        //RIGHT
        currLongestIncrPath = Math.max(currLongestIncrPath,
                longestIncreasingPathInMatrixFromAnyPoint_Graph_Memoization_DFS(
                        matrix, row, col + 1, matrix[row][col], memo) + 1);

        //cache the currLongestIncrPath at curr row,col
        memo.put(key, currLongestIncrPath);

        return currLongestIncrPath;
    }

    public void longestIncreasingPathInMatrixFromAnyPoint_Graph_Memoization(int[][] matrix) {
        //https://leetcode.com/problems/longest-increasing-path-in-a-matrix/
        //explanation: https://youtu.be/wCc_nd-GiEc
        int row = matrix.length;
        int col = matrix[0].length;

        //<"row,col", currLongestIncrPath>
        Map<String, Integer> memo = new HashMap<>();

        int longestIncrPath = 0;

        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                longestIncrPath = Math.max(longestIncrPath,
                        longestIncreasingPathInMatrixFromAnyPoint_Graph_Memoization_DFS(
                                matrix, r, c, -1, memo));
            }
        }
        //output
        System.out.println("Longest increasing path in matrix: " + longestIncrPath);
    }

    public int swimInRisingWater_Graph(int[][] grid) {
        //https://leetcode.com/problems/swim-in-rising-water/
        //explanation: https://youtu.be/amvrKlMLuGY
        class Node {

            int time;
            int x;
            int y;

            public Node(int time, int x, int y) {
                this.time = time;
                this.x = x;
                this.y = y;
            }

        }

        int n = grid.length;
        Set<String> visited = new HashSet<>();
        PriorityQueue<Node> minHeapTime = new PriorityQueue<>(
                (n1, n2) -> n1.time - n2.time
        );

        int[][] dirs = {
            {1, 0},
            {-1, 0},
            {0, 1},
            {0, -1}
        };

        int currX = 0;
        int currY = 0;

        visited.add(currX + "," + currY);
        minHeapTime.add(new Node(grid[currX][currY], currX, currY));

        while (!minHeapTime.isEmpty()) {

            Node curr = minHeapTime.poll();

            if (curr.x == n - 1 && curr.y == n - 1) {
                return curr.time;
            }
            int currTime = curr.time;

            for (int[] dir : dirs) {

                int newX = curr.x + dir[0];
                int newY = curr.y + dir[1];

                if (newX < 0 || newX >= n
                        || newY < 0 || newY >= n
                        || visited.contains(newX + "," + newY)) {
                    continue;
                }
                minHeapTime.add(new Node(
                        Math.max(currTime, grid[newX][newY]),
                        newX, newY));
            }
        }
        return -1;
    }

    public void minCostToConnectAllPoints_Graph(int[][] points) {
        //https://leetcode.com/problems/min-cost-to-connect-all-points/
        //explanation: https://youtu.be/f7JOBJIC-NA
        //BASED on prim's algo
        int n = points.length;
        //<srcVertex, List<List<dist, destVertex>>>
        Map<Integer, List<List<Integer>>> graph = new HashMap<>();
        for (int node = 0; node < n; node++) {
            graph.put(node, new ArrayList<>());
        }
        for (int src = 0; src < n; src++) {
            int x1 = points[src][0];
            int y1 = points[src][1];
            for (int dest = src + 1; dest < n; dest++) {
                int x2 = points[dest][0];
                int y2 = points[dest][1];
                //manhattan dist
                int dist = Math.abs(x1 - x2) + Math.abs(y1 - y2);
                graph.get(src).add(Arrays.asList(dist, dest));
                graph.get(dest).add(Arrays.asList(dist, src));
            }
        }

        int totalDist = 0;
        int currSrc = 0;
        int currDist = 0;

        Set<Integer> visited = new HashSet<>();
        //List<dist, destVertex>
        PriorityQueue<List<Integer>> minHeapDist = new PriorityQueue<>(
                (l1, l2) -> l1.get(0) - l2.get(0)
        );

        minHeapDist.add(Arrays.asList(currDist, currSrc));

        while (visited.size() < n) {

            List<Integer> currEdge = minHeapDist.poll();
            currDist = currEdge.get(0);
            currSrc = currEdge.get(1);

            if (visited.contains(currSrc)) {
                continue;
            }

            totalDist += currDist;
            visited.add(currSrc);

            for (List<Integer> childEdge : graph.get(currSrc)) {
                int childVertex = childEdge.get(1);
                if (visited.contains(childVertex)) {
                    continue;
                }
                minHeapDist.add(childEdge);
            }
        }
        //output
        System.out.println("Min cost to connect all points: " + totalDist);
    }

    class Coord {

        int row;
        int col;

        public Coord(int row, int col) {
            this.row = row;
            this.col = col;
        }

    }

    private boolean pacificAtlanticWaterFlow_IsOutOfBounds(int row, int col, int ROW, int COL) {
        return row < 0 || row >= ROW || col < 0 || col >= COL;
    }

    private void pacificAtlanticWaterFlow_HelperBFS(int[][] heights, Queue<Coord> queue,
            boolean[][] visited, int[][] dirs, int ROW, int COL) {

        while (!queue.isEmpty()) {

            Coord curr = queue.poll();

            visited[curr.row][curr.col] = true;

            for (int[] dir : dirs) {
                int newRow = curr.row + dir[0];
                int newCol = curr.col + dir[1];

                if (pacificAtlanticWaterFlow_IsOutOfBounds(newRow, newCol, ROW, COL)
                        || visited[newRow][newCol]
                        || heights[newRow][newCol] < heights[curr.row][curr.col]) {
                    continue;
                }
                queue.add(new Coord(newRow, newCol));
            }
        }
    }

    public void pacificAtlanticWaterFlow(int[][] heights) {
        //https://leetcode.com/problems/pacific-atlantic-water-flow/
        //https://www.geeksforgeeks.org/atlantic-pacific-water-flow/

        int ROW = heights.length;
        int COL = heights[0].length;

        List<List<Integer>> result = new ArrayList<>();

        int topEdge = 0;
        int bottomEdge = ROW - 1;
        int leftEdge = 0;
        int rightEdge = COL - 1;

        int[][] dirs = {
            {-1, 0},
            {1, 0},
            {0, -1},
            {0, 1}
        };

        Queue<Coord> queueAtlantic = new LinkedList<>();
        Queue<Coord> queuePacific = new LinkedList<>();

        boolean[][] visitedAtlantic = new boolean[ROW][COL];
        boolean[][] visitedPacific = new boolean[ROW][COL];

        //save all the top & bottom row edges that are touched by atlantic and pacific ocean
        for (int col = 0; col < COL; col++) {
            queueAtlantic.add(new Coord(bottomEdge, col));
            queuePacific.add(new Coord(topEdge, col));
        }

        //save all the left & right col edges that are touched by atlantic and pacific ocean
        for (int row = 0; row < ROW; row++) {
            queueAtlantic.add(new Coord(row, rightEdge));
            queuePacific.add(new Coord(row, leftEdge));
        }

        //Do BFS from all the atlantic ocean are corrds
        pacificAtlanticWaterFlow_HelperBFS(
                heights, queueAtlantic, visitedAtlantic, dirs, ROW, COL);

        //Do BFS from all the pacific ocean are corrds
        pacificAtlanticWaterFlow_HelperBFS(
                heights, queuePacific, visitedPacific, dirs, ROW, COL);

        for (int row = 0; row < ROW; row++) {
            for (int col = 0; col < COL; col++) {
                if (visitedAtlantic[row][col] && visitedPacific[row][col]) {
                    result.add(Arrays.asList(row, col));
                }
            }
        }
        //output
        System.out.println("All coordinates form pacific to atlantic ocean water flow: " + result);
    }

    public void minimumCostToFillGivenBag_DP_Memoization(int[] cost, int W) {

        //0-1Knapsack problem
        //problem statement: https://practice.geeksforgeeks.org/problems/minimum-cost-to-fill-given-weight-in-a-bag1956/1
        //create normal data
        List<Integer> value = new ArrayList<>();
        List<Integer> weight = new ArrayList<>();

        int actualSize = 0;
        for (int i = 0; i < cost.length; i++) {
            if (cost[i] != -1) {
                value.add(cost[i]);
                weight.add(i + 1);
                actualSize++;
            }
        }

        int[][] memo = new int[actualSize + 1][W + 1];
        for (int x = 0; x < actualSize + 1; x++) {
            for (int y = 0; y < W + 1; y++) {
                if (x == 0) {
                    memo[x][y] = Integer.MAX_VALUE;
                }
                if (y == 0) {
                    memo[x][y] = 0;
                }
            }
        }

        for (int x = 1; x < actualSize + 1; x++) {
            for (int y = 1; y < W + 1; y++) {
                if (weight.get(x - 1) > y) {
                    memo[x][y] = memo[x - 1][y];
                } else {
                    memo[x][y] = Math.min(value.get(x - 1) + memo[x][y - weight.get(x - 1)],
                            memo[x - 1][y]);
                }
            }
        }

        //output
        System.out.println("Min cost: " + memo[actualSize][W]);
    }

    public void minCoinsRequiredToMakeChangeInUnlimitedSupplyOfCoins_DP_Memoization(int[] coins, int change) {

        //Explanation ; SomePracticeQuestion.minNoOfCoinsUsedForChange()
        int N = coins.length;
        int[][] memo = new int[N + 1][change + 1];

        //base cond
        for (int x = 0; x < N + 1; x++) {
            for (int y = 0; y < change + 1; y++) {

                //if no coisn are available, we might need infinite-coins to make that change
                if (x == 0) {
                    memo[x][y] = Integer.MAX_VALUE - 1;
                }

                //if coins are available, but change we need to make is 0, wee need 0 coins
                if (y == 0) {
                    memo[x][y] = 0;
                }

                if (x == 1 && y >= 1) {
                    if (y % coins[x - 1] == 0) {
                        memo[x][y] = 1;
                    } else {
                        memo[x][y] = Integer.MAX_VALUE - 1;
                    }
                }
            }
        }

        for (int x = 1; x < N + 1; x++) {
            for (int y = 1; y < change + 1; y++) {
                //if the amount of coins is greater than the change(y) we are making
                //then just leave that coin, and move from that without making any change in y
                if (coins[x - 1] > y) {
                    memo[x][y] = memo[x - 1][y];
                } else {
                    //two choices 
                    //1. take that coin and adjust the change y with the amount of that coin
                    //and add 1 as picking up 1 coins in min
                    //2. don't take that coins and move to next coin, without making any adjustment in change y
                    memo[x][y] = Math.min(memo[x][y - coins[x - 1]] + 1,
                            memo[x - 1][y]);
                }
            }
        }

        //output
        System.out.println("Minimum coins required to make change: " + memo[N][change]);
    }

    public boolean checkIfStringCIsInterleavingOfStringAAndB_DP_Memoization(String a, String b, String c) {

        //.............................T: O(M * N) M = a.length(), N = b.length()
        //https://leetcode.com/problems/interleaving-string/
        //explanation: https://youtu.be/3Rw3p9LrgvE
        if (c.length() != a.length() + b.length()) {
            return false;
        }

        boolean dp[] = new boolean[b.length() + 1];
        for (int i = 0; i <= a.length(); i++) {
            for (int j = 0; j <= b.length(); j++) {
                if (i == 0 && j == 0) {
                    dp[j] = true;
                } else if (i == 0) {
                    dp[j] = dp[j - 1] && b.charAt(j - 1) == c.charAt(i + j - 1);
                } else if (j == 0) {
                    dp[j] = dp[j] && a.charAt(i - 1) == c.charAt(i + j - 1);
                } else {
                    dp[j] = (dp[j] && a.charAt(i - 1) == c.charAt(i + j - 1))
                            || (dp[j - 1] && b.charAt(j - 1) == c.charAt(i + j - 1));
                }
            }
        }
        return dp[b.length()];
    }

    public void minimumDiffPartition_DP_Memoization(int[] arr) {

        //https://practice.geeksforgeeks.org/problems/minimum-sum-partition3317/1#
        int sumArr = 0;
        for (int e : arr) {
            sumArr += e;
        }
        int n = arr.length;
        boolean[][] memo = new boolean[n + 1][sumArr + 1];
        for (int x = 0; x < n + 1; x++) {
            for (int y = 0; y < sumArr + 1; y++) {
                if (x == 0) {
                    memo[x][y] = false;
                }
                if (y == 0) {
                    memo[x][y] = true;
                }
            }
        }

        for (int x = 1; x < n + 1; x++) {
            for (int y = 1; y < sumArr + 1; y++) {
                if (arr[x - 1] > y) {
                    memo[x][y] = memo[x - 1][y];
                } else {
                    memo[x][y] = memo[x - 1][y] || memo[x - 1][y - arr[x - 1]];
                }
            }
        }

        int minDiff = Integer.MAX_VALUE;
        for (int i = (sumArr + 1) / 2; i >= 0; i--) {
            if (memo[n][i]) {
                minDiff = Math.abs(Math.min(minDiff, sumArr - 2 * i));
            }
        }

        //output
        System.out.println("Min diff between two sum of two partition of the array: " + minDiff);
    }

    public void maximalSquare_DP_Memoization(int[][] matrix) {
        //......................T: O(M*N), M = matrix row, N = matrix col
        //......................S: O(M*N), Using DP[M + 1][N + 1] OR O(1), Without using DP[][]
        //https://leetcode.com/problems/maximal-square/
        //https://practice.geeksforgeeks.org/problems/largest-square-formed-in-a-matrix0806/1
        int R = matrix.length;
        int C = matrix[0].length;
        int maxSqrLen = 0;

        //Using extra DP[][] space
//        int[][] memo = new int[R + 1][C + 1];
//        for (int r = 1; r < memo.length; r++) {
//            for (int c = 1; c < memo[r].length; c++) {
//                if (matrix[r - 1][c - 1] == 1) {
//                    memo[r][c] = Math.min(
//                            //diagonal
//                            memo[r - 1][c - 1],
//                            //up, left
//                            Math.min(memo[r - 1][c], memo[r][c - 1])
//                    ) + 1;
//                    maxSqrLen = Math.max(maxSqrLen, memo[r][c]);
//                }
//            }
//        }
        //Wihtout using extra DP[][] space
        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {
                if (matrix[r][c] == 1) {

                    if (r == 0 || c == 0) {
                        maxSqrLen = Math.max(maxSqrLen, matrix[r][c]);
                        continue;
                    }

                    matrix[r][c] = Math.min(
                            //diagonal
                            matrix[r - 1][c - 1],
                            Math.min(
                                    //up
                                    matrix[r - 1][c],
                                    //left
                                    matrix[r][c - 1])
                    ) + 1;
                    maxSqrLen = Math.max(maxSqrLen, matrix[r][c]);
                }
            }
        }

        //Output:
        System.out.println("Max square in binary matrix: length: " + maxSqrLen + " area: " + (maxSqrLen * maxSqrLen));
    }

    public void countAllSquareWithOneInBinaryMatrix_DP_Memoization(int[][] matrix) {
        //......................T: O(M*N), M = matrix row, N = matrix col
        //......................S: O(M*N), Using DP[M + 1][N + 1] OR O(1), Without using DP[][]
        //https://leetcode.com/problems/count-square-submatrices-with-all-ones
        //explanation: https://youtu.be/Z2h3rkVXPeQ 
        int totalSquares = 0;
        int R = matrix.length;
        int C = matrix[0].length;

        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {

                //consider only 1 for our squares
                if (matrix[r][c] == 1) {
                    //first row & col, if they have 1 there, we can simply
                    //add in out total beacuse they at the top & left edge of matrix 
                    //they will not be forming any squares
                    if (r == 0 || c == 0) {
                        totalSquares += matrix[r][c]; //basically 1 at the top & left edge
                        continue;
                    }
                    matrix[r][c] = Math.min(
                            //diagonal
                            matrix[r - 1][c - 1],
                            Math.min(
                                    //row above
                                    matrix[r - 1][c],
                                    //left col
                                    matrix[r][c - 1]
                            )
                    ) + 1;
                    totalSquares += matrix[r][c];
                }
            }
        }
        //output:
        System.out.println("Total squares in given matrix: " + totalSquares);
    }

    public void longestStringChain_DP_Memoization(String[] words) {
        //https://leetcode.com/problems/longest-string-chain/
        //https://leetcode.com/problems/longest-string-chain/discuss/2094249/Java-DP

        //map<word, chain>, this map stores the no of chains can be formed by each word
        Map<String, Integer> memo = new HashMap<>();
        for (String word : words) {
            //every single word is in itself is a chain, thats why 1
            memo.put(word, 1);
        }

        //incr order of their lengths
        Arrays.sort(words, (s1, s2) -> s1.length() - s2.length());
        int maxChain = 0;
        for (String word : words) {
            //for each word in words[], try to remove one char and
            //check if that new string matches any previous string
            int currWordLength = word.length();
            for (int i = 0; i < currWordLength; i++) {
                //escape ith char from current word
                //because question says 
                //'we can insert exactly one letter anywhere in previousWord without
                //changing the order of the other characters to make it equal to curr word'
                String previousWord = word.substring(0, i) + word.substring(i + 1);
                if (memo.containsKey(previousWord)) {
                    memo.put(word,
                            //taking max because
                            //forming a chain with curr word may create a longer chain
                            //or the previousWord already have the longer chain( with their previous words)
                            //+ 1 for this curr word
                            Math.max(
                                    memo.get(word),
                                    memo.get(previousWord) + 1
                            ));
                }
            }
            maxChain = Math.max(maxChain, memo.get(word));
        }
        //output
        System.out.println("Max string chain formed: " + maxChain);
    }

    private void minimumUnfairDistributionOfCookiesToKStudent_Backtracking_findUnfairness(
            int[] cookieSumToKthStudent) {

        int currMaxSum = cookieSumToKthStudent[0];
        for (int sum : cookieSumToKthStudent) {
            currMaxSum = Math.max(currMaxSum, sum);
        }
        minimumUnfairDistributionOfCookiesToKStudent_Result = Math.min(
                minimumUnfairDistributionOfCookiesToKStudent_Result,
                currMaxSum);
    }

    private void minimumUnfairDistributionOfCookiesToKStudent_Backtracking_Helper(
            int[] cookies, int[] cookieSumToKthStudent, int currCookieTaken) {

        if (currCookieTaken >= cookies.length) {
            minimumUnfairDistributionOfCookiesToKStudent_Backtracking_findUnfairness(cookieSumToKthStudent);
            return;
        }

        for (int i = 0; i < cookieSumToKthStudent.length; i++) {
            cookieSumToKthStudent[i] += cookies[currCookieTaken];
            minimumUnfairDistributionOfCookiesToKStudent_Backtracking_Helper(
                    cookies, cookieSumToKthStudent, currCookieTaken + 1);
            cookieSumToKthStudent[i] -= cookies[currCookieTaken];
        }
    }

    private int minimumUnfairDistributionOfCookiesToKStudent_Result;

    public void minimumUnfairDistributionOfCookiesToKStudent_Backtracking(int[] cookies, int k) {
        //https://leetcode.com/problems/fair-distribution-of-cookies/
        //https://leetcode.com/problems/fair-distribution-of-cookies/discuss/2212309/JAVA-oror-SIMPLE-BACKTRACKING
        int[] cookieSumToKthStudent = new int[k];
        minimumUnfairDistributionOfCookiesToKStudent_Result = Integer.MAX_VALUE;
        minimumUnfairDistributionOfCookiesToKStudent_Backtracking_Helper(
                cookies, cookieSumToKthStudent, 0);
        //output:
        System.out.println("Min unfairness in distribution of cookies: "
                + minimumUnfairDistributionOfCookiesToKStudent_Result);
    }

    private boolean partitionToKEqualSumSubset_Backtracking_Helper(int[] nums, int k,
            int index, int currSubsetSum, int sumPerKSubset, Set<Integer> indexUsed) {

        if (k == 0) {
            return true;
        }

        if (currSubsetSum == sumPerKSubset) {
            return partitionToKEqualSumSubset_Backtracking_Helper(nums, k - 1,
                    0, 0, sumPerKSubset, indexUsed);
        }

        for (int i = index; i < nums.length; i++) {
            if (indexUsed.contains(i)
                    //handle duplicates
                    || (i - 1 >= 0 && nums[i] == nums[i - 1] && !indexUsed.contains(i - 1))
                    //handles case where adding curr nums[i] to currSubsetSum
                    //will not give any solution
                    || currSubsetSum + nums[i] > sumPerKSubset) {
                continue;
            }

            indexUsed.add(i);
            if (partitionToKEqualSumSubset_Backtracking_Helper(
                    nums, k, i + 1, currSubsetSum + nums[i], sumPerKSubset, indexUsed)) {
                return true;
            }
            indexUsed.remove(i);
        }
        return false;
    }

    public boolean partitionToKEqualSumSubset_Backtracking(int[] nums, int k) {
        //.............................T: O(2^(k*N)), N is length of array, we will be
        //trying all possible k subsets by making decision if a index to be used in subset
        //for next time or not.
        //https://leetcode.com/problems/partition-to-k-equal-sum-subsets/
        //https://leetcode.com/problems/partition-to-k-equal-sum-subsets/discuss/2360226/Java-ororBeats-90-oror-Hardcore-expl'n-!!
        //explanation: https://youtu.be/mBk4I0X46oI

        //we cant make k partitions(non-empty subsets) if array element is less than k
        if (k > nums.length) {
            return false;
        }

        int arrSum = 0;
        for (int val : nums) {
            arrSum += val;
        }

        //k subset partition should have equal sum for each subset
        //if the total array sum is not divisible by k then one of the k 
        //partition will have more sum than the others
        //ex: nums = [5,5,5,5], k = 4 ==> arrSum = 20 where arrSum % k == 0
        //that means subset = {{5},{5},{5},{5}} all sub set have equal sum of 5
        //ex: nums = [5,5,5,6], k = 4 ==> arrSum = 21 where arrSum % k != 0
        //that means subset = {{5},{5},{5},{6}} all sub set doesn't have equal sum
        if (arrSum % k != 0) {
            return false;
        }
        //sum of each subset
        int sumPerKSubset = arrSum / k;
        Set<Integer> indexUsed = new HashSet<>();
        //sort nums so that we can handle duplicates while making decisions
        Arrays.sort(nums);
        return partitionToKEqualSumSubset_Backtracking_Helper(nums, k, 0, 0, sumPerKSubset, indexUsed);
    }

    private void nQueens_Backtracking_Helper_HashSetCheck(
            char[][] board, int col, int n,
            List<List<String>> queenPlacements, Set<String> previousPlacedQueens) {

        //if we have placed our queens successfully
        //starting from col == 0 to col == n - 1
        //then when col == n is called, we can say the curr placement of queens
        //are valid palcements
        if (col == n) {
            //as per output format, converting each row to string format
            List<String> currPlacement = new ArrayList<>();
            for (char[] row : board) {
                currPlacement.add(String.valueOf(row));
            }
            queenPlacements.add(currPlacement);
            return;
        }

        //for a given col we will try to place our queen in each row
        //of that col, if it is safe to place there we will move to next col
        //to try to place our next queen safely
        for (int row = 0; row < n; row++) {
            //first we will check if it is safe to place our curr queen
            //at a given row and col
            //safety rule:
            //1. no queen should already be there in upper-left diagonal
            //2. no queen should already be there in curr row straight-left col
            //3. no queen should already be there in bottom-left diagonal
            //if a queen is already placed on any of the below locations we don't
            //want to place our curr queen in alignment with previous queen
            if (previousPlacedQueens.contains("UPPER-LEFT-DIAGONAL" + (n - 1 + col - row))
                    || previousPlacedQueens.contains("STRAIGHT-LEFT" + row)
                    || previousPlacedQueens.contains("BOTTOM-LEFT-DIAGONAL" + (row + col))) {
                continue;
            }
            //we will try place our Queen in each row
            //in a given col

            previousPlacedQueens.add("UPPER-LEFT-DIAGONAL" + (n - 1 + col - row));
            previousPlacedQueens.add("STRAIGHT-LEFT" + row);
            previousPlacedQueens.add("BOTTOM-LEFT-DIAGONAL" + (row + col));

            board[row][col] = 'Q';

            nQueens_Backtracking_Helper_HashSetCheck(
                    board, col + 1, n, queenPlacements, previousPlacedQueens);

            board[row][col] = '.';

            previousPlacedQueens.remove("UPPER-LEFT-DIAGONAL" + (n - 1 + col - row));
            previousPlacedQueens.remove("STRAIGHT-LEFT" + row);
            previousPlacedQueens.remove("BOTTOM-LEFT-DIAGONAL" + (row + col));
        }
    }

    private boolean nQueens_Backtracking_IsSafeToPlaceQueen(
            char[][] board, int row, int col, int n) {
        /*
         //since we are moving col wise by placing our queen in curr col
         //and we are moving like [0 <= col < n] therefore from our col there will
         //be no queens placed in right, only before this curr col.
         //so keeping this in mind we will only check all the prev dires from the curr col
         //to check if there were any queen placed in previous itrations or not
         upper-left diagonal
         straight-left col
         bootom-left diagonal
         .\
         ..\
         ...\
         ....\
        
         ----- [COL]
        
         ..../
         .../
         ../
         ./
        
         */
        int currRow = row;
        int currCol = col;

        //upper-left diagonal
        while (currRow >= 0 && currCol >= 0) {
            if (board[currRow][currCol] == 'Q') {
                return false;
            }
            currRow--;
            currCol--;
        }

        currRow = row;
        currCol = col;
        //same row straight-left col
        while (currCol >= 0) {
            if (board[currRow][currCol] == 'Q') {
                return false;
            }
            currCol--;
        }

        currRow = row;
        currCol = col;
        //bottom-left diagonal
        while (currRow < n && currCol >= 0) {
            if (board[currRow][currCol] == 'Q') {
                return false;
            }
            currRow++;
            currCol--;
        }
        return true;
    }

    private void nQueens_Backtracking_Helper(
            char[][] board, int col, int n, List<List<String>> queenPlacements) {

        //if we have placed our queens successfully
        //starting from col == 0 to col == n - 1
        //then when col == n is called, we can say the curr placement of queens
        //are valid palcements
        if (col == n) {
            //as per output format, converting each row to string format
            List<String> currPlacement = new ArrayList<>();
            for (char[] row : board) {
                currPlacement.add(String.valueOf(row));
            }
            queenPlacements.add(currPlacement);
            return;
        }

        //for a given col we will try to place our queen in each row
        //of that col, if it is safe to place there we will move to next col
        //to try to place our next queen safely
        for (int row = 0; row < n; row++) {
            //first we will check if it is safe to place our curr queen
            //at a given row and col
            //safety rule:
            //1. no queen should already be there in upper-left diagonal
            //2. no queen should already be there in curr row straight-left col
            //3. no queen should already be there in bottom-left diagonal
            if (nQueens_Backtracking_IsSafeToPlaceQueen(board, row, col, n)) {
                //we will try place our Queen in each row
                //in a given col
                board[row][col] = 'Q';
                nQueens_Backtracking_Helper(board, col + 1, n, queenPlacements);
                board[row][col] = '.';
            }
        }
    }

    public void nQueens_Backtracking(int n) {
        //https://leetcode.com/problems/n-queens/
        //explanation: https://youtu.be/i05Ju7AftcM
        List<List<String>> queenPlacements = new ArrayList<>();
        char[][] board = new char[n][n];
        for (int r = 0; r < n; r++) {
            for (int c = 0; c < n; c++) {
                board[r][c] = '.';
            }
        }
        nQueens_Backtracking_Helper(board, 0, n, queenPlacements);

        //on leetcode this set based checking was slow,
        //keep this approach if in case it was asked
//        Set<String> previousPlacedQueens = new HashSet<>();
//        nQueens_Backtracking_Helper_HashSetCheck(board, 0, n, queenPlacements, previousPlacedQueens);
        //output:
        System.out.println("All possible placement of N-Queens: " + queenPlacements);
    }

    private boolean sudokuSolver_Backtracking_IsValidToPutNum(
            char[][] board, int row, int col, char num) {
        for (int i = 0; i < board.length; i++) {
            //if in row wise dir already have same num as we are tyring to put
            if (board[i][col] == num) {
                return false;
            }
            //if in col wise dir already have same num as we are tyring to put
            if (board[row][i] == num) {
                return false;
            }
            //if the curr 3*3 submatrix already have same num as we are tyring to put
            if (board[3 * (row / 3) + i / 3][3 * (col / 3) + i % 3] == num) {
                return false;
            }
        }
        return true;
    }

    private boolean sudokuSolver_Backtracking_Helper(char[][] board) {

        for (int r = 0; r < board.length; r++) {
            for (int c = 0; c < board[r].length; c++) {
                //if the curr spot is empty
                if (board[r][c] == '.') {
                    //with this loop we are trying to put in all the possible num
                    //in the given row and col
                    for (char num = '1'; num <= '9'; num++) {
                        //for curr num we will check if it is valid to place this num
                        //in curr row and col, if it is safe we will put that num into
                        //the curr empty spot and move to new location by recursion
                        if (sudokuSolver_Backtracking_IsValidToPutNum(board, r, c, num)) {
                            board[r][c] = num;
                            //if by recursion we find the sudoku is solved we will return true
                            //if it can't be solved keep the original empty spot in the curr row and col
                            if (sudokuSolver_Backtracking_Helper(board)) {
                                return true;
                            } else {
                                board[r][c] = '.';
                            }
                        }
                    }
                    //if above loop of [1 to 9] doesn't satisfy the sudoku board
                    return false;
                }
            }
        }
        return true;
    }

    public void sudokuSolver_Backtracking(char[][] board) {
        //https://leetcode.com/problems/sudoku-solver
        //explanation: https://youtu.be/FWAIf_EVUKE
        //actual:
        System.out.println("Empty sudoku board");
        for (int r = 0; r < board.length; r++) {
            for (int c = 0; c < board[r].length; c++) {
                System.out.print(board[r][c] + "\t");
            }
            System.out.println();
        }

        sudokuSolver_Backtracking_Helper(board);

        //output
        System.out.println("Solved sudoku");
        for (int r = 0; r < board.length; r++) {
            for (int c = 0; c < board[r].length; c++) {
                System.out.print(board[r][c] + "\t");
            }
            System.out.println();
        }
    }

    public void wordBreakTwo_Backtracking_Helper(
            String str, String currStr, Set<String> wordSet, List<String> result) {

        int n = str.length();

        if (n == 0) {
            result.add(currStr.trim());
            return;
        }

        for (int i = 1; i <= n; i++) {
            //prefix substring
            String substr = str.substring(0, i);
            //if some prefix substr is there in wordSet
            //then we can form our currStr with this substr and
            //recursively check on the new left over substring str[i, n] like
            //how many prefix substr is matching in that new substr
            if (wordSet.contains(substr)) {
                wordBreakTwo_Backtracking_Helper(
                        //left over new substr
                        str.substring(i, n),
                        //pass currStr as new string object with curr combination
                        //of substr as (currStr + " " + substr)
                        //ex at i = 3 : these are recur combinations
                        //"" + " " + "cat" ==> recur
                        //" cat" + " " + "sand" ==> recur
                        //" cat sand dog" ==> recur then str.length() == 0 (Base Cond)
                        //at i = 4 : these are recur combinations
                        //"" + " " + "cats" ==> recur
                        //" cats" + " " + "and" ==> recur
                        //" cats and dog" ==> recur then str.length() == 0 (Base Cond)
                        String.valueOf(currStr + " " + substr),
                        wordSet,
                        result);
            }
        }
    }

    public void wordBreakTwo_Backtracking(String str, String[] wordDict) {
        //https://leetcode.com/problems/word-break-ii/
        //based on wordBreak_Recursive
        List<String> result = new ArrayList<>();
        Set<String> wordSet = new HashSet<>();
        wordSet.addAll(Arrays.asList(wordDict));
        wordBreakTwo_Backtracking_Helper(str, "", wordSet, result);
        //output
        System.out.println("Word break all string combinations (Backtracking): " + result);
    }

    public void LRUCacheDesignImpl(List<String> operations, List<List<Integer>> inputs) {

        LRUCacheDesign lruObj = null;
        for (int i = 0; i < operations.size(); i++) {
            String operation = operations.get(i);
            switch (operation) {
                case "LRUCache":
                    int capacity = inputs.get(i).get(0);
                    lruObj = new LRUCacheDesign(capacity);
                    System.out.println("Object created:" + operation);
                    break;
                case "put":
                    int key = inputs.get(i).get(0);
                    int value = inputs.get(i).get(1);
                    lruObj.put(key, value);
                    System.out.println("Put: " + key + " " + value);
                    break;
                case "get":
                    key = inputs.get(i).get(0);
                    System.out.println("Get: " + lruObj.get(key));
                    break;
            }
        }
    }

    //MY AMAZON ONLINE ASSESSMENT
    public boolean robotRodeo(String command) {

        //initial state of robot
        int x = 0;
        int y = 0;
        int dir = 0; //0 = North, 1 = East, 2 = South, 3 = West (in clockwise direction of actual N-E-S-W dir)
        for (char move : command.toCharArray()) {

            if (move == 'R') {
                //for a given curr dir, R would be on +1 side in clockwise way
                //where %4 will bound our dir upto 4 actual dir(N-E-S-W dir)
                dir = (dir + 1) % 4;
            } else if (move == 'L') {
                //for a given curr dir, L would be on +3 side in clockwise way
                //where %4 will bound our dir upto 4 actual dir(N-E-S-W dir)
                dir = (dir + 3) % 4;
            } else {
                //now move = G
                if (dir == 0) {
                    //North and G, move robot to vertical up(or to north) from given (x,y)
                    y++;
                } else if (dir == 1) {
                    //East and G, move robot to horizontal right(or to east side) from given (x,y)
                    x++;
                } else if (dir == 2) {
                    //South and G, move robot to vertical down(or to south side) from given (x,y)
                    y--;
                } else { //dir == 3
                    //West and G, move robot to horizontal left(or to west side) from given (x,y)
                    x--;
                }
            }
        }

        //if the robot returned to same inital (x,y) = (0,0) after all moves in given command
        //that means there exists a cycle
        return dir != 0 || (x == 0 && y == 0);
    }

    public int swapsRequiredToSortArray(int[] arr) {

        //https://www.geeksforgeeks.org/minimum-number-swaps-required-sort-array/
        int result = 0;
        Map<Integer, Integer> index = new HashMap<>();
        int[] sortedArr = arr.clone();

        for (int i = 0; i < arr.length; i++) {
            index.put(arr[i], i);
        }

        Arrays.sort(sortedArr);
        for (int i = 0; i < arr.length; i++) {

            if (arr[i] != sortedArr[i]) {
                result++;
                int init = arr[i];

                //swap
                swapIntArray(arr, i, index.get(sortedArr[i]));
                //adjust the indexes after swapping
                index.put(init, index.get(sortedArr[i]));
                index.put(sortedArr[i], i);
            }
        }

        return result;
    }

    public void checkBinaryNumberStreamIsDivisibleByN(int[] binaryStream, int N) {

        //explanation: https://www.geeksforgeeks.org/check-divisibility-binary-stream/
        //see method 2
        /*
        
         formula: 
         if bit is 1 new decimal = 2 * prevDecimal + 1
         if bit is 0 new decimal = 2 * prevDecimal
        
         Ex:
         binaryStream = [1,0,1,0,1]
         prevDecimal = 0
         binaryFormed = ""
         i = 0
         bit = 1 -> binaryFormed.append(bit) = "1" == actualDecimal(binaryFormed) = 1
         if bit == 1: prevDecimal = (2 * prevDecimal) + 1
         ---> prevDecimal = 2 * 0 + 1 = 1
        
         i = 1
         bit = 0 -> binaryFormed.append(bit) = "10" == actualDecimal(binaryFormed) = 2
         if bit == 0: prevDecimal = (2 * prevDecimal)
         ---> prevDecimal = 2 * 1 = 2
        
         i = 2
         bit = 1 -> binaryFormed.append(bit) = "101" == actualDecimal(binaryFormed) = 5
         if bit == 1: prevDecimal = (2 * prevDecimal) + 1
         ---> prevDecimal = 2 * 2 + 1= 5
        
         so on...
         */
        int remainder = 0;
        int decimal = 0;
        StringBuilder sb = new StringBuilder(); //just for output purpose, not necessary to use
        for (int bit : binaryStream) {

            if (bit == 0) {
                remainder = (2 * remainder) % N;
                decimal = 2 * decimal;
            } else if (bit == 1) {
                remainder = (2 * remainder + 1) % N;
                decimal = 2 * decimal + 1;
            }

            sb.append(bit);
            if (remainder == 0) { //another way if(decimal % N) but TLE
                System.out.println("Binary formed: " + sb.toString() + " dec(" + decimal + ") is divisible by " + N);
            } else {
                System.out.println("Binary formed: " + sb.toString() + " dec(" + decimal + ") is not divisible by " + N);
            }
        }
    }

    private String convertNumberToWords_Helper(int n, String suff) {

        // Strings at index 0 is not used, it is to make array 
        // indexing simple 
        String one[] = {"", "one ", "two ", "three ", "four ",
            "five ", "six ", "seven ", "eight ",
            "nine ", "ten ", "eleven ", "twelve ",
            "thirteen ", "fourteen ", "fifteen ",
            "sixteen ", "seventeen ", "eighteen ",
            "nineteen "};

        // Strings at index 0 and 1 are not used, they is to 
        // make array indexing simple 
        String ten[] = {"", "", "twenty ", "thirty ", "forty ",
            "fifty ", "sixty ", "seventy ", "eighty ",
            "ninety "};

        String str = "";
        // if n is more than 19, divide it 
        if (n > 19) {
            str += ten[n / 10] + one[n % 10];
        } else {
            str += one[n];
        }

        // if n is non-zero 
        if (n != 0) {
            str += suff;
        }

        return str;
    }

    public void convertNumberToWords(long n) {

        //https://www.geeksforgeeks.org/program-to-convert-a-given-number-to-words-set-2/
        StringBuilder sb = new StringBuilder();

        // handles digits at ten millions and hundred 
        // millions places (if any) 
        sb.append(convertNumberToWords_Helper((int) (n / 1000_000_0) % 100, "crore "));

        // handles digits at hundred thousands and one 
        // millions places (if any) 
        sb.append(convertNumberToWords_Helper((int) ((n / 100_000) % 100), "lakh "));

        // handles digits at thousands and tens thousands 
        // places (if any) 
        sb.append(convertNumberToWords_Helper((int) ((n / 1000) % 100), "thousand "));

        // handles digit at hundreds places (if any) 
        sb.append(convertNumberToWords_Helper((int) ((n / 100) % 10), "hundred "));

        if (n > 100 && n % 100 > 0) {
            sb.append("and ");
        }

        // handles digits at ones and tens places (if any) 
        sb.append(convertNumberToWords_Helper((int) (n % 100), ""));

        //output
        System.out.println("In words: \n" + (sb.toString().equals("") ? "zero" : sb.toString()));
    }

    public void printPascalTriangle_SimpleAddition(int rows) {

        //https://leetcode.com/problems/pascals-triangle/
        List<List<Integer>> triangle = new ArrayList<>();

        if (rows == 0) {
            return;
        }

        triangle.add(new ArrayList<>());
        triangle.get(0).add(1); //base 

        for (int r = 1; r < rows; r++) {
            List<Integer> currRow = new ArrayList<>();
            List<Integer> prevRow = triangle.get(r - 1);

            currRow.add(1); //first 1 of the row
            for (int c = 1; c < r; c++) {

                currRow.add(prevRow.get(c - 1) + prevRow.get(c));
            }
            currRow.add(1); //last 1 of the row
            triangle.add(currRow);
        }

        //output
        System.out.println("Pascal tiangle: " + triangle);
    }

    public void printPascalTriangle_BinomialCoeff(int rows) {

        //https://leetcode.com/problems/pascals-triangle/
        //https://www.geeksforgeeks.org/pascal-triangle/
        List<List<Integer>> result = new ArrayList<>();

        if (rows == 0) {
            return;
        }

        for (int r = 1; r <= rows; r++) {
            int X = 1;
            result.add(new ArrayList<>());
            for (int c = 1; c <= r; c++) {
                result.get(r - 1).add(X);
                X = X * (r - c) / c;
            }
        }

        //output
        System.out.println("Pascal tiangle: " + result);
    }

    private boolean findCelebrityInNPepole_knows(int a, int b) {
        //if i knows j then its 1 else 0
        //for N * N people matrix
        int[][] whoKnowsWhom = {
            {0, 0, 1, 0},
            {0, 0, 1, 0},
            {0, 0, 0, 0},
            {0, 0, 1, 0}}; //SHOULD BE GLOBAL
        return whoKnowsWhom[a][b] == 1;
    }

    public int findCelebrityInNPepole(int N) {

        //......................T: O(N * N)
        //......................S: O(N), indegree[]
        //https://www.geeksforgeeks.org/the-celebrity-problem/
        int[] indegree = new int[N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (findCelebrityInNPepole_knows(i, j)) {
                    indegree[j]++;
                    indegree[i]--;
                }
            }
        }

        for (int i = 0; i < N; i++) {
            if (indegree[i] == N - 1) {
                return i;
            }
        }
        return -1;
    }

    public int findCelebrityInNPepole_Optimized(int N) {
        //......................T: O(N)
        //......................S: O(1)
        //https://www.geeksforgeeks.org/the-celebrity-problem/
        //OPTIMISED
        //TWO POINTER APPROACH
        int potentialCelebrityStart = 0;
        int potentialCelebrityEnd = N - 1;

        //until endPerson and startPerson becomes equal
        while (potentialCelebrityEnd > potentialCelebrityStart) {

            if (findCelebrityInNPepole_knows(potentialCelebrityStart, potentialCelebrityEnd)) {
                potentialCelebrityStart++;
            } else {
                potentialCelebrityEnd--;
            }
        }

        //startPerson == endPerson consider to be a potential celebrity
        for (int ordinaryPerson = 0; ordinaryPerson < N; ordinaryPerson++) {
            // If any person doesn't
            // know 'startPerson' or 'startPerson' doesn't
            // know any person, return -1
            if (ordinaryPerson != potentialCelebrityStart
                    && (findCelebrityInNPepole_knows(potentialCelebrityStart, ordinaryPerson) /*if celebrity know any person then he can't be celebrity*/
                    || !findCelebrityInNPepole_knows(ordinaryPerson, potentialCelebrityStart) /*if any person doesn't know celebrity then he can't be celebrity*/)) {
                return -1;
            }
        }
        return potentialCelebrityStart;
    }

    private int maximumDistanceCoveredInRobotWalkingSimulation_UpdateY(int currX, int currY,
            int currDir, int dist, Set<String> obstacles) {

        //for y dir can either be 0 = N OR 2 = S
        if (currDir == 0) { //y + dist
            for (int i = 0; i < dist; i++) {
                currY++;
                if (obstacles.contains(currX + "-" + currY)) {
                    //updated x-y coordinate is in obstacle
                    //take a step back, and return as we can't move further
                    currY--;
                    return currY;
                }
            }
        } else { //y - dist
            for (int i = 0; i < dist; i++) {
                currY--;
                if (obstacles.contains(currX + "-" + currY)) {
                    //updated x-y coordinate is in obstacle
                    //take a step ahead, and return as we can't move further
                    currY++;
                    return currY;
                }
            }
        }
        return currY;
    }

    private int maximumDistanceCoveredInRobotWalkingSimulation_UpdateX(int currX, int currY,
            int currDir, int dist, Set<String> obstacles) {

        //for x dir can either be 1 = E OR 3 = W
        if (currDir == 1) { //x + dist
            for (int i = 0; i < dist; i++) {
                currX++;
                if (obstacles.contains(currX + "-" + currY)) {
                    //updated x-y coordinate is in obstacle
                    //take a step back, and return as we can't move further
                    currX--;
                    return currX;
                }
            }
        } else { //x - dist
            for (int i = 0; i < dist; i++) {
                currX--;
                if (obstacles.contains(currX + "-" + currY)) {
                    //updated x-y coordinate is in obstacle
                    //take a step ahead, and return as we can't move further
                    currX++;
                    return currX;
                }
            }
        }
        return currX;
    }

    public void maximumDistanceCoveredInRobotWalkingSimulation(int[] commands, int[][] obstacles) {

        //https://leetcode.com/problems/walking-robot-simulation/
        Set<String> set = new HashSet<>();
        for (int[] obstacle : obstacles) {
            set.add(obstacle[0] + "-" + obstacle[1]);
        }

        //max distance covered in commands
        int maxDistCoveredFromOrigin = 0;

        //initial position
        int x = 0;
        int y = 0;

        //initial dir
        int dir = 0; // 0 = N, 1 = E, 2 = S, 3 = W

        for (int move : commands) {

            if (move == -1) { // -1 == TURN RIGHT
                dir = (dir + 1) % 4;
            } else if (move == -2) { // -2 == TURN LEFT
                dir = (dir + 3) % 4;
            } else { // MAKE STEPS AHEAD, BUT KEEP CHECK ON OBSTACLES
                if (dir == 0) { // NORTH y++
                    y = maximumDistanceCoveredInRobotWalkingSimulation_UpdateY(x, y, dir, move, set);
                } else if (dir == 1) { // EAST x++
                    x = maximumDistanceCoveredInRobotWalkingSimulation_UpdateX(x, y, dir, move, set);
                } else if (dir == 2) { // SOUTH y--
                    y = maximumDistanceCoveredInRobotWalkingSimulation_UpdateY(x, y, dir, move, set);
                } else { // WEST x--
                    x = maximumDistanceCoveredInRobotWalkingSimulation_UpdateX(x, y, dir, move, set);
                }

                //calculate distance 
                //robot's curr position(x,y) is the farthest then update maxDistCoveredFromOrigin
                maxDistCoveredFromOrigin = Math.max(maxDistCoveredFromOrigin,
                        (x * x + y * y)); //x^2 + y^2
            }
        }

        //output
        System.out.println("Max distance covered by robot from origin : " + maxDistCoveredFromOrigin);
    }

    public void brokenCalculatorMakeXEqualToY(int x, int y) {

        //............................T: O(LogY)
        //https://leetcode.com/problems/broken-calculator/
        //explanation: https://leetcode.com/problems/broken-calculator/solution/
        //Work backwards
        int res = 0;
        while (y > x) {
            res++;
            if (y % 2 == 1) {
                y++;
            } else {
                y /= 2;
            }
        }

        //output
        System.out.println("Min operations on X make equal to Y: " + (res + x - y));
    }

    public boolean rectangleOverlappingAndArea(int[] rec1, int[] rec2) {
        //explanation: https://youtu.be/zGv3hOORxh0
        //https://leetcode.com/problems/rectangle-overlap/
        //https://leetcode.com/problems/rectangle-area/
        //input format:
        //rec[4] = bottm-left coordinate(x, y) = rec[0], rec[1] & top-right coordinate(x, y) = rec[2], rec[3]

        /*
         ::::::::::R1-----------
         ::::::::::::|         |
         ::::::::::::|         |
         :::::R2----------     |
         :::::::|    |=O=|     |
         :::::::|    -----------
         :::::::|        |
         :::::::|        |
         ::::::::---------
         */
        //overlapped rectangle's bottom-left(x, y)
        int bottomLeftX = Math.max(rec1[0], rec2[0]);
        int bottomLeftY = Math.max(rec1[1], rec2[1]);

        //overlapped rectangle's top-right(x, y)
        int topRightX = Math.min(rec1[2], rec2[2]);
        int topRightY = Math.max(rec1[3], rec2[3]);

        int lengthX = topRightX - bottomLeftX;
        int lengthY = topRightY - bottomLeftY;

        //if rec1 & rec2 are not overlapping case
        if (lengthX < 0 || lengthY < 0) {
            return false;
        }
        //given 2 rectangles, if they are connected by any
        //one corner or edge they are not overlapped.
        //area = length * bredth
        int area = lengthX * lengthY;
        System.out.println("Area of overlapped rectangle: " + area);
        return area > 0;
    }

    public void addTwoNumsWithoutPlusOrMinus(int a, int b) {
        //https://leetcode.com/problems/sum-of-two-integers
        //explanation: https://youtu.be/gIlZOcZHtlQ
        //^ = XOR
        int sum = a ^ b;
        //& = AND
        int carry = a & b;
        while (carry != 0) {
            carry = carry << 1;
            int tempSum = sum ^ carry;
            int tempCarry = sum & carry;
            sum = tempSum;
            carry = tempCarry;
        }
        //output
        System.out.println("Adding two nums: " + a + " " + b + " without + or - : " + sum);
    }

    public void threeConsecutiveNumberThatSumsToGivenNumber(int num) {
        //https://leetcode.com/problems/find-three-consecutive-integers-that-sum-to-a-given-number/
        //https://leetcode.com/problems/find-three-consecutive-integers-that-sum-to-a-given-number/discuss/2164778/the-three-integers-will-be-in-arithmetic-progression
        if (num % 3 == 0) {
            int first = (num / 3) - 1;
            int second = (num / 3);
            int third = (num / 3) + 1;
            System.out.println("Three consecutive number that sums to " + num + " : "
                    + first + ", " + second + ", " + third);
            return;
        }
        System.out.println("Three consecutive number that sums to " + num + " : Not possible");
    }

    public void detectSquares(List<int[]> points, List<int[]> queryPoints) {
        //Input is little bit modified from the actual question 
        //https://leetcode.com/problems/detect-squares/
        //explanation: https://youtu.be/bahebearrDc
        //<"x,y", freq>
        Map<String, Integer> pointMap = new HashMap<>();
        for (int[] point : points) {
            int x = point[0];
            int y = point[1];
            String key = x + "," + y;
            pointMap.put(key, pointMap.getOrDefault(key, 0) + 1);
        }

        for (int[] query : queryPoints) {
            int qX = query[0];
            int qY = query[1];
            int sqauresDetectedFromCurrQueryPoint = 0;
            for (int[] point : points) {
                int x = point[0];
                int y = point[1];

                //if curr point[x, y] is not forming a diagonal with query[qX, qY]
                // or any of the curr x, y is same as qX, qY then they can't form diagnal
                //with query[qX, qY]
                boolean isDiagonal = Math.abs(x - qX) == Math.abs(y - qY);
                if (!isDiagonal || x == qY || y == qY) {
                    continue;
                }

                //otherwise now here find two coord points
                //top-left such a way that its coord [x, qY]
                //bottom-right such a way that its coord [qX, y]
                //if they exists in out pointMap in some freq that much freq
                //will form sqaures
                String topLeftKey = x + "," + qY;
                String bottomRightKey = qX + "," + y;

                sqauresDetectedFromCurrQueryPoint
                        += pointMap.getOrDefault(topLeftKey, 0) * pointMap.getOrDefault(bottomRightKey, 0);
            }
            System.out.println("Squares detected from query[qX, qY]: [" + qX + ", " + qY + "] : "
                    + sqauresDetectedFromCurrQueryPoint);
        }
    }

    public void serverAllocationToTasks(int[] servers, int[] tasks) {
        //https://leetcode.com/problems/process-tasks-using-servers
        //https://www.geeksforgeeks.org/google-interview-experience-for-software-engineer-l3-bangalore-6-years-experienced/
        class Server {

            int index;
            int weight;
            //taskArrivalTime + taskProcessingTime
            int bookedTime;

            public Server(int index, int weight) {
                this.index = index;
                this.weight = weight;
            }
        }

        int serverLen = servers.length;
        int taskLen = tasks.length;

        int[] serverIdxAllotedPerTask = new int[taskLen];

        PriorityQueue<Server> freeServer = new PriorityQueue<>(
                //choose server with lowest weight, if weights are same
                //choose server with lowest index value
                (s1, s2) -> s1.weight == s2.weight
                        ? s1.index - s2.index
                        : s1.weight - s2.weight
        );

        PriorityQueue<Server> busyServer = new PriorityQueue<>(
                //server's booked time(booked time == taskArrivalTime + taskProcessiingTime) are same
                (s1, s2) -> s1.bookedTime == s2.bookedTime
                        //choose server with lowest weight, if weights are same
                        //choose server with lowest index value
                        ? (s1.weight == s2.weight
                                ? s1.index - s2.index
                                : s1.weight - s2.weight)
                        //if booked time was not same choose the lowest bookedtime
                        //(i.e whoose task will finish early)
                        : s1.bookedTime - s2.bookedTime
        );

        for (int idx = 0; idx < serverLen; idx++) {
            freeServer.add(new Server(idx, servers[idx]));
        }

        for (int time = 0; time < taskLen; time++) {

            int processTime = tasks[time];

            //at any time 'time' if we have some busy servers that can finish their task
            //before this 'time' that means it will get free by time 'time' comes.
            //add it back to free server heap, so that we can utilize it later
            while (!busyServer.isEmpty() && busyServer.peek().bookedTime <= time) {
                freeServer.add(busyServer.poll());
            }

            if (freeServer.isEmpty()) {
                //if there are no free servers, we have to use the most optimal
                //busy server (i.e, either that currBusyServer has lowest bookedTime
                //and if bookedTime are same then it should have
                //lowest weight and if weights are same then it should have lowest index)
                //purpose of taking the optimal busy server is, it will finish early
                //and when it will finish, we want to assign curr processTime to it.
                //that's why (currBusyServer.bookedTime += processTime) this simulates
                //this curr task[time] will be immediately be assigned to it.
                Server currBusyServer = busyServer.poll();
                currBusyServer.bookedTime += processTime;
                //add in our currBusyServer back to all busy servers
                busyServer.add(currBusyServer);
                //we need to tell that this curr task[time] is assigned
                //to which server(based on its index), so task at time 'time' is assigned
                //currBusyServer.index server
                serverIdxAllotedPerTask[time] = currBusyServer.index;
                continue;
            }

            //if we have free servers available, take the optimal currFreeServer
            //book this server's bookedTime upto total time of time + processTime
            //since we have used one free server that means it is busy now
            Server currFreeServer = freeServer.poll();
            currFreeServer.bookedTime = time + processTime;
            //so move our currFreeServer to busy server
            busyServer.add(currFreeServer);
            //as our result we need to tell which curr task[time] is assigned
            //to which server(based on its index), so task at time 'time' is assigned
            //currFreeServer.index server
            serverIdxAllotedPerTask[time] = currFreeServer.index;
        }
        //output:
        for (int i = 0; i < taskLen; i++) {
            System.out.println("For task: " + tasks[i] + " alloted server with index: " + serverIdxAllotedPerTask[i]);
        }
    }

    public void maxPatientTreatedInGivenInAnyNRoom(int[][] patients, int totalRooms) {
        //https://leetcode.com/problems/process-tasks-using-servers
        //https://www.geeksforgeeks.org/google-interview-experience-for-software-engineer-l3-bangalore-6-years-experienced/
        //approach simmilar to serverAllocationToTasks()
        class Room {

            int roomNo;
            int patientTreated;
            int bookedTime;

            public Room(int roomNo) {
                this.roomNo = roomNo;
            }
        }

        int maxPatientTreated = 0;
        int roomNoOfMaxpatientTreated = -1;

        PriorityQueue<Room> filledRoomMinHeap = new PriorityQueue<>(
                (r1, r2) -> r1.bookedTime - r2.bookedTime);
        PriorityQueue<Room> freeRoomMinHeap = new PriorityQueue<>(
                (r1, r2) -> r1.roomNo - r2.roomNo);

        for (int roomNo = 1; roomNo <= totalRooms; roomNo++) {
            freeRoomMinHeap.add(new Room(roomNo));
        }

        for (int[] patient : patients) {

            int entry = patient[0];
            int duration = patient[1];

            while (!filledRoomMinHeap.isEmpty() && filledRoomMinHeap.peek().bookedTime < entry) {
                freeRoomMinHeap.add(filledRoomMinHeap.poll());
            }

            if (freeRoomMinHeap.isEmpty()) {
                continue;
            }

            Room currFreeRoom = freeRoomMinHeap.poll();
            currFreeRoom.patientTreated++;
            currFreeRoom.bookedTime = entry + duration;

            filledRoomMinHeap.add(currFreeRoom);

            if (currFreeRoom.patientTreated > maxPatientTreated) {
                maxPatientTreated = currFreeRoom.patientTreated;
                roomNoOfMaxpatientTreated = currFreeRoom.roomNo;
            }
        }
        //output
        System.out.println("Room no in which max patient treated : "
                + roomNoOfMaxpatientTreated + " max patient : " + maxPatientTreated);
    }

    class SkylineProblemBuildingCoord {

        int x;
        int height;

        public SkylineProblemBuildingCoord(int x, int height) {
            this.x = x;
            this.height = height;
        }

    }

    private List<SkylineProblemBuildingCoord> skylineProblem_BreakBuildingInCoords(
            int[][] buildings) {

        List<SkylineProblemBuildingCoord> coords = new ArrayList<>();
        for (int[] building : buildings) {
            //for start points of building
            int start = building[0]; //x
            int height = building[2]; //height
            //all start points have height -ve
            coords.add(new SkylineProblemBuildingCoord(start, -height));

            //for end points of building
            int end = building[1]; //x
            height = building[2]; //height
            //all end points have height +ve
            coords.add(new SkylineProblemBuildingCoord(end, height));
        }

        Collections.sort(coords, (c1, c2) -> c1.x == c2.x
                ? c1.height - c2.height
                : c1.x - c2.x);
        return coords;
    }

    public void skylineProblem(int[][] buildings) {
        //https://leetcode.com/problems/the-skyline-problem/
        //https://leetcode.com/problems/the-skyline-problem/discuss/2257654/With-Algorithm-Java-Solution-O(NlogN)
        //explanation: https://youtu.be/GSBLe8cKu0s
        List<List<Integer>> skylinePoints = new ArrayList<>();
        List<SkylineProblemBuildingCoord> coords = skylineProblem_BreakBuildingInCoords(buildings);

        PriorityQueue<Integer> maxHeapHeights = new PriorityQueue<>(Collections.reverseOrder());
        maxHeapHeights.add(0); // default building height

        int prevMaxHeight = 0;

        for (SkylineProblemBuildingCoord coord : coords) {
            //System.out.println(coord.x + " " + coord.height + " " + maxHeapHeights.peek());
            //if curr height is -ve that means its a start point
            //so put that height in maxHeap heights as original == abs(height)
            if (coord.height < 0) {
                //height of building start point which we made -ve
                maxHeapHeights.add(Math.abs(coord.height));
            } else {
                //removing object from PriorityQueue
                //takes O(N) time as it search for the object first
                //if the height is here that means its a end point
                //any time we reach the end point we will remove the height
                //associated with this end point
                maxHeapHeights.remove(coord.height);
            }
            //after adding or removing the heights from maxHeap heights
            //we will have a new max height
            int currMaxHeight = maxHeapHeights.peek();

            if (currMaxHeight == prevMaxHeight) {
                continue;
            }

            skylinePoints.add(Arrays.asList(coord.x, currMaxHeight));
            prevMaxHeight = currMaxHeight;
        }
        //output;
        System.out.println("Skyline coordinates of the given buildings: " + skylinePoints);
    }

    public void skylineProblem_TreeMap(int[][] buildings) {
        //..........................T: O(LogN), as treemap supports all operations in LogN time
        //OPTIMIZED much faster as compared to above priority queue approach,
        //as priority queue remove operation is O(N) time
        //https://leetcode.com/problems/the-skyline-problem/
        //https://leetcode.com/problems/the-skyline-problem/discuss/2257654/With-Algorithm-Java-Solution-O(NlogN)
        //explanation: https://youtu.be/GSBLe8cKu0s
        List<List<Integer>> skylinePoints = new ArrayList<>();
        List<SkylineProblemBuildingCoord> coords = skylineProblem_BreakBuildingInCoords(buildings);

        //<height, freq> = will store keys in maxHeap way, that means max heights on root
        //freq is required because there may be multiple buildings with same height
        TreeMap<Integer, Integer> maxHeapHeights = new TreeMap<>(Collections.reverseOrder());
        maxHeapHeights.put(0, 1); // default building height and its freq

        int prevMaxHeight = 0;

        for (SkylineProblemBuildingCoord coord : coords) {
            //System.out.println(coord.x + " " + coord.height + " " + maxHeapHeights.firstKey());
            //if curr height is -ve that means its a start point
            //so put that height in maxHeap heights as original == abs(height)
            if (coord.height < 0) {
                //height of building start point which we made -ve
                int startHeight = Math.abs(coord.height);
                //each time we see same height just increase freq if already exist
                maxHeapHeights.put(startHeight, maxHeapHeights.getOrDefault(startHeight, 0) + 1);
            } else {
                //if the height is here that means its a end point
                //any time we reach the end point we will remove the height
                //associated with this end point
                maxHeapHeights.put(coord.height, maxHeapHeights.getOrDefault(coord.height, 0) - 1);
                if (maxHeapHeights.get(coord.height) <= 0) {
                    maxHeapHeights.remove(coord.height);
                }
            }
            //after adding or removing the heights from maxHeap heights
            //we will have a new max height
            int currMaxHeight = maxHeapHeights.firstKey();

            if (currMaxHeight == prevMaxHeight) {
                continue;
            }

            skylinePoints.add(Arrays.asList(coord.x, currMaxHeight));
            prevMaxHeight = currMaxHeight;
        }
        //output;
        System.out.println("Skyline coordinates of the given buildings: " + skylinePoints);
    }

    public void minAngleBetweeHourAndMinuteHands(int hour, int min) {
        //https://leetcode.com/problems/angle-between-hands-of-a-clock/
        /*
         In 12 hour, hour hand make 360deg
         12hr = 360deg
         1hr = 360/12deg = 30deg
         1hr = 60min
         60min = 30deg
         1min = 30/60deg = 0.5deg
         ..........
         To complete 1hr, min hand makes 360deg
         1hr = 60min = 360deg
         1min = 360/60deg = 6deg
         */

        double hourAngle = (30 * hour) + (0.5 * min);
        double minuteAngle = (0 * hour) + (6 * min);
        double angle = Math.abs(hourAngle - minuteAngle);
        double minAngle = angle < 180 ? angle : 360.0 - angle;
        //output
        System.out.println("Min angle between hour and min hands: " + minAngle);
    }

    class ColorNumber {

        String color;
        int number;

        public ColorNumber(String color, int number) {
            this.color = color;
            this.number = number;
        }
    }

    public boolean cardOf12_Helper(List<ColorNumber> colorNumbers) {
        Map<String, List<Integer>> allNumbersOfSameCardMap = new HashMap<>();

        for (ColorNumber colorNumber : colorNumbers) {

            String color = colorNumber.color;
            int cardNum = colorNumber.number;
            allNumbersOfSameCardMap.putIfAbsent(color, new ArrayList<>());
            allNumbersOfSameCardMap.get(color).add(cardNum);
        }

        for (String color : allNumbersOfSameCardMap.keySet()) {
            if (allNumbersOfSameCardMap.get(color).size() % 3 != 0) {
                return false;
            }
            Collections.sort(allNumbersOfSameCardMap.get(color));
        }

        for (String color : allNumbersOfSameCardMap.keySet()) {
            List<Integer> cards = allNumbersOfSameCardMap.get(color);
            for (int i = 0; i < cards.size(); i += 3) {
                int cardNum1 = cards.get(i);
                int cardNum2 = cards.get(i + 1);
                int cardNum3 = cards.get(i + 2);

                //either 3 cards that picked up
                //1. of same num i.e, 1,1,1
                //OR 2. are consecutive (1,2,3)
                //cNum1 + 1 == cNum2 ==> 1 + 1 == 2 && cNum1 + 2 == cNum3 ==> 1 + 2 == 3
                boolean hasPassed = (cardNum1 == cardNum2 && cardNum2 == cardNum3)
                        || (cardNum1 + 1 == cardNum2 && cardNum1 + 2 == cardNum3);
                if (!hasPassed) {
                    return false;
                }
            }
        }
        return true;
    }

    public void cardOf12() {
        //https://leetcode.com/discuss/interview-experience/2279548/Google-or-Phone-Screen-or-Question-or-India
        //generating input to the question here
        List<ColorNumber> colorNumbers = Arrays.asList(
                new ColorNumber("RED", 1),
                new ColorNumber("RED", 1),
                new ColorNumber("RED", 1),
                new ColorNumber("RED", 1),
                new ColorNumber("BLUE", 2),
                new ColorNumber("BLUE", 2),
                new ColorNumber("BLUE", 2),
                new ColorNumber("BLUE", 2),
                new ColorNumber("GREEN", 4),
                new ColorNumber("GREEN", 4),
                new ColorNumber("GREEN", 4),
                new ColorNumber("GREEN", 4)
        );

        System.out.println("3 group of cards possible from given 12 cards: "
                + cardOf12_Helper(colorNumbers));

        colorNumbers = Arrays.asList(
                new ColorNumber("RED", 1),
                new ColorNumber("RED", 1),
                new ColorNumber("RED", 1),
                new ColorNumber("RED", 2),
                new ColorNumber("RED", 2),
                new ColorNumber("RED", 2),
                new ColorNumber("BLUE", 1),
                new ColorNumber("BLUE", 2),
                new ColorNumber("BLUE", 3),
                new ColorNumber("GREEN", 5),
                new ColorNumber("GREEN", 5),
                new ColorNumber("GREEN", 5)
        );

        System.out.println("3 group of cards possible from given 12 cards: "
                + cardOf12_Helper(colorNumbers));
    }

    public void implementIncreamentalStack() {
        //explanation: https://youtu.be/L8tY9gSfHz4
        ImplementIncreamentalStack stack = new ImplementIncreamentalStack();
        stack.push(1);
        stack.push(2);
        stack.push(3);
        stack.push(4);
        stack.print();
        System.out.println("POP: " + stack.pop());
        System.out.println("PEEK: " + stack.peek());
        stack.print();
        stack.increament(2, 10);
        stack.push(5);
        stack.push(6);
        stack.push(7);
        stack.increament(5, 100);
        stack.print();
        while (!stack.isEmpty()) {
            System.out.println("POP: " + stack.pop());
        }
        stack.print();
    }

    public static void main(String[] args) {

        //Object to access method
        DSA450Questions obj = new DSA450Questions();

        //......................................................................
//        Row: 6
//        System.out.println("Reverse array");
//        int[] a1 = {1, 2, 3, 4, 5};
//        obj.reverseArray(a1);
//        int[] a2 = {1, 2, 3, 4};
//        obj.reverseArray(a2);
        //......................................................................
//        Row: 56
//        System.out.println("Reverse string");
//        String str1 = "Sangeet";
//        obj.reverseString(str1);
//        String str2 = "ABCD";
//        obj.reverseString(str2);
        //......................................................................
//        Row: 57 
//        System.out.println("Is string pallindrome");
//        String str3 = "Sangeet";
//        System.out.println(str3+" "+obj.isStringPallindrome(str3));
//        String str4 = "ABBA";
//        System.out.println(str4+" "+obj.isStringPallindrome(str4));
        //......................................................................
//        Row: 58
//        System.out.println("Print duplicates char in string");
//        String str5 = "AABBCDD";
//        obj.printDuplicatesCharInString(str5);
//        String str6 = "XYZPQRS";
//        obj.printDuplicatesCharInString(str6);
        //......................................................................
//        Row: 139
//        System.out.println("Reverse a linked list iterative/recursive");
//        Node<Integer> node1 = new Node<>(1);
//        node1.setNext(new Node<>(2));
//        node1.getNext().setNext(new Node<>(3));
//        obj.reverseLinkedList_Iterative(node1);
//        Node<Integer> node2 = new Node<>(1);
//        node2.setNext(new Node<>(2));
//        node2.getNext().setNext(new Node<>(3));
//        node2.getNext().getNext().setNext(new Node<>(4));
//        node2.getNext().getNext().getNext().setNext(new Node<>(5));
//        obj.reverseLinkedList_Recursive(node2);
        //......................................................................
//        Row: 177
//        System.out.println("Level order traversal of tree iterative & recursive");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.levelOrderTraversal_Iterative(root1);
//        obj.levelOrderTraversal_Iterative2(root1); //size based approach
//        obj.levelOrderTraversal_Recursive(root1);
        //......................................................................
//        Row: 179
//        System.out.println("Height of tree");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        System.out.println(obj.heightOfTree(root1));
//        TreeNode<Integer> root2 = new TreeNode<>(1);
//        root2.setLeft(new TreeNode(2));
//        System.out.println(obj.heightOfTree(root2));
        //......................................................................
//        Row: 181
//        System.out.println("Invert/mirror of tree");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        //actual
//        BinaryTree bt = new BinaryTree<>(root1);
//        bt.treeBFS();
//        obj.mirrorOfTree(root1);
//        System.out.println();
//        //output
//        bt = new BinaryTree<>(root1);
//        bt.treeBFS();
        //......................................................................
//        Row: 299
//        System.out.println("Middle element in the stack");
//        Stack<Integer> stack = new Stack<>();
//        stack.addAll(Arrays.asList(1, 2, 3, 4, 5, 6, 7));
//        obj.middleElementInStack(stack);
//        stack.clear();
//        stack.addAll(Arrays.asList(1, 2, 3, 4));
//        obj.middleElementInStack(stack);
//        stack.clear();
//        //empty stack!!
//        obj.middleElementInStack(stack);
        //......................................................................
//        Row: 182
//        System.out.println("Inorder traversal of tree Iterative/recursive");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        //actual
//        BinaryTree bt = new BinaryTree<>(root1);
//        bt.treeBFS();
//        System.out.println();
//        obj.inOrderTraversal_Iterative(root1);
//        obj.inOrderTraversal_Recursive(root1);
        //......................................................................
//        Row: 183
//        System.out.println("Preorder traversal of tree Iterative/recursive");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        //actual
//        BinaryTree bt = new BinaryTree<>(root1);
//        bt.treeBFS();
//        System.out.println();
//        obj.preOrderTraversal_Iterative(root1);
//        obj.preOrderTraversal_Recursive(root1);
        //......................................................................
//        Row: 184
//        System.out.println("Postsorder traversal of tree Iterative/recursive");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        //actual
//        BinaryTree bt = new BinaryTree<>(root1);
//        bt.treeBFS();
//        System.out.println();
//        obj.postOrderTraversal_Iterative(root1);
//        obj.postOrderTraversal_recursive(root1);
        //......................................................................
//        Row: 148
//        System.out.println("Add two numbers represented by linked list");
//        Node<Integer> n1 = new Node<>(4);
//        n1.setNext(new Node<>(5));
//        Node<Integer> n2 = new Node<>(3);
//        n2.setNext(new Node<>(4));
//        n2.getNext().setNext(new Node<>(5));
//        obj.sumOfNumbersAsLinkedList_ByStack(n1, n2);
//        obj.sumOfNumbersAsLinkedList_ByReversingList(n1, n2);
        //......................................................................
//        Row: 178
//        System.out.println("Reverse level order traversal");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.reverseLevelOrderTraversal(root1);
        //......................................................................
//        Row: 185
//        System.out.println("Left view of tree");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.leftViewOfTree(root1);
//        obj.leftViewOfTreeWithoutExtraSpace(root1);
        //......................................................................
//        Row: 186
//        System.out.println("Right view of tree");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.rightViewOfTree(root1);
//        obj.rightViewOfTreeWithoutExtraSpace(root1);
        //......................................................................
//        Row: 187
//        System.out.println("Top view of tree");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.topViewOfTree(root1);
        //......................................................................
//        Row: 188
//        System.out.println("Bottom view of tree");
//        //https://practice.geeksforgeeks.org/problems/bottom-view-of-binary-tree/1
//        TreeNode<Integer> root1 = new TreeNode<>(20);
//        root1.setLeft(new TreeNode(8));
//        root1.getLeft().setLeft(new TreeNode(5));
//        root1.getLeft().setRight(new TreeNode(3));
//        root1.getLeft().getRight().setLeft(new TreeNode(10));
//        root1.setRight(new TreeNode(22));
//        root1.getRight().setLeft(new TreeNode(4));
//        root1.getRight().setRight(new TreeNode(25));
//        root1.getRight().getLeft().setRight(new TreeNode(14));
//        obj.bottomViewOfTree(root1);
        //......................................................................
//        Row: 189
//        System.out.println("Zig zag traversal of tree");
//        //https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.zigZagTreeTraversal(root1, true);
//        root1 = new TreeNode<>(20);
//        root1.setLeft(new TreeNode(8));
//        root1.getLeft().setLeft(new TreeNode(5));
//        root1.getLeft().setRight(new TreeNode(3));
//        root1.getLeft().getRight().setLeft(new TreeNode(10));
//        root1.setRight(new TreeNode(22));
//        root1.getRight().setLeft(new TreeNode(4));
//        root1.getRight().setRight(new TreeNode(25));
//        root1.getRight().getLeft().setRight(new TreeNode(14));
//        obj.zigZagTreeTraversal(root1, false);
        //......................................................................
//        Row: 30
//        System.out.println("All the element from array[N] and given K that occurs more than N/K times");
//        obj.arrayElementMoreThan_NDivK(new int[]{3, 1, 2, 2, 1, 2, 3, 3}, 4);
        //......................................................................
//        Row: 81
//        System.out.println("Roman numeral string to decimal");
//        //https://leetcode.com/problems/roman-to-integer/submissions/
//        obj.romanStringToDecimal("III");
//        obj.romanStringToDecimal("CI");
//        obj.romanStringToDecimal("IM");
//        obj.romanStringToDecimal("V");
//        obj.romanStringToDecimal("XI");
//        obj.romanStringToDecimal("IX");
//        obj.romanStringToDecimal("IV");
//        obj.integerToRomanString(100);
//        obj.integerToRomanString(101);
//        obj.integerToRomanString(4999);
        //......................................................................
//        Row: 86
//        System.out.println("Longest common subsequence");
//        obj.longestCommonSubsequence("ababcba", "ababcba");
//        obj.longestCommonSubsequence("abxayzbcpqba", "kgxyhgtzpnlerq");
//        obj.longestCommonSubsequence("abcd", "pqrs");
//        obj.longestCommonSubsequence("abcd", "");
//        obj.longestCommonSubsequence("", "pqrs");
        //......................................................................
//        Row: 144
//        System.out.println("Remove duplicates from sorted linked list");
//        Node<Integer> node1 = new Node<>(1);
//        node1.setNext(new Node<>(1));
//        node1.getNext().setNext(new Node<>(2));
//        node1.getNext().getNext().setNext(new Node<>(2));
//        node1.getNext().getNext().getNext().setNext(new Node<>(2));
//        node1.getNext().getNext().getNext().getNext().setNext(new Node<>(3));
//        node1.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(4));
//        node1.getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(5));
//        node1.getNext().getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(5));
//        obj.removeDuplicateFromSortedLinkedList(node1);
//        node1 = new Node<>(1);
//        node1.setNext(new Node<>(2));
//        node1.getNext().setNext(new Node<>(2));
//        node1.getNext().getNext().setNext(new Node<>(3));
//        node1.getNext().getNext().getNext().setNext(new Node<>(3));
//        node1.getNext().getNext().getNext().getNext().setNext(new Node<>(3));
//        node1.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(4));
//        node1.getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(4));
//        node1.getNext().getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(5));
//        obj.removeDuplicateFromSortedLinkedList(node1);
//        node1 = new Node<>(1);
//        node1.setNext(new Node<>(2));
//        node1.getNext().setNext(new Node<>(3));
//        node1.getNext().getNext().setNext(new Node<>(4));
//        node1.getNext().getNext().getNext().setNext(new Node<>(5));
//        node1.getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        obj.removeDuplicateFromSortedLinkedList(node1);
        //......................................................................
//        Row: 194
//        System.out.println("Convert tree to doubly linked list");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.treeToDoublyLinkedList(root1);
//        root1 = new TreeNode<>(20);
//        root1.setLeft(new TreeNode(8));
//        root1.getLeft().setLeft(new TreeNode(5));
//        root1.getLeft().setRight(new TreeNode(3));
//        root1.getLeft().getRight().setLeft(new TreeNode(10));
//        root1.setRight(new TreeNode(22));
//        root1.getRight().setLeft(new TreeNode(4));
//        root1.getRight().setRight(new TreeNode(25));
//        root1.getRight().getLeft().setRight(new TreeNode(14));
//        obj.treeToDoublyLinkedList(root1);
        //......................................................................
//        Row: 199
//        System.out.println("Check if all the leaf nodes of tree are at same level");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.checkIfAllLeafNodeOfTreeAtSameLevel(root1);
//        root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode(2));
//        root1.setRight(new TreeNode(3));
//        obj.checkIfAllLeafNodeOfTreeAtSameLevel(root1);
        //......................................................................
//        Row: 216
//        System.out.println("Min & max in the BST");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.minAndMaxInBST(root1);
        //......................................................................
//        Row: 218
//        System.out.println("Check if a tree is BST");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.isTreeBST(root1);
//        root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(10)); //BST break cond.
//        root1.getRight().setRight(new TreeNode(9));
//        obj.isTreeBST(root1);
        //......................................................................
//        Row: 225
//        System.out.println("Kth largest node in the BST");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.kTHLargestNodeInBST(root1, 4);
//        obj.kTHLargestNodeInBSTWithoutHeap(root1, 4);
//        obj.kTHLargestNodeInBST(root1, 21);
//        obj.kTHLargestNodeInBSTWithoutHeap(root1, 21);
        //......................................................................
//        Row: 226
//        System.out.println("Kth smallest node in the BST");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.kTHSmallestNodeInBST(root1, 4);
//        obj.kTHSmallestNodeInBST(root1, 1);
//        obj.kTHSmallestNodeInBST(root1, 21);
        //......................................................................
//        Row: 169
//        System.out.println("Merge K sorted linked lists");
//        //https://leetcode.com/problems/merge-k-sorted-lists/
//        Node<Integer> n1 = new Node<>(1);
//        n1.setNext(new Node<>(2));
//        n1.getNext().setNext(new Node<>(3));
//        Node<Integer> n2 = new Node<>(4);
//        n2.setNext(new Node<>(10));
//        n2.getNext().setNext(new Node<>(15));
//        Node<Integer> n3 = new Node<>(3);
//        n3.setNext(new Node<>(9));
//        n3.getNext().setNext(new Node<>(27));
//        int K = 3;
//        Node<Integer>[] nodes = new Node[K];
//        nodes[0] = n1;
//        nodes[1] = n2;
//        nodes[2] = n3;
//        obj.mergeKSortedLinkedList(nodes);
        //......................................................................
//        Row: 173
//        System.out.println("Print the Kth node from the end of a linked list 3 approaches");
//        //https://www.geeksforgeeks.org/nth-node-from-the-end-of-a-linked-list/
//        Node<Integer> n1 = new Node<>(1);
//        n1.setNext(new Node<>(2));
//        n1.getNext().setNext(new Node<>(3));
//        n1.getNext().getNext().setNext(new Node<>(5));
//        n1.getNext().getNext().getNext().setNext(new Node<>(9));
//        n1.getNext().getNext().getNext().getNext().setNext(new Node<>(15));
//        obj.kThNodeFromEndOfLinkedList_1(n1, 3);
//        obj.kThNodeFromEndOfLinkedList_2(n1, 3);
//        obj.kThNodeFromEndOfLinkedList_3(n1, 3); //OPTIMISED O(N)
//        obj.kThNodeFromEndOfLinkedList_3(n1, 6); //OPTIMISED O(N)
//        obj.kThNodeFromEndOfLinkedList_3(n1, 8); //OPTIMISED O(N)
        //......................................................................
//        Row: 190
//        System.out.println("Check if a tree is height balanced or not");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.isTreeHeightBalanced(root1);
//        root1 = new TreeNode<>(1); //SKEWED TREE
//        root1.setLeft(new TreeNode(10));
//        root1.getLeft().setLeft(new TreeNode(15));
//        obj.isTreeHeightBalanced(root1);
        //......................................................................
//        Row: 201
//        System.out.println("Check if 2 trees are mirror or not");
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.setRight(new TreeNode<>(3));
//        TreeNode<Integer> root2 = new TreeNode<>(1);
//        root2.setLeft(new TreeNode<>(3));
//        root2.setRight(new TreeNode<>(2));
//        System.out.println("2 tree are mirror: "+obj.checkTwoTreeAreMirror(root1, root2));
//        root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.setRight(new TreeNode<>(3));
//        root2 = new TreeNode<>(1);
//        root2.setLeft(new TreeNode<>(2)); //SAME 
//        root2.setRight(new TreeNode<>(3)); //SAME
//        System.out.println("2 tree are mirror: "+obj.checkTwoTreeAreMirror(root1, root2));
        //......................................................................
//        Row: 333
//        System.out.println("Next smaller element to right in array");
//        obj.nextSmallerElementInRightInArray(new int[]{4, 8, 5, 2, 25});
        //......................................................................
//        Row: 309
//        System.out.println("Reverse a stack using recursion");
//        Stack<Integer> stack = new Stack<>();
//        stack.addAll(Arrays.asList(1, 2, 3, 4, 5));
//        obj.reverseStack(stack);
        //......................................................................
//        Row: 7
//        System.out.println("Min & max in array");
//        obj.minMaxInArray_1(new int[]{1000, 11, 445, 1, 330, 3000});
//        obj.minMaxInArray_2(new int[]{1000, 11, 445, 1, 330, 3000});
        //......................................................................
//        Row: 8
//        System.out.println("Kth smallest and largest element in array");
//        obj.kThSmallestElementInArray(new int[]{7, 10, 4, 3, 20, 15}, 3);
//        obj.kThLargestElementInArray(new int[]{7, 10, 4, 3, 20, 15}, 3);
        //......................................................................
//        Row: 9
//        System.out.println("Sort the array containing elements 0, 1, 2");
//        //https://leetcode.com/problems/sort-colors/
//        obj.sortArrayOf012_1(new int[]{0, 1, 1, 0, 1, 2, 1, 2, 0, 0, 0, 1});
//        obj.sortArrayOf012_2(new int[]{0, 1, 1, 0, 1, 2, 1, 2, 0, 0, 0, 1}); //DUTCH NATIONAL FLAG ALGO
        //......................................................................
//        Row: 51
//        System.out.println("Rotate a matrix 90 degrees clockwise/anticlockwise");
//        //https://leetcode.com/problems/rotate-image
//        int[][] mat = new int[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
//        obj.rotateMatrixClockWise90Deg(mat);
//        mat = new int[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
//        obj.rotateMatrixAntiClockWise90Deg(mat);
        //......................................................................
//        Row: 62
//        System.out.println("Count and say");
//        //https://leetcode.com/problems/count-and-say/
//        obj.countAndSay(1);
//        obj.countAndSay(2);
//        obj.countAndSay(3);
//        obj.countAndSay(10);
        //......................................................................
//        Row: 93
//        System.out.println("Remove consecutive duplicate char in string");
//        obj.removeConsecutiveDuplicateInString("aababbccd");
//        obj.removeConsecutiveDuplicateInString("aaabbbcccbbbbaaaa");
//        obj.removeConsecutiveDuplicateInString("xyzpqrs");
//        obj.removeConsecutiveDuplicateInString("abcppqrspplmn");
//        obj.removeConsecutiveDuplicateInString("abcdlllllmmmmm");
//        obj.removeConsecutiveDuplicateInString("aaaaaaaaaaaa");
        //......................................................................
//        Row: 108
//        System.out.println("Majority Element");
//        obj.majorityElement_1(new int[] { 1, 3, 3, 1, 2 });
//        obj.majorityElement_1(new int[] { 1, 3, 3, 3, 2 });
//        obj.majorityElement_2(new int[] { 1, 3, 3, 1, 2 }); //MOORE'S VOTING ALGO
//        obj.majorityElement_2(new int[] { 1, 3, 3, 3, 2 }); //MOORE'S VOTING ALGO
        //......................................................................
//        Row: 195
//        System.out.println("Convert tree to its sum tree");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.convertTreeToSumTree(root1); //EXTRA QUEUE SPACE IS USED
//        //reset root
//        root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.convertTreeToSumTree_Recursion(root1); //NO EXTRA QUEUE SPACE IS USED - OPTIMISED
        //......................................................................
//        Row: 206
//        System.out.println("K sum path from any node top to down");
//        //https://leetcode.com/problems/path-sum-iii/
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode(3));
//        root1.getLeft().setLeft(new TreeNode(2));
//        root1.getLeft().setRight(new TreeNode(1));
//        root1.getLeft().getRight().setLeft(new TreeNode(1));
//        root1.setRight(new TreeNode(-1));
//        root1.getRight().setLeft(new TreeNode(4));
//        root1.getRight().getLeft().setLeft(new TreeNode(1));
//        root1.getRight().getLeft().setRight(new TreeNode(2));
//        root1.getRight().setRight(new TreeNode(5));
//        root1.getRight().getRight().setRight(new TreeNode(6));
//        obj.printKSumPathAnyNodeTopToDown(root1, 5);
        //......................................................................
//        Row: 349, 269
//        System.out.println("Min cost to combine ropes of diff lengths into one big rope");
//        obj.minCostOfRope(new int[]{4, 3, 2, 6});
        //......................................................................
//        Row: 344, 89, 271
//        System.out.println("Reorganise string");
//        //https://leetcode.com/problems/reorganize-string/
//        System.out.println("Reorganise string output: "+obj.reorganizeString("aab"));
//        System.out.println("Reorganise string output: "+obj.reorganizeString("aaab"));
//        System.out.println("Reorganise string output: "+obj.reorganizeString("bbbbb"));
//        System.out.println("Reorganise string output: "+obj.reorganizeString("geeksforgeeks"));
        //......................................................................
//        Row: 98
//        System.out.println("Print all sentences that can be formed from list/array of words");
//        String[][] arr = {{"you", "we", ""},
//        {"have", "are", ""},
//        {"sleep", "eat", "drink"}};
//        obj.printSentencesFromCollectionOfWords(arr); //GRAPH LIKE DFS
        //......................................................................
//        Row: 74
//        System.out.println("KMP pattern matching algo");
//        //https://leetcode.com/problems/implement-strstr/
//        obj.KMP_PatternMatching_Algorithm("ABABDABACDABABCABAB", "ABABCABAB");
//        obj.KMP_PatternMatching_Algorithm("sangeeangt", "ang");
//        obj.KMP_PatternMatching_Algorithm("sangeeangt", "xyz");
//        obj.longestPrefixAlsoSuffixInString_KMPAlgo("abab");
//        obj.longestPrefixAlsoSuffixInString_KMPAlgo("aaaa");
//        obj.longestPrefixAlsoSuffixInString_KMPAlgo("aabcavefaabca");
//        obj.longestPrefixAlsoSuffixInString_KMPAlgo("abcdef");
        //......................................................................
//        Row: 82
//        System.out.println("Longest common prefix in list of strings");
//        obj.longestCommonPrefix(new String[]{"flower", "flow", "flight"});
//        obj.longestCommonPrefix(new String[]{"flower", "flower", "flower"});
//        obj.longestCommonPrefix(new String[]{"dog", "racecar", "car"});
//        obj.longestCommonPrefix(new String[]{"a"});
//        obj.longestCommonPrefix(new String[]{"abc", "abcdef", "abcdlmno"});
        //......................................................................
//        Row: 114
//        System.out.println("Merge 2 sorted arrays without using extra space");
//        int arr1[] = new int[]{1, 5, 9, 10, 15, 20};
//        int arr2[] = new int[]{2, 3, 8, 13};
//        obj.mergeTwoSortedArraysWithoutExtraSpace(arr1, arr2, arr1.length, arr2.length);
        //......................................................................
//        Row: 140
//        System.out.println("Swap Linked List Nodes In Pairs");
//        //https://leetcode.com/problems/swap-nodes-in-pairs/
//        Node<Integer> node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        node.getNext().getNext().setNext(new Node<>(4));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        new LinkedListUtil<>(obj.swapLinkedListNodesInPair(node)).print();
//        node = new Node<>(3);
//        node.setNext(new Node<>(8));
//        node.getNext().setNext(new Node<>(7));
//        node.getNext().getNext().setNext(new Node<>(2));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(3));
//        new LinkedListUtil<>(obj.swapLinkedListNodesInPair(node)).print();
        //......................................................................
//        Row: 140
//        System.out.println("Reverse a linked list in K groups");
//        //https://leetcode.com/problems/reverse-nodes-in-k-group/
//        Node<Integer> node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        node.getNext().getNext().setNext(new Node<>(4));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        new LinkedListUtil<>(obj.reverseLinkedListInKGroups(node, 3)).print();
//        node = new Node<>(3);
//        node.setNext(new Node<>(8));
//        node.getNext().setNext(new Node<>(7));
//        node.getNext().getNext().setNext(new Node<>(2));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(3));
//        new LinkedListUtil<>(obj.reverseLinkedListInKGroups(node, 4)).print();
        //......................................................................
//        Row: 207, 220
//        System.out.println("Lowest common ancestor of two given node/ node values for binary tree and binary search tree both");
//        TreeNode<Integer> root1 = new TreeNode<>(5);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode<>(3));
//        root1.getLeft().setRight(new TreeNode<>(4));
//        obj.lowestCommonAncestorOfTree(root1, 3, 4);
//        root1 = new TreeNode<>(5);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode<>(3));
//        root1.getLeft().setRight(new TreeNode<>(4));
//        root1.setRight(new TreeNode<>(6));
//        obj.lowestCommonAncestorOfTree(root1, 3, 6);
//        //CASE OF BST
//        root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.lowestCommonAncestorOfTree(root1, 0, 5);
//        root1 = new TreeNode<>(5);
//        root1.setLeft(new TreeNode(4));
//        root1.getLeft().setLeft(new TreeNode<>(3));
//        root1.setRight(new TreeNode(6));
//        root1.getRight().setRight(new TreeNode(7));
//        root1.getRight().getRight().setRight(new TreeNode(8));
//        obj.lowestCommonAncestorOfTree(root1, 7, 8);
        //......................................................................
//        Row: 69, 416
//        System.out.println("Edit distance recursion/ DP memoization");
//        String s1 = "sunday";
//        String s2 = "saturday";
//        System.out.println("Edit distance recursion: "+obj.editDistance_Recursion(s1, s2, s1.length(), s2.length()));
//        System.out.println("Edit distance dp memoization: "+obj.editDistance_DP_Memoization(s1, s2));
        //......................................................................
//        Row: 84
//        System.out.println("Second most occuring word in list");
//        obj.secondMostOccuringWordInStringList(new String[]{"aaa", "bbb", "ccc", "bbb", "aaa", "aaa"});
        //......................................................................
//        Row: 101
//        System.out.println("Find first and last occurence of K in sorted array");
//        //https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
//        //https://leetcode.com/problems/find-target-indices-after-sorting-array/
//        obj.findFirstAndLastOccurenceOfKInSortedArray(new int[]{1, 3, 5, 5, 5, 5, 67, 123, 125}, 5);
//        obj.findFirstAndLastOccurenceOfKInSortedArray(new int[]{1, 3, 5, 5, 5, 5, 67, 123, 125}, 9);
//        obj.findFirstAndLastOccurenceOfKInSortedArray(new int[]{1, 3, 5, 67, 123, 125}, 5);
        //......................................................................
//        Row: 141, 143
//        System.out.println("Detect and print starting node of a loop cycle in linked list 2 approaches");
//        Node<Integer> node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        node.getNext().getNext().setNext(new Node<>(4));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        node.getNext().getNext().getNext().getNext().setNext(node.getNext().getNext()); //Node 5 connects to Node 3
//        System.out.println("Is there a loop in linked list: "+obj.detectLoopCycleInLinkedList_HashBased(node));
//        System.out.println("Is there a loop in linked list: "+obj.detectLoopCycleInLinkedList_Iterative(node)); //T: O(N), S: O(1) //OPTIMISED
//        node = new Node<>(3);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(0));
//        node.getNext().getNext().setNext(new Node<>(-4));
//        node.getNext().getNext().getNext().setNext(node.getNext());
//        System.out.println("Is there a loop in linked list: "+obj.detectLoopCycleInLinkedList_HashBased(node));
//        System.out.println("Is there a loop in linked list: "+obj.detectLoopCycleInLinkedList_Iterative(node)); //T: O(N), S: O(1) //OPTIMISED
        //......................................................................
//        Row: 142
//        System.out.println("Detect and remove loop cycle in linked list 2 approaches");
//        Node<Integer> node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        node.getNext().getNext().setNext(new Node<>(4));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        node.getNext().getNext().getNext().getNext().setNext(node.getNext().getNext()); //Node 5 connects to Node 3
//        obj.detectAndRemoveLoopCycleInLinkedList_HashBased(node);
//        node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        node.getNext().getNext().setNext(new Node<>(4));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        node.getNext().getNext().getNext().getNext().setNext(node.getNext().getNext()); //Node 5 connects to Node 3
//        obj.detectAndRemoveLoopCycleInLinkedList_Iterative(node); //OPTIMISED
        //......................................................................
//        Row: 145
//        System.out.println("Remove duplicates element in unsorted linked list 2 different outputs");
//        Node<Integer> node = new Node<>(3);
//        node.setNext(new Node<>(4));
//        node.getNext().setNext(new Node<>(5));
//        node.getNext().getNext().setNext(new Node<>(5));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(3));
//        node.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        node.getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        obj.removeDuplicatesFromUnSortedLinkedListOnlyConsecutive(node);
//        node = new Node<>(3);
//        node.setNext(new Node<>(4));
//        node.getNext().setNext(new Node<>(5));
//        node.getNext().getNext().setNext(new Node<>(5));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(3));
//        node.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        node.getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        obj.removeDuplicatesFromUnSortedLinkedListAllExtraOccuernce(node);
        //......................................................................
//        Row: 153
//        System.out.println("Find the middle element of the linked list");
//        Node<Integer> node = new Node<>(3);
//        node.setNext(new Node<>(5));
//        node.getNext().setNext(new Node<>(2));
//        node.getNext().getNext().setNext(new Node<>(4));
//        node.getNext().getNext().getNext().setNext(new Node<>(1));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(7));
//        node.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        System.out.println("Middle element: "+obj.findMiddleNodeOfLinkedList(node).getData());
//        node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        node.getNext().getNext().setNext(new Node<>(4));
//        System.out.println("Middle element: "+obj.findMiddleNodeOfLinkedList(node).getData());
        //......................................................................
//        Row: 151
//        System.out.println("Sort linked list using merge sort");
//        Node<Integer> node = new Node<>(3);
//        node.setNext(new Node<>(5));
//        node.getNext().setNext(new Node<>(2));
//        node.getNext().getNext().setNext(new Node<>(4));
//        node.getNext().getNext().getNext().setNext(new Node<>(1));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(7));
//        node.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        new LinkedListUtil<Integer>(obj.mergeSortDivideAndMerge(node)).print();
//        node = new Node<>(3);
//        node.setNext(new Node<>(3));
//        node.getNext().setNext(new Node<>(7));
//        node.getNext().getNext().setNext(new Node<>(1));
//        node.getNext().getNext().getNext().setNext(new Node<>(1));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(2));
//        node.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(5));
//        new LinkedListUtil<Integer>(obj.mergeSortDivideAndMerge(node)).print();
        //......................................................................
//        Row: 198
//        System.out.println("Check if a tree is sum tree");
//        TreeNode<Integer> root = new TreeNode<>(10);
//        root.setLeft(new TreeNode<>(20));
//        root.getLeft().setLeft(new TreeNode<>(10));
//        root.getLeft().setRight(new TreeNode<>(10));
//        root.setRight(new TreeNode<>(30)); //NOT A SUM TREE
//        obj.checkTreeIsSumTree(root);
//        root = new TreeNode<>(3);
//        root.setLeft(new TreeNode<>(2));
//        root.setRight(new TreeNode<>(1)); //SUM TREE
//        obj.checkTreeIsSumTree(root);
        //......................................................................
//        Row: 410
//        System.out.println("Coin change DP problem");
//        //https://leetcode.com/problems/coin-change-2/
//        int[] coins = {1, 2, 3};
//        int N = coins.length;
//        int K = 4;
//        System.out.println("Possible ways to make change using recursion: "+obj.coinChange_Recursion(coins, N, K));
//        obj.coinChange_DP_Memoization(coins, K);
        //......................................................................
//        Row: 411
//        System.out.println("0-1 knap sack DP problem");
//        int[] weight = {4,5,1};
//        int[] value = {1,2,3};
//        int N = value.length;
//        int W = 4;
//        System.out.println("The maximum profit can be made with given knap sack using recursion: "+obj.knapSack01_Recusrion(W, weight, value, N));
//        obj.knapSack01_DP_Memoization(W, weight, value, N);
//        weight = new int[]{4,5,6};
//        value = new int[]{1,2,3};
//        N = value.length;
//        W = 3;
//        System.out.println("The maximum profit can be made with given knap sack using recursion: "+obj.knapSack01_Recusrion(W, weight, value, N));
//        obj.knapSack01_DP_Memoization(W, weight, value, N);
        //......................................................................
//        Row: 417, 282
//        System.out.println("Subset sum DP problem");
//        //https://leetcode.com/problems/partition-equal-subset-sum/
//        int[] arr = new int[]{1, 5, 5, 11};
//        int N = arr.length;
//        int sum = 11;
//        System.out.println("The sub set for the given sum is possible: "+obj.subsetSum_Recursion(arr, sum, N));
//        obj.subsetSum_DP_Memoization(arr, sum, N);
//        System.out.println("Equal sum partition for the given array is possible or not");
//        obj.equalsSumPartition_SubsetSum(arr, N);
//        //arr to be different
//        arr = new int[]{1,5,5,12};
//        obj.equalsSumPartition_SubsetSum(arr, N);
        //......................................................................
//        Row: 423
//        System.out.println("Longest common sub sequence of 2 strings DP problem");
//        //https://leetcode.com/problems/longest-common-subsequence
//        String s1 = "ABCDGH";
//        String s2 = "AEDFHR";
//        System.out.println("The longest common sub sequence length for the given 2 strings: "+obj.longestCommonSubsequence_Recursion(s1, s2, s1.length(), s2.length()));
//        obj.longestCommonSubsequence_DP_Memoization(s1, s2, s1.length(), s2.length());
//        s1 = "ABCDGH";
//        s2 = "";
//        System.out.println("The longest common sub sequence length for the given 2 strings: "+obj.longestCommonSubsequence_Recursion(s1, s2, s1.length(), s2.length()));
//        obj.longestCommonSubsequence_DP_Memoization(s1, s2, s1.length(), s2.length());
//        System.out.println("Uncrossed Lines / Longest common sub sequence of 2 int arrays DP problem");
//        //https://leetcode.com/problems/uncrossed-lines/
//        obj.uncrossedLines_DP_Memoization(new int[]{1,4,2}, new int[]{1,2,4});
//        obj.uncrossedLines_DP_Memoization(new int[]{2,5,1,2,5}, new int[]{10,5,2,1,5,2});
//        obj.uncrossedLines_DP_Memoization(new int[]{1,3,7,1,7,5}, new int[]{1,9,2,5,1});
        //......................................................................
//        Row: SEPARATE IMPORTANT QUESTION
//        System.out.println("Longest Pallindromic Subsequence DP problem");
//        //https://leetcode.com/problems/longest-palindromic-subsequence
//        obj.longestPallindromicSubsequence_DP_Memoization("bbbab");
//        obj.longestPallindromicSubsequence_DP_Memoization("cbbs");
        //......................................................................
//        Row: SEPARATE IMPORTANT QUESTION
//        System.out.println("Delete Operation for Two Strings DP problem");
//        //https://leetcode.com/problems/delete-operation-for-two-strings/
//        // delete s from sea ==> "ea" insert t to "ea"  ==> eat
//        obj.deleteOperationOfTwoStrings_DP_Memoization("sea", "eat"); 
//        obj.deleteOperationOfTwoStrings_DP_Memoization("leetcode", "etco");
        //......................................................................
//        Row: 97
//        System.out.println("Check two strings are isomorphic or not");
//        //https://leetcode.com/problems/word-pattern/
//        //https://www.geeksforgeeks.org/check-if-two-given-strings-are-isomorphic-to-each-other/
//        String s1 = "aab";
//        String s2 = "xxy";
//        System.out.println("Is isomorphic strings 1: "+obj.checkIsomorphicStrings_1(s1, s2));
//        System.out.println("Is isomorphic strings 2: "+obj.checkIsomorphicStrings_2(s1, s2));
//        s1 = "aab";
//        s2 = "xyz";
//        System.out.println("Is isomorphic strings 1: "+obj.checkIsomorphicStrings_1(s1, s2));
//        System.out.println("Is isomorphic strings 2: "+obj.checkIsomorphicStrings_2(s1, s2));
//        s1 = "13";
//        s2 = "42";
//        System.out.println("Is isomorphic strings 1: "+obj.checkIsomorphicStrings_1(s1, s2));
//        System.out.println("Is isomorphic strings 2: "+obj.checkIsomorphicStrings_2(s1, s2));
        //......................................................................
//        Row: 96
//        System.out.println("Transform one string to another with min gievn no of operations");
//        //https://www.geeksforgeeks.org/transform-one-string-to-another-using-minimum-number-of-given-operation/
//        System.out.println("Transform operations required: " + obj.transformOneStringToAnotherWithMinOprn("EACBD", "EABCD"));
//        System.out.println("Transform operations required: " + obj.transformOneStringToAnotherWithMinOprn("EACBD", "EACBD"));
//        System.out.println("Transform operations required: " + obj.transformOneStringToAnotherWithMinOprn("EACCD", "EABCD"));
        //......................................................................
//        Row: 154
//        System.out.println("Check if a linked list is circular linked list");
//        Node<Integer> node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        node.getNext().getNext().setNext(new Node<>(4));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        node.getNext().getNext().getNext().getNext().getNext().setNext(node); //CIRCULAR 6 -> 1
//        System.out.println("Check if given linked list is circular linked list: " + obj.checkIfLinkedListIsCircularLinkedList(node));
//        node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        node.getNext().getNext().setNext(new Node<>(4));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(6)); //NOT CIRCULAR 6 -> NULL
//        System.out.println("Check if given linked list is circular linked list: " + obj.checkIfLinkedListIsCircularLinkedList(node));
        //......................................................................
//        Row: 152
//        System.out.println("Quick sort in linked list");
//        //https://www.geeksforgeeks.org/quick-sort/
//        Node<Integer> node = new Node<>(10);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(5));
//        node.getNext().getNext().setNext(new Node<>(1));
//        node.getNext().getNext().getNext().setNext(new Node<>(3));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        obj.quickSortInLinkedList(node);
        //......................................................................
//        Row: 304
//        System.out.println("Find next greater element");
//        obj.nextGreaterElementInRightInArray(new int[]{1,3,2,4});
//        obj.nextGreaterElementInRightInArray(new int[]{1,2,3,4,5}); //STACK WILL HOLD N ELEMENT S: O(N)
//        obj.nextGreaterElementInRightInArray(new int[]{5,4,3,2,1}); //STACK WILL NOT HOLD N ELEMENT S: O(1)
        //......................................................................
//        Row: 339
//        System.out.println("First K largest element in array");
//        obj.kLargestElementInArray(new int[]{12, 5, 787, 1, 23}, 2);
//        obj.kLargestElementInArray(new int[]{1, 23, 12, 9, 30, 2, 50}, 3);
        //......................................................................
//        Row: 20, 70
//        System.out.println("Next permutation");
//        //https://leetcode.com/problems/next-permutation
//        obj.nextPermutation(new int[]{1,2,3});
//        obj.nextPermutation(new int[]{4,3,2,1});
//        obj.nextPermutation(new int[]{1,3,1,4,7,6,2});
//        obj.nextPermutation(new int[]{2,7,4,3,2});
//        obj.nextPermutation(new int[]{1, 2, 3, 6, 5, 4});
        //......................................................................
//        Row: 27
//        System.out.println("Factorial of large number");
//        //https://www.geeksforgeeks.org/factorial-large-number/
//        obj.factorialLargeNumber(1);
//        obj.factorialLargeNumber(5);
//        obj.factorialLargeNumber(10);
//        obj.factorialLargeNumber(897);
        //......................................................................
//        Row: 103
//        System.out.println("Search in rotated sorted array");
//        //https://leetcode.com/problems/search-in-rotated-sorted-array
//        System.out.println("The target is found at location: "+ obj.searchInRotatedSortedArray(new int[]{4,5,6,7,0,1,2}, 0));
//        System.out.println("The target is found at location: "+ obj.searchInRotatedSortedArray(new int[]{4,5,6,7,0,1,2}, 4));
//        System.out.println("The target is found at location: "+ obj.searchInRotatedSortedArray(new int[]{4,5,6,7,0,1,2}, 3));
//        //https://leetcode.com/problems/search-in-rotated-sorted-array-ii/
//        System.out.println("The target is found at location in arr with duplicate elements: "
//                + obj.searchInRotatedSortedArrayWithDuplicateArrayElement(new int[]{1,0,1,1,1}, 0));
//        System.out.println("The target is found at location in arr with duplicate elements: "
//                + obj.searchInRotatedSortedArrayWithDuplicateArrayElement(new int[]{2,5,6,0,0,1,2}, 0));
//        System.out.println("The target is found at location in arr with duplicate elements: "
//                + obj.searchInRotatedSortedArrayWithDuplicateArrayElement(new int[]{2,5,6,0,0,1,2}, 3));
        //......................................................................
//        Row: 146
//        System.out.println("Move last node of linked list to front");
//        Node<Integer> node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        node.getNext().getNext().setNext(new Node<>(4));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        obj.moveLastNodeToFrontOfLinkedList(node);
        //......................................................................
//        Row: 147
//        System.out.println("Add 1 to linked list");
//        Node<Integer> node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        node.getNext().getNext().setNext(new Node<>(4));
//        obj.addOneToLinkedList(node);
//        node = new Node<>(9); //COND WHEN METHOD WILL CREATE NEWHEAD TO STORE EXTRA CARRY IN THE SUM RECURSION
//        node.setNext(new Node<>(9));
//        node.getNext().setNext(new Node<>(9));
//        node.getNext().getNext().setNext(new Node<>(9));
//        obj.addOneToLinkedList(node);
        //......................................................................
//        Row: 167
//        System.out.println("Sort linked list of 0s, 1s, 2s using 2 approaches");
//        //https://www.geeksforgeeks.org/sort-a-linked-list-of-0s-1s-or-2s/
//        Node<Integer> node = new Node<>(0);
//        node.setNext(new Node<>(1));
//        node.getNext().setNext(new Node<>(0));
//        node.getNext().getNext().setNext(new Node<>(2));
//        node.getNext().getNext().getNext().setNext(new Node<>(1));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(1));
//        node.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(2));
//        node.getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(1));
//        node.getNext().getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(2));
//        new LinkedListUtil<>(obj.mergeSortDivideAndMerge(node)).print(); //SIMPLE MERGE SORT APPROACH T: O(N.LogN)
//        node = new Node<>(0);
//        node.setNext(new Node<>(1));
//        node.getNext().setNext(new Node<>(0));
//        node.getNext().getNext().setNext(new Node<>(2));
//        node.getNext().getNext().getNext().setNext(new Node<>(1));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(1));
//        node.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(2));
//        node.getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(1));
//        node.getNext().getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(2));
//        obj.sortLinkedListOf012_2(node); //SIMPLE MANIPULATION OF NODE T: O(N)
        //......................................................................
//        Row: 202
//        System.out.println("Sum of node on the longest path of tree from root to leaf");
//        TreeNode<Integer> root = new TreeNode<>(4);
//        root.setLeft(new TreeNode<>(2));
//        root.getLeft().setLeft(new TreeNode<>(7));
//        root.getLeft().setRight(new TreeNode<>(1));
//        root.getLeft().getRight().setLeft(new TreeNode<>(6)); //LONGEST PATH
//        root.setRight(new TreeNode<>(5));
//        root.getRight().setLeft(new TreeNode<>(2));
//        root.getRight().setRight(new TreeNode<>(3));
//        obj.longestPathNodeSum(root);
        //......................................................................
//        Row: 34
//        System.out.println("Rain water trapping 2 approaches");
//        //https://leetcode.com/problems/trapping-rain-water/
//        obj.rainWaterTrappingUsingStack(new int[]{3,0,0,2,0,4});
//        obj.rainWaterTrappingUsingTwoPointers(new int[]{3,0,0,2,0,4});
//        obj.rainWaterTrappingUsingStack(new int[]{6,9,9});
//        obj.rainWaterTrappingUsingTwoPointers(new int[]{6,9,9});
//        obj.rainWaterTrappingUsingStack(new int[]{7,4,0,9});
//        obj.rainWaterTrappingUsingTwoPointers(new int[]{7,4,0,9});
//        obj.rainWaterTrappingUsingTwoPointers(new int[]{3,4,1});
        //......................................................................
//        Row: 28
//        System.out.println("Find maximum product subarray");
//        //https://leetcode.com/problems/maximum-product-subarray/
//        obj.findMaximumProductSubarray(new int[]{2,3,-2,4});
//        obj.findMaximumProductSubarray(new int[]{-2,0,-1});
//        obj.findMaximumProductSubarray(new int[]{-1,8});
//        obj.findMaximumProductSubarray(new int[]{-1,-8});
        //......................................................................
//        Row: 217
//        System.out.println("Find predecessor and successor of given node in BST");
//        //https://www.geeksforgeeks.org/inorder-predecessor-successor-given-key-bst/
//        //predecessors and successor can be found when we do the inorder traversal of tree
//        //inorder traversal of BST is sorted list of node data
//        //for below BST inorder list [0,2,3,4,5,6,7,8,9]
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.findPredecessorAndSuccessorInBST(root1, 6);
//        obj.findPredecessorAndSuccessorInBST(root1, 2);
//        obj.findPredecessorAndSuccessorInBST(root1, 5);
//        obj.findPredecessorAndSuccessorInBST(root1, 10); //ONLY PREDECESSOR IS POSSIBLE
//        obj.findPredecessorAndSuccessorInBST(root1, -1); //ONLY SUCCESSOR IS POSSIBLE
        //......................................................................
//        Row: 424, 64
//        System.out.println("Longest Repeating Subsequence DP problem");
//        System.out.println("Longest repeating subsequence: "+obj.longestRepeatingSubsequence_Recursion("axxxy", 5)); //xx, xx
//        obj.longestRepeatingSubsequence_DP_Memoization("axxxy"); //xx, xx
        //......................................................................
//        Row: 441
//        System.out.println("Longest common substring DP problem");
//        //https://leetcode.com/problems/maximum-length-of-repeated-subarray/
//        obj.longestCommonSubstring_DP_Memoization("ABCDGH", "ACDGHR");
        //......................................................................
//        Row: 441
//        System.out.println("Maximum length of pair chain 2 approaches");
//        //https://leetcode.com/problems/maximum-length-of-pair-chain/
//        System.out.println("maximum length of pair chain DP approach: "+
//                obj.maximumLengthOfPairChain_DP_Approach(new int[][]{
//                    {1,2},
//                    {3,4},
//                    {2,3}}));
//        System.out.println("maximum length of pair chain Greedy approach: "+
//                obj.maximumLengthOfPairChain_Greedy_Approach(new int[][]{
//                    {1,2},
//                    {3,4},
//                    {2,3}}));
        //......................................................................
//        Row: 412
//        System.out.println("Binomial coefficient DP problem");
//        //https://www.geeksforgeeks.org/binomial-coefficient-dp-9/
//        System.out.println("Binomial coefficient recursive way: "+obj.findBinomialCoefficient_Recursion(5, 2));
//        obj.findBinomialCoefficient_DP_Memoization(5, 2);
//        System.out.println("Binomial coefficient recursive way: "+obj.findBinomialCoefficient_Recursion(5, 6));
//        obj.findBinomialCoefficient_DP_Memoization(5, 6);
//        System.out.println("Binomial coefficient recursive way: "+obj.findBinomialCoefficient_Recursion(5, 5));
//        obj.findBinomialCoefficient_DP_Memoization(5, 5);
//        System.out.println("Binomial coefficient recursive way: "+obj.findBinomialCoefficient_Recursion(5, 0));
//        obj.findBinomialCoefficient_DP_Memoization(5, 0);
        //......................................................................
//        Row: 229
//        System.out.println("Count BST nodes that lie in the given range");
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.countNodesThatLieInGivenRange(root1, 1, 4);
//        obj.countNodesThatLieInGivenRange(root1, 6, 9);
        //......................................................................
//        Row: 235
//        System.out.println("Flatten BST to linked list (skewed tree)");
//        //https://leetcode.com/problems/flatten-binary-tree-to-linked-list/
//        //https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.flattenBSTToLinkedList(root1);
//        //obj.flattenBSTToLinkedList_Recursion(root1);
        //......................................................................
//        Row: 158
//        System.out.println("Reverse a doubly linked list");
//        Node<Integer> node = new Node<>(3);
//        Node<Integer> next = new Node<>(4);
//        node.setNext(next);
//        next.setPrevious(node);
//        Node<Integer> nextToNext = new Node<>(5);
//        next.setNext(nextToNext);
//        nextToNext.setPrevious(next);
//        obj.reverseDoublyLinkedList(node);
        //......................................................................
//        Row: 18, 13
//        System.out.println("Kaden's algorithm approaches");
//        //https://www.geeksforgeeks.org/largest-sum-contiguous-subarray/
//        int[] a = new int[]{-2, -3, 4, -1, -2, 1, 5, -3};
//        System.out.println("Maximum sum subarray: "+ obj.kadaneAlgorithm(a));
//        obj.kadaneAlgorithm_PointingIndexes(a);
//        a = new int[]{-1, -2, -3, -4};
//        System.out.println("Maximum sum subarray: "+ obj.kadaneAlgorithm(a));
//        obj.kadaneAlgorithm_PointingIndexes(a);
//        a = new int[]{-1, -2, 0, -4};
//        System.out.println("Maximum sum subarray: "+ obj.kadaneAlgorithm(a));
//        obj.kadaneAlgorithm_PointingIndexes(a);
//        a = new int[]{1, 2, -3, -4};
//        System.out.println("Maximum sum subarray: "+ obj.kadaneAlgorithm(a));
//        obj.kadaneAlgorithm_PointingIndexes(a);
        //......................................................................
//        Row: 10
//        System.out.println("Move all negative elements to one side of array");
//        obj.moveNegativeElementsToOneSideOfArray(new int[]{-12, 11, -13, -5, 6, -7, 5, -3, -6});
//        obj.moveNegativeElementsToOneSideOfArray(new int[]{-1, 2, -3, 4, 5, 6, -7, 8, 9});
//        obj.moveNegativeElementsToOneSideOfArray(new int[]{-1, -2, -3, -1, -10, -7});
//        obj.moveNegativeElementsToOneSideOfArray(new int[]{1, 2, 3, 1, 10, 7});
//        obj.moveNegativeElementsToOneSideOfArray(new int[]{1, -2, -3, -1, -10, -7});
        //......................................................................
//        Row: 11
//        System.out.println("Find union and intersection of two arrays");
//        obj.findUnionAndIntersectionOfTwoArrays(new int[]{1,2,3,4,5}, new int[]{1,2,3});
//        obj.findUnionAndIntersectionOfTwoArrays(new int[]{4,9,5}, new int[]{9,4,9,8,4});
        //......................................................................
//        Row: 12
//        System.out.println("Cyclically rotate element in array by 1");
//        //https://leetcode.com/problems/rotate-array/
//        obj.rotateArrayByK_BruteForce(new int[]{1, 2, 3, 4, 5}, 1);
//        obj.rotateArrayByK_BruteForce(new int[]{1, 2, 3, 4, 5}, 4);
//        obj.rotateArrayByK(new int[]{1, 2, 3, 4, 5}, 1);
//        obj.rotateArrayByK(new int[]{1, 2, 3, 4, 5}, 4);
        //......................................................................
//        Row: 14
//        System.out.println("Minimize the difference between the heights");
//        //https://www.geeksforgeeks.org/minimize-the-maximum-difference-between-the-heights/
//        obj.minimizeDifferenceBetweenHeights(new int[]{1, 5, 8, 10}, 2);
//        obj.minimizeDifferenceBetweenHeights(new int[]{4, 6}, 10);
        //......................................................................
//        Row: 191
//        System.out.println("Diagonal traversal of tree");
//        //https://www.geeksforgeeks.org/diagonal-traversal-of-binary-tree/
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.diagonalTraversalOfTree(root1);
        //......................................................................
//        Row: 180
//        System.out.println("Diameter of tree DP on tree problem");
//        //https://leetcode.com/problems/diameter-of-binary-tree/
//        //https://practice.geeksforgeeks.org/problems/diameter-of-binary-tree/1/
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode(0));
//        root1.getLeft().setRight(new TreeNode(4));
//        root1.getLeft().getRight().setLeft(new TreeNode(3));
//        root1.getLeft().getRight().setRight(new TreeNode(5));
//        root1.setRight(new TreeNode(8));
//        root1.getRight().setLeft(new TreeNode(7));
//        root1.getRight().setRight(new TreeNode(9));
//        obj.diameterOfTree(root1);
//        root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode(2));
//        root1.getLeft().setLeft(new TreeNode<>(3));
//        root1.getLeft().setRight(new TreeNode<>(4));
//        root1.getLeft().getRight().setRight(new TreeNode<>(5));
//        obj.diameterOfTree(root1);
        //......................................................................
//        Row: 238, 251
//        System.out.println("N meeting in a room/ Activity selection");
//        //https://practice.geeksforgeeks.org/problems/n-meetings-in-one-room-1587115620/1
//        int[] startTime = {1, 3, 0, 5, 8, 5};
//        int[] finishTime = {2, 4, 6, 7, 9, 9};
//        obj.nMeetingRooms_Greedy(startTime, finishTime);
        //......................................................................
//        Row: 357, 358
//        System.out.println("BFS/DFS directed graph");
//        List<List<Integer>> adjList = new ArrayList<>();
//        adjList.add(0, Arrays.asList(1, 2, 3));
//        adjList.add(1, Arrays.asList());
//        adjList.add(2, Arrays.asList(4));
//        adjList.add(3, Arrays.asList());
//        adjList.add(4, Arrays.asList());
//        obj.graphBFSAdjList_Graph(adjList.size(), adjList);
//        obj.graphDFSAdjList_Graph(adjList.size(), adjList);
//        obj.graphDFSAdjList_Recursive_Graph(adjList.size(), adjList);
//        adjList = new ArrayList<>();
//        adjList.add(0, Arrays.asList(1, 2, 3));
//        adjList.add(1, Arrays.asList(5));
//        adjList.add(2, Arrays.asList(4));
//        adjList.add(3, Arrays.asList());
//        adjList.add(4, Arrays.asList(3));
//        adjList.add(5, Arrays.asList());
//        obj.graphBFSAdjList_Graph(adjList.size(), adjList);
//        obj.graphDFSAdjList_Graph(adjList.size(), adjList);
//        obj.graphDFSAdjList_Recursive_Graph(adjList.size(), adjList);
//        adjList = new ArrayList<>();
//        adjList.add(0, Arrays.asList(1));
//        adjList.add(1, Arrays.asList(2));
//        adjList.add(2, Arrays.asList(3));
//        adjList.add(3, Arrays.asList(4));
//        adjList.add(4, Arrays.asList(5));
//        adjList.add(5, Arrays.asList());
//        obj.graphBFSAdjList_Graph(adjList.size(), adjList);
//        obj.graphDFSAdjList_Graph(adjList.size(), adjList);
//        obj.graphDFSAdjList_Recursive_Graph(adjList.size(), adjList);
        //......................................................................
//        Row: 361, 275
//        System.out.println("Search in maze");
//        int[][] maze = new int[][]{
//            {1, 0, 0, 0},
//            {1, 1, 0, 1},
//            {1, 1, 0, 0},
//            {0, 1, 1, 1}
//        };
//        obj.findPathRatInMaze_Graph(maze, maze.length);
//        maze = new int[][]{
//            {1, 0, 0, 0},
//            {1, 1, 0, 1},
//            {1, 1, 0, 0},
//            {0, 1, 1, 0}
//        };
//        obj.findPathRatInMaze_Graph(maze, maze.length);
        //......................................................................
//        Row: 371
//        System.out.println("No. of Island");
//        //https://leetcode.com/problems/number-of-islands
//        int[][] grid = {
//            {0, 1, 1, 1, 0, 0, 0},
//            {0, 0, 1, 1, 0, 1, 0}};
//        obj.numberOfIslands_Graph(grid);
        //......................................................................
//        Row: 348
//        System.out.println("Check if binary tree is heap (max heap)");
//        TreeNode<Integer> root = new TreeNode<>(10);
//        root.setLeft(new TreeNode<>(7));
//        root.getLeft().setLeft(new TreeNode<>(6));
//        root.getLeft().setRight(new TreeNode<>(5));
//        root.setRight(new TreeNode<>(8));
//        root.getRight().setLeft(new TreeNode<>(4));
//        obj.checkIfBinaryTreeIsMaxHeap(root);
//        root = new TreeNode<>(10);
//        root.setLeft(new TreeNode<>(7));
//        root.getLeft().setLeft(new TreeNode<>(6));
//        root.getLeft().setRight(new TreeNode<>(5));
//        root.setRight(new TreeNode<>(8));
//        root.getRight().setLeft(new TreeNode<>(9)); 
//        obj.checkIfBinaryTreeIsMaxHeap(root);
//        root = new TreeNode<>(10);
//        root.setLeft(new TreeNode<>(7));
//        root.getLeft().setLeft(new TreeNode<>(6));
//        root.getLeft().setRight(new TreeNode<>(5));
//        root.setRight(new TreeNode<>(8));
//        root.getRight().setLeft(new TreeNode<>(4)); 
//        root.getRight().setRight(new TreeNode<>(3)); 
//        obj.checkIfBinaryTreeIsMaxHeap(root);
        //......................................................................
//        Row: 91
//        System.out.println("Arrange all anagrams together");
//        obj.arrangeAllWordsAsTheirAnagrams(Arrays.asList("act", "god", "cat", "dog", "tac"));
        //......................................................................
//        Row: 90
//        System.out.println("Minimum character added at front of string to make it pallindrome/ Shortest pallindrome");
//        //https://leetcode.com/problems/shortest-palindrome
//        obj.characterAddedAtFrontToMakeStringPallindrome("ABC"); // 2 char = B,C (ex CBABC)
//        obj.characterAddedAtFrontToMakeStringPallindrome("ABA"); // 0 char already pallindrome
//        obj.shortestPallindrome("a");
//        obj.shortestPallindrome("aacecaaa");
//        obj.shortestPallindrome("abcd");
//        obj.shortestPallindrome("aaaa");
        //......................................................................
//        Row: 360
//        System.out.println("Detect cycle in undirected graph DFS");
//        //https://www.geeksforgeeks.org/detect-cycle-undirected-graph/
//        List<List<Integer>> adjList = new ArrayList<>(); //CYCLE //0 <--> 1 <--> 2 <--> 0
//        adjList.add(0, Arrays.asList(1, 2));
//        adjList.add(1, Arrays.asList(0, 2));
//        adjList.add(2, Arrays.asList(0, 1));
//        System.out.println("Is there a cycle in undirected graph: " + obj.detectCycleInUndirectedGraphDFS_Graph(adjList.size(), adjList));
//        adjList = new ArrayList<>(); //NO CYCLE 0 <--> 1 <--> 2
//        adjList.add(0, Arrays.asList(1));       
//        adjList.add(1, Arrays.asList(0, 2));
//        adjList.add(2, Arrays.asList(1));
//        System.out.println("Is there a cycle in undirected graph: " + obj.detectCycleInUndirectedGraphDFS_Graph(adjList.size(), adjList));
//        adjList = new ArrayList<>(); //CYCLE //FAIL CASE // 0 <--> 1 //CASE OF DIRECTED GRAPH
//        adjList.add(0, Arrays.asList(1));
//        adjList.add(1, Arrays.asList(0));
//        System.out.println("Is there a cycle in undirected graph: " + obj.detectCycleInUndirectedGraphDFS_Graph(adjList.size(), adjList));
        //......................................................................
//        Row: 368
//        System.out.println("Topological sort graph");    
//        List<List<Integer>> adjList = new ArrayList<>();
//        adjList.add(0, Arrays.asList());
//        adjList.add(1, Arrays.asList());
//        adjList.add(2, Arrays.asList(3));
//        adjList.add(3, Arrays.asList(1));
//        adjList.add(4, Arrays.asList(0,1));
//        adjList.add(5, Arrays.asList(0,2));
//        obj.topologicalSort_Graph(adjList.size(), adjList);
        //......................................................................
//        Row: 439
//        System.out.println("Minimum cost to fill the given bag");
//        //https://www.geeksforgeeks.org/minimum-cost-to-fill-given-weight-in-a-bag/
//        obj.minimumCostToFillGivenBag_DP_Memoization(new int[]{20, 10, 4, 50, 100}, 5);
//        obj.minimumCostToFillGivenBag_DP_Memoization(new int[]{-1, -1, 4, 3, -1}, 5);
        //......................................................................
//        Row: 22
//        System.out.println("Best time to buy and sell stock");
//        //https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
//        obj.bestProfitToBuySellStock(new int[]{7,1,5,3,6,4});
//        obj.bestProfitToBuySellStock(new int[]{7,6,4,3,1});
        //......................................................................
//        System.out.println("Maximum profit by buying seling stocks,"
//                + "can hold stock atmost one but can buy/sell same stock in same day");
//        //https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/
//        obj.bestProfitToBuySellStockCanHoldAtmostOneStock(new int[]{2, 30, 15, 10, 8, 25, 80});
//        obj.bestProfitToBuySellStockCanHoldAtmostOneStock(new int[]{2, 30, 80, 10, 8, 25, 60});
//        obj.bestProfitToBuySellStockCanHoldAtmostOneStock(new int[]{3, 3, 5, 0, 0, 3, 1, 4});
        //......................................................................
//        Row: 31
//        System.out.println("Maximum profit by buying seling stocks atmost twice");
//        //https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/
//        obj.bestProfitToBuySellStockAtMostTwice(new int[]{2, 30, 15, 10, 8, 25, 80});
//        obj.bestProfitToBuySellStockAtMostTwice(new int[]{2, 30, 80, 10, 8, 25, 60});
//        obj.bestProfitToBuySellStockAtMostTwice(new int[]{3, 3, 5, 0, 0, 3, 1, 4});
        //......................................................................
//        Row: 23
//        System.out.println("Find all pairs in array whose sum is given to K");
//        //https://www.geeksforgeeks.org/count-pairs-with-given-sum/
//        obj.countAllPairsInArrayThatSumIsK(new int[]{1, 5, 7, 1}, 6);
//        obj.countAllPairsInArrayThatSumIsK(new int[]{1, 1, 1, 1}, 2);
        //......................................................................
//        Row: 60
//        System.out.println("Check if one string is rotation of other string");
//        //https://www.geeksforgeeks.org/a-program-to-check-if-strings-are-rotations-of-each-other/
//        System.out.println("Check if one string is rotation: "+obj.checkIfOneStringRotationOfOtherString("AACD", "ACDA"));
        //......................................................................
//        Row: 312
//        System.out.println("Largest area of histogram");
//        //https://leetcode.com/problems/largest-rectangle-in-histogram
//        //https://www.geeksforgeeks.org/largest-rectangle-under-histogram/
//        obj.largestAreaInHistogram(new int[]{6, 2, 5, 4, 5, 1, 6});
//        obj.largestAreaInHistogram(new int[]{6,9,8});
        //......................................................................
//        Row: 418
//        System.out.println("Friends pairing DP problem");
//        //https://www.geeksforgeeks.org/friends-pairing-problem/
//        System.out.println("No. of ways friends can be paired recursion: "+obj.friendsPairingProblem_Recursion(4));
//        obj.friendsPairingProblem_DP_Memoization(4);
//        obj.friendsPairingProblem(4);
        //......................................................................
//        Row: 341
//        System.out.println("Merge k sorted arrays (heap)");
//        int[][] arr = new int[][]{
//            {1,2,3},
//            {4,5,6},
//            {7,8,9}
//        };
//        obj.mergeKSortedArrays_1(arr);
//        obj.mergeKSortedArrays_2(arr);
        //......................................................................
//        Row: 343
//        System.out.println("Kth largest sum from contigous subarray");
//        //https://www.geeksforgeeks.org/k-th-largest-sum-contiguous-subarray/
//        obj.kThLargestSumFromContigousSubarray(new int[]{10, -10, 20, -40}, 6);
//        obj.kThLargestSumFromContigousSubarray(new int[]{20, -5, -1}, 3);
        //......................................................................
//        Row: 329
//        System.out.println("Check if all levels in two trees are anagrams of each other");
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.setRight(new TreeNode<>(3));
//        root1.getRight().setLeft(new TreeNode<>(4));
//        root1.getRight().setRight(new TreeNode<>(5));
//        TreeNode<Integer> root2 = new TreeNode<>(1);
//        root2.setLeft(new TreeNode<>(3));
//        root2.setRight(new TreeNode<>(2));
//        root2.getRight().setLeft(new TreeNode<>(5));
//        root2.getRight().setRight(new TreeNode<>(4));
//        System.out.println("Check if all levels of two trees are anagrams 1: "+
//                obj.checkIfAllLevelsOfTwoTreesAreAnagrams_1(root1, root2));
//        System.out.println("Check if all levels of two trees are anagrams 2: "+
//                obj.checkIfAllLevelsOfTwoTreesAreAnagrams_2(root1, root2));
//        root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.getLeft().setLeft(new TreeNode<>(6));
//        root1.getLeft().setRight(new TreeNode<>(7));
//        root1.setRight(new TreeNode<>(3));
//        root1.getRight().setLeft(new TreeNode<>(4));
//        root1.getRight().setRight(new TreeNode<>(5));
//        root2 = new TreeNode<>(1);
//        root2.setLeft(new TreeNode<>(3));
//        root2.setRight(new TreeNode<>(2));
//        root2.getRight().setLeft(new TreeNode<>(5));
//        root2.getRight().setRight(new TreeNode<>(4));
//        System.out.println("Check if all levels of two trees are anagrams 1: "+
//                obj.checkIfAllLevelsOfTwoTreesAreAnagrams_1(root1, root2));
//        System.out.println("Check if all levels of two trees are anagrams 2: "+
//                obj.checkIfAllLevelsOfTwoTreesAreAnagrams_2(root1, root2));
//        root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.setRight(new TreeNode<>(2));
//        root1.getRight().setLeft(new TreeNode<>(4));
//        root1.getRight().setRight(new TreeNode<>(5));
//        root2 = new TreeNode<>(1);
//        root2.setLeft(new TreeNode<>(3));
//        root2.setRight(new TreeNode<>(2));
//        root2.getRight().setLeft(new TreeNode<>(5));
//        root2.getRight().setRight(new TreeNode<>(4));
//        System.out.println("Check if all levels of two trees are anagrams 1: "+
//                obj.checkIfAllLevelsOfTwoTreesAreAnagrams_1(root1, root2));
//        System.out.println("Check if all levels of two trees are anagrams 2: "+
//                obj.checkIfAllLevelsOfTwoTreesAreAnagrams_2(root1, root2));
//        root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.setRight(new TreeNode<>(3));
//        root1.getRight().setLeft(new TreeNode<>(4));
//        root1.getRight().setRight(new TreeNode<>(5));
//        root2 = new TreeNode<>(1);
//        root2.setLeft(new TreeNode<>(2));
//        root2.setRight(new TreeNode<>(3));
//        root2.getRight().setLeft(new TreeNode<>(4));
//        root2.getRight().setRight(new TreeNode<>(5));
//        root2.getRight().getRight().setLeft(new TreeNode<>(6));
//        root2.getRight().getRight().setRight(new TreeNode<>(7));
//        System.out.println("Check if all levels of two trees are anagrams 1: "+
//                obj.checkIfAllLevelsOfTwoTreesAreAnagrams_1(root1, root2));
//        System.out.println("Check if all levels of two trees are anagrams 2: "+
//                obj.checkIfAllLevelsOfTwoTreesAreAnagrams_2(root1, root2));
        //......................................................................
//        Row: 78
//        System.out.println("Count the presence of given string in char array");
//        char[][] charArr = new char[][]{
//            {'D','D','D','G','D','D'},
//            {'B','B','D','E','B','S'},
//            {'B','S','K','E','B','K'},
//            {'D','D','D','D','D','E'},
//            {'D','D','D','D','D','E'},
//            {'D','D','D','D','D','G'}
//           };
//        String str= "GEEKS";
//        obj.countOccurenceOfGivenStringInCharArray(charArr, str);
//        charArr = new char[][]{
//            {'B','B','M','B','B','B'},
//            {'C','B','A','B','B','B'},
//            {'I','B','G','B','B','B'},
//            {'G','B','I','B','B','B'},
//            {'A','B','C','B','B','B'},
//            {'M','C','I','G','A','M'}
//           };
//        str= "MAGIC";
//        obj.countOccurenceOfGivenStringInCharArray(charArr, str);
        //......................................................................
//        Row: 149
//        System.out.println("Intersection of two sorted linked list");
//        Node<Integer> node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        node.getNext().getNext().setNext(new Node<>(4));
//        node.getNext().getNext().getNext().setNext(new Node<>(5));
//        node.getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        Node<Integer> node2 = new Node<>(2);
//        node2.setNext(new Node<>(4));
//        node2.getNext().setNext(new Node<>(4));
//        node2.getNext().getNext().setNext(new Node<>(6));
//        node2.getNext().getNext().getNext().setNext(new Node<>(7));
//        obj.intersectionOfTwoSortedLinkedList(node, node2);
        //......................................................................
//        Row: 26
//        System.out.println("Check if any sub array with sum 0 is present or not");
//        System.out.println("Is there with subarray sum 0 "+obj.checkIfSubarrayWithSum0(new int[]{4, 2, -3, 1, 6}));
//        System.out.println("Is there with subarray sum 0 "+obj.checkIfSubarrayWithSum0(new int[]{2, -3, 1}));
//        System.out.println("Is there with subarray sum 0 "+obj.checkIfSubarrayWithSum0(new int[]{4, 2, 0, -1}));
        //......................................................................
//        Row: 107, 16
//        System.out.println("Find repeating and missing in unsorted array");
//        //https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/
//        //https://www.geeksforgeeks.org/find-a-repeating-and-a-missing-number/
//        obj.findRepeatingAndMissingInUnsortedArray_1(new int[]{7, 3, 4, 5, 5, 6, 2 });
//        obj.findRepeatingAndMissingInUnsortedArray_1(new int[]{3,1,3});
//        obj.findRepeatingAndMissingInUnsortedArray_2(new int[]{7, 3, 4, 5, 5, 6, 2 });
//        obj.findRepeatingAndMissingInUnsortedArray_2(new int[]{3,1,3});
//        obj.findRepeatingAndMissingInUnsortedArray_2(new int[]{4,3,2,7,8,2,3,1});
        //......................................................................
//        Row: 110
//        System.out.println("Check if any pair possible in an array having given difference");
//        System.out.println("Check if any pair is possible in the array having given diff: "+
//                obj.checkIfPairPossibleInArrayHavingGivenDiff(new int[]{5, 20, 3, 2, 5, 80}, 78));
//        System.out.println("Check if any pair is possible in the array having given diff: "+
//                obj.checkIfPairPossibleInArrayHavingGivenDiff(new int[]{90, 70, 20, 80, 50}, 45));
        //......................................................................
//        Row: 150
//        System.out.println("Intersection point in two given linked list (by ref linkage) 2 approach");
//        //https://leetcode.com/problems/intersection-of-two-linked-lists
//        Node<Integer> common = new Node<>(15);
//        common.setNext(new Node<>(30));
//        Node<Integer> node1 = new Node<>(3);
//        node1.setNext(new Node<>(9));
//        node1.getNext().setNext(new Node<>(6));
//        node1.getNext().getNext().setNext(common);
//        Node<Integer> node2 = new Node<>(10);
//        node2.setNext(common);
//        obj.intersectionPointOfTwoLinkedListByRef(node1, node2);
//        obj.intersectionPointOfTwoLinkedListByRef_HashBased(node1, node2);
//        obj.intersectionPointOfTwoLinkedListByRef_Iterative(node1, node2);
//        common = new Node<>(4);
//        common.setNext(new Node<>(5));
//        common.getNext().setNext(new Node<>(6));
//        node1 = new Node<>(1);
//        node1.setNext(new Node<>(2));
//        node1.getNext().setNext(new Node<>(3));
//        node1.getNext().getNext().setNext(common);
//        node2 = new Node<>(10);
//        node2.setNext(new Node<>(20));
//        node2.getNext().setNext(common);
//        obj.intersectionPointOfTwoLinkedListByRef(node1, node2);
//        obj.intersectionPointOfTwoLinkedListByRef_HashBased(node1, node2);
//        obj.intersectionPointOfTwoLinkedListByRef_Iterative(node1, node2);
        //......................................................................
//        Row: 359
//        System.out.println("Detect cycle in directed graph using DFS");
//        //https://www.geeksforgeeks.org/detect-cycle-in-a-graph/
//        List<List<Integer>> adjList = new ArrayList<>();
//        adjList.add(0, Arrays.asList(1)); //CYCLE 0 --> 1 --> 2 --> 0
//        adjList.add(1, Arrays.asList(2));
//        adjList.add(2, Arrays.asList(0));
//        System.out.println("Is there a cycle in directed graph: " + obj.detectCycleInDirectedGraphDFS_Graph(adjList.size(), adjList));
//        adjList = new ArrayList<>(); //NO CYCLE // 0 --> 1 --> 2
//        adjList.add(0, Arrays.asList(1));
//        adjList.add(1, Arrays.asList(2));
//        adjList.add(2, Arrays.asList());
//        System.out.println("Is there a cycle in directed graph: " + obj.detectCycleInDirectedGraphDFS_Graph(adjList.size(), adjList));
//        adjList = new ArrayList<>(); //CYCLE // 0 --> 1 --> 0
//        adjList.add(0, Arrays.asList(1));
//        adjList.add(1, Arrays.asList(0));
//        System.out.println("Is there a cycle in directed graph: " + obj.detectCycleInDirectedGraphDFS_Graph(adjList.size(), adjList));
//        adjList = new ArrayList<>(); //CYCLE // disjoint graph 0 --> 1 --> 2 | 3 --> 4 --> 5 --> 3
//        adjList.add(0, Arrays.asList(1));
//        adjList.add(1, Arrays.asList(2));
//        adjList.add(2, Arrays.asList());
//        adjList.add(3, Arrays.asList(4));
//        adjList.add(4, Arrays.asList(5));
//        adjList.add(5, Arrays.asList(3));
//        System.out.println("Is there a cycle in directed graph: " + obj.detectCycleInDirectedGraphDFS_Graph(adjList.size(), adjList));
        //......................................................................
//        Row: 363
//        System.out.println("Flood fill");
//        //https://leetcode.com/problems/flood-fill/
//        int[][] image = new int[][]{
//            {1,1,1},
//            {1,1,0},
//            {1,0,1}
//        };
//        obj.floodFill(image, 1, 1, 2);
//        image = new int[][]{
//            {0,0,0},
//            {0,0,0}
//        };
//        obj.floodFill(image, 0, 0, 2);
//        image = new int[][]{
//            {0,0,0,0,0},
//            {0,1,1,1,0},
//            {0,1,1,1,0},
//            {0,1,1,1,0},
//            {0,0,0,0,0},
//        };
//        obj.floodFill(image, 2, 2, 3);
//        image = new int[][]{
//            {0,0,0},
//            {0,1,1},
//        };
//        obj.floodFill(image, 1, 1, 1); //Edge case could have caused StackOverflowError, visited[][] helped
        //......................................................................
//        Row: 49
//        System.out.println("Maximum size of rectangle in binary matrix");
//        //https://leetcode.com/problems/maximal-rectangle
//        int[][] mat = new int[][]{
//            {0, 1, 1, 0},
//            {1, 1, 1, 1},
//            {1, 1, 1, 1},
//            {1, 1, 0, 0},};
//        obj.maxAreaOfRectangleInBinaryMatrix(mat);
//        mat = new int[][]{
//            {0, 0, 0, 0},
//            {0, 1, 1, 0},
//            {0, 1, 1, 0},
//            {0, 0, 0, 0},};
//        obj.maxAreaOfRectangleInBinaryMatrix(mat);
        //......................................................................
//        Row: 19
//        System.out.println("Merge intervals");
//        //https://leetcode.com/problems/merge-intervals/
//        //https://leetcode.com/problems/non-overlapping-intervals/
//        int[][] intervals = new int[][]{
//            {1, 3}, {2, 6}, {8, 10}, {15, 18}
//        };
//        obj.mergeIntervals_1(intervals);
//        obj.mergeIntervals_2(intervals);
//        intervals = new int[][]{
//            {1, 4}, {4, 5}
//        };
//        obj.mergeIntervals_1(intervals);
//        obj.mergeIntervals_2(intervals);
//        intervals = new int[][]{
//            {1, 4}, {0, 4}
//        };
//        obj.mergeIntervals_1(intervals);
//        obj.mergeIntervals_2(intervals);
//        intervals = new int[][]{
//            {1, 7}, {2, 5}
//        };
//        obj.mergeIntervals_1(intervals);
//        obj.mergeIntervals_2(intervals);
        //......................................................................
//        Row: 47
//        System.out.println("Row with maximum 1s in the matrix");
//        //https://www.geeksforgeeks.org/find-the-row-with-maximum-number-1s/
//        int[][] mat = new int[][]{
//            {0, 1, 1, 1},
//            {0, 0, 1, 1},
//            {1, 1, 1, 1},
//            {0, 0, 0, 0}
//        };
//        obj.maximumOnesInRowOfABinarySortedMatrix_1(mat);
//        obj.maximumOnesInRowOfABinarySortedMatrix_2(mat); //OPTIMISED
//        mat = new int[][]{
//            {0, 0, 0, 0}
//        };
//        obj.maximumOnesInRowOfABinarySortedMatrix_1(mat);
//        obj.maximumOnesInRowOfABinarySortedMatrix_2(mat); //OPTIMISED
        //......................................................................
//        Row: 45
//        System.out.println("Find a value in row wise sorted matrix");
//        //https://leetcode.com/problems/search-a-2d-matrix-ii/
//        int[][] mat = new int[][]{
//            {1, 3, 5, 7}, 
//            {10, 11, 16, 20}, 
//            {23, 30, 34, 60}
//        };
//        obj.findAValueInRowWiseSortedMatrix(mat, 13);
//        mat = new int[][]{
//            {1, 3, 5, 7}, 
//            {10, 11, 16, 20}, 
//            {23, 30, 34, 60}
//        };
//        obj.findAValueInRowWiseSortedMatrix(mat, 11);
        //......................................................................
//        Row: 65
//        System.out.println("Print all subsequences of the given string");
//        //https://www.geeksforgeeks.org/print-subsequences-string/
//        obj.printAllSubSequencesOfAString("abc");
//        obj.printAllSubSequencesOfAString("aaaa");
        //......................................................................
//        Row: 44
//        System.out.println("Spiral matrix traversal");
//        int[][] mat = new int[][]{
//            {1, 2, 3, 4},
//            {5, 6, 7, 8},
//            {9, 10, 11, 12},
//            {13, 14, 15, 16}
//        };
//        obj.spiralMatrixTraversal(mat);
//        mat = new int[][]{
//            {1, 2, 3, 4},
//           {5, 6, 7, 8},
//           {9, 10, 11, 12}
//        };
//        obj.spiralMatrixTraversal(mat);
        //......................................................................
//        Row: 156
//        System.out.println("Check singly linked list is pallindrome or not");
//        //https://leetcode.com/problems/palindrome-linked-list
//        Node<Integer> node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(1));
//        System.out.println("Is linked list pallindrome: "+obj.checkIfLinkedListPallindrome_1(node));
//        System.out.println("Is linked list pallindrome OPTIMISED: "+obj.checkIfLinkedListPallindrome_2(node));
//        node = new Node<>(1);
//        node.setNext(new Node<>(2));
//        node.getNext().setNext(new Node<>(3));
//        System.out.println("Is linked list pallindrome: "+obj.checkIfLinkedListPallindrome_1(node));
//        System.out.println("Is linked list pallindrome OPTIMISED: "+obj.checkIfLinkedListPallindrome_2(node));
        //......................................................................
//        Row: 71, 301
//        System.out.println("Balanced parenthesis evaluation");
//        //https://leetcode.com/problems/valid-parentheses/
//        System.out.println(obj.balancedParenthesisEvaluation("()"));
//        System.out.println(obj.balancedParenthesisEvaluation("({[]})"));
//        System.out.println(obj.balancedParenthesisEvaluation(")}]"));
//        System.out.println(obj.balancedParenthesisEvaluation("({)}"));
        //......................................................................
//        Row: 104
//        System.out.println("Square root of a number");
//        //https://leetcode.com/problems/sqrtx/
//        System.out.println("Square root of a number precise double value: "+obj.squareRootOfANumber_PreciseDoubleValue(4));
//        System.out.println("Square root of a number precise double value: "+obj.squareRootOfANumber_PreciseDoubleValue(1));
//        System.out.println("Square root of a number precise double value: "+obj.squareRootOfANumber_PreciseDoubleValue(3));
//        System.out.println("Square root of a number precise double value: "+obj.squareRootOfANumber_PreciseDoubleValue(1.5));
//        System.out.println("Square root of a number rounded to int value: "+obj.squareRootOfANumber_RoundedIntValue(4));
//        System.out.println("Square root of a number rounded to int value: "+obj.squareRootOfANumber_RoundedIntValue(1));
//        System.out.println("Square root of a number rounded to int value: "+obj.squareRootOfANumber_RoundedIntValue(3));
//        System.out.println("Square root of a number rounded to int value: "+obj.squareRootOfANumber_RoundedIntValue(2));
        //......................................................................
//        Row: 211
//        System.out.println("Tree isomorphic (Flip Equivalent Binary Trees)");
//        //https://leetcode.com/problems/flip-equivalent-binary-trees/
//        //https://www.geeksforgeeks.org/tree-isomorphism-problem/
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.getLeft().setLeft(new TreeNode<>(4));
//        root1.setRight(new TreeNode<>(3));
//        TreeNode<Integer> root2 = new TreeNode<>(1);
//        root2.setLeft(new TreeNode<>(3));
//        root2.getLeft().setLeft(new TreeNode<>(4));
//        root2.setRight(new TreeNode<>(2));
//        System.out.println("Are two tres isomorphic: "+obj.areTwoTreeIsoMorphic(root1, root2));
//        root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.getLeft().setLeft(new TreeNode<>(4));
//        root1.setRight(new TreeNode<>(3));
//        root2 = new TreeNode<>(1);
//        root2.setLeft(new TreeNode<>(3));
//        root2.setRight(new TreeNode<>(2));
//        root2.getRight().setRight(new TreeNode<>(4));
//        System.out.println("Are two tres isomorphic: "+obj.areTwoTreeIsoMorphic(root1, root2));
        //......................................................................
//        Row: 210
//        System.out.println("Duplicate subtrees in a tree");
//        //https://leetcode.com/problems/find-duplicate-subtrees
//        //https://www.geeksforgeeks.org/find-duplicate-subtrees/
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.getLeft().setLeft(new TreeNode<>(4));
//        root1.setRight(new TreeNode<>(3));
//        root1.getRight().setLeft(new TreeNode<>(2));
//        root1.getRight().getLeft().setLeft(new TreeNode<>(4));
//        root1.getRight().setRight(new TreeNode<>(4));
//        obj.findDuplicateSubtreeInAGivenTree(root1);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Print all the nodes that are at K distance from the target node");
//        //https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/
//        TreeNode<Integer> root1 = new TreeNode<>(3);
//        root1.setLeft(new TreeNode<>(5));
//        root1.getLeft().setLeft(new TreeNode<>(6));
//        root1.getLeft().setRight(new TreeNode<>(2));
//        root1.getLeft().getRight().setLeft(new TreeNode<>(7));
//        root1.getLeft().getRight().setRight(new TreeNode<>(4));
//        root1.setRight(new TreeNode<>(1));
//        root1.getRight().setLeft(new TreeNode<>(0));
//        root1.getRight().setRight(new TreeNode<>(8));
//        obj.printAllTheNodesAtKDistanceFromTargetNode(root1, 5, 2);
//        obj.printAllTheNodesAtKDistanceFromTargetNode(root1, 3, 3);
//        obj.printAllTheNodesAtKDistanceFromTargetNode(root1, 6, 3);
        //......................................................................
//        Row: 39
//        System.out.println("Minimum no of operations required to make an array pallindrome");
//        obj.minOperationsToMakeArrayPallindrome(new int[]{10, 15, 10});
//        obj.minOperationsToMakeArrayPallindrome(new int[]{1, 4, 5, 9, 1});
//        obj.minOperationsToMakeArrayPallindrome(new int[]{1, 2, 3, 4});
//        obj.minOperationsToMakeArrayPallindrome(new int[]{1});
//        obj.minOperationsToMakeArrayPallindrome(new int[]{1, 2});
        //......................................................................
//        Row: 112
//        System.out.println("maximum sum such that no 2 elements are adjacent / Stickler thief DP problem");
//        //https://leetcode.com/problems/house-robber
//        //https://leetcode.com/problems/house-robber-ii
//        int[] houses = new int[]{5, 5, 10, 100, 10, 5};
//        System.out.println("The maximum amount stickler thief can pick from alternate houses: " + obj.sticklerThief_Recursion(houses, houses.length));
//        System.out.println("The maximum amount stickler thief can pick from alternate houses (DP): "
//                + obj.sticklerThief_DP_Memoization(houses));
//        houses = new int[]{1, 2, 3};
//        System.out.println("The maximum amount stickler thief can pick from alternate houses: " + obj.sticklerThief_Recursion(houses, houses.length));
//        System.out.println("The maximum amount stickler thief can pick from alternate houses (DP): "
//                + obj.sticklerThief_DP_Memoization(houses));
//        houses = new int[]{5};
//        System.out.println("The maximum amount stickler thief can pick from alternate houses: " + obj.sticklerThief_Recursion(houses, houses.length));
//        System.out.println("The maximum amount stickler thief can pick from alternate houses (DP): "
//                + obj.sticklerThief_DP_Memoization(houses));
//        obj.sticklerThiefTwo_DP_Memoization(new int[]{2,3,2});
//        obj.sticklerThiefTwo_DP_Memoization(new int[]{1,2,3});
//        obj.sticklerThiefTwo_DP_Memoization(new int[]{1,2,3,1});
        //......................................................................
//        Row: 203
//        System.out.println("Check if given undirected graph is a binary tree or not");
//        //https://www.geeksforgeeks.org/check-given-graph-tree/#:~:text=Since%20the%20graph%20is%20undirected,graph%20is%20connected%2C%20otherwise%20not.
//        List<List<Integer>> adjList = new ArrayList<>();
//        adjList.add(0, Arrays.asList(1, 2, 3));
//        adjList.add(1, Arrays.asList(0));
//        adjList.add(2, Arrays.asList(0));
//        adjList.add(3, Arrays.asList(0, 4));
//        adjList.add(4, Arrays.asList(3));
//        System.out.println("Is graph is binary tree: " + obj.checkIfGivenUndirectedGraphIsBinaryTree(adjList.size(), adjList));
//        adjList = new ArrayList<>();
//        adjList.add(0, Arrays.asList(1, 2, 3));
//        adjList.add(1, Arrays.asList(0, 2)); // CYCLE 0 <--> 1 <--> 2
//        adjList.add(2, Arrays.asList(0, 1));
//        adjList.add(3, Arrays.asList(0, 4));
//        adjList.add(4, Arrays.asList(3));
//        System.out.println("Is graph is binary tree: " + obj.checkIfGivenUndirectedGraphIsBinaryTree(adjList.size(), adjList));
//        adjList = new ArrayList<>();
//        adjList.add(0, Arrays.asList(1, 2, 3));
//        adjList.add(1, Arrays.asList(0));
//        adjList.add(2, Arrays.asList(0));
//        adjList.add(3, Arrays.asList(0, 4));
//        adjList.add(4, Arrays.asList(3, 5, 6)); // CYCLE 4 <--> 5 <--> 6 but not starting with 0
//        adjList.add(5, Arrays.asList(4, 6));
//        adjList.add(6, Arrays.asList(4, 5));
//        System.out.println("Is graph is binary tree: " + obj.checkIfGivenUndirectedGraphIsBinaryTree(adjList.size(), adjList));
//        adjList = new ArrayList<>();
//        adjList.add(0, Arrays.asList(1, 2, 3));
//        adjList.add(1, Arrays.asList(0));
//        adjList.add(2, Arrays.asList(0));
//        adjList.add(3, Arrays.asList(0));
//        adjList.add(4, Arrays.asList(5, 6)); // CYCLE 4 <--> 5 <--> 6 but 3 <--> 4 not connected
//        adjList.add(5, Arrays.asList(4, 6));
//        adjList.add(6, Arrays.asList(4, 5));
//        System.out.println("Is graph is binary tree: " + obj.checkIfGivenUndirectedGraphIsBinaryTree(adjList.size(), adjList));
//        adjList = new ArrayList<>();
//        adjList.add(0, Arrays.asList(1, 2, 3));
//        adjList.add(1, Arrays.asList(0));
//        adjList.add(2, Arrays.asList(0));
//        adjList.add(3, Arrays.asList(0));
//        adjList.add(4, Arrays.asList()); // vertex 4 is not connected(3 <--> 4 not connected)
//        System.out.println("Is graph is binary tree: " + obj.checkIfGivenUndirectedGraphIsBinaryTree(adjList.size(), adjList));
        //......................................................................
//        Row: 174
//        System.out.println("First non repeating character from the stream of character");
//        obj.firstNonRepeatingCharacterFromStream("geeksforgeeksandgeeksquizfor");
//        obj.firstNonRepeatingCharacterFromStream("aaaaa");
//        obj.firstNonRepeatingCharacterFromStream("abcd");
//        obj.firstNonRepeatingCharacterFromStream("aabbccdd");
        //......................................................................
//        Row: 425
//        System.out.println("Longest increasing subsequence");
//        //https://leetcode.com/problems/longest-increasing-subsequence
//        //https://leetcode.com/problems/number-of-longest-increasing-subsequence
//        //https://www.geeksforgeeks.org/longest-increasing-subsequence-dp-3/
//        int[] arr = new int[]{10, 22, 9, 33, 21, 50, 41, 60}; //LONGEST INC SEQ 10, 22, 33, 50, 60: length = 5
//        System.out.println("Recursion Longest increasing subsequnec in the givve array: " + obj.longestIncreasingSubsequence_Recursion(arr, arr.length));
//        obj.longestIncreasingSubsequence_DP_Memoization(arr, arr.length);
//        arr = new int[]{3, 10, 2, 1, 20};
//        System.out.println("Recursion Longest increasing subsequnec in the givve array: " + obj.longestIncreasingSubsequence_Recursion(arr, arr.length));
//        obj.longestIncreasingSubsequence_DP_Memoization(arr, arr.length);
//        arr = new int[]{3, 2};
//        System.out.println("Recursion Longest increasing subsequnec in the givve array: " + obj.longestIncreasingSubsequence_Recursion(arr, arr.length));
//        obj.longestIncreasingSubsequence_DP_Memoization(arr, arr.length);
//        arr = new int[]{0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15}; //LONGEST INC SEQ 0, 2, 6, 9, 13, 15: length 6
//        System.out.println("Recursion Longest increasing subsequnec in the givve array: " + obj.longestIncreasingSubsequence_Recursion(arr, arr.length));
//        obj.longestIncreasingSubsequence_DP_Memoization(arr, arr.length);
//        arr = new int[]{5,8,7,1,9}; //LONGEST INC SEQ 5,8,9 or 5,7,9
//        System.out.println("Recursion Longest increasing subsequnec in the givve array: " + obj.longestIncreasingSubsequence_Recursion(arr, arr.length));
//        obj.longestIncreasingSubsequence_DP_Memoization(arr, arr.length);
//        arr = new int[]{5}; 
//        System.out.println("Recursion Longest increasing subsequnec in the givve array: " + obj.longestIncreasingSubsequence_Recursion(arr, arr.length));
//        obj.longestIncreasingSubsequence_DP_Memoization(arr, arr.length);
        //......................................................................
//        Row: 425
//        System.out.println("Maximum sum increasing subsequence");
//        //https://www.geeksforgeeks.org/maximum-sum-increasing-subsequence-dp-14/
//        //Max sum & also Incr subseq: [1,2,4,100] = 106
//        obj.maxSumIncreasingSubsequence_DP_Memoization(new int[]{1, 101, 2, 3, 100, 4, 5});
//        //Max sum & also Incr subseq: [3,10,20] = 33
//        obj.maxSumIncreasingSubsequence_DP_Memoization(new int[]{3, 10, 2, 1, 20});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Pair of movie watching during a flight");
//        //problem: https://www.geeksforgeeks.org/amazon-interview-experience-sde-2-10/
//        //solution: https://leetcode.com/discuss/interview-question/313719/Amazon-Online-Assessment-2019-Tho-sum-closest/291502
//        obj.pairsOfMoviesCanBeWatchedDuringFlightDurationK_Greedy(new int[]{90, 85, 75, 60, 120, 150, 125}, 250);
//        obj.pairsOfMoviesCanBeWatchedDuringFlightDurationK_Greedy(new int[]{27, 1,10, 39, 12, 52, 32, 67, 76}, 77);
        //......................................................................
//        Row: 261
//        System.out.println("Choclate distributions");
//        //https://www.geeksforgeeks.org/chocolate-distribution-problem/
//        System.out.println("Min diff in distribution of choclates among the students: "
//                +obj.choclateDistribution_Greedy(new int[]{12, 4, 7, 9, 2, 23,
//                    25, 41, 30, 40, 28,
//                    42, 30, 44, 48, 43,
//                   50}, 7));
//        System.out.println("Min diff in distribution of choclates among the students: "
//                +obj.choclateDistribution_Greedy(new int[]{7, 3, 2, 4, 9, 12, 56}, 3));
        //......................................................................
//        Row: 123
//        System.out.println("Kth element in 2 sorted array 2 approaches");
//        //https://www.geeksforgeeks.org/k-th-element-two-sorted-arrays/
//        obj.kThElementInTwoSortedArrays_1(new int[]{2, 3, 6, 7, 9 }, new int[]{1, 4, 8, 10}, 5);
//        obj.kThElementInTwoSortedArrays_2(new int[]{2, 3, 6, 7, 9 }, new int[]{1, 4, 8, 10}, 5); //OPTIMISED
        //......................................................................
//        Row: 72
//        System.out.println("Word break 1 and 2");
//        //https://leetcode.com/problems/word-break/
//        Set<String> set = new HashSet<>();
//        set.addAll(Arrays.asList("mobile","samsung","sam","sung","man","mango","icecream","and",  
//                            "go","i","like","ice","cream"));
//        System.out.println("Word break possible recursive: "+obj.wordBreak_Recursive("ilikesamsung", set)); 
//        System.out.println("Word break possible recursive: "+obj.wordBreak_Recursive("ilike", set));
//        System.out.println("Word break possible recursive: "+obj.wordBreak_Recursive("ilikedhokhla", set));
//        System.out.println("Word break possible recursive: "+obj.wordBreak_Recursive("andicecreamhill", set));
//        System.out.println("Word break possible dp: "+obj.wordBreak_DP_Problem("ilikesamsung", set)); 
//        System.out.println("Word break possible dp: "+obj.wordBreak_DP_Problem("ilike", set));
//        System.out.println("Word break possible dp: "+obj.wordBreak_DP_Problem("ilikedhokhla", set));
//        System.out.println("Word break possible dp: "+obj.wordBreak_DP_Problem("andicecreamhill", set));
//        //https://leetcode.com/problems/word-break-ii/
//        obj.wordBreakTwo_Backtracking("catsanddog", new String[]{"cat","cats","and","sand","dog"});
//        //one 'd' is missing here 
//        obj.wordBreakTwo_Backtracking("catsandog", new String[]{"cats","dog","sand","and","cat"});
//        obj.wordBreakTwo_Backtracking("aaaaaaa", new String[]{"aaaa","aa","a"});
        //......................................................................
//        Row: 245
//        System.out.println("Minimum platform needed");
//        //https://www.geeksforgeeks.org/minimum-number-platforms-required-railwaybus-station/
//        obj.minimumPlatformNeeded_BruteForce(new int[]{900, 940, 950, 1100, 1500, 1800}, 
//                new int[]{910, 1200, 1120, 1130, 1900, 2000});
//        obj.minimumPlatformNeeded_Greedy(new int[]{900, 940, 950, 1100, 1500, 1800}, 
//                new int[]{910, 1200, 1120, 1130, 1900, 2000});
        //......................................................................
//        Row: 242
//        System.out.println("Fractional knapsack");
//        //https://www.geeksforgeeks.org/fractional-knapsack-problem/
//        obj.fractionalKnapsack(new int[]{10,20,30}, new int[]{60, 100, 120}, 50);
//        obj.fractionalKnapsack(new int[]{10,20}, new int[]{60, 100}, 50);
//        obj.fractionalKnapsack(new int[]{60,70}, new int[]{60, 100}, 50);
        //......................................................................
//        Row: 326
//        System.out.println("Rotten oranges");
//        //https://leetcode.com/problems/rotting-oranges/
//        //Hash BASED: this approach rot all those oranges that are adjacent to a rotten orange in 1 unit of time
//        System.out.println("Rottening all the fresh oranges are possile in time: "
//                + obj.rottenOranges_HashBased(new int[][]{
//                    {2, 1, 1}, {1, 1, 0}, {0, 1, 1}}));
//        System.out.println("Rottening all the fresh oranges are possile in time: "
//                + obj.rottenOranges_HashBased(new int[][]{
//                    {2, 1, 1}, {0, 1, 1}, {1, 0, 1}}));
//        System.out.println("Rottening all the fresh oranges are possile in time: "
//                + obj.rottenOranges_HashBased(new int[][]{
//                    {0, 2}}));
//        //DFS BASED: this approach rot all the oranges that are connected to a rotten orange
//        //this follows flood fill way
//        obj.rottenOranges_DFS(new int[][]{
//            {2, 1, 1}, {1, 1, 0}, {0, 1, 1}});
//        obj.rottenOranges_DFS(new int[][]{
//            {2, 1, 1}, {0, 1, 1}, {1, 0, 1}});
//        obj.rottenOranges_DFS(new int[][]{
//            {0, 2}});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT : MY AMAZON ONLINE ASSESSMENT
//        System.out.println("Robot rodeo/ Robot bounded in the circle");
//        //https://leetcode.com/problems/robot-bounded-in-circle/
//        System.out.println("Is robot moving in circle: " + obj.robotRodeo("GGLLGG"));
//        System.out.println("Is robot moving in circle: " + obj.robotRodeo("GG"));
//        System.out.println("Is robot moving in circle: " + obj.robotRodeo("GL"));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT : MY AMAZON ONLINE ASSESSMENT
//        System.out.println("Swaps required to sort the array");
//        //https://leetcode.com/discuss/interview-question/346621/Google-or-Phone-Screen-or-Min-swaps-to-sort-array
//        //https://www.geeksforgeeks.org/minimum-number-swaps-required-sort-array/
//        System.out.println("Min swaps: " + obj.swapsRequiredToSortArray(new int[]{1, 5, 4, 3, 2}));
//        System.out.println("Min swaps: " + obj.swapsRequiredToSortArray(new int[]{7, 1, 2}));
//        System.out.println("Min swaps: " + obj.swapsRequiredToSortArray(new int[]{5, 1, 3, 2}));
//        System.out.println("Min swaps: " + obj.swapsRequiredToSortArray(new int[]{1, 2, 3, 4}));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("All path from source to target in directed acyclic graph");
//        //https://leetcode.com/problems/all-paths-from-source-to-target/
//        obj.allPathFromSourceToTargetInDirectedAcyclicGraph(new int[][]{
//            {1, 2}, {3}, {3}, {}
//        });
//        obj.allPathFromSourceToTargetInDirectedAcyclicGraph(new int[][]{
//            {4, 3, 1}, {3, 2, 4}, {3}, {4}, {}
//        });
//        obj.allPathFromSourceToTargetInDirectedAcyclicGraph(new int[][]{
//            {2}, {}, {1}
//        });
//        obj.allPathFromSourceToTargetInDirectedAcyclicGraph_SameButShort(new int[][]{
//            {1, 2}, {3}, {3}, {}
//        });
//        obj.allPathFromSourceToTargetInDirectedAcyclicGraph_SameButShort(new int[][]{
//            {4, 3, 1}, {3, 2, 4}, {3}, {4}, {}
//        });
//        obj.allPathFromSourceToTargetInDirectedAcyclicGraph_SameButShort(new int[][]{
//            {2}, {}, {1}
//        });
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Longest substring without repeating characters");
//        //https://leetcode.com/problems/longest-substring-without-repeating-characters/
//        obj.longestSubstringWithoutRepeatingChar("abcabcbb");
//        obj.longestSubstringWithoutRepeatingChar("bbbbb");
//        obj.longestSubstringWithoutRepeatingChar("pwwkew");
//        obj.longestSubstringWithoutRepeatingChar("");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Minimum window substring containing all the character of string t in string s");
//        //https://leetcode.com/problems/minimum-window-substring/
//        obj.minimumWindowSubstring("", "");
//        obj.minimumWindowSubstring("ADOBECODEBANC", "ABC");
//        obj.minimumWindowSubstring("a", "a");
//        obj.minimumWindowSubstring("aa", "aa");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Delete tree nodes and return forest");
//        //https://leetcode.com/problems/delete-nodes-and-return-forest/
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(3));
//        root1.setRight(new TreeNode<>(2));
//        root1.getRight().setLeft(new TreeNode<>(5));
//        root1.getRight().getLeft().setLeft(new TreeNode<>(6));
//        root1.getRight().getLeft().getLeft().setLeft(new TreeNode<>(7));
//        root1.getRight().getLeft().setRight(new TreeNode<>(8));
//        root1.getRight().setRight(new TreeNode<>(4));
//        obj.deleteTreeNodesAndReturnForest(root1, new int[]{8, 1, 6});
//        root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(3));
//        root1.setRight(new TreeNode<>(2));
//        obj.deleteTreeNodesAndReturnForest(root1, new int[]{3, 2});
        //......................................................................
//        Row: 321
//        System.out.println("LRU cache design");
//        List<String> operations = Arrays.asList("LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get");
//        List<List<Integer>> inputs = Arrays.asList(
//                Arrays.asList(2),
//                Arrays.asList(1, 1),
//                Arrays.asList(2, 2),
//                Arrays.asList(1),
//                Arrays.asList(3, 3),
//                Arrays.asList(2),
//                Arrays.asList(4, 4),
//                Arrays.asList(1),
//                Arrays.asList(3),
//                Arrays.asList(4)
//        );
//        obj.LRUCacheDesignImpl(operations, inputs);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Product of array excluding element itself");
//        obj.productOfArrayExcludingElementItself_BruteForce(new int[]{1, 2, 3, 4});
//        obj.productOfArrayExcludingElementItself_Optimised1(new int[]{1, 2, 3, 4});
//        obj.productOfArrayExcludingElementItself_Optimised2(new int[]{1, 2, 3, 4}); //TIME & SPACE OPTIMISED
        //......................................................................
//        Row: 196
//        System.out.println("Construct binary tree from inorder and preorder/ postorder");
//        //https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
//        //https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/
//        obj.constructBinaryTreeFromInorderPreorderArray(new int[]{9, 3, 15, 20, 7}, new int[]{3, 9, 20, 15, 7});
//        obj.constructBinaryTreeFromInorderPostorderArray(new int[]{9, 3, 15, 20, 7}, new int[]{9,15,7,20,3});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Construct BST from given preorder/postorder array");
//        //https://practice.geeksforgeeks.org/problems/preorder-to-postorder4423/1/
//        //PREORDER ARRAY
//        obj.constructBinarySearchTreeFromPreorderArray(new int[]{40, 30, 35, 80, 100});
//        obj.constructBinarySearchTreeFromPreorderArray(new int[]{40, 30, 32, 35, 80, 90, 100, 120});
//        //POSTORDER ARRAY
//        obj.constructBinarySearchTreeFromPostorderArray(new int[]{35, 30, 100, 80, 40});
//        obj.constructBinarySearchTreeFromPostorderArray(new int[]{35, 32, 30, 120, 100, 90, 80, 40});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Count all the occurences of anagrams of given pattern in the text SLIDING WINDOW");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-324-sde2/
//        //https://leetcode.com/problems/permutation-in-string/
//        obj.countAllOccurencesOfPatternInGivenString("forxxorfxdofr", "for");
//        obj.countAllOccurencesOfPatternInGivenString("foorxxorfxdofr", "for");
//        obj.countAllOccurencesOfPatternInGivenString("aabaabaa", "aaba");
//        obj.countAllOccurencesOfPatternInGivenString("aabaabaa", "xyz");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Convert all the leaves of tree to DLL and remove leaves from tree");
//        //https://practice.geeksforgeeks.org/problems/leaves-to-dll/1/
//        TreeNode<Integer> root = new TreeNode<>(1);
//        root.setLeft(new TreeNode<>(2));
//        root.getLeft().setLeft(new TreeNode<>(4));
//        root.getLeft().setRight(new TreeNode<>(5));
//        root.setRight(new TreeNode<>(3));
//        obj.leavesOfTreeToDoublyLinkedListAndRemoveLeaves(root);
//        root = new TreeNode<>(1);
//        root.setLeft(new TreeNode<>(2));
//        root.getLeft().setLeft(new TreeNode<>(4));
//        root.getLeft().setRight(new TreeNode<>(5));
//        root.setRight(new TreeNode<>(3));
//        root.getRight().setLeft(new TreeNode<>(6));
//        root.getRight().setRight(new TreeNode<>(7));
//        obj.leavesOfTreeToDoublyLinkedListAndRemoveLeaves(root);
//        root = new TreeNode<>(1); //EDGE CASE
//        obj.leavesOfTreeToDoublyLinkedListAndRemoveLeaves(root);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Maximum of all subarray in size of K");
//        //https://leetcode.com/problems/sliding-window-maximum/
//        //https://www.geeksforgeeks.org/sliding-window-maximum-maximum-of-all-subarrays-of-size-k/
//        //https://practice.geeksforgeeks.org/problems/maximum-of-all-subarrays-of-size-k3101/1/
//        obj.maximumOfAllSubArrayOfSizeK(new int[]{1, 2, 3, 1, 4, 5, 2, 3, 6}, 3);
//        obj.maximumOfAllSubArrayOfSizeK(new int[]{8, 5, 10, 7, 9, 4, 15, 12, 90, 13}, 4);
//        obj.maximumOfAllSubArrayOfSizeK(new int[]{1, 3, 1, 2, 0, 5}, 3);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Check two N-ary trees are mirror image of each other");
//        //https://youtu.be/UGzXSDZv-SY
//        List<List<Integer>> tree1 = Arrays.asList(
//                Arrays.asList(1, 2, 3, 4), //0
//                Arrays.asList(5, 6), //1
//                Arrays.asList(7), //2
//                Arrays.asList(), //3
//                Arrays.asList(), //4
//                Arrays.asList(), //5
//                Arrays.asList(), //6
//                Arrays.asList() //7
//        );
//        List<List<Integer>> tree2 = Arrays.asList(
//                Arrays.asList(4, 3, 2, 1), //0
//                Arrays.asList(6, 5), //1
//                Arrays.asList(7), //2
//                Arrays.asList(), //3
//                Arrays.asList(), //4
//                Arrays.asList(), //5
//                Arrays.asList(), //6
//                Arrays.asList() //7
//        );
//        System.out.println("Check if two N-ary trees are mirror of each other: "
//                + obj.checkIfTwoNAryTreeAreMirror(tree1, tree2));
//        tree1 = Arrays.asList(
//                Arrays.asList(1, 2, 3, 4), //0
//                Arrays.asList(5, 6), //1
//                Arrays.asList(7), //2
//                Arrays.asList(), //3
//                Arrays.asList(), //4
//                Arrays.asList(), //5
//                Arrays.asList(), //6
//                Arrays.asList() //7
//        );
//        //BOTH TREE ARE SAME NOW THIS SHOULD GIVE FALSE
//        tree2 = Arrays.asList(
//                Arrays.asList(1, 2, 3, 4), //0
//                Arrays.asList(5, 6), //1
//                Arrays.asList(7), //2
//                Arrays.asList(), //3
//                Arrays.asList(), //4
//                Arrays.asList(), //5
//                Arrays.asList(), //6
//                Arrays.asList() //7
//        );
//        System.out.println("Check if two N-ary trees are mirror of each other: "
//                + obj.checkIfTwoNAryTreeAreMirror(tree1, tree2));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Find minimum in rotated sorted array");
//        //https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/
//        obj.findMinimumInRotatedSortedArray(new int[]{3,4,5,1,2});
//        obj.findMinimumInRotatedSortedArray(new int[]{0,1,2,3,4,5});
//        obj.findMinimumInRotatedSortedArray(new int[]{4,6,8,10,1,3});
//        obj.findMinimumInRotatedSortedArray(new int[]{5,4,3,2,1});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Deepest leaves sum in the tree");
//        //https://leetcode.com/problems/deepest-leaves-sum/
//        TreeNode<Integer> root = new TreeNode<>(1);
//        root.setLeft(new TreeNode<>(2));
//        root.getLeft().setLeft(new TreeNode<>(4));
//        root.getLeft().setRight(new TreeNode<>(5));
//        root.getLeft().getLeft().setLeft(new TreeNode<>(7));
//        root.setRight(new TreeNode<>(3));
//        root.getRight().setRight(new TreeNode<>(6));
//        root.getRight().getRight().setRight(new TreeNode<>(8));
//        obj.deepestLeavesSumOfTree_Iterative(root);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Average waiting time");
//        //https://leetcode.com/problems/average-waiting-time/
//        obj.averageWaitingTime(new int[][]{{1, 2}, {2, 5}, {4, 3}});
//        obj.averageWaitingTime(new int[][]{{5, 2}, {5, 4}, {10, 3}, {20, 1}});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Partition labels");
//        //https://leetcode.com/problems/partition-labels/
//        obj.partitionLabels("ababcbacadefegdehijhklij");
//        obj.partitionLabels("aabbaaccddcc");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Longest repeating character replacement (SLIDING WINDOW)");
//        //https://leetcode.com/problems/longest-repeating-character-replacement/
//        obj.longestRepeatingCharacterByKReplacement("ABAB", 2);
//        obj.longestRepeatingCharacterByKReplacement("AABABBA", 1);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("3Sum");
//        //https://leetcode.com/problems/3sum/
//        obj.threeSum(new int[]{-1,0,1,2,-1,-4});
//        obj.threeSum(new int[]{0}); //no trplet sum is possible
//        obj.threeSum(new int[]{});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Minimum time differenec");
//        //https://leetcode.com/problems/minimum-time-difference/
//        System.out.println("Min time difference: "+obj.minimumTimeDifference(Arrays.asList("23:59","00:00")));
//        System.out.println("Min time difference: "+obj.minimumTimeDifference(Arrays.asList("15:45","16:00")));
//        System.out.println("Min time difference: "+obj.minimumTimeDifference(Arrays.asList("00:00","23:59","00:00")));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Length of contigous array (subarray with equal no of 0 and 1)");
//        //https://leetcode.com/problems/contiguous-array/
//        obj.contigousArrayWithEqualZeroAndOne(new int[]{0,1});
//        obj.contigousArrayWithEqualZeroAndOne(new int[]{0,1,0,1,0});
//        obj.contigousArrayWithEqualZeroAndOne(new int[]{1,0,0,1,0});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Generate balanced parenthesis");
//        //https://www.geeksforgeeks.org/print-all-combinations-of-balanced-parentheses/
//        //https://leetcode.com/problems/generate-parentheses/
//        obj.generateBalancedParenthesis(1);
//        obj.generateBalancedParenthesis(2);
//        obj.generateBalancedParenthesis(3);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Find whether path exists");
//        //https://practice.geeksforgeeks.org/problems/find-whether-path-exist5238/1
//        System.out.println("Path exists: " + obj.checkIfPathExistsFromSourceToDestination(new int[][]{
//            {1, 3},
//            {3, 2}
//        }));
//        System.out.println("Path exists: " + obj.checkIfPathExistsFromSourceToDestination(new int[][]{
//            {3, 0, 3, 0, 0}, {3, 0, 0, 0, 3}, {3, 3, 3, 3, 3}, {0, 2, 3, 0, 0}, {3, 0, 0, 1, 3}
//        }));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Reverse Linked list in K groups alternatively");
//        //https://www.interviewbit.com/problems/reverse-alternate-k-nodes/
//        Node<Integer> head = new Node<>(3);
//        head.setNext(new Node<>(4));
//        head.getNext().setNext(new Node<>(7));
//        head.getNext().getNext().setNext(new Node<>(5));
//        head.getNext().getNext().getNext().setNext(new Node<>(6));
//        head.getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        head.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(15));
//        head.getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(61));
//        head.getNext().getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(16));
//        new LinkedListUtil<Integer>(obj.reverseLinkedListInKGroupsAlternatively(head, 3)).print();
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Smallest String in a tree starting from leaf to root");
//        //https://leetcode.com/problems/smallest-string-starting-from-leaf/
//        TreeNode<Integer> root = new TreeNode<>(0);
//        root.setLeft(new TreeNode<>(1));
//        root.getLeft().setLeft(new TreeNode<>(3));
//        root.getLeft().setRight(new TreeNode<>(4));
//        root.setRight(new TreeNode<>(2));
//        root.getRight().setLeft(new TreeNode<>(3));
//        root.getRight().setRight(new TreeNode<>(4));
//        obj.smallestStringInTreeFromLeafToRoot(root);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Print the sum where each root-to-leaf path of tree represent a number");
//        //https://leetcode.com/problems/sum-root-to-leaf-numbers/
//        TreeNode<Integer> root = new TreeNode<>(1);
//        root.setLeft(new TreeNode<>(2));
//        root.setRight(new TreeNode<>(3));
//        obj.printSumWhereRootToLeafPathIsANumber(root);
//        root = new TreeNode<>(1);
//        root.setLeft(new TreeNode<>(2));
//        root.getLeft().setLeft(new TreeNode<>(4));
//        root.getLeft().setRight(new TreeNode<>(5));
//        root.getLeft().getLeft().setLeft(new TreeNode<>(7));
//        root.setRight(new TreeNode<>(3));
//        root.getRight().setRight(new TreeNode<>(6));
//        root.getRight().getRight().setRight(new TreeNode<>(8));
//        obj.printSumWhereRootToLeafPathIsANumber(root);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Max sum path in two sorted arrays");
//        //https://www.geeksforgeeks.org/maximum-sum-path-across-two-arrays/
//        obj.maxSumPathInTwoSortedArrays(new int[]{2, 3, 7, 10, 12}, new int[]{1, 5, 7, 8});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Asteroid collision");
//        //https://leetcode.com/problems/asteroid-collision/
//        obj.asteroidCollision(new int[] {5,10,-5});
//        obj.asteroidCollision(new int[] {8, -8});
//        obj.asteroidCollision(new int[] {10,2,-5});
//        obj.asteroidCollision(new int[] {3,2,1,-10});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Longest pallindromic substring");
//        //https://leetcode.com/problems/longest-palindromic-substring/
//        obj.longestPallindromicSubstring("racecar");
//        obj.longestPallindromicSubstring("babccaa");
//        obj.longestPallindromicSubstring("aabbaa");
//        obj.longestPallindromicSubstring("cbb");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Next greater element 2 (consider array to cyclic)");
//        //https://leetcode.com/problems/next-greater-element-ii/
//        obj.nextGreaterElement2_CyclicArray(new int[]{1, 2, 1});
//        obj.nextGreaterElement2_CyclicArray(new int[]{5, 4, 3, 2, 1});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Valid sudoku");
//        //https://leetcode.com/problems/valid-sudoku/
//        //https://leetcode.com/problems/check-if-every-row-and-column-contains-all-numbers/
//        obj.validSudoku(new String[][]{
//            {"5", "3", ".", ".", "7", ".", ".", ".", "."}, 
//            {"6", ".", ".", "1", "9", "5", ".", ".", "."}, 
//            {".", "9", "8", ".", ".", ".", ".", "6", "."}, 
//            {"8", ".", ".", ".", "6", ".", ".", ".", "3"}, 
//            {"4", ".", ".", "8", ".", "3", ".", ".", "1"}, 
//            {"7", ".", ".", ".", "2", ".", ".", ".", "6"}, 
//            {".", "6", ".", ".", ".", ".", "2", "8", "."}, 
//            {".", ".", ".", "4", "1", "9", ".", ".", "5"}, 
//            {".", ".", ".", ".", "8", ".", ".", "7", "9"}});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Diagonal matrix traversal");
//        //https://www.geeksforgeeks.org/print-matrix-diagonal-pattern/
//        obj.diagonalMatrixTraversal(new int[][]{
//            {1, 2, 3},
//            {4, 5, 6},
//            {7, 8, 9}});
//        obj.diagonalMatrixTraversal(new int[][]{
//            {1, 2, 3},
//            {4, 5, 6}});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Score of parenthesis");
//        //https://leetcode.com/problems/score-of-parentheses
//        obj.scoreOfParenthesis("()");
//        obj.scoreOfParenthesis("()()");
//        obj.scoreOfParenthesis("(())");
//        obj.scoreOfParenthesis("(()())");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("minimun remove of character to make valid parenthesis");
//        //https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/
//        obj.minimumCharRemovalToMakeValidParenthesis("lee(t(c)o)de)");
//        obj.minimumCharRemovalToMakeValidParenthesis("a)b(c)d");
//        obj.minimumCharRemovalToMakeValidParenthesis(")))(((");
//        obj.minimumCharRemovalToMakeValidParenthesis("()()");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Repeated substring pattern");
//        //https://leetcode.com/problems/repeated-substring-pattern/
//        System.out.println("Repeated substring pattern possible: "+obj.repeatedSubstringPattern("abab"));
//        System.out.println("Repeated substring pattern possible: "+obj.repeatedSubstringPattern("aba"));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Find median in data stream");
//        //https://leetcode.com/problems/find-median-from-data-stream/
//        obj.findMedianInDataStream(new int[] {5, 15, 1, 3, 2, 8, 7, 9, 10, 6, 11, 4});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Pairs of Songs With Total Durations Divisible by 60");
//        //https://leetcode.com/problems/pairs-of-songs-with-total-durations-divisible-by-60/
//        obj.numPairsDivisibleBy60(new int[]{30,20,150,100,40});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Count inversion in the arrays (using modified merge sort)");
//        obj.countInversion(new int[]{1, 20, 6, 4, 5});
//        obj.countInversion(new int[]{1, 3, 2}); 
//        obj.countInversion(new int[]{7,1,2});
//        obj.countInversion(new int[]{5,3,2,4,1});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Minimum window length where subarray is greater OR equal to target sum");
//        //https://leetcode.com/problems/minimum-size-subarray-sum/
//        obj.minimumWindowSubarrayForTargetSumK(new int[] {2,3,1,2,4,3}, 7);
//        obj.minimumWindowSubarrayForTargetSumK(new int[] {1,4,4}, 4);
//        obj.minimumWindowSubarrayForTargetSumK(new int[] {1,1,1,1,1,1,1,1}, 11);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Maximum length of repeated subarray (DP PROBLEM)");
//        //https://leetcode.com/problems/maximum-length-of-repeated-subarray/
//        obj.maximumLengthOfRepeatedSubarray_DP_Memoization(new int[]{1,2,3,2,1}, new int[]{3,2,1,4,7});
//        obj.maximumLengthOfRepeatedSubarray_DP_Memoization(new int[]{0,1,1,1,1,1}, new int[]{0,1,0,1,0,1});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Fix two swapped nodes of the BST");
//        //https://www.geeksforgeeks.org/fix-two-swapped-nodes-of-bst/
//        //https://leetcode.com/problems/recover-binary-search-tree/
//        TreeNode<Integer> root = new TreeNode<>(10);
//        root.setLeft(new TreeNode<>(5));
//        root.getLeft().setLeft(new TreeNode<>(2));
//        root.getLeft().setRight(new TreeNode<>(20)); //FIRST
//        root.setRight(new TreeNode<>(8)); //LAST
//        obj.fixTwoSwappedNodesInBST(root);
//        root = new TreeNode<>(3); //MID
//        root.setLeft(new TreeNode<>(5)); //FIRST
//        obj.fixTwoSwappedNodesInBST(root);
//        root = new TreeNode<>(3);
//        root.setLeft(new TreeNode<>(5)); //FIRST
//        root.setRight(new TreeNode<>(1)); //LAST
//        obj.fixTwoSwappedNodesInBST(root);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Merge two binary trees");
//        //https://leetcode.com/problems/merge-two-binary-trees/
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(3));
//        root1.getLeft().setLeft(new TreeNode<>(5));
//        root1.setRight(new TreeNode<>(2));
//        TreeNode<Integer> root2 = new TreeNode<>(2);
//        root2.setLeft(new TreeNode<>(1));
//        root2.getLeft().setRight(new TreeNode<>(4));
//        root2.setRight(new TreeNode<>(3));
//        root2.getRight().setRight(new TreeNode<>(7));
//        obj.mergeTwoBinaryTree(root1, root2);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Flip M 0s in binary array and find maximum length of consecutive 1s");
//        //https://leetcode.com/problems/max-consecutive-ones-iii/
//        obj.flipMZerosFindMaxLengthOfConsecutiveOnes(new int[]{1, 0, 1}, 1);
//        obj.flipMZerosFindMaxLengthOfConsecutiveOnes(new int[]{1, 0, 0, 1, 1, 0, 1, 0, 1, 1}, 3);
//        obj.flipMZerosFindMaxLengthOfConsecutiveOnes(new int[]{0,0,0,}, 1);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Local minima and local maxima(Peak element) (BINARY SEARCH)");
//        //https://www.geeksforgeeks.org/find-local-minima-array/
//        //https://leetcode.com/problems/find-peak-element/        
//        obj.findLocalMinima(new int[]{1, 2, 3});
//        obj.findLocalMinima(new int[]{23, 8, 15, 2, 3});
//        obj.findLocalMinima(new int[]{9, 6, 3, 14, 5, 7, 4});
//        obj.findLocalMinima(new int[]{3, 2, 1});
//        //MAXIMA OR PEAK ELEMENT
//        obj.findLocalMaxima(new int[]{3, 2, 1});
//        obj.findLocalMaxima(new int[]{1, 2, 3});
//        obj.findLocalMaxima(new int[]{9, 6, 3, 14, 5, 7, 4});
//        obj.findLocalMaxima(new int[]{23, 8, 15, 2, 3});
//        obj.findLocalMaxima(new int[]{23, 8});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Reorder linked list");
//        //https://leetcode.com/problems/reorder-list/
//        Node<Integer> head = new Node<Integer>(1);
//        head.setNext(new Node<>(2));
//        head.getNext().setNext(new Node<>(3));
//        head.getNext().getNext().setNext(new Node<>(4));
//        head.getNext().getNext().getNext().setNext(new Node<>(5));
//        obj.reorderLinkedList(head);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Remove K digits and create the smallest num");
//        //https://leetcode.com/problems/remove-k-digits/
//        obj.removeKDigitsToCreateSmallestNumber("1432219", 3);
//        obj.removeKDigitsToCreateSmallestNumber("10", 2);
//        obj.removeKDigitsToCreateSmallestNumber("10200", 1);
//        obj.removeKDigitsToCreateSmallestNumber("4321", 2);
//        obj.removeKDigitsToCreateSmallestNumber("1234", 2);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Find the most competetive subsequence of length K in the given array");
//        //https://leetcode.com/problems/find-the-most-competitive-subsequence/
//        obj.findTheMostCompetetiveSubsequenceOfSizeKFromArray(new int[]{3,5,2,6}, 2);
//        obj.findTheMostCompetetiveSubsequenceOfSizeKFromArray(new int[]{2,4,3,3,5,4,9,6}, 4);
//        obj.findTheMostCompetetiveSubsequenceOfSizeKFromArray(new int[]{5,4,3,2,1}, 2);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Odd even linked list");
//        //https://leetcode.com/problems/odd-even-linked-list/
//        Node<Integer> head = new Node<Integer>(1);
//        head.setNext(new Node<>(2));
//        head.getNext().setNext(new Node<>(3));
//        head.getNext().getNext().setNext(new Node<>(4));
//        head.getNext().getNext().getNext().setNext(new Node<>(5));
//        obj.rearrangeLinkedListAsOddIndexFirstAndEvenIndexAtEnd(head);
//        head = new Node<Integer>(1);
//        head.setNext(new Node<>(2));
//        head.getNext().setNext(new Node<>(3));
//        head.getNext().getNext().setNext(new Node<>(4));
//        head.getNext().getNext().getNext().setNext(new Node<>(5));
//        head.getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        obj.rearrangeLinkedListAsOddIndexFirstAndEvenIndexAtEnd(head);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Number of ways to create BST and BT with given N (req info: Catalan number series)");
//        //https://leetcode.com/problems/unique-binary-search-trees/
//        //https://www.geeksforgeeks.org/total-number-of-possible-binary-search-trees-with-n-keys/#
//        obj.numberOfWaysToCreateBSTAndBTWithGivenN(3);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Check if binary tree is complete binary tree or not");
//        //https://leetcode.com/problems/check-completeness-of-a-binary-tree/
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.getLeft().setLeft(new TreeNode<>(4));
//        root1.getLeft().setRight(new TreeNode<>(5));
//        root1.setRight(new TreeNode<>(3));
//        root1.getRight().setLeft(new TreeNode<>(6)); //COMPLETE BINARY TREE
//        System.out.println("Is tree complete binary tree: "+obj.checkIfBinaryTreeIsCompleteOrNot(root1));
//        root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.getLeft().setLeft(new TreeNode<>(4));
//        root1.getLeft().setRight(new TreeNode<>(5));
//        root1.setRight(new TreeNode<>(3));
//        root1.getRight().setRight(new TreeNode<>(7)); //NOT-COMPLETE BINARY TREE
//        System.out.println("Is tree complete binary tree: "+obj.checkIfBinaryTreeIsCompleteOrNot(root1));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("maximum width of binary tree");
//        //https://leetcode.com/problems/maximum-width-of-binary-tree/
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(3));
//        root1.getLeft().setLeft(new TreeNode<>(5));
//        root1.getLeft().setRight(new TreeNode<>(3));
//        root1.setRight(new TreeNode<>(2));
//        root1.getRight().setRight(new TreeNode<>(9));
//        obj.maximumWidthOfBinaryTree(root1);
//        root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(3));
//        root1.getLeft().setLeft(new TreeNode<>(5));
//        root1.getLeft().getLeft().setLeft(new TreeNode<>(6));
//        root1.setRight(new TreeNode<>(2));
//        root1.getRight().setRight(new TreeNode<>(9));
//        root1.getRight().getRight().setRight(new TreeNode<>(7));
//        obj.maximumWidthOfBinaryTree(root1);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Sum of elements in matrix except given row and col");
//        //https://www.geeksforgeeks.org/find-sum-of-all-elements-in-a-matrix-except-the-elements-in-given-row-andor-column-2/
//        int[][] rowAndCol = new int[][]{
//            {0, 0},
//            {1, 1},
//            {0, 1}
//        };
//        int[][] matrix = new int[][]{
//            {1, 1, 2},
//            {3, 4, 6},
//            {5, 3, 2}
//        };
//        obj.sumOfElementsInMatrixExceptGivenRowAndCol(matrix, rowAndCol);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Count number elements from second array less than or equal to element in first array");
//        //https://www.geeksforgeeks.org/element-1st-array-count-elements-less-equal-2nd-array/
//        obj.countElementsFromSecondArrayLessOrEqualToElementInFirstArray(
//                new int[]{1, 2, 3, 4, 7, 9},
//                new int[]{0, 1, 2, 1, 1, 4});
//        obj.countElementsFromSecondArrayLessOrEqualToElementInFirstArray(
//                new int[]{5, 10, 2, 6, 1, 8, 6, 12},
//                new int[]{6, 5, 11, 4, 2, 3, 7});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Check if input binary number stream is divisible by N");
//        //https://leetcode.com/problems/binary-prefix-divisible-by-5/
//        //https://www.geeksforgeeks.org/check-divisibility-binary-stream/
//        obj.checkBinaryNumberStreamIsDivisibleByN(new int[]{1,0,1,0,1}, 3);
//        obj.checkBinaryNumberStreamIsDivisibleByN(new int[]{1,0,1}, 5);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Convert number to words");
//        //https://www.geeksforgeeks.org/program-to-convert-a-given-number-to-words-set-2/
//        obj.convertNumberToWords(438237764);
//        obj.convertNumberToWords(0);
//        obj.convertNumberToWords(101);
//        obj.convertNumberToWords(1000);
//        obj.convertNumberToWords(222);
//        obj.convertNumberToWords(999999999);
//        obj.convertNumberToWords(1234567891);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Maximum swaps to form greater number by swaping 2 digit atmost ");
//        //https://leetcode.com/problems/maximum-swap/
//        obj.swapTwoDigitAtMostToFormAGreaterNumber_Greedy(2736);
//        obj.swapTwoDigitAtMostToFormAGreaterNumber_Greedy(9973);
//        obj.swapTwoDigitAtMostToFormAGreaterNumber_Greedy(1002);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("String comparision after processing backspace character");
//        //https://leetcode.com/problems/backspace-string-compare/
//        System.out.println("Are two strings same after processing backspace char: "+
//                obj.stringComparisionAfterProcessingBackspaceChar("ab#c", "ad#c"));
//        System.out.println("Are two strings same after processing backspace char: "+
//                obj.stringComparisionAfterProcessingBackspaceChar("a#c", "b"));
//        System.out.println("Are two strings same after processing backspace char: "+
//                obj.stringComparisionAfterProcessingBackspaceChar("a#", "b#"));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Find negative number in window size of K from the array");
//        //https://practice.geeksforgeeks.org/problems/first-negative-integer-in-every-window-of-size-k/0
//        obj.firstNegativeNumberInWindowKFromArray(new int[]{-8, 2, 3, -6, 10}, 2);
//        obj.firstNegativeNumberInWindowKFromArray(new int[]{12, -1, -7, 8, -15, 30, 16, 28}, 3);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("String zig zag");
//        //https://leetcode.com/problems/zigzag-conversion/
//        obj.stringZigZag("PAYPALISHIRING", 3);
//        obj.stringZigZag("PAYPALISHIRING", 4);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Can we divide the given array into consecutive sequence of length W");
//        //https://leetcode.com/problems/hand-of-straights/
//        //https://leetcode.com/problems/divide-array-in-sets-of-k-consecutive-numbers
//        System.out.println("Given array can be divided into consecutive groups of length W: "
//                +obj.handOfStraight(new int[]{1,2,3,6,2,3,4,7,8}, 3));
//        System.out.println("Given array can be divided into consecutive groups of length W: "
//                + obj.handOfStraight(new int[]{1,2,3,4,5}, 4));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Can we visit all the rooms");
//        //https://leetcode.com/problems/keys-and-rooms/
//        List<List<Integer>> rooms = Arrays.asList(
//                Arrays.asList(1),
//                Arrays.asList(2),
//                Arrays.asList(3),
//                Arrays.asList()
//        );
//        System.out.println("Can we visit all the rooms: "+obj.canWeVisitAllTheRooms(rooms));
//        rooms = Arrays.asList(
//                Arrays.asList(1, 3),
//                Arrays.asList(3, 0, 1),
//                Arrays.asList(2),
//                Arrays.asList(0)
//        );
//        System.out.println("Can we visit all the rooms: "+obj.canWeVisitAllTheRooms(rooms));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("number of Steps taken to Open the lock");
//        //https://leetcode.com/problems/open-the-lock/
//        String[] deadends = new String[]{"0201", "0101", "0102", "1212", "2002"};
//        String target = "0202";
//        System.out.println("Steps required: " + obj.stepsToOpenTheLock(deadends, target));
//        deadends = new String[]{"8887", "8889", "8878", "8898", "8788", "8988", "7888", "9888"};
//        target = "8888";
//        System.out.println("Steps required: " + obj.stepsToOpenTheLock(deadends, target));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Squares of sorted array");
//        //https://leetcode.com/problems/squares-of-a-sorted-array/
//        obj.sortedSquaresOfSortedArray_1(new int[]{-4, -1, 0, 3, 10});
//        obj.sortedSquaresOfSortedArray_2(new int[]{-4, -1, 0, 3, 10}); //OPTIMISED
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Move zeros to end");
//        //https://leetcode.com/problems/move-zeroes/
//        obj.moveZeroesToEnd(new int[]{0, 1, 0, 3, 12});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Distribute coins in binary tree");
//        //https://leetcode.com/problems/distribute-coins-in-binary-tree/
//        TreeNode<Integer> root = new TreeNode<>(3);
//        root.setLeft(new TreeNode<>(0));
//        root.setRight(new TreeNode<>(0));
//        obj.distributeCoinsInBinaryTree(root);
//        root = new TreeNode<>(0);
//        root.setLeft(new TreeNode<>(3));
//        root.setRight(new TreeNode<>(0));
//        obj.distributeCoinsInBinaryTree(root);
//        root = new TreeNode<>(1);
//        root.setLeft(new TreeNode<>(0));
//        root.setRight(new TreeNode<>(2));
//        obj.distributeCoinsInBinaryTree(root);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Next warmer day in the given weather recordings (Next greater element to right)");
//        //https://youtu.be/0mcAy91rPzE
//        //https://leetcode.com/problems/daily-temperatures/
//        obj.nextWarmerDayInTheGivenWeatherRecordings(new int[] {60, 90, 76, 80, 100, 62, 90});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Container with most water");
//        //https://leetcode.com/problems/container-with-most-water/
//        obj.containerWithMostWater(new int[]{1,8,6,2,5,4,8,3,7});
//        obj.containerWithMostWater(new int[]{5,6,8,9,8,6,5});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Check if binary tree is odd-even binary tree");
//        //https://leetcode.com/problems/even-odd-tree
//        TreeNode<Integer> root = new TreeNode<>(1);
//        root.setLeft(new TreeNode<>(10));
//        root.getLeft().setLeft(new TreeNode<>(3));
//        root.getLeft().getLeft().setLeft(new TreeNode<>(12));
//        root.getLeft().getLeft().setRight(new TreeNode<>(8));
//        root.setRight(new TreeNode<>(4));
//        root.getRight().setLeft(new TreeNode<>(7));
//        root.getRight().getLeft().setLeft(new TreeNode<>(6));
//        root.getRight().setRight(new TreeNode<>(9));
//        root.getRight().getRight().setRight(new TreeNode<>(2));
//        System.out.println("Check is binary tree is odd-even tree: "+obj.checkIfBinaryTreeIsOddEvenTree(root));
//        root = new TreeNode<>(1);
//        root.setLeft(new TreeNode<>(11)); //LEVEL - 1 (ODD) data == ODD which is FALSE
//        root.getLeft().setLeft(new TreeNode<>(3));
//        root.getLeft().getLeft().setLeft(new TreeNode<>(12));
//        root.getLeft().getLeft().setRight(new TreeNode<>(8));
//        root.setRight(new TreeNode<>(4));
//        root.getRight().setLeft(new TreeNode<>(7));
//        root.getRight().getLeft().setLeft(new TreeNode<>(6));
//        root.getRight().setRight(new TreeNode<>(9));
//        root.getRight().getRight().setRight(new TreeNode<>(2));
//        System.out.println("Check is binary tree is odd-even tree: "+obj.checkIfBinaryTreeIsOddEvenTree(root));
//        root = new TreeNode<>(1);
//        root.setLeft(new TreeNode<>(10)); 
//        root.getLeft().setLeft(new TreeNode<>(3));
//        root.getLeft().getLeft().setLeft(new TreeNode<>(12));
//        root.getLeft().getLeft().setRight(new TreeNode<>(8));
//        root.setRight(new TreeNode<>(4));
//        root.getRight().setLeft(new TreeNode<>(3)); //LEVEL - 2 (EVEN) nodes at this level should be strictly incr but level 2[3,3, 9] FALSE
//        root.getRight().getLeft().setLeft(new TreeNode<>(6));
//        root.getRight().setRight(new TreeNode<>(9));
//        root.getRight().getRight().setRight(new TreeNode<>(2));
//        System.out.println("Check is binary tree is odd-even tree: "+obj.checkIfBinaryTreeIsOddEvenTree(root));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Longest Substring With K Unique Characters (SLIDING WINDOW)");
//        //https://practice.geeksforgeeks.org/problems/longest-k-unique-characters-substring0853/1
//        obj.longestSubstringWithKUniqueCharacter("aabacbebebe", 3);
//        obj.longestSubstringWithKUniqueCharacter("aabcaccbeb", 3);
//        obj.longestSubstringWithKUniqueCharacter("aaaa", 2);
//        obj.longestSubstringWithKUniqueCharacter("hq", 2);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Smallest Substring With K Unique Characters (SLIDING WINDOW)");
//        //https://www.codingninjas.com/codestudio/problems/smallest-subarray-with-k-distinct-elements_630523?leftPanelTab=0
//        obj.smallestSubstringWithKUniqueCharacter("aabacbebebe", 3);
//        obj.smallestSubstringWithKUniqueCharacter("aabcaccbeb", 3);
//        obj.smallestSubstringWithKUniqueCharacter("aaaa", 2);
//        obj.smallestSubstringWithKUniqueCharacter("hq", 2);
//        obj.smallestSubstringWithKUniqueCharacter("aabab", 3);
//        obj.smallestSubstringWithKUniqueCharacter("aaabbbccc", 3);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Smallest Subarray With K Distinct Elements (SLIDING WINDOW)");
//        //https://www.codingninjas.com/codestudio/problems/smallest-subarray-with-k-distinct-elements_630523?leftPanelTab=0
//        obj.smallestSubarrayWithKDistinctElements(new int[]{1, 1, 2, 1, 2}, 3);
//        obj.smallestSubarrayWithKDistinctElements(new int[]{4, 2, 2, 2, 3, 4, 4, 3}, 3);
//        obj.smallestSubarrayWithKDistinctElements(new int[]{1,1,1,2,2,2,3,3,3}, 3);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Smallest missing positive missing number");
//        //https://practice.geeksforgeeks.org/problems/smallest-positive-missing-number-1587115621/1#
//        System.out.println("Smallest missing positive number: " + obj.smallestMissingPositiveNumber(new int[]{1, 2, 3, 4, 5}));
//        System.out.println("Smallest missing positive number: " + obj.smallestMissingPositiveNumber(new int[]{0, -10, 1, 3, -20}));
//        System.out.println("Smallest missing positive number: " + obj.smallestMissingPositiveNumber(new int[]{}));
//        System.out.println("Smallest missing positive number: " + obj.smallestMissingPositiveNumber(new int[]{-2,2,7,1}));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Sort the matrix diaonally");
//        //https://leetcode.com/problems/sort-the-matrix-diagonally
//        int[][] mat = new int[][]{{3, 3, 1, 1}, {2, 2, 1, 2}, {1, 1, 1, 2}};
//        obj.sortTheMatrixDiagonally(mat);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Subarray sum equals K");
//        //https://leetcode.com/problems/subarray-sum-equals-k/
//        obj.subarraySumEqualsK(new int[]{1, 1, 1}, 2);
//        obj.subarraySumEqualsK(new int[]{1, 2, 3}, 3);
//        obj.subarraySumEqualsK(new int[]{1}, 0);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Subarray product less than K (SLIDING WINDOW)");
//        //https://leetcode.com/problems/subarray-product-less-than-k/
//        obj.subarrayProductLessThanK(new int[]{10, 5, 2, 6}, 100);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Linked list component");
//        //https://leetcode.com/problems/linked-list-components/
//        Node<Integer> head = new Node<>(0);
//        head.setNext(new Node<>(1));
//        head.getNext().setNext(new Node<>(2));
//        head.getNext().getNext().setNext(new Node<>(3));
//        head.getNext().getNext().getNext().setNext(new Node<>(4));
//        obj.linkedListComponent(head, new int[]{0,3,1,4});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Linked list partition list");
//        //https://leetcode.com/problems/partition-list
//        Node<Integer> head = new Node<>(1);
//        head.setNext(new Node<>(4));
//        head.getNext().setNext(new Node<>(3));
//        head.getNext().getNext().setNext(new Node<>(2));
//        head.getNext().getNext().getNext().setNext(new Node<>(5));
//        head.getNext().getNext().getNext().getNext().setNext(new Node<>(2));
//        obj.partitionList(head, 3);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Max sum in any path of the tree");
//        //https://leetcode.com/problems/binary-tree-maximum-path-sum/
//        TreeNode<Integer> root = new TreeNode<>(1);
//        root.setLeft(new TreeNode<>(2));
//        root.setRight(new TreeNode<>(3));
//        obj.maxSumInAnyPathOfTree(root);
//        root = new TreeNode<>(42);
//        root.setLeft(new TreeNode<>(-2));
//        root.setRight(new TreeNode<>(2));
//        obj.maxSumInAnyPathOfTree(root);
//        root = new TreeNode<>(-10);
//        root.setLeft(new TreeNode<>(9));
//        root.setRight(new TreeNode<>(20));
//        root.getRight().setLeft(new TreeNode<>(15));
//        root.getRight().setRight(new TreeNode<>(7));
//        obj.maxSumInAnyPathOfTree(root); //MAX SUM accross path: 15<->20<->7
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("longest length of edge between tree nodes having same values");
//        //https://leetcode.com/problems/longest-univalue-path/
//        TreeNode<Integer> root = new TreeNode<>(1);
//        root.setLeft(new TreeNode<>(4));
//        root.getLeft().setLeft(new TreeNode<>(4));
//        root.getLeft().setRight(new TreeNode<>(4));
//        root.setRight(new TreeNode<>(5));
//        root.getRight().setRight(new TreeNode<>(5));
//        obj.longestEdgeLengthBetweenTreeNodesWithSameValue(root); //EDGE b/w 4<->4<->4
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Global and local inversions");
//        //https://leetcode.com/problems/global-and-local-inversions/
//        System.out.println("Are counts for global and local inversion are equal: "
//                + obj.globalAndLocalInversionCountAreEqual(new int[]{1, 0, 2}));
//        System.out.println("Are counts for global and local inversion are equal: "
//                + obj.globalAndLocalInversionCountAreEqual(new int[]{1, 2, 0}));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Pascal triangle");
//        //https://leetcode.com/problems/pascals-triangle/
//        obj.printPascalTriangle_SimpleAddition(6);
//        obj.printPascalTriangle_BinomialCoeff(6);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Largest number from the given set of number/ Smallest Value of the Rearranged Number");
//        //https://leetcode.com/problems/largest-number/
//        obj.largestNumberFromSetOfNumbers(new String[]{"3", "30", "34", "5", "9"});
//        obj.largestNumberFromSetOfNumbers(new String[]{"54", "546", "548", "60"});
//        obj.largestNumberFromSetOfNumbers(new String[]{"0", "0"});
//        //https://leetcode.com/problems/smallest-value-of-the-rearranged-number
//        System.out.println("Smallest number from the given num: " + obj.smallestNumber(310));
//        System.out.println("Smallest number from the given num: " + obj.smallestNumber(-7650));
//        System.out.println("Smallest number from the given num: " + obj.smallestNumber(-12345));
//        System.out.println("Smallest number from the given num: " + obj.smallestNumber(54321));
//        System.out.println("Smallest number from the given num: " + obj.smallestNumber(54321000));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("String compression");
//        //https://leetcode.com/problems/string-compression/
//        obj.stringCompression(new char[]{'a', 'a', 'b', 'b', 'c', 'c'});
//        obj.stringCompression(new char[]{'a'});
//        obj.stringCompression(new char[]{'a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Rotate linked list K times");
//        //https://leetcode.com/problems/rotate-list/
//        Node<Integer> head = new Node<>(0);
//        head.setNext(new Node<>(1));
//        head.getNext().setNext(new Node<>(2));
//        obj.rotateLinkedListKTimes(head, 4);
//        head = new Node<>(1);
//        head.setNext(new Node<>(2));
//        head.getNext().setNext(new Node<>(3));
//        head.getNext().getNext().setNext(new Node<>(4));
//        head.getNext().getNext().getNext().setNext(new Node<>(5));
//        obj.rotateLinkedListKTimes(head, 2);
//        //same linked list approach 2
//        head = new Node<>(0);
//        head.setNext(new Node<>(1));
//        head.getNext().setNext(new Node<>(2));
//        obj.rotateLinkedListKTimes2(head, 4);
//        head = new Node<>(1);
//        head.setNext(new Node<>(2));
//        head.getNext().setNext(new Node<>(3));
//        head.getNext().getNext().setNext(new Node<>(4));
//        head.getNext().getNext().getNext().setNext(new Node<>(5));
//        obj.rotateLinkedListKTimes2(head, 2);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Sort the linked list in relative order of the given arr");
//        //https://leetcode.com/problems/relative-sort-array/
//        //https://www.geeksforgeeks.org/sort-linked-list-order-elements-appearing-array/
//        Node<Integer> head = new Node<>(3);
//        head.setNext(new Node<>(2));
//        head.getNext().setNext(new Node<>(5));
//        head.getNext().getNext().setNext(new Node<>(8));
//        head.getNext().getNext().getNext().setNext(new Node<>(5));
//        head.getNext().getNext().getNext().getNext().setNext(new Node<>(2));
//        head.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(1));
//        obj.sortLinkedListInRelativeOrderOfArr(head, new int[]{5, 1, 3, 2, 8});
//        head = new Node<>(3);
//        head.setNext(new Node<>(2));
//        head.getNext().setNext(new Node<>(5));
//        head.getNext().getNext().setNext(new Node<>(8));
//        head.getNext().getNext().getNext().setNext(new Node<>(5));
//        head.getNext().getNext().getNext().getNext().setNext(new Node<>(2));
//        head.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(1));
//        head.getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(19)); //EXTRA NOT IN arr[]
//        head.getNext().getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(7)); //EXTRA NOT IN arr[]
//        obj.sortLinkedListInRelativeOrderOfArr(head, new int[]{5, 1, 3, 2, 8});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Count nodes in complete binary tree");
//        //https://leetcode.com/problems/count-complete-tree-nodes/
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.getLeft().setLeft(new TreeNode<>(4));
//        root1.getLeft().setRight(new TreeNode<>(5));
//        root1.setRight(new TreeNode<>(3));
//        root1.getRight().setLeft(new TreeNode<>(6));
//        root1.getRight().setRight(new TreeNode<>(7));
//        System.out.println("Nnumber of nodes in complete binary tree: " + obj.countNodesInCompleteBinaryTree(root1));
//        root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.getLeft().setLeft(new TreeNode<>(4));
//        root1.getLeft().setRight(new TreeNode<>(5));
//        root1.setRight(new TreeNode<>(3));
//        root1.getRight().setLeft(new TreeNode<>(6));
//        System.out.println("Nnumber of nodes in complete binary tree: " + obj.countNodesInCompleteBinaryTree(root1));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("All source to destination path with weight sum in N-Ary tree ");
//        //https://www.geeksforgeeks.org/amazon-interview-experience-set-424-sde-2/
//        List<List<VertexWithWeight>> adjList = Arrays.asList(
//                /*0*/Arrays.asList(new VertexWithWeight(1, 10),
//                        new VertexWithWeight(2, 50),
//                        new VertexWithWeight(3, 20)),
//                /*1*/ Arrays.asList(new VertexWithWeight(0, 10),
//                        new VertexWithWeight(4, 30),
//                        new VertexWithWeight(5, 40)),
//                /*2*/ Arrays.asList(new VertexWithWeight(0, 50)),
//                /*3*/ Arrays.asList(new VertexWithWeight(0, 20)),
//                /*4*/ Arrays.asList(new VertexWithWeight(1, 30)),
//                /*5*/ Arrays.asList(new VertexWithWeight(1, 40))
//        );
//        obj.vertexWithWeightAllSourceToDestinationPath(adjList);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Count good nodes in binary tree");
//        //https://leetcode.com/problems/count-good-nodes-in-binary-tree/
//        TreeNode<Integer> root1 = new TreeNode<>(3);
//        root1.setLeft(new TreeNode<>(1));
//        root1.getLeft().setLeft(new TreeNode<>(3));
//        root1.setRight(new TreeNode<>(4));
//        root1.getRight().setLeft(new TreeNode<>(1));
//        root1.getRight().setRight(new TreeNode<>(5));
//        System.out.println("Good nodes counts in tree approach 1: "+obj.countGoodNodesInBinaryTree_1(root1, Integer.MIN_VALUE));
//        obj.countGoodNodesInBinaryTree_2(root1); //EASIER APPROACH
//        root1 = new TreeNode<>(3);
//        root1.setLeft(new TreeNode<>(3));
//        root1.getLeft().setLeft(new TreeNode<>(4));
//        root1.getLeft().setRight(new TreeNode<>(2));
//        System.out.println("Good nodes counts in tree approach 1: "+obj.countGoodNodesInBinaryTree_1(root1, Integer.MIN_VALUE));
//        obj.countGoodNodesInBinaryTree_2(root1); //EASIER APPROACH
//        root1 = new TreeNode<>(1);
//        System.out.println("Good nodes counts in tree approach 1: "+obj.countGoodNodesInBinaryTree_1(root1, Integer.MIN_VALUE));
//        obj.countGoodNodesInBinaryTree_2(root1); //EASIER APPROACH
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Min distance between 2 nodes in given binary tree");
//        //https://www.geeksforgeeks.org/find-distance-between-two-nodes-of-a-binary-tree/
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.setRight(new TreeNode<>(3));
//        obj.minDistanceBetweenGivenTwoNodesInBinaryTree(root1, 2, 3);
//        obj.minDistanceBetweenGivenTwoNodesInBinaryTree(root1, 3, 3);
//        root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.getLeft().setLeft(new TreeNode<>(4));
//        root1.getLeft().setRight(new TreeNode<>(5));
//        root1.setRight(new TreeNode<>(3));
//        root1.getRight().setLeft(new TreeNode<>(6));
//        root1.getRight().getLeft().setRight(new TreeNode<>(8));
//        root1.getRight().setRight(new TreeNode<>(7));
//        obj.minDistanceBetweenGivenTwoNodesInBinaryTree(root1, 4, 5);
//        obj.minDistanceBetweenGivenTwoNodesInBinaryTree(root1, 4, 6);
//        obj.minDistanceBetweenGivenTwoNodesInBinaryTree(root1, 3, 4);
//        obj.minDistanceBetweenGivenTwoNodesInBinaryTree(root1, 2, 4);
//        obj.minDistanceBetweenGivenTwoNodesInBinaryTree(root1, 8, 5);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Celebrity problem");
//        //https://www.geeksforgeeks.org/the-celebrity-problem/
//        System.out.println("Id of celebrity person: " + obj.findCelebrityInNPepole(4));
//        System.out.println("Id of celebrity person: " + obj.findCelebrityInNPepole_Optimized(4));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Max product in splitted binary tree");
//        //https://leetcode.com/problems/maximum-product-of-splitted-binary-tree/
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.getLeft().setLeft(new TreeNode<>(4));
//        root1.getLeft().setRight(new TreeNode<>(5));
//        root1.setRight(new TreeNode<>(3));
//        root1.getRight().setLeft(new TreeNode<>(6));
//        obj.maxProductIfBinaryTreeIsSplitIntoTwo(root1);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Longest consecutive sequenec");
//        //https://leetcode.com/problems/longest-consecutive-sequence/
//        //https://practice.geeksforgeeks.org/problems/longest-consecutive-subsequence2449/1
//        System.out.println("Longest consecutive seq: "+obj.longestConsecutiveSequence(new int[]{2,6,1,9,4,5,3}));
//        System.out.println("Longest consecutive seq: "+obj.longestConsecutiveSequence(new int[]{2,2,4,5,1,1,1,3,4,5,6,7,8,9,9}));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Max difference of indexes");
//        //https://www.geeksforgeeks.org/given-an-array-arr-find-the-maximum-j-i-such-that-arrj-arri/
//        obj.maxDifferenceOfIndexes(new int[]{34, 8, 10, 3, 2, 80, 30, 33, 1});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Maximum difference between node and its ancestor");
//        //https://www.geeksforgeeks.org/maximum-difference-between-node-and-its-ancestor-in-binary-tree/
//        TreeNode<Integer> root1 = new TreeNode<>(7);
//        root1.setLeft(new TreeNode<>(2));
//        root1.setRight(new TreeNode<>(3));
//        root1.getRight().setRight(new TreeNode<>(1)); //Node = 1 and ancestor = 7 max diff becomes 6
//        obj.maximumDifferenceBetweenNodeAndItsAncestor(root1);
//        root1 = new TreeNode<>(1);
//        root1.setRight(new TreeNode<>(34));
//        obj.maximumDifferenceBetweenNodeAndItsAncestor(root1);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Delete nodes from binary search tree");
//        //https://leetcode.com/problems/delete-node-in-a-bst/
//        TreeNode<Integer> root1 = new TreeNode<>(3);
//        root1.setLeft(new TreeNode<>(0));
//        root1.getLeft().setRight(new TreeNode<>(2));
//        root1.getLeft().getRight().setLeft(new TreeNode<>(1));
//        root1.setRight(new TreeNode<>(4));
//        new BinaryTree<Integer>(obj.deleteTreeNodeFromBinarySearchTree(root1, 3)).treeBFS(); //DELETE ROOT OF TREE
//        new BinaryTree<Integer>(obj.deleteTreeNodeFromBinarySearchTree(root1, 1)).treeBFS(); //DELETE LEAF
//        new BinaryTree<Integer>(obj.deleteTreeNodeFromBinarySearchTree(root1, 0)).treeBFS(); //DELETE ROOT HAS ONE CHILD (RIGHT)
//        new BinaryTree<Integer>(obj.deleteTreeNodeFromBinarySearchTree(root1, 2)).treeBFS(); //DELETE ROOT HAS ONE CHILD (LEFT)
//        System.out.println();
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Delete nodes from binary search tree that are not in range");
//        //https://leetcode.com/problems/trim-a-binary-search-tree/
//        TreeNode<Integer> root1 = new TreeNode<>(3);
//        root1.setLeft(new TreeNode<>(0));
//        root1.getLeft().setRight(new TreeNode<>(2));
//        root1.getLeft().getRight().setLeft(new TreeNode<>(1));
//        root1.setRight(new TreeNode<>(4));
//        obj.deleteTreeNodeFromBinarySearchTreeNotInRange(root1, 1, 3);
//        root1 = new TreeNode<>(3);
//        root1.setLeft(new TreeNode<>(1));
//        root1.getLeft().setLeft(new TreeNode<>(0));
//        root1.getLeft().setRight(new TreeNode<>(2));
//        root1.setRight(new TreeNode<>(4));
//        obj.deleteTreeNodeFromBinarySearchTreeNotInRange(root1, 2, 3);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Rearrange array elements");
//        //https://www.geeksforgeeks.org/rearrange-given-array-place/
//        obj.rearrangeArrayElements(new int[]{4, 0, 2, 1, 3});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Minimum range that contains atleast one element from K sorted List");
//        //https://leetcode.com/problems/smallest-range-covering-elements-from-k-lists/
//        //https://www.geeksforgeeks.org/find-smallest-range-containing-elements-from-k-lists/
//        obj.minimumRangeContainingAtleastOneElementFromKSortedList(new int[][]{
//            {1, 3, 5, 7, 9},
//            {0, 2, 4, 6, 8},
//            {2, 3, 5, 7, 11}
//        });
//        obj.minimumRangeContainingAtleastOneElementFromKSortedList(new int[][]{
//            {1, 2, 3, 4},
//            {5, 6, 7, 8},
//            {9, 10, 11, 12}
//        });
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Tuples with same product");
//        //https://leetcode.com/problems/tuple-with-same-product/
//        obj.tupleWithSameProduct(new int[]{2, 3, 4, 6});
//        obj.tupleWithSameProduct(new int[]{2, 3, 5, 7});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Check if two nodes are cousin of each other");
//        //https://leetcode.com/problems/cousins-in-binary-tree/
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.getLeft().setLeft(new TreeNode<>(4));
//        root1.setRight(new TreeNode<>(3));
//        obj.checkIfTwoTreeNodesAreCousin(root1, 4, 3);
//        obj.checkIfTwoTreeNodesAreCousin(root1, 2, 3);
//        root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.getLeft().setRight(new TreeNode<>(4));
//        root1.setRight(new TreeNode<>(3));
//        root1.getRight().setRight(new TreeNode<>(5));
//        obj.checkIfTwoTreeNodesAreCousin(root1, 4, 5);
//        obj.checkIfTwoTreeNodesAreCousin(root1, 2, 5);
//        obj.checkIfTwoTreeNodesAreCousin(root1, 2, 6); //6 don't exist
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Closest Strings distance");
//        //https://practice.geeksforgeeks.org/problems/closest-strings0611/1#
//        System.out.println("Closest string distance: " + obj.closestStringDistance(
//                Arrays.asList("geeks", "for", "geeks", "contribute", "practice"), "geeks", "practice"));
//        System.out.println("Closest string distance: " + obj.closestStringDistance(
//                Arrays.asList("geeks", "for", "geeks", "contribute", "practice"), "geeks", "geeks"));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Connect all nodes at same level in a tree by the random pointer (Recursive/ Iterative)");
//        //https://leetcode.com/problems/populating-next-right-pointers-in-each-node/
//        //https://practice.geeksforgeeks.org/problems/connect-nodes-at-same-level/1#
//        TreeNode<Integer> root1 = new TreeNode<>(10);
//        root1.setLeft(new TreeNode<>(3));
//        root1.getLeft().setLeft(new TreeNode<>(4));
//        root1.getLeft().setRight(new TreeNode<>(1));
//        root1.setRight(new TreeNode<>(5));
//        root1.getRight().setRight(new TreeNode<>(2));
//        obj.connectTreeNodesAtSameLevel_Recursive(root1);
//        obj.connectTreeNodesAtSameLevel_Iterative(root1);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("K - diff pairs in array");
//        //https://leetcode.com/problems/k-diff-pairs-in-an-array/
//        //https://leetcode.com/problems/count-number-of-pairs-with-absolute-difference-k/
//        obj.kDiffPairsInArray(new int[]{3, 1, 4, 1, 5}, 2);
//        obj.kDiffPairsInArray(new int[]{-1, -2, -3}, 1);
//        obj.kDiffPairsInArray(new int[]{1, 3, 1, 5, 4}, 0);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Least number of unique integers left after K removals");
//        //https://leetcode.com/problems/least-number-of-unique-integers-after-k-removals/
//        obj.leastNumberOfUniqueIntegersLeftAfterKRemoval(new int[]{5, 5, 4}, 1);
//        obj.leastNumberOfUniqueIntegersLeftAfterKRemoval(new int[]{4, 3, 1, 1, 3, 3, 2}, 3);
//        obj.leastNumberOfUniqueIntegersLeftAfterKRemoval(new int[]{4, 3, 1, 1, 3, 3, 2}, 7);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Min number of coines required to make change in unlimited supply of coins(DP_PROBLEM)");
//        //https://practice.geeksforgeeks.org/problems/number-of-coins1824/1
//        obj.minCoinsRequiredToMakeChangeInUnlimitedSupplyOfCoins_DP_Memoization(new int[]{25, 10, 5}, 30);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Long pressed names");
//        //https://leetcode.com/problems/long-pressed-name/
//        System.out.println("Both actual name and typed name supposed to equal: " + obj.longPressedNames("alex", "aaleex"));
//        System.out.println("Both actual name and typed name supposed to equal: " + obj.longPressedNames("laex", "aaleex"));
//        System.out.println("Both actual name and typed name supposed to equal: " + obj.longPressedNames("saeed", "ssaaedd"));
//        System.out.println("Both actual name and typed name supposed to equal: " + obj.longPressedNames("leelee", "lgeelege"));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Binary search tree to greater sum tree");
//        //https://leetcode.com/problems/convert-bst-to-greater-tree/
//        TreeNode<Integer> root1 = new TreeNode<>(4);
//        root1.setLeft(new TreeNode<>(1));
//        root1.getLeft().setLeft(new TreeNode<>(0));
//        root1.getLeft().setRight(new TreeNode<>(2));
//        root1.getLeft().getRight().setRight(new TreeNode<>(3));
//        root1.setRight(new TreeNode<>(6));
//        root1.getRight().setLeft(new TreeNode<>(5));
//        root1.getRight().setRight(new TreeNode<>(7));
//        root1.getRight().getRight().setRight(new TreeNode<>(8));
//        obj.binarySearchTreeToGreaterSumTree(root1);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Print all interleavings of given two strings");
//        //https://www.geeksforgeeks.org/print-all-interleavings-of-given-two-strings/
//        obj.interleavingOfTwoStrings("AB", "CD");
//        obj.interleavingOfTwoStrings("ABC", "DEF");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Check if String C is interleaving of String A & B");
//        //https://leetcode.com/problems/interleaving-string/
//        System.out.println("Third string is interleaving of other two: "
//                + obj.checkIfStringCIsInterleavingOfStringAAndB_DP_Memoization("AB", "CD", "ACBG"));
//        System.out.println("Third string is interleaving of other two: "
//                + obj.checkIfStringCIsInterleavingOfStringAAndB_DP_Memoization("AB", "CD", "ACDB"));
//        System.out.println("Third string is interleaving of other two: "
//                + obj.checkIfStringCIsInterleavingOfStringAAndB_DP_Memoization("AB", "CD", "AC"));
//        System.out.println("Third string is interleaving of other two: "
//                + obj.checkIfStringCIsInterleavingOfStringAAndB_DP_Memoization("aabcc", "dbbca", "aadbbcbcac"));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Print all permutation of distinct char in string");
//        //https://leetcode.com/problems/permutations/
//        //https://leetcode.com/problems/permutations-ii/
//        obj.printAllPermutationOfDistinctCharInString("ABC");
//        obj.printAllPermutationOfDistinctCharInString("ABCD");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Print all permutation of a distinct integer");
//        //https://leetcode.com/problems/permutations/
//        //https://leetcode.com/problems/permutations-ii/
//        obj.printAllPermutationOfDistinctIntegerArray(new int[]{1, 2, 3});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Max distance covered by robot walking simulation");
//        //https://leetcode.com/problems/walking-robot-simulation/
//        obj.maximumDistanceCoveredInRobotWalkingSimulation(new int[]{4, -1, 4, -2, 4},
//                new int[][]{{2, 4}});
//        obj.maximumDistanceCoveredInRobotWalkingSimulation(new int[]{4, -1, 3},
//                new int[][]{});
//        obj.maximumDistanceCoveredInRobotWalkingSimulation(new int[]{2, -1, 2 - 1, 2},
//                new int[][]{});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Longest substring having all vowels in order");
//        //https://leetcode.com/problems/longest-substring-of-all-vowels-in-order/
//        obj.longestSubstringHavingAllVowelsInOrder("aeiaaioaaaaeiiiiouuuooaauuaeiu");
//        obj.longestSubstringHavingAllVowelsInOrder("aeeeiiiioooauuuaeiou");
//        obj.longestSubstringHavingAllVowelsInOrder("a");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Evaluate arithematic expression(Basic calculator)");
//        //https://leetcode.com/problems/basic-calculator-ii/
//        obj.arithematicExpressionEvaluationBasicCalculator(" 3/2 ");
//        obj.arithematicExpressionEvaluationBasicCalculator(" 35 + 5 / 4 ");
//        obj.arithematicExpressionEvaluationBasicCalculator(" 3 + 5 / 4*2");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Single threaded CPU Greedy");
//        //https://leetcode.com/problems/single-threaded-cpu/
//        obj.singleThreadedCPU_Greedy(new int[][]{
//            {1, 2}, {2, 4}, {3, 2}, {4, 1}
//        });
//        obj.singleThreadedCPU_Greedy(new int[][]{
//            {7, 10}, {7, 12}, {7, 5}, {7, 4}, {7, 2}
//        });
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Replace bracket pattern with given value in list");
//        //https://leetcode.com/problems/evaluate-the-bracket-pairs-of-a-string/
//        List<List<String>> keyReplacement = new ArrayList<>();
//        keyReplacement.add(Arrays.asList("name", "bob"));
//        keyReplacement.add(Arrays.asList("age", "two"));
//        obj.evaluateBracketPatternAndReplaceWithGivenWord("(name)is(age)yearsold",
//                keyReplacement);
//        keyReplacement = new ArrayList<>();
//        keyReplacement.add(Arrays.asList("a", "b"));
//        obj.evaluateBracketPatternAndReplaceWithGivenWord("hi(name)",
//                keyReplacement);
//        keyReplacement = new ArrayList<>();
//        keyReplacement.add(Arrays.asList("a", "yes"));
//        obj.evaluateBracketPatternAndReplaceWithGivenWord("(a)(a)(a)aaa",
//                keyReplacement);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("All phone digits letter combinations");
//        //https://leetcode.com/problems/letter-combinations-of-a-phone-number
//        obj.allPhoneDigitLetterCombinations("2");
//        obj.allPhoneDigitLetterCombinations("23");
//        obj.allPhoneDigitLetterCombinations("7979");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Combination sum 1 and 3");
//        //https://leetcode.com/problems/combination-sum
//        //https://leetcode.com/problems/combinations/
//        obj.combinationSum_1(new int[]{1}, 2);
//        obj.combinationSum_1(new int[]{2}, 1); //No Combinations Possible
//        obj.combinationSum_1(new int[]{2, 3, 5}, 8);
//        //https://leetcode.com/problems/combination-sum-iii/
//        obj.combinationSum_3(3, 7);
//        obj.combinationSum_3(3, 9);
//        obj.combinationSum_3(4, 1);
//        obj.combinationSum_3(9, 60);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Combination sum-2");
//        //https://leetcode.com/problems/combination-sum-ii/
//        obj.combinationSum_2(new int[]{10, 1, 2, 7, 6, 1, 5}, 8);
//        obj.combinationSum_2(new int[]{2, 5, 2, 1, 2}, 5);
//        obj.combinationSum_2(new int[]{1}, 5);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Shortest unsorted contigous subarray");
//        //https://leetcode.com/problems/shortest-unsorted-continuous-subarray
//        System.out.println("Length of shortest unsorted contigous subarray "
//                + obj.shortestUnsortedContigousSubarray(new int[]{2, 6, 4, 8, 10, 9, 15}));
//        System.out.println("Length of shortest unsorted contigous subarray "
//                + obj.shortestUnsortedContigousSubarray(new int[]{1, 2, 3, 4})); //ALREADY SORTED
//        System.out.println("Length of shortest unsorted contigous subarray "
//                + obj.shortestUnsortedContigousSubarray(new int[]{4})); //ALREADY SORTED
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Minimum operations to make array increasing");
//        //https://leetcode.com/problems/minimum-operations-to-make-the-array-increasing/
//        obj.minimumOperationsToMakeArrayStrictlyIncr(new int[]{1, 1, 1});
//        obj.minimumOperationsToMakeArrayStrictlyIncr(new int[]{4});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Broken calculator (make X equal to Y with decreament Or Multiplication oprn on X)");
//        //https://leetcode.com/problems/broken-calculator/
//        obj.brokenCalculatorMakeXEqualToY(2, 3);
//        obj.brokenCalculatorMakeXEqualToY(3, 10);
//        obj.brokenCalculatorMakeXEqualToY(1024, 1);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Minimum Number of Vertices to Reach All Nodes");
//        //https://leetcode.com/problems/minimum-number-of-vertices-to-reach-all-nodes/
//        obj.vertexThroughAllOtherVertexCanBeReachedInDirectedAcyclicGraph_Graph(6, new int[][]{
//            {0, 1}, {0, 2}, {2, 5}, {3, 4}, {4, 2}
//        });
//        obj.vertexThroughAllOtherVertexCanBeReachedInDirectedAcyclicGraph_Graph(5, new int[][]{
//            {0, 1}, {2, 1}, {3, 1}, {4, 1}
//        });
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Check if atmost one char swap make strings equal");
//        //https://leetcode.com/problems/check-if-one-string-swap-can-make-strings-equal/
//        System.out.println("String are equal with atmost one char swap: "
//                + obj.checkIfOneCharSwapMakeStringEqual("bank", "kanb"));
//        System.out.println("String are equal with atmost one char swap: "
//                + obj.checkIfOneCharSwapMakeStringEqual("kelb", "kelb"));
//        System.out.println("String are equal with atmost one char swap: "
//                + obj.checkIfOneCharSwapMakeStringEqual("abcd", "dcba"));
//        System.out.println("String are equal with atmost one char swap: "
//                + obj.checkIfOneCharSwapMakeStringEqual("attack", "defend"));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Largest Substring Between Two Equal Characters");
//        //https://leetcode.com/problems/largest-substring-between-two-equal-characters/
//        obj.largestSubstringBetweenTwoSameChar("aa");
//        obj.largestSubstringBetweenTwoSameChar("cbzxy");
//        obj.largestSubstringBetweenTwoSameChar("cabbac");
//        obj.largestSubstringBetweenTwoSameChar("abca");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Smallest subtree with all the deepest nodes in a tree");
//        //https://leetcode.com/problems/lowest-common-ancestor-of-deepest-leaves/
//        //https://leetcode.com/problems/smallest-subtree-with-all-the-deepest-nodes/
//        TreeNode<Integer> root1 = new TreeNode<>(3);
//        root1.setLeft(new TreeNode<>(5));
//        root1.getLeft().setLeft(new TreeNode<>(6));
//        root1.getLeft().setRight(new TreeNode<>(2));
//        root1.getLeft().getRight().setLeft(new TreeNode<>(7));
//        root1.getLeft().getRight().setRight(new TreeNode<>(4));
//        root1.setRight(new TreeNode<>(1));
//        root1.getRight().setLeft(new TreeNode<>(0));
//        root1.getRight().setRight(new TreeNode<>(8));
//        obj.subtreeWithAllDeepestNodes(root1); //DEEP LEAF = 7,4 subtree = [2,7,4]
//        root1 = new TreeNode<>(0);
//        root1.setLeft(new TreeNode<>(1));
//        root1.getLeft().setRight(new TreeNode<>(2));
//        root1.setRight(new TreeNode<>(3));
//        obj.subtreeWithAllDeepestNodes(root1); //DEEP LEAF = 2 subtree = [2]
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Determine if two strings are close");
//        //https://leetcode.com/problems/determine-if-two-strings-are-close/
//        System.out.println(obj.determineIfTwoStringCanBeMadeClose("a", "aa"));
//        System.out.println(obj.determineIfTwoStringCanBeMadeClose("abc", "cba"));
//        System.out.println(obj.determineIfTwoStringCanBeMadeClose("cabbba", "abbccc"));
//        System.out.println(obj.determineIfTwoStringCanBeMadeClose("cabbba", "aabbss"));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Minimum char required to make string t anagram of string s");
//        //https://leetcode.com/problems/minimum-number-of-steps-to-make-two-strings-anagram/
//        //https://leetcode.com/problems/minimum-number-of-steps-to-make-two-strings-anagram-ii/
//        obj.minCharacterRequiredToMakeStringTAnagramOfS("bab", "aba");
//        obj.minCharacterRequiredToMakeStringTAnagramOfS("leetcode", "practice");
//        obj.minCharacterRequiredToMakeStringTAnagramOfS("xxyyzz", "xxyyzz");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Minimum char removed to make string t & s anagram");
//        //https://practice.geeksforgeeks.org/problems/anagram-of-string
//        //https://www.geeksforgeeks.org/remove-minimum-number-characters-two-strings-become-anagram/
//        obj.minCharacterRemovedToMakeStringTAndSAnagrams("bcadeh", "hea"); //remove b,c,d from S
//        obj.minCharacterRemovedToMakeStringTAndSAnagrams("bcadeh", "heaz"); //remove b,c,d from S, z from T
//        obj.minCharacterRemovedToMakeStringTAndSAnagrams("cddgk", "gcd"); //remove k,d from S
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Check if graph is bipartite");
//        //https://leetcode.com/problems/is-graph-bipartite/
//        System.out.println("Graph is bipartite: "
//                + obj.checkIfGraphIsBipartite_Graph(new int[][]{{1, 3}, {0, 2}, {1, 3}, {0, 2}}));
//        System.out.println("Graph is bipartite: "
//                + obj.checkIfGraphIsBipartite_Graph(new int[][]{{1, 2, 3}, {0, 2}, {0, 1, 3}, {0, 2}}));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Pseudo-Palindromic Paths in a Binary Tree");
//        //https://leetcode.com/problems/pseudo-palindromic-paths-in-a-binary-tree/
//        TreeNode<Integer> root1 = new TreeNode<>(2);
//        root1.setLeft(new TreeNode<>(3));
//        root1.getLeft().setLeft(new TreeNode<>(3));
//        root1.getLeft().setRight(new TreeNode<>(1));
//        root1.setRight(new TreeNode<>(1));
//        root1.getRight().setRight(new TreeNode<>(1));
//        obj.pseudoPallindromicPathInBinaryTree(root1);
//        root1 = new TreeNode<>(3);
//        root1.setLeft(new TreeNode<>(3));
//        root1.getLeft().setLeft(new TreeNode<>(3));
//        root1.setRight(new TreeNode<>(1));
//        obj.pseudoPallindromicPathInBinaryTree(root1);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Split linked list in K parts");
//        //https://leetcode.com/problems/split-linked-list-in-parts/
//        Node<Integer> head = new Node<>(1);
//        head.setNext(new Node<>(2));
//        head.getNext().setNext(new Node<>(3));
//        obj.splitLinkedListInKParts(head, 5);
//        head = new Node<>(1);
//        head.setNext(new Node<>(2));
//        head.getNext().setNext(new Node<>(3));
//        head.getNext().getNext().setNext(new Node<>(4));
//        head.getNext().getNext().getNext().setNext(new Node<>(5));
//        head.getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        head.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(7));
//        head.getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(8));
//        head.getNext().getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(9));
//        head.getNext().getNext().getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(10));
//        obj.splitLinkedListInKParts(head, 3);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Remove given value from linked list");
//        //https://leetcode.com/problems/remove-linked-list-elements
//        Node<Integer> head = new Node<>(6);
//        head.setNext(new Node<>(1));
//        head.getNext().setNext(new Node<>(2));
//        head.getNext().getNext().setNext(new Node<>(3));
//        head.getNext().getNext().getNext().setNext(new Node<>(6));
//        head.getNext().getNext().getNext().getNext().setNext(new Node<>(4));
//        head.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(5));
//        head.getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        obj.trimLinkedListAndRemoveAllOccurencesOfGivenVal(head, 6);
//        head = new Node<>(6);
//        head.setNext(new Node<>(6));
//        head.getNext().setNext(new Node<>(6));
//        head.getNext().getNext().setNext(new Node<>(6));
//        head.getNext().getNext().getNext().setNext(new Node<>(6));
//        head.getNext().getNext().getNext().getNext().setNext(new Node<>(6));
//        obj.trimLinkedListAndRemoveAllOccurencesOfGivenVal(head, 6);
//        head = new Node<>(6);
//        obj.trimLinkedListAndRemoveAllOccurencesOfGivenVal(head, 6);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Count number of jumps to reach the end");
//        //https://www.geeksforgeeks.org/minimum-number-jumps-reach-endset-2on-solution/
//        System.out.println("Number of jumps t reach end: "
//                +obj.countNumberOfJumpsToReachTheEnd(new int[]{1, 3, 5, 8, 9, 2, 6, 7, 6, 8, 9}));
//        System.out.println("Number of jumps t reach end: "
//                +obj.countNumberOfJumpsToReachTheEnd(new int[]{1, 0, 5, 8, 9}));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Minimum difference of sum of two partition of the array DP Problem");
//        //https://practice.geeksforgeeks.org/problems/minimum-sum-partition3317/1#
//        obj.minimumDiffPartition_DP_Memoization(new int[]{1, 6, 11, 5});
//        obj.minimumDiffPartition_DP_Memoization(new int[]{1, 4});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Longest subarray of consecutive ones after deleting one element in the binary array");
//        //https://leetcode.com/problems/longest-subarray-of-1s-after-deleting-one-element/
//        obj.longestSubarrayOfConsecutiveOnesAfterDeletingOneElement(new int[]{0,0,0});
//        obj.longestSubarrayOfConsecutiveOnesAfterDeletingOneElement(new int[]{1,1,1,1});
//        obj.longestSubarrayOfConsecutiveOnesAfterDeletingOneElement(new int[]{1,1,0,1});
//        obj.longestSubarrayOfConsecutiveOnesAfterDeletingOneElement(new int[]{0,1,1,1,0,1,1,0,1});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Count subarray with odd sum");
//        //https://leetcode.com/problems/number-of-sub-arrays-with-odd-sum
//        obj.countSubarrayWithOddSum(new int[]{1, 3, 5});
//        obj.countSubarrayWithOddSum(new int[]{2, 4, 6});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Merge new interval in between");
//        //https://leetcode.com/problems/insert-interval/
//        obj.mergeNewInterval(new int[][]{{1, 3}, {6, 9}}, new int[]{2, 5});
//        obj.mergeNewInterval(new int[][]{{1, 2}, {3, 5}, {6, 7}, {8, 10}, {12, 16}}, new int[]{4, 8});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Count the number of turns in between two nodes of a tree");
//        //https://www.geeksforgeeks.org/number-turns-reach-one-node-binary-tree/
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.getLeft().setLeft(new TreeNode<>(4));
//        root1.getLeft().getLeft().setLeft(new TreeNode<>(8));
//        root1.getLeft().setRight(new TreeNode<>(5));
//        root1.setRight(new TreeNode<>(3));
//        root1.getRight().setLeft(new TreeNode<>(6));
//        root1.getRight().getLeft().setLeft(new TreeNode<>(9));
//        root1.getRight().getLeft().setRight(new TreeNode<>(10));
//        root1.getRight().setRight(new TreeNode<>(7));
//        System.out.println("Count turns between two nodes: "
//                +obj.countNumberOfTurnsBetweenTwoNodesOfTree(root1, 9, 10));
//        System.out.println("Count turns between two nodes: "
//                +obj.countNumberOfTurnsBetweenTwoNodesOfTree(root1, 5, 6));
//        System.out.println("Count turns between two nodes: "
//                +obj.countNumberOfTurnsBetweenTwoNodesOfTree(root1, 1, 4));
//        System.out.println("Count turns between two nodes: "
//                +obj.countNumberOfTurnsBetweenTwoNodesOfTree(root1, 5, 10));        
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Longest zig zag path in the binary tree");
//        //https://leetcode.com/problems/longest-zigzag-path-in-a-binary-tree/
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(2));
//        root1.getLeft().setRight(new TreeNode<>(4));
//        root1.getLeft().getRight().setLeft(new TreeNode<>(5));
//        root1.getLeft().getRight().getLeft().setRight(new TreeNode<>(7));
//        root1.getLeft().getRight().setRight(new TreeNode<>(6));
//        root1.setRight(new TreeNode<>(3));
//        obj.longestZigZagPathInTree(root1);
//        obj.longestZigZagPathInTree2(root1);
//        root1 = new TreeNode<>(1);
//        obj.longestZigZagPathInTree(root1);
//        obj.longestZigZagPathInTree2(root1);
//        root1 = new TreeNode<>(1);
//        root1.setRight(new TreeNode<>(2));
//        root1.getRight().setLeft(new TreeNode<>(3));
//        root1.getRight().setRight(new TreeNode<>(4));
//        root1.getRight().getRight().setLeft(new TreeNode<>(5));
//        root1.getRight().getRight().getLeft().setRight(new TreeNode<>(7));
//        root1.getRight().getRight().getLeft().getRight().setRight(new TreeNode<>(8));
//        root1.getRight().getRight().setRight(new TreeNode<>(6));
//        obj.longestZigZagPathInTree(root1);
//        obj.longestZigZagPathInTree2(root1);
//        root1 = new TreeNode<>(1); //SKEWED
//        root1.setLeft(new TreeNode<>(2));
//        root1.getLeft().setLeft(new TreeNode<>(3));
//        root1.getLeft().getLeft().setLeft(new TreeNode<>(4));
//        obj.longestZigZagPathInTree(root1);
//        obj.longestZigZagPathInTree2(root1);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Duplicate zero in-place");
//        //https://leetcode.com/problems/duplicate-zeros/
//        obj.duplicateZeroInArray(new int[]{1, 0, 2, 3, 0, 4, 5, 0});
//        obj.duplicateZeroInArray(new int[]{1, 0, 0, 3, 0, 4, 5, 0});
//        obj.duplicateZeroInArray(new int[]{1, 2, 3});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Array Nesting");
//        //https://leetcode.com/problems/array-nesting/
//        obj.arrayNesting(new int[]{5, 4, 0, 3, 1, 6, 2});
//        obj.arrayNesting(new int[]{0, 1, 2});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Escaping ghost");
//        //https://leetcode.com/problems/escape-the-ghosts/
//        System.out.println("User escaped all the ghosts to reach target: "
//                + obj.escapingGhost(new int[][]{{1, 0}, {0, 3}}, new int[]{0, 1}));
//        System.out.println("User escaped all the ghosts to reach target: "
//                + obj.escapingGhost(new int[][]{{1, 0}}, new int[]{2, 0})); //GHOST is in between user and target
//        System.out.println("User escaped all the ghosts to reach target: "
//                + obj.escapingGhost(new int[][]{{2, 0}}, new int[]{1, 0})); //GHOST and user reach target at same time
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Remove duplicate from sorted array 2 (elements can occur at most twice)");
//        //https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/
//        obj.removeDuplicateInSortedArray2WhereElementCanHaveAtMostTwiceOccur(new int[]{1, 1, 1, 2, 2, 3});
//        obj.removeDuplicateInSortedArray2WhereElementCanHaveAtMostTwiceOccur(new int[]{0, 0, 1, 1, 1, 1, 2, 3, 3});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Calculate power(x, n)");
//        //https://leetcode.com/problems/powx-n/
//        obj.nPowerOfX(5.0, 0); //5 ^ 0 = 1
//        obj.nPowerOfX(5.0, 1); //5 ^ 1 = 5
//        obj.nPowerOfX(2.0, 10);
//        obj.nPowerOfX(2.0, -3);
//        obj.nPowerOfX(-2.0, 2);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Convert postfix exprssion to infix expression");
//        //https://www.geeksforgeeks.org/postfix-to-infix/
//        obj.convertPostfixToInfixExpression("abc++");
//        obj.convertPostfixToInfixExpression("52*5+");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Convert infix exprssion to postfix expression");
//        //https://www.geeksforgeeks.org/stack-set-2-infix-to-postfix/
//        //https://leetcode.com/problems/basic-calculator/
//        obj.convertInfixToPostfixExpression("a*b+c/d");
//        obj.convertInfixToPostfixExpression("a+b*c/d-e");
//        obj.convertInfixToPostfixExpression("(a*b)+(c/d)-(e^f)");
//        obj.convertInfixToPostfixExpression("a+b*(c^d-e)^(f+g*h)-i");
//        obj.convertInfixToPostfixExpression("(1+(4+5+2)-3)+(6+8)");
        //......................................................................
//        Row: 307
//        System.out.println("Postfix expression evaluation");
//        //https://leetcode.com/problems/evaluate-reverse-polish-notation/
//        //https://leetcode.com/problems/basic-calculator/
//        obj.postfixExpressionEvaluation_SingleDigit("23+");
//        obj.postfixExpressionEvaluation_SingleDigit("231*+9-");
//        obj.postfixExpressionEvaluation_MultipleDigit("10 20 +");
//        obj.postfixExpressionEvaluation_MultipleDigit("100 200 * 10 /");
//        obj.postfixExpressionEvaluation_MultipleDigit("100 200 + 10 / 1000 +");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Course Schedule 2");
//        //https://leetcode.com/problems/course-schedule-ii/
//        obj.courseSchedule2_Graph(new int[][]{{1, 0}}, 2);
//        obj.courseSchedule2_Graph(new int[][]{{1, 0}, {0, 1}}, 2); //CYCLE
//        obj.courseSchedule2_Graph(new int[][]{{1, 0}, {2, 0}, {3, 1}, {3, 2}}, 4);
//        /*
//         //https://leetcode.com/discuss/interview-question/742238/Amazon-or-Student-Order
//         Given a result of a competition among all the students of a class, 
//         write a program to make students stand in a order such that every 
//         student must have lost to the student in his/her immediate left and 
//         won against the student to his/her immediate right.
//        
//         // student[][] = {{0,1},{1,2}} Here 0 lost to 1, 1 lost to 2
//         output of above:  2 1 0
//         0 loses to 1 so 1 is on left of 0
//         1 0
//         1 loses to 2 so 2 is on left of 1
//         2 1 0
//         */
//        obj.courseSchedule2_Graph(new int[][]{{0, 1}, {1, 2}}, 3); //3 is total student
//        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Boats to Save People/ Efficient janitor");
//        //https://leetcode.com/problems/boats-to-save-people/
//        //https://leetcode.com/discuss/interview-question/490066/Efficient-Janitor-Efficient-Vineet-(Hackerrank-OA)
//        //both the approaches works for boats-to-save-people & efficient janitor
//        obj.efficientJanitor_Greedy(new double[]{1.01, 1.01, 3.0, 2.7, 1.99, 2.3, 1.7});
//        obj.efficientJanitor_Greedy(new double[]{1.01, 1.991, 1.32, 1.4});
//        obj.efficientJanitor2_Greedy(new double[]{1.01, 1.01, 3.0, 2.7, 1.99, 2.3, 1.7});
//        obj.efficientJanitor2_Greedy(new double[]{1.01, 1.991, 1.32, 1.4});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Flip string to monotone increase");
//        //https://leetcode.com/problems/flip-string-to-monotone-increasing/
//        obj.flipStringToMonotoneIncrease("00110");
//        obj.flipStringToMonotoneIncrease("00110000");
//        obj.flipStringToMonotoneIncrease("00000011"); //ALREADY MONOTONE
//        obj.flipStringToMonotoneIncrease("000000"); //ALREADY MONOTONE
//        obj.flipStringToMonotoneIncrease("11111111"); //ALREADY MONOTONE
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Subarray sum divisible by K");
//        //https://leetcode.com/problems/subarray-sums-divisible-by-k/
//        obj.subarraySumDivisibleByK(new int[]{4, 5, 0, -2, -3, 1}, 5);
//        obj.subarraySumDivisibleByK(new int[]{-1,2,9}, 2);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Gas station");
//        //https://leetcode.com/problems/gas-station/
//        obj.gasStationCompleteCircuit(new int[]{1, 2, 3, 4, 5}, new int[]{3, 4, 5, 1, 2});
//        obj.gasStationCompleteCircuit(new int[]{2, 3, 4}, new int[]{3, 4, 3});
//        obj.gasStationCompleteCircuit(new int[]{3,1,1}, new int[]{1,2,2});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Minimum deletion cost to avoid repeating charracters");
//        //https://leetcode.com/problems/minimum-time-to-make-rope-colorful/
//        //https://leetcode.com/problems/minimum-deletion-cost-to-avoid-repeating-letters/
//        obj.minDeletionCostToAvoidRepeatingChar("abaac", new int[]{1, 2, 3, 4, 5});
//        obj.minDeletionCostToAvoidRepeatingChar("abc", new int[]{1, 2, 3});
//        obj.minDeletionCostToAvoidRepeatingChar("aabaa", new int[]{1, 2, 3, 4, 1});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Alien dictionary (Graph)");
//        //https://www.geeksforgeeks.org/given-sorted-dictionary-find-precedence-characters/
//        obj.alienDictionary_Graph(new String[]{"caa", "aaa", "aab"}, 3);
//        obj.alienDictionary_Graph(new String[]{"baa", "abcd", "abca", "cab", "cad"}, 4);
//        obj.alienDictionary2_Graph(new String[]{"wrt", "wrf", "er", "ett", "rftt"});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Range update and get queries");
//        //https://www.geeksforgeeks.org/binary-indexed-tree-range-updates-point-queries/
//        obj.rangeUpdateAndPointQueries(new int[]{0, 0, 0, 0, 0});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Remove all adjacent duplicate K chars in the strings and print remaining");
//        //https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string
//        //https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string-ii/
//        obj.removeAdjacentDuplicateKCharInString("pbbcggttciiippooaais", 2);
//        obj.removeAdjacentDuplicateKCharInString("abcd", 2);
//        obj.removeAdjacentDuplicateKCharInString("deeedbbcccbdaa", 3);
//        obj.removeAdjacentDuplicateKCharInString("XABCDFFDCBA", 2);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Minimum Number of Swaps to Make the Binary String Alternating");
//        //https://leetcode.com/problems/minimum-number-of-swaps-to-make-the-binary-string-alternating/
//        System.out.println("Minswaps required to make binary string alternate: "
//                + obj.minSwapRequiredToMakeBinaryStringAlternate("111000")); //101010
//        System.out.println("Minswaps required to make binary string alternate: "
//                + obj.minSwapRequiredToMakeBinaryStringAlternate("00")); //Not Possible
//        System.out.println("Minswaps required to make binary string alternate: "
//                + obj.minSwapRequiredToMakeBinaryStringAlternate("1110")); //Not Possible
//        System.out.println("Minswaps required to make binary string alternate: "
//                + obj.minSwapRequiredToMakeBinaryStringAlternate("010")); //Already alternate
//        System.out.println("Minswaps required to make binary string alternate: "
//                + obj.minSwapRequiredToMakeBinaryStringAlternate("1")); //Already alternate
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Rectangle overlapping");
//        //https://leetcode.com/problems/rectangle-overlap/
//        //https://leetcode.com/problems/rectangle-area/
//        System.out.println("Rectangle overlapping & area: "
//                + obj.rectangleOverlappingAndArea(new int[]{0, 0, 2, 2}, new int[]{1, 1, 3, 3})); //overlapping
//        System.out.println("Rectangle overlapping & area: "
//                + obj.rectangleOverlappingAndArea(new int[]{-2, 0, 0, 2}, new int[]{-3, 1, -1, 3})); //Other quadrant
//        System.out.println("Rectangle overlapping & area: "
//                + obj.rectangleOverlappingAndArea(new int[]{0, 0, 1, 1}, new int[]{2, 2, 2, 3})); //No overlapping
//        System.out.println("Rectangle overlapping & area: "
//                + obj.rectangleOverlappingAndArea(new int[]{1, 1, 2, 2}, new int[]{2, 1, 3, 2})); //Only edge touching
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("All common element in 3 sorted arrays");
//        //https://practice.geeksforgeeks.org/problems/common-elements1132/1
//        obj.allCommonElementIn3SortedArray(
//                new int[]{1, 5, 10, 20, 40, 80},
//                new int[]{6, 7, 20, 80, 100},
//                new int[]{3, 4, 15, 20, 30, 70, 80, 120});
//        obj.allCommonElementIn3SortedArray(
//                new int[]{3, 3, 3, 3},
//                new int[]{3, 3, 3, 3},
//                new int[]{3, 3, 3, 3});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Rearrage elements of array randomly but in-place & in O(N)");
//        obj.randomlyRearrangeElementsOfArray(new int[]{1, 0, 7, 2, 10});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Time Needed to Inform All Employees");
//        //https://leetcode.com/problems/time-needed-to-inform-all-employees/
//        obj.timeNeededToInformAllEmployee_NAryTree(1, 0, new int[]{-1}, new int[]{0});
//        obj.timeNeededToInformAllEmployee_NAryTree(6, 2, new int[]{2, 2, -1, 2, 2, 2}, new int[]{0, 0, 1, 0, 0, 0});
//        obj.timeNeededToInformAllEmployee_NAryTree(15, 0,
//                new int[]{-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6},
//                new int[]{1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0});
//        obj.timeNeededToInformAllEmployee_DFS(1, 0, new int[]{-1}, new int[]{0});
//        obj.timeNeededToInformAllEmployee_DFS(6, 2, new int[]{2, 2, -1, 2, 2, 2}, new int[]{0, 0, 1, 0, 0, 0});
//        obj.timeNeededToInformAllEmployee_DFS(15, 0,
//                new int[]{-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6},
//                new int[]{1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Minimum Index Sum of Two Lists");
//        //https://leetcode.com/problems/minimum-index-sum-of-two-lists/
//        obj.minimumIndexSumOfTwoStringArray(new String[]{"Shogun", "Tapioca Express", "Burger King", "KFC"},
//                new String[]{"KFC", "Burger King", "Tapioca Express", "Shogun"});
//        obj.minimumIndexSumOfTwoStringArray(new String[]{"Shogun", "Tapioca Express", "Burger King", "KFC"},
//                new String[]{"Piatti", "The Grill at Torrey Pines", "Hungry Hunter Steakhouse", "Shogun"});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Maximal square DP");
//        //https://leetcode.com/problems/maximal-square/
//        //https://practice.geeksforgeeks.org/problems/largest-square-formed-in-a-matrix0806/1
//        obj.maximalSquare_DP_Memoization(new int[][]{
//            {1, 0, 1, 0, 0},
//            {1, 0, 1, 1, 1},
//            {1, 1, 1, 1, 1},
//            {1, 0, 0, 1, 0},}); //SQR: mat[1][2] to mat[2][3]
//        obj.maximalSquare_DP_Memoization(new int[][]{
//            {0, 0, 0, 0},
//            {0, 0, 1, 0},
//            {0, 0, 0, 0},}); //SQR: mat[1][2]
//        obj.maximalSquare_DP_Memoization(new int[][]{
//            {0}}); //SQR: 0
//        obj.maximalSquare_DP_Memoization(new int[][]{
//            {1, 1, 1, 1},
//            {1, 1, 1, 1},
//            {1, 1, 1, 1},
//            {1, 1, 1, 1},}); //SQR: mat[N][N]
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Count all squares with one in binary matrix/ maximal squares approach");
//        //https://leetcode.com/problems/count-square-submatrices-with-all-ones
//        obj.countAllSquareWithOneInBinaryMatrix_DP_Memoization(new int[][]{
//            {0, 0, 0, 0},
//            {0, 0, 1, 0},
//            {0, 0, 0, 0},});
//        obj.countAllSquareWithOneInBinaryMatrix_DP_Memoization(new int[][]{
//            {0, 1, 1, 1},
//            {1, 1, 1, 1},
//            {0, 1, 1, 1},});
//        obj.countAllSquareWithOneInBinaryMatrix_DP_Memoization(new int[][]{
//            {1, 1, 0, 0},
//            {1, 1, 0, 0},
//            {0, 0, 0, 0},});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Teemo attacking ashee");
//        //https://leetcode.com/problems/teemo-attacking
//        //https://leetcode.com/discuss/interview-question/280433/Google-or-Phone-screen-or-Program-scheduling
//        System.out.println("Total time till ashee remained poisined: "
//                + obj.teemoAttackingAshee(new int[]{1, 4}, 2));
//        System.out.println("Total time till ashee remained poisined: "
//                + obj.teemoAttackingAshee(new int[]{1, 2}, 2));
//        System.out.println("Total time till ashee remained poisined: "
//                + obj.teemoAttackingAshee(new int[]{1, 10}, 5));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Min operation to make array equal of size n");
//        //https://leetcode.com/problems/minimum-operations-to-make-array-equal/
//        obj.minOperationToMakeArrayOfSizeNEqual(3);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("First Unique Character in a String");
//        //https://leetcode.com/problems/first-unique-character-in-a-string/
//        System.out.println("Index of first unique char: " + obj.firstUniqueCharacterInString("aabbcc"));
//        System.out.println("Index of first unique char: " + obj.firstUniqueCharacterInString("leetcode"));
//        System.out.println("Index of first unique char: " + obj.firstUniqueCharacterInString("loveleetcode"));
//        System.out.println("Index of first unique char: " + obj.firstUniqueCharacterInString("aaaa"));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Find Pivot Index");
//        //https://leetcode.com/problems/find-pivot-index/
//        //https://leetcode.com/problems/find-the-middle-index-in-array
//        //https://leetcode.com/problems/number-of-ways-to-split-array
//        System.out.println("Pivot index: " + obj.findPivotIndex(new int[]{1, 7, 3, 6, 5, 6}));
//        System.out.println("Pivot index: " + obj.findPivotIndex(new int[]{1, 2, 3}));
//        System.out.println("Pivot index: " + obj.findPivotIndex(new int[]{2, 1, -1}));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Maximum subarray sum with unique elements/ Maximum Erasure Value");
//        //https://leetcode.com/problems/maximum-erasure-value/
//        obj.maximumSubarraySumWithUniqueElements(new int[]{4, 2, 4, 5, 6}); //sunarr: [2,4,5,6]
//        obj.maximumSubarraySumWithUniqueElements(new int[]{5, 2, 1, 2, 5, 2, 1, 2, 5}); //sunarr: [5,2,1] or [1,2,5]
//        obj.maximumSubarraySumWithUniqueElements(new int[]{1, 1, 1, 1}); //subarr: [1]
//        obj.maximumSubarraySumWithUniqueElements(new int[]{1, 2, 3, 4}); //subarr: [1,2,3,4]
//        obj.maximumSubarraySumWithUniqueElements(new int[]{1, 1, 2, 2, 3, 3, 4, 4}); //subarr: [3,4]
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Minimum path sum in grid(top-left to bottom-right)");
//        //https://leetcode.com/problems/minimum-path-sum/
//        obj.minimumPathSumInGrid(new int[][]{
//            {1, 3, 1}, {1, 5, 1}, {4, 2, 1}
//        });
//        obj.minimumPathSumInGrid(new int[][]{
//            {1, 2, 3}, {4, 5, 6}
//        });
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Partition Array Into Three Parts With Equal Sum");
//        //https://leetcode.com/problems/partition-array-into-three-parts-with-equal-sum/
//        System.out.println("Pratition possible: "
//                + obj.partitionArrayIntoThreePartsWithEqualSum(new int[]{0, 2, 1, -6, 6, -7, 9, 1, 2, 0, 1}));
//        System.out.println("Pratition possible: "
//                + obj.partitionArrayIntoThreePartsWithEqualSum(new int[]{0, 2, 1, -6, 6, 7, 9, -1, 2, 0, 1}));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Substring with Concatenation of All Words");
//        //https://leetcode.com/problems/substring-with-concatenation-of-all-words/
//        obj.substringWithConcatenationsOfGivenWords("barfoothefoobarman",
//                new String[]{"foo", "bar"});
//        obj.substringWithConcatenationsOfGivenWords("wordgoodgoodgoodbestword",
//                new String[]{"word", "good", "best", "word"});
//        obj.substringWithConcatenationsOfGivenWords("barfoofoobarthefoobarman",
//                new String[]{"bar", "foo", "the"});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Remove all sequences of consecutive linked list node sum to zero");
//        //https://leetcode.com/problems/remove-zero-sum-consecutive-nodes-from-linked-list/
//        Node<Integer> head = new Node<>(1);
//        head.setNext(new Node<>(2));
//        head.getNext().setNext(new Node<>(3));
//        head.getNext().getNext().setNext(new Node<>(-3));
//        head.getNext().getNext().getNext().setNext(new Node<>(-2));
//        obj.removeZeroSumConsecutiveNodesFromLinkedList(head);
//        head = new Node<>(1);
//        head.setNext(new Node<>(2));
//        head.getNext().setNext(new Node<>(3));
//        head.getNext().getNext().setNext(new Node<>(-3));
//        head.getNext().getNext().getNext().setNext(new Node<>(4));
//        obj.removeZeroSumConsecutiveNodesFromLinkedList(head);
//        head = new Node<>(1);
//        head.setNext(new Node<>(2));
//        head.getNext().setNext(new Node<>(3));
//        head.getNext().getNext().setNext(new Node<>(-3));
//        head.getNext().getNext().getNext().setNext(new Node<>(1));
//        obj.removeZeroSumConsecutiveNodesFromLinkedList(head);
//        head = new Node<>(0);
//        head.setNext(new Node<>(2));
//        head.getNext().setNext(new Node<>(3));
//        head.getNext().getNext().setNext(new Node<>(0));
//        head.getNext().getNext().getNext().setNext(new Node<>(4));
//        obj.removeZeroSumConsecutiveNodesFromLinkedList(head);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Add two nums without usiing + or -");
//        //https://leetcode.com/problems/sum-of-two-integers
//        obj.addTwoNumsWithoutPlusOrMinus(4, 6);
//        obj.addTwoNumsWithoutPlusOrMinus(4, -6);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Top k frequent elements in an array");
//        //https://leetcode.com/problems/top-k-frequent-elements/
//        obj.topKFrequentElements(new int[]{1, 1, 1, 2, 2, 3}, 2);
//        obj.topKFrequentElements(new int[]{1}, 1);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Maximum words that you can type");
//        //https://leetcode.com/problems/maximum-number-of-words-you-can-type/
//        obj.maximumWordsThatYouCanType("hello world", "ad"); //hello can be typed, world is broken on char d
//        obj.maximumWordsThatYouCanType("world world", "ad"); //no words can be typed, world is broken on char d
//        obj.maximumWordsThatYouCanType("leet code", "lt"); //code can be typed, leet is broken on char l(even one char can make it broken)
//        obj.maximumWordsThatYouCanType("hello world", "xy"); //hello, world both can be typed, no words contains broken chars x & y
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Maximum length AP in binary tree");
//        //https://www.geeksforgeeks.org/longest-path-to-the-bottom-of-a-binary-tree-forming-an-arithmetic-progression/
//        //https://www.geeksforgeeks.org/longest-arithmetic-progression-path-in-given-binary-tree/
//        TreeNode<Integer> root1 = new TreeNode<>(6);
//        root1.setRight(new TreeNode<>(9)); //common diff from prev node is 3
//        root1.getRight().setRight(new TreeNode<>(12)); //common diff from prev node is 3
//        root1.getRight().getRight().setRight(new TreeNode<>(15)); //common diff from prev node is 3
//        root1.getRight().setLeft(new TreeNode<>(7));
//        obj.longestPathArithemeticProgressionInBinaryTree(root1);
//        root1 = new TreeNode<>(6);
//        obj.longestPathArithemeticProgressionInBinaryTree(root1);
//        root1 = new TreeNode<>(6);
//        root1.setRight(new TreeNode<>(9));
//        obj.longestPathArithemeticProgressionInBinaryTree(root1);
//        root1 = new TreeNode<>(15);
//        root1.setRight(new TreeNode<>(12)); //common diff from prev node is 3
//        root1.getRight().setRight(new TreeNode<>(9)); //common diff from prev node is 3
//        root1.getRight().getRight().setRight(new TreeNode<>(6)); //common diff from prev node is 3
//        root1.getRight().setLeft(new TreeNode<>(7));
//        obj.longestPathArithemeticProgressionInBinaryTree(root1);
//        root1 = new TreeNode<>(15);
//        root1.setRight(new TreeNode<>(12)); //common diff from prev node is 3
//        root1.getRight().setRight(new TreeNode<>(11)); //common diff from prev node is NOT 3, hence breaking AP, maxLen - 15 -> 12 = 2
//        root1.getRight().getRight().setRight(new TreeNode<>(9));
//        root1.getRight().setLeft(new TreeNode<>(7));
//        obj.longestPathArithemeticProgressionInBinaryTree(root1);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Longest string chain");
//        //https://leetcode.com/problems/longest-string-chain/
//        obj.longestStringChain_DP_Memoization(new String[]{"a", "abc", "ab", "abcd"}); //4
//        obj.longestStringChain_DP_Memoization(new String[]{"a", "b", "ba", "bca", "bda", "bdca"});
//        obj.longestStringChain_DP_Memoization(new String[]{"l", "mn", "op", "qrst"}); //any string can be a single lengthed chain = 1
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("maximum sum circular subarray");
//        //https://leetcode.com/problems/maximum-sum-circular-subarray
//        obj.maximumSumCircularSubarray(new int[]{5, -1, -2, 5});
//        obj.maximumSumCircularSubarray(new int[]{-1, 5, 5, -2});
//        obj.maximumSumCircularSubarray(new int[]{-4, -3, -2, -1});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Find town judge Graph");
//        //https://leetcode.com/problems/find-the-town-judge/
//        System.out.println("Judge is : " + obj.findTownJudge_Graph(2, new int[][]{{1, 2}}));
//        System.out.println("Judge is : " + obj.findTownJudge_Graph(3, new int[][]{{1, 3}, {2, 3}}));
//        System.out.println("Judge is : " + obj.findTownJudge_Graph(3, new int[][]{{1, 3}, {2, 3}, {3, 1}}));
//        System.out.println("Judge is : " + obj.findTownJudge_Graph(4, new int[][]{{1, 2}, {2, 3}, {3, 4}}));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Check if a Parentheses String Can Be Valid");
//        //https://leetcode.com/problems/check-if-a-parentheses-string-can-be-valid/
//        System.out.println("Valid parenthesis possible: " + obj.checkIfParenthesisStringCanBeValid("))()))", "010100"));
//        System.out.println("Valid parenthesis possible: " + obj.checkIfParenthesisStringCanBeValid(")))(((", "001100"));
//        System.out.println("Valid parenthesis possible: " + obj.checkIfParenthesisStringCanBeValid("()()", "0000"));
//        System.out.println("Valid parenthesis possible: " + obj.checkIfParenthesisStringCanBeValid(")", "0"));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Largest Odd Number in String");
//        //https://leetcode.com/problems/largest-odd-number-in-string/
//        obj.largestOddNumInGivenNumString("52");
//        obj.largestOddNumInGivenNumString("4206");
//        obj.largestOddNumInGivenNumString("864278642");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Non-decreasing Array");
//        //https://leetcode.com/problems/non-decreasing-array/
//        System.out.println("Non decreasing array with atmost one element change possible: "
//                + obj.nonDecreasingArrayWithAtmostOneChange(new int[]{4, 2, 3}));
//        System.out.println("Non decreasing array with atmost one element change possible: "
//                + obj.nonDecreasingArrayWithAtmostOneChange(new int[]{4, 2}));
//        System.out.println("Non decreasing array with atmost one element change possible: "
//                + obj.nonDecreasingArrayWithAtmostOneChange(new int[]{4, 2, 1}));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Surrounded Regions");
//        //https://leetcode.com/problems/surrounded-regions/
//        obj.surroundedRegions_Graph(new char[][]{
//            {'X', 'X', 'X', 'X'},
//            {'X', 'O', 'O', 'X'},
//            {'X', 'X', 'O', 'X'},
//            {'X', 'O', 'X', 'X'}
//        });
//        obj.surroundedRegions_Graph(new char[][]{
//            {'X', 'O', 'X', 'X', 'X'},
//            {'X', 'O', 'X', 'O', 'X'},
//            {'X', 'O', 'X', 'O', 'X'},
//            {'X', 'O', 'X', 'X', 'X'}
//        });
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Longest Increasing Path in a Matrix");
//        //https://leetcode.com/problems/longest-increasing-path-in-a-matrix/
//        obj.longestIncreasingPathInMatrixFromAnyPoint_Graph_Memoization(new int[][]{
//            {9, 9, 4}, {6, 6, 8}, {2, 1, 1}
//        });
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Two Sum/ Two Sum II - Input Array Is Sorted/ Four Sum");
//        //https://leetcode.com/problems/two-sum/
//        //https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/
//        obj.twoSum_UnsortedArray(new int[]{2, 7, 11, 15}, 9);
//        obj.twoSum_UnsortedArray(new int[]{3, 2, 4}, 6);
//        obj.twoSum2_SortedArray(new int[]{2, 7, 11, 15}, 9);
//        obj.twoSum2_SortedArray(new int[]{-1, 0}, -1);
//        //https://leetcode.com/problems/4sum/
//        obj.fourSum(new int[]{1, 0, -1, 0, -2, 2}, 0);
//        obj.fourSum(new int[]{2, 2, 2, 2, 2}, 8);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Smallest String With A Given Numeric Value");
//        //https://leetcode.com/problems/smallest-string-with-a-given-numeric-value/
//        obj.smallestStringWithGivenLengthNAndCharSumValueK(3, 27);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Replace All ?'s to Avoid Consecutive Repeating Characters");
//        //https://leetcode.com/problems/replace-all-s-to-avoid-consecutive-repeating-characters/
//        obj.replaceAllQuestionMarksWithACharAndNoConsecutiveRepeatingChar("?zs");
//        obj.replaceAllQuestionMarksWithACharAndNoConsecutiveRepeatingChar("ubv?w");
//        obj.replaceAllQuestionMarksWithACharAndNoConsecutiveRepeatingChar("?z?a?");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Simplify Path");
//        //https://leetcode.com/problems/simplify-path/
//        System.out.println("Canonical path: " + obj.simplifyPath("/home/"));
//        System.out.println("Canonical path: " + obj.simplifyPath("/../"));
//        System.out.println("Canonical path: " + obj.simplifyPath("/home///foo/./bar/zoo/./../"));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Maximum Length of Subarray With Positive Product");
//        //https://leetcode.com/problems/maximum-length-of-subarray-with-positive-product/
//        obj.maximumLengthOfSubarrayWithPositiveProduct(new int[]{1, -2, -3, 4}); //lenght = 4 as 1 * -2 * -3 * 4 > 0
//        obj.maximumLengthOfSubarrayWithPositiveProduct(new int[]{0, 1, -2, -3, -4});
//        obj.maximumLengthOfSubarrayWithPositiveProduct(new int[]{-1, 8, 8, -2});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Swim in Rising Water");
//        //https://leetcode.com/problems/swim-in-rising-water/
//        System.out.println("Time taken to swim in rising water to bottom-right corner: "
//                + obj.swimInRisingWater_Graph(new int[][]{
//                    {0, 2}, {1, 3}}));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Find Common Characters");
//        //https://leetcode.com/problems/find-common-characters/
//        obj.findCommonCharacters(new String[]{"bella", "label", "roller"});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
        System.out.println("Minimum Interval to Include Each Query");
        //https://leetcode.com/problems/minimum-interval-to-include-each-query/
        obj.minimumIntervalToIncludeEachQuery(new int[][]{
            {1, 4}, {2, 4}, {3, 6}, {4, 4}}, new int[]{2, 3, 4, 5});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Find Three Consecutive Integers That Sum to a Given Number");
//        //https://leetcode.com/problems/find-three-consecutive-integers-that-sum-to-a-given-number/
//        obj.threeConsecutiveNumberThatSumsToGivenNumber(33);
//        obj.threeConsecutiveNumberThatSumsToGivenNumber(4);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Merge Nodes in Between Zeros");
//        //https://leetcode.com/problems/merge-nodes-in-between-zeros/
//        Node<Integer> head = new Node<>(0);
//        head.setNext(new Node<>(3));
//        head.getNext().setNext(new Node<>(1));
//        head.getNext().getNext().setNext(new Node<>(0));
//        head.getNext().getNext().getNext().setNext(new Node<>(4));
//        head.getNext().getNext().getNext().getNext().setNext(new Node<>(5));
//        head.getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(2));
//        head.getNext().getNext().getNext().getNext().getNext().getNext().setNext(new Node<>(0));
//        obj.mergeNodesInBetweenZeros(head);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Search Suggestions System");
//        //https://leetcode.com/problems/search-suggestions-system/
//        obj.searchSuggestionSystem(new String[]{"mobile", "mouse", "moneypot", "monitor", "mousepad"}, "mouse");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Maximum length prefix of one string that occurs as subsequence in another");
//        //https://www.geeksforgeeks.org/maximum-length-prefix-one-string-occurs-subsequence-another/?ref=rp
//        obj.maximumLengthOfSubstringThatExistsAsSubseqInOtherString("biggerdiagram", "digger");
//        obj.maximumLengthOfSubstringThatExistsAsSubseqInOtherString("abcdef", "xyz");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Is Subsequence");
//        //https://leetcode.com/problems/is-subsequence/
//        System.out.println("Is string curr exists as any subseq in main: "
//                + obj.isSubsequence("ahbgdc", "abc"));
//        //whole digger string doesn't exists as any subseq in main string
//        System.out.println("Is string curr exists as any subseq in main: "
//                + obj.isSubsequence("biggerdiagram", "digger"));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT MY GOOGLE INTERVIEW QUESTION
//        System.out.println("Minimum partitions in curr string where its prefix exists as subseq in main string");
//        obj.partitionsInCurrStringWherePrefixExistsAsSubseqInMainString("aaaabbc", "aacbbabc");
//        obj.partitionsInCurrStringWherePrefixExistsAsSubseqInMainString("aaaabbc", "lmno");
//        obj.partitionsInCurrStringWherePrefixExistsAsSubseqInMainString("aaaabbc", "abcbbabc");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Number of Matching Subsequences");
//        //https://leetcode.com/problems/number-of-matching-subsequences/
//        obj.numberOfMatchingSubseq("abcde", new String[]{"a", "bb", "acd", "ace"});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Binary Tree Cameras");
//        //https://leetcode.com/problems/binary-tree-cameras/
//        TreeNode<Integer> root1 = new TreeNode<>(0);
//        root1.setLeft(new TreeNode<>(0));
//        root1.getLeft().setLeft(new TreeNode<>(0));
//        root1.getLeft().setRight(new TreeNode<>(0));
//        obj.binaryTreeCameras(root1);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Convert Sorted Array to Height Balanced Binary Search Tree");
//        //https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/
//        obj.convertSortedArrayToHeightBalancedBinarySearchTree(new int[]{-10, -3, 0, 5, 9});
//        obj.convertSortedArrayToHeightBalancedBinarySearchTree(new int[]{1, 2, 3, 4, 5});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Minimum Number of Arrows to Burst Balloons");
//        //https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/
//        obj.minimumArrowsToBurstBalloons_Greedy(new int[][]{
//            {1, 6}, {7, 12}});
//        obj.minimumArrowsToBurstBalloons_Greedy(new int[][]{
//            {10, 16}, {2, 8}, {1, 6}, {7, 12}});
//        obj.minimumArrowsToBurstBalloons_Greedy(new int[][]{
//            {1, 2}, {3, 4}, {5, 6}, {7, 8}});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Triangle - minimum path sum from top to bottom");
//        //https://leetcode.com/problems/triangle/
//        List<List<Integer>> triangle = new ArrayList<>();
//        triangle.add(Arrays.asList(2));
//        triangle.add(Arrays.asList(3, 4));
//        triangle.add(Arrays.asList(6, 5, 7));
//        triangle.add(Arrays.asList(4, 1, 8, 3));
//        System.out.println("Min path sum top to bottom in triangle 2d matrix: "
//                + obj.triangleMinPathSumTopToBottom(triangle));
//        triangle = new ArrayList<>();
//        triangle.add(Arrays.asList(2));
//        System.out.println("Min path sum top to bottom in triangle 2d matrix: "
//                + obj.triangleMinPathSumTopToBottom(triangle));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Candy Distribution");
//        //https://leetcode.com/problems/candy/
//        obj.candyDistributionToNStudent(new int[]{1, 0, 2}); //EASY UNDERSTANDING
//        obj.candyDistributionToNStudent2(new int[]{1, 0, 2}); //SPACE OPTIMISED
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Partition Array Such That Maximum Difference Is K");
//        //https://leetcode.com/problems/partition-array-such-that-maximum-difference-is-k/
//        obj.partitionArrSuchThatMaxDiffIsK_Greedy(new int[]{3, 6, 1, 2, 5}, 2);
//        obj.partitionArrSuchThatMaxDiffIsK_Greedy(new int[]{2, 2, 4, 5}, 0);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Find Triangular Sum of an Array");
//        //https://leetcode.com/problems/find-triangular-sum-of-an-array/
//        //https://leetcode.com/problems/min-max-game/
//        obj.findTriangularSumOfArray(new int[]{1, 2, 3, 4, 5});
//        obj.findTriangularSumOfArray(new int[]{5});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Fair Distribution of Cookies");
//        //https://leetcode.com/problems/fair-distribution-of-cookies/
//        obj.minimumUnfairDistributionOfCookiesToKStudent_Backtracking(new int[]{8, 15, 10, 20, 8}, 2);
//        obj.minimumUnfairDistributionOfCookiesToKStudent_Backtracking(new int[]{6, 1, 3, 2, 2, 4, 1, 2}, 3);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("132 Pattern");
//        //https://leetcode.com/problems/132-pattern/
//        System.out.println("132 Pattern: " + obj.has132Pattern(new int[]{1, 2, 3, 4}));
//        System.out.println("132 Pattern: " + obj.has132Pattern(new int[]{3, 1, 4, 2}));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Minimum Number of Swaps to Make the String Balanced");
//        //https://leetcode.com/problems/minimum-number-of-swaps-to-make-the-string-balanced/
//        obj.minimumSwapsToMakeParenthesisStringBalanced("][][");
//        obj.minimumSwapsToMakeParenthesisStringBalanced("]]][[[");
//        obj.minimumSwapsToMakeParenthesisStringBalanced("[[]][]");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Minimum Add to Make Parentheses Valid");
//        //https://leetcode.com/problems/minimum-add-to-make-parentheses-valid/
//        obj.minimumAdditionsToMakeParenthesisStringValid("())");
//        obj.minimumAdditionsToMakeParenthesisStringValid("(())");
//        obj.minimumAdditionsToMakeParenthesisStringValid(")))(((");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Furthest Building You Can Reach");
//        //https://leetcode.com/problems/furthest-building-you-can-reach/
//        System.out.println("Index of the farthest building we can reach: "
//                + obj.farthestBuildingWeCanReachUsingBricksAndLadders_Greedy(new int[]{4, 2, 7, 6, 9, 14, 12}, 5, 1));
//        System.out.println("Index of the farthest building we can reach: "
//                + obj.farthestBuildingWeCanReachUsingBricksAndLadders_Greedy(new int[]{1, 5, 1, 2, 3, 4, 10000}, 4, 1));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Verifying an Alien Dictionary");
//        //https://leetcode.com/problems/verifying-an-alien-dictionary/
//        System.out.println("Alien word dict are sorted acc to alien aplhabet: "
//                + obj.areAlienWordsSorted(new String[]{"hello", "leetcode"}, "hlabcdefgijkmnopqrstuvwxyz"));
//        System.out.println("Alien word dict are sorted acc to alien aplhabet: "
//                + obj.areAlienWordsSorted(new String[]{"apple", "app"}, "abcdefghijklmnopqrstuvwxyz"));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Detect Squares");
//        //https://leetcode.com/problems/detect-squares/
//        List<int[]> points = Arrays.asList(
//                new int[]{3, 10},
//                new int[]{11, 2},
//                new int[]{3, 2}
//        );
//        List<int[]> queryPoints = Arrays.asList(
//                new int[]{11, 10},
//                new int[]{14, 8}
//        );
//        obj.detectSquares(points, queryPoints);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Amount To Paint The Area");
//        //https://leetcode.com/problems/amount-of-new-area-painted-each-day/
//        //https://www.geeksforgeeks.org/google-interview-experience-for-software-engineer-l3-bangalore-6-years-experienced/
//        //https://leetcode.com/discuss/interview-question/2072036/Google-or-Onsite-or-banglore-or-May-2022-or-Paint-a-line
//        obj.amountToPaintTheArea(new int[][]{
//            {4, 10}, {7, 13}, {16, 20}, {1, 40}});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Process Tasks Using Servers / Find Max Patient Treated In Any Given N Rooms");
//        //https://leetcode.com/problems/process-tasks-using-servers
//        //https://www.geeksforgeeks.org/google-interview-experience-for-software-engineer-l3-bangalore-6-years-experienced/
//        //https://leetcode.com/discuss/interview-question/2072047/Google-or-Onsite-or-Banglore-or-May-2022-or-Patient-Queue
//        obj.serverAllocationToTasks(new int[]{3, 3, 2}, new int[]{1, 2, 3, 2, 1, 2});
//        // room 2 as (1,2) will be alloted first and will go first then (6,4) will be alloted
//        obj.maxPatientTreatedInGivenInAnyNRoom(new int[][]{
//            {1, 8}, {1, 2}, {6, 4}}, 2);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Number following a pattern");
//        //https://practice.geeksforgeeks.org/problems/number-following-a-pattern3126/1#
//        obj.generateNumberFollowingPattern("D");
//        obj.generateNumberFollowingPattern("IIDDD");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
        System.out.println("Pacific Atlantic Water Flow");
        //https://leetcode.com/problems/pacific-atlantic-water-flow/
        obj.pacificAtlanticWaterFlow(new int[][]{
            {1, 2, 2, 3, 5},
            {3, 2, 3, 4, 4},
            {2, 4, 5, 3, 1},
            {6, 7, 1, 4, 5},
            {5, 1, 1, 2, 4}});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Mth Element After K Array Rotation");
//        //https://www.geeksforgeeks.org/cpp-program-to-find-the-mth-element-of-the-array-after-k-left-rotations/
//        obj.mThElementAfterKArrayRotation(new int[]{1, 2, 3, 4, 5}, 2, 3);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("The Skyline Problem");
//        //https://leetcode.com/problems/the-skyline-problem/
//        obj.skylineProblem(new int[][]{
//            {2, 9, 10}, {3, 7, 15}, {5, 12, 12}, {15, 20, 10}, {19, 24, 8}
//        });
//        //OPTIMIZED with tree map
//        obj.skylineProblem_TreeMap(new int[][]{
//            {2, 9, 10}, {3, 7, 15}, {5, 12, 12}, {15, 20, 10}, {19, 24, 8}
//        });
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Angle Between Hands of a Clock");
//        //https://leetcode.com/problems/angle-between-hands-of-a-clock/
//        obj.minAngleBetweeHourAndMinuteHands(12, 30);
//        obj.minAngleBetweeHourAndMinuteHands(2, 60); // 3:00
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Minimum Inserts/ Deletes To Make String Pallindrome");
//        //https://leetcode.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/
//        //https://www.geeksforgeeks.org/java-program-to-find-minimum-insertions-to-form-a-palindrome-dp-28/
//        obj.minInsertsToMakeStringPallindrome_DP_Memoization("abcda"); //insert d,b like this abdcdba
//        obj.minInsertsToMakeStringPallindrome_DP_Memoization("aba");
//        obj.minDeletesToMakeStringPallindrome_DP_Memoization("abcda"); //deletes d,b like this aca
//        obj.minDeletesToMakeStringPallindrome_DP_Memoization("aba");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Maximize sum of given array after removing valleys");
//        //https://www.geeksforgeeks.org/maximize-sum-of-given-array-after-removing-valleys/
//        obj.maximizeSumAfterRemovingValleys(new int[]{5, 1, 8}); //valley removed [1,1,8] = 10
//        obj.maximizeSumAfterRemovingValleys(new int[]{8, 1, 10, 1, 8}); // valley removed [1,1,10,1,1] = 14
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Minimum Number of Moves to Make Palindrome");
//        //https://leetcode.com/problems/minimum-number-of-moves-to-make-palindrome/
//        // a_swap(a,b)_b ==> abab ==> swap(a,b)_ab ==> baab == pallindrome in 2 swaps
//        obj.minMovesToMakeStringPallindrome("aabb");
//        obj.minMovesToMakeStringPallindrome("zzazz");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Largest 3-Same-Digit Number in String");
//        //https://leetcode.com/problems/largest-3-same-digit-number-in-string/
//        obj.largestThreeSameDigitNumInString("6777133339");
//        obj.largestThreeSameDigitNumInString("2300019");
//        obj.largestThreeSameDigitNumInString("42352338");
//        obj.largestThreeSameDigitNumInString("00042352338");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Number of Visible People in a Queue");
//        //https://leetcode.com/problems/number-of-visible-people-in-a-queue/
//        obj.numberOfVisiblePeopleInQueue(new int[]{10, 6, 8, 5, 11, 9});
//        obj.numberOfVisiblePeopleInQueue(new int[]{5, 1, 2, 3, 10});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
        System.out.println("Min Cost to Connect All Points/ Prim's Algo Based");
        //https://leetcode.com/problems/min-cost-to-connect-all-points/
        obj.minCostToConnectAllPoints_Graph(new int[][]{
            {0, 0}, {2, 2}, {3, 10}, {5, 2}, {7, 0}});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Split Array Largest Sum");
//        //https://leetcode.com/problems/split-array-largest-sum/
//        obj.splitArrayInLargestSum(new int[]{7, 2, 5, 10, 8}, 2);
//        obj.splitArrayInLargestSum(new int[]{1, 2, 3, 4, 5}, 2);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Capacity To Ship Packages Within D Days");
//        //https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/
//        //https://leetcode.com/problems/minimized-maximum-of-products-distributed-to-any-store
//        obj.shipWeightsWithinGivenDays(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 1);
//        obj.shipWeightsWithinGivenDays(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 5);
//        obj.shipWeightsWithinGivenDays(new int[]{3, 2, 2, 4, 1, 4}, 3);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Minimize Page Allocations To Students");
//        //https://www.interviewbit.com/problems/allocate-books/
//        obj.minimizePageAllocationsToStudents(new int[]{12, 34, 67, 90}, 2);
//        obj.minimizePageAllocationsToStudents(new int[]{12, 34, 67, 90}, 1);
//        obj.minimizePageAllocationsToStudents(new int[]{12, 34, 67, 90}, 4);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Koko Eating Bananas");
//        //https://leetcode.com/problems/koko-eating-bananas/
//        obj.kokoEatingBananas(new int[]{3, 6, 7, 11}, 8);
//        obj.kokoEatingBananas(new int[]{30, 11, 23, 4, 20}, 5);
//        obj.kokoEatingBananas(new int[]{30, 11, 23, 4, 20}, 6);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Find Subsequence of Length K With the Largest Sum");
//        //https://leetcode.com/problems/find-subsequence-of-length-k-with-the-largest-sum/
//        obj.subseqOfLengthKWithLargestSum(new int[]{2, 1, 3, 3}, 2);
//        obj.subseqOfLengthKWithLargestSum(new int[]{-1, -2, 3, 4}, 3);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Minimum Number of Flips to Make the Binary String Alternating");
//        //https://leetcode.com/problems/minimum-number-of-flips-to-make-the-binary-string-alternating/
//        obj.minFlipsToMakeBinaryStringAlternating("111000");
//        obj.minFlipsToMakeBinaryStringAlternating("010");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("12 Color Card Possible");
//        //https://leetcode.com/discuss/interview-experience/2279548/Google-or-Phone-Screen-or-Question-or-India
//        obj.cardOf12();
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Paint Fence");
//        //https://www.geeksforgeeks.org/painting-fence-algorithm/
//        obj.paintFence_DP_Memoization(1, 2);
//        obj.paintFence_DP_Memoization(2, 2);
//        obj.paintFence_DP_Memoization(3, 2);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Smallest Rectangle Enclosing Black Pixels");
//        //https://leetcode.com/problems/smallest-rectangle-enclosing-black-pixels/
//        obj.smallestRectangleEnclosingBlackPixels(fnew int[][]{
//            {0, 0, 1, 0}, {0, 1, 1, 0}, {0, 1, 0, 0}});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Jump game(s)");
//        //https://leetcode.com/problems/jump-game/
//        obj.jumpGame(new int[]{2, 3, 1, 1, 4});
//        obj.jumpGame(new int[]{3, 2, 1, 0, 4});
//        obj.jumpGame(new int[]{1});
//        obj.jumpGame(new int[]{0});
//        //https://leetcode.com/problems/jump-game-iii/
//        obj.jumpGameThree(new int[]{4, 2, 3, 0, 3, 1, 2}, 5);
//        obj.jumpGameThree(new int[]{3, 0, 2, 1, 2}, 2);
//        //https://leetcode.com/problems/jump-game-iv/
//        System.out.println("Steps to reach end: "
//                + obj.jumpGameFour(new int[]{100, -23, -23, 404, 100, 23, 23, 23, 3, 404}));
//        System.out.println("Steps to reach end: "
//                + obj.jumpGameFour(new int[]{7}));
//        System.out.println("Steps to reach end: "
//                + obj.jumpGameFour(new int[]{7, 6, 9, 6, 9, 6, 9, 7}));
//        //https://leetcode.com/problems/jump-game-vii/
//        System.out.println("Can we reach end of str: " + obj.jumpGameSeven("011010", 2, 3));
//        System.out.println("Can we reach end of str: " + obj.jumpGameSeven("01101110", 2, 3));
//        //https://leetcode.com/problems/frog-jump/
//        System.out.println("Frog can reach the end of river: " + obj.frogJump(new int[]{0, 1, 3, 5, 6, 8, 12, 17}));
//        System.out.println("Frog can reach the end of river: " + obj.frogJump(new int[]{0, 1, 2, 3, 4, 8, 9, 11}));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Djikstra Algorithm Graph");
//        obj.djikstraAlgorithm_Graph();
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Network Delay Time");
//        //https://leetcode.com/problems/network-delay-time/
//        obj.networkTimeDelay_Graph(new int[][]{{2, 1, 1}, {2, 3, 1}, {3, 4, 1}}, 4, 2);
//        obj.networkTimeDelay_Graph(new int[][]{{1, 2, 1}}, 2, 1);
//        obj.networkTimeDelay_Graph(new int[][]{{1, 2, 1}}, 2, 2);
//        obj.networkTimeDelay_Graph(new int[][]{{1, 2, 1}, {2, 1, 1}}, 2, 1);
//        //max network delay is 100 because max time it will take from 1 to 2 
//        //and other nodes will take 1 unit time each 
//        obj.networkTimeDelay_Graph(new int[][]{{1, 2, 100}, {1, 3, 1}, {3, 4, 1}, {4, 5, 1}, {5, 6, 1}}, 6, 1);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Task Schedular");
//        //https://leetcode.com/problems/task-scheduler/
//        obj.taskSchedular_Greedy(new char[]{'A', 'A', 'A', 'B', 'B', 'B'}, 2);
//        obj.taskSchedular_Greedy(new char[]{'A', 'A', 'A', 'B', 'B', 'B'}, 0);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Check If A Move Is Legal");
//        //https://leetcode.com/problems/check-if-move-is-legal/
//        obj.checkIfMoveIsLegal(
//                new String[][]{
//                    {".", ".", ".", "B", ".", ".", ".", "."},
//                    {".", ".", ".", "W", ".", ".", ".", "."},
//                    {".", ".", ".", "W", ".", ".", ".", "."},
//                    {".", ".", ".", "W", ".", ".", ".", "."},
//                    {"W", "B", "B", ".", "W", "W", "W", "B"},
//                    {".", ".", ".", "B", ".", ".", ".", "."},
//                    {".", ".", ".", "B", ".", ".", ".", "."},
//                    {".", ".", ".", "W", ".", ".", ".", "."}
//                }, 4, 3, "B"
//        );
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Island Perimeter");
//        //https://leetcode.com/problems/island-perimeter/
//        obj.islandPerimeter(new int[][]{{0, 1, 0}, {0, 1, 0}});
//        obj.islandPerimeter(new int[][]{{1, 1, 0}});
//        obj.islandPerimeter(new int[][]{
//            {0, 1, 0, 0}, {1, 1, 1, 0}, {0, 1, 0, 0}, {1, 1, 0, 0}});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Decode String");
//        //https://leetcode.com/problems/decode-string/
//        obj.decodedString("3[a]2[bc]");
//        obj.decodedString("3[a2[c]]");
//        obj.decodedString("2[abc]3[cd]ef");
//        obj.decodedString("20[abc]3[cd]ef");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Valid Palindrome II");
//        //https://leetcode.com/problems/valid-palindrome-ii/
//        System.out.println("Valid pallindrome two: " + obj.validPallindromeTwo("abc"));
//        System.out.println("Valid pallindrome two: " + obj.validPallindromeTwo("aba"));
//        System.out.println("Valid pallindrome two: " + obj.validPallindromeTwo("abca"));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Decode Ways DP Problem");
//        //https://leetcode.com/problems/decode-ways/
//        obj.decodeWays_Recursive_Memoization("12");
//        obj.decodeWays_Recursive_Memoization("226");
//        obj.decodeWays_Recursive_Memoization("06");
//        obj.decodeWays_Recursive_Memoization("11106");
//        obj.decodeWays_DP_Memoization("12");
//        obj.decodeWays_DP_Memoization("226");
//        obj.decodeWays_DP_Memoization("06");
//        obj.decodeWays_DP_Memoization("11106");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Two City Scheduling");
//        //https://leetcode.com/problems/two-city-scheduling/
//        obj.twoCityScheduling_Greedy(new int[][]{
//            {10, 20}, {30, 200}, {400, 50}, {30, 20}});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT MY GOOGLE ONSITE INTERVIEW
//        System.out.println("Filling Bookcase Shelves");
//        //https://leetcode.com/problems/filling-bookcase-shelves/
//        obj.fillingBooksInShelves_DP_Recusrive_Memoization(new int[][]{
//            {1, 1}, {2, 3}, {2, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 2}}, 4);
//        obj.fillingBooksInShelves_DP_Recusrive_Memoization(new int[][]{
//            {1, 3}, {2, 4}, {3, 2}}, 6);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
        System.out.println("Perfect Squares");
        //https://leetcode.com/problems/perfect-squares
        //possible perfect sqrs that sum upto 12
        //1. sqr(3) + sqr(1) + sqr(1) + sqr(1) ==> 9 + 1 + 1 + 1 == 12
        //2. sqr(2) + sqr(2) + sqr(2) ==> 4 + 4 + 4 == 12 also this is MIN hence our result
        obj.perfectSquares_DP_Recursive_Memoization(12);
        obj.perfectSquares_DP_Recursive_Memoization(13);
        obj.perfectSquares_DP_Recursive_Memoization(1);
        //TLE on leetcode
        obj.perfectSquares_DP_Memoization(12);
        obj.perfectSquares_DP_Memoization(13);
        obj.perfectSquares_DP_Memoization(1);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
        System.out.println("Out of Boundary Paths");
        //https://leetcode.com/problems/out-of-boundary-paths/
        obj.outOfBoundaryPaths_DP_Recursive_Memoization(2, 2, 2, 0, 0);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
        System.out.println("Range Sum Query (1D/ 2D) - Immutable");
        //https://leetcode.com/problems/range-sum-query-immutable/
        //https://leetcode.com/problems/range-sum-query-2d-immutable/
        obj.rangeSumQueries_BruteForce(
                new int[]{-2, 0, 3, -5, 2, -1},
                new int[][]{{0, 2}, {2, 5}, {0, 5}});
        obj.rangeSumQueries(
                new int[]{-2, 0, 3, -5, 2, -1},
                new int[][]{{0, 2}, {2, 5}, {0, 5}});
        obj.rangeSumQuery2D(
                new int[][]{
                    {3, 0, 1, 4, 2},
                    {5, 6, 3, 2, 1},
                    {1, 2, 0, 1, 5},
                    {4, 1, 0, 1, 7},
                    {1, 0, 3, 0, 5}},
                new int[][]{
                    {0, 0, 4, 4},
                    {2, 1, 4, 3},
                    {1, 1, 2, 2},
                    {1, 2, 2, 4}});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Binary Search Tree Iterator");
//        //https://leetcode.com/problems/binary-search-tree-iterator
//        TreeNode<Integer> root1 = new TreeNode<>(2);
//        root1.setLeft(new TreeNode<>(1));
//        root1.setRight(new TreeNode<>(3));
//        obj.binarySearchTreeIterator(root1);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
        System.out.println("Partition to K Equal Sum Subsets");
        //https://leetcode.com/problems/partition-to-k-equal-sum-subsets/
        System.out.println("Partition to K equal subset sum possible: "
                + obj.partitionToKEqualSumSubset_Backtracking(new int[]{4, 3, 2, 3, 5, 2, 1}, 4));
        System.out.println("Partition to K equal subset sum possible: "
                + obj.partitionToKEqualSumSubset_Backtracking(new int[]{1, 2, 3, 4}, 3));
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
        System.out.println("Implement Increamental Stack");
        obj.implementIncreamentalStack();
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
        System.out.println("N-Queens");
        //https://leetcode.com/problems/n-queens/
        obj.nQueens_Backtracking(1);
        obj.nQueens_Backtracking(4);
        obj.nQueens_Backtracking(9);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
        System.out.println("Sudoku Solver");
        //https://leetcode.com/problems/sudoku-solver
        obj.sudokuSolver_Backtracking(new char[][]{
            {'5', '3', '.', '.', '7', '.', '.', '.', '.'},
            {'6', '.', '.', '1', '9', '5', '.', '.', '.'},
            {'.', '9', '8', '.', '.', '.', '.', '6', '.'},
            {'8', '.', '.', '.', '6', '.', '.', '.', '3'},
            {'4', '.', '.', '8', '.', '3', '.', '.', '1'},
            {'7', '.', '.', '.', '2', '.', '.', '.', '6'},
            {'.', '6', '.', '.', '.', '.', '2', '8', '.'},
            {'.', '.', '.', '4', '1', '9', '.', '.', '5'},
            {'.', '.', '.', '.', '8', '.', '.', '7', '9'}});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
//        System.out.println("Linked List in Binary Tree");
//        //https://leetcode.com/problems/linked-list-in-binary-tree/
//        TreeNode<Integer> root1 = new TreeNode<>(1);
//        root1.setLeft(new TreeNode<>(4));
//        root1.getLeft().setRight(new TreeNode<>(2));
//        root1.getLeft().getRight().setLeft(new TreeNode<>(1));
//        root1.setRight(new TreeNode<>(4));
//        root1.getRight().setLeft(new TreeNode<>(2));
//        root1.getRight().getLeft().setLeft(new TreeNode<>(6));
//        root1.getRight().getLeft().setRight(new TreeNode<>(8));
//        root1.getRight().getLeft().getRight().setLeft(new TreeNode<>(1));
//        root1.getRight().getLeft().getRight().setRight(new TreeNode<>(3));
//        Node<Integer> head = new Node<>(4);
//        head.setNext(new Node<>(2));
//        head.getNext().setNext(new Node<>(8));
//        obj.linkedListInBinaryTree(head, root1);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
        System.out.println("Text Justification");
        //https://leetcode.com/problems/text-justification/
        obj.textJustification(new String[]{"This", "is", "an", "example", "of", "text", "justification."}, 16);
        obj.textJustification(new String[]{"What", "must", "be", "acknowledgment", "shall", "be"}, 16);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
        System.out.println("Positions of Large Groups");
        //https://leetcode.com/problems/positions-of-large-groups/
        obj.positionsOfLargeGroups("abbxxxxzzy");
        obj.positionsOfLargeGroups("aaa");
        obj.positionsOfLargeGroups("abc");
        obj.positionsOfLargeGroups("nnnhaaannnm");
        obj.positionsOfLargeGroups_Optimized("abbxxxxzzy");
        obj.positionsOfLargeGroups_Optimized("aaa");
        obj.positionsOfLargeGroups_Optimized("abc");
        obj.positionsOfLargeGroups_Optimized("nnnhaaannnm");
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
        System.out.println("Frog Jump DP Problem");
        //https://www.codingninjas.com/codestudio/problems/frog-jump_3621012?leftPanelTab=0
        obj.frogJump_Recursive_And_Memoization(new int[]{10, 20, 30, 10});
        obj.frogJump_DP_Memoization(new int[]{10, 20, 30, 10});
        obj.frogJump_DP_Memoization_SpaceOptimization(new int[]{10, 20, 30, 10});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
        System.out.println("Ninja’s Training");
        //https://www.codingninjas.com/codestudio/problems/ninja-s-training_3621003?leftPanelTab=0
        obj.ninjaTraining_Recursive_And_Memoization(new int[][]{
            {1, 2, 5}, {3, 1, 1}, {3, 3, 3}});
        obj.ninjaTraining_Recursive_And_Memoization(new int[][]{
            {10, 40, 70}, {20, 50, 80}, {30, 60, 90}});
        obj.ninjaTraining_DP_Memoization(new int[][]{
            {1, 2, 5}, {3, 1, 1}, {3, 3, 3}});
        obj.ninjaTraining_DP_Memoization(new int[][]{
            {10, 40, 70}, {20, 50, 80}, {30, 60, 90}});
        obj.ninjaTraining_DP_Memoization_SpaceOptimization(new int[][]{
            {1, 2, 5}, {3, 1, 1}, {3, 3, 3}});
        obj.ninjaTraining_DP_Memoization_SpaceOptimization(new int[][]{
            {10, 40, 70}, {20, 50, 80}, {30, 60, 90}});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
        System.out.println("Partition Array According to Given Pivot");
        //https://leetcode.com/problems/partition-array-according-to-given-pivot/
        //https://leetcode.com/problems/rearrange-array-elements-by-sign
        obj.partitionArrayOnGivenPivot(new int[]{9, 12, 5, 10, 14, 3, 10}, 10);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
        System.out.println("Print Binary Tree In String Array Format");
        //https://leetcode.com/problems/print-binary-tree/
        TreeNode<Integer> root1 = new TreeNode<>(1);
        root1.setLeft(new TreeNode<>(4));
        root1.getLeft().setRight(new TreeNode<>(2));
        root1.getLeft().getRight().setLeft(new TreeNode<>(1));
        root1.setRight(new TreeNode<>(4));
        root1.getRight().setLeft(new TreeNode<>(2));
        root1.getRight().getLeft().setLeft(new TreeNode<>(6));
        root1.getRight().getLeft().setRight(new TreeNode<>(8));
        root1.getRight().getLeft().getRight().setLeft(new TreeNode<>(1));
        root1.getRight().getLeft().getRight().setRight(new TreeNode<>(3));
        obj.printBinaryTreeInStringMatrixFormat(root1);
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
        System.out.println("Find Eventual Safe States Graph Problem");
        //https://leetcode.com/problems/find-eventual-safe-states/
        obj.findEventualSafeNodes_Graph(new int[][]{{1, 2}, {2, 3}, {5}, {0}, {5}, {}, {}});
        obj.findEventualSafeNodes_Graph(new int[][]{{1, 2, 3, 4}, {1, 2}, {3, 4}, {0, 4}, {}});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
        System.out.println("Number of Provinces");
        //https://leetcode.com/problems/number-of-provinces/
        obj.numberOfProvince_Graph(new int[][]{{1, 1, 0}, {1, 1, 0}, {0, 0, 1}});
        obj.numberOfProvince_Graph(new int[][]{{1, 0, 1}, {0, 1, 0}, {1, 0, 1}});
        obj.numberOfProvince_Graph(new int[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
        //......................................................................
//        Row: SEPARATE QUESTION IMPORTANT
        System.out.println("Distant Barcodes");
        //https://leetcode.com/problems/distant-barcodes/
        obj.distinctBarcodes(new int[]{1, 1, 1, 2, 2, 2});
        obj.distinctBarcodes(new int[]{1, 1, 1, 1, 2, 2, 3, 3});
    }

}
