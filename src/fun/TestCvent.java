/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package fun;

/**
 *
 * @author sangeetdas
 */
public class TestCvent {
    
    private static int equilibrium(int[] arr){
        int n = arr.length;
        int totalArraySum = 0;
        int prefixSum = 0;
        int[] prefixSumArr = new int[n];
        
        for(int index = 0; index < n; index++){
            prefixSum += arr[index];
            prefixSumArr[index] = prefixSum;
        }
        
        totalArraySum = prefixSum;
        int leftPrefixSum = 0;
        
        for(int index = 0; index < n; index++){
            if(leftPrefixSum == totalArraySum - prefixSumArr[index]) return index;
            leftPrefixSum += arr[index];
        }
        return -1;
    }
    
    
    public static void main(String[] args) {
        
        /*
        Given a sequence arr[] of size n, Write a function int equilibrium(int[] arr,
        int n) that returns an equilibrium index
        (if any) or -1 if no equilibrium index exists.
        
        Input: A[] = {-7, 1, 5, 2, -4, 3, 0} 
        Output: 3 
        
        3 is an equilibrium index, because: 
        A[0] + A[1] + A[2] = A[4] + A[5] + A[6]
        
        {-7, 1, 5, 2, -4, 3, 0}
        prefixSum[] = [-7, -6, -1, 1, -3, 0, 0]
        totalArraySum = prefixSum[n - 1];
        
        leftPrefixSum = 0;
        index = 0
        
        loop
        leftPrefixSum += arr[index]
        if(leftPrefixSum == totalArraySum - prefixSum[index]) return index;
        
        */
        
        System.out.println(equilibrium(new int[]{-7, 1, 5, 2, -4, 3, 0}));
        
        
    }
    
}
