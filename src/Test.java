
import java.util.Arrays;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author sangeetdas
 */
public class Test {
    
    
    static void merge(int[] arr, int start, int mid, int end){
        
        int[] leftArr = Arrays.copyOfRange(arr, start, mid + 1);
        int[] rightArr = Arrays.copyOfRange(arr, mid + 1, end + 1);
        
        int i = 0; //leftArr
        int j = 0; //rightArr
        int index = start;
        
        while(i < leftArr.length && j < rightArr.length){
            
            if(leftArr[i] <= rightArr[j]){
                arr[index++] = leftArr[i++];
            }else {
                arr[index++] = rightArr[j++];
            }
        }
        
        while(i < leftArr.length){
            arr[index++] = leftArr[i++];
        }
        
        while(j < rightArr.length){
            arr[index++] = rightArr[j++];
        }
        
    }
    
    static void divide(int[] arr, int start, int end){
        
        if(end >= start){
            
            int mid = start + (end - start) / 2;
            divide(arr, start, mid);
            divide(arr, mid + 1, end);
            merge(arr, start, mid, end);
        }
    }
    
    
    public static void main(String[] args) {
        
        int[] arr = {4,5,6,4,5,5,6,4,4,6};
        int start = 0;
        int end = arr.length - 1;
        
        for(int element : arr){
            System.out.print(element + " ");
        }
        System.out.println();
        
        divide(arr, start, end);
        
        for(int element : arr){
            System.out.print(element + " ");
        }
        System.out.println();
        
    }
    
}
