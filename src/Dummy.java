
import java.util.Arrays;
import java.util.stream.IntStream;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author sangeetdas
 */
public class Dummy {

    static void merge(int[] arr, int start, int mid, int end) {

        int[] left = Arrays.copyOfRange(arr, start, mid + 1);
        int[] right = Arrays.copyOfRange(arr, mid + 1, end + 1);

        int i = 0;
        int j = 0;
        int index = start;

        while (i < left.length && j < right.length) {
            if (left[i] <= right[j]) {
                arr[index++] = left[i++];
            } else {
                arr[index++] = right[j++];
            }
        }

        while (i < left.length) {
            arr[index++] = left[i++];
        }

        while (j < right.length) {
            arr[index++] = right[j++];
        }
    }

    static void divide(int[] arr, int start, int end) {

        if (end > start) {

            int mid = start + (end - start) / 2;

            divide(arr, start, mid);
            divide(arr, mid + 1, end);
            merge(arr, start, mid, end);
        }
    }

    static int secondLargest(int[] arr) {

        int n = arr.length;
        int start = 0;
        int end = n - 1;
        divide(arr, start, end);
        return arr[n - 2];
    }

    public static void main(String[] args) {
        int[] arr = {3, 1, 6, 8, 4, 5};

        int sum = IntStream.of(arr).sum();
//        System.out.println(sum);

//        System.out.println(secondLargest(arr));

        
        /*
        If the value p being boxed is an integer literal of type int 
        between -128 and 127 inclusive (§3.10.1), or the boolean literal true or 
        false (§3.10.3), or a character literal between '\u0000' and '\u007f' 
        inclusive (§3.10.4), then let a and b be the results of any two boxing 
        conversions of p. It is always the case that a == b.
        */
//        Integer num1 = 100;
//        Integer num2 = 100;
//        Integer num3 = 500;
//        Integer num4 = 500;
//        if (num1 == num2) {
//            System.out.println("num1 == num2");
//        } else {
//            System.out.println("num1 != num2");
//        }
//        if (num3 == num4) {
//            System.out.println("num3 == num4");
//        } else {
//            System.out.println("num3 != num4");
//        }

//        int odd = 1;
//        if (odd) {
//            System.out.println("odd");
//        } else {
//            System.out.println("even");
//        }

//        method(null);
        
//        try{
//            System.exit(0);
//        }catch(Exception e){
//            System.out.println(e.getMessage());
//        }finally{
//            System.out.println("finally 1");
//        }
        
//        try{
//            throw new Error();
//        }catch(Exception e){
//            System.out.println(e.getMessage());
//        }finally{
//            System.out.println("finally 2");
//        }
//        
//        try{
//            return;
//        }catch(Exception e){
//            System.out.println(e.getMessage());
//        }finally{
//            System.out.println("finally 3");
//        }

    }

    public static void method(Object o) {
        System.out.println("Object method");
    }

    public static void method(String s) {
        System.out.println("String method");
    }

}
