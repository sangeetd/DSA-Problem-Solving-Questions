/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.dsaproblemsolvingquestions.project.fun;

import java.util.Arrays;

/**
 *
 * @author RAVI
 */
public class MergeSort {

    //https://leetcode.com/problems/sort-an-array/description/
    private static void merge(int[] nums, int start, int mid, int end) {
        int[] left = Arrays.copyOfRange(nums, start, mid + 1);
        int[] right = Arrays.copyOfRange(nums, mid + 1, end + 1);

        int i = 0;
        int j = 0;
        int k = start;

        while (i < left.length && j < right.length) {
            if (left[i] <= right[j]) {
                nums[k++] = left[i++];
            } else {
                nums[k++] = right[j++];
            }
        }

        while (i < left.length) {
            nums[k++] = left[i++];
        }

        while (j < right.length) {
            nums[k++] = right[j++];
        }
    }

    private static void divide(int[] nums, int start, int end) {
        if (start >= end) {
            return;
        }
        int mid = start + (end - start) / 2;
        divide(nums, start, mid);
        divide(nums, mid + 1, end);
        merge(nums, start, mid, end);
    }

    public static void main(String[] args) {

        int[] nums = {9, 4, 3, 6, 7, 1, 2, 11, 5};

        divide(nums, 0, nums.length - 1);

        System.out.println("Merge sort :" + Arrays.toString(nums));

    }

}
