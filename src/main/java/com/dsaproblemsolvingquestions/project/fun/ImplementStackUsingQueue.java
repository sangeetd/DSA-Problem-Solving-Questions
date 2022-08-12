/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.dsaproblemsolvingquestions.project.fun;

import java.util.LinkedList;
import java.util.Queue;

/**
 *
 * @author sangeetdas
 */
public class ImplementStackUsingQueue {
    //https://leetcode.com/problems/implement-stack-using-queues/
    //explanation: https://youtu.be/rW4vm0-DLYc
    Queue<Integer> queue;
    
    public ImplementStackUsingQueue() {
        queue = new LinkedList<>();
    }
    
    public void push(int val){
        queue.add(val);
    }
    
    public int pop(){
        int size = queue.size();
        for(int i = 0; i < size - 1; i++){
             //remove first element in queue
            int pop = queue.poll();
            //put them at back of queue
            push(pop);
        }
        //until the very last element
        //remove it
        return queue.remove();
    }
    
    public int peek(){
        int size = queue.size();
        for(int i = 0; i < size - 1; i++){
            //remove first element in queue
            int pop = queue.poll();
            //put them at back of queue
            push(pop);
        }
        //until the very last element
        //remove it first it will be peek
        int peek = queue.remove();
        //add at back this peek to queue
        push(peek);
        return peek;
    }
    
    public boolean isEmpty(){
        return queue.isEmpty();
    }
    
}
