/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.dsaproblemsolvingquestions.project.fun;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author sangeetdas
 */
public class ImplementIncreamentalStack {
    //https://youtu.be/L8tY9gSfHz4
    //To implement a stack that perfoms basic push pop peek operations in O(1)
    //but also supports increament(index, increaseBy) method in O(1)
    //this method will increase the value of elements in the stack by 'increaseBy' for [0 to index]

    private List<Integer> stack;
    private List<Integer> increament;

    public ImplementIncreamentalStack() {
        this.stack = new ArrayList<>();
        this.increament = new ArrayList<>();
    }

    public void push(int val) {
        stack.add(val);
        //for each added value val, the default increamental value will be 0
        increament.add(0);
    }

    public int pop() {

        if (stack.isEmpty()) {
            return -1;
        }

        //the actual value of stack
        int popped = stack.get(stack.size() - 1);
        stack.remove(stack.size() - 1);

        //any increamental value that needs to be added to our popped stack value
        //if not it will be 0 
        int currSize = increament.size();
        int peekIncreamentalValue = increament.get(currSize - 1);
        increament.remove(currSize - 1);

        if (!increament.isEmpty()) {
            //set the total increamental value for the a previous stack element, if req
            currSize = increament.size();
            increament.set(currSize - 1, increament.get(currSize - 1) + peekIncreamentalValue);
        }

        return popped + peekIncreamentalValue;
    }

    public int peek() {
        int currSize = stack.size();
        //the actual value of stack
        int peek = stack.get(currSize - 1);

        //any increamental value that needs to be added to our popped stack value
        //if not it will be 0 
        currSize = increament.size();
        int peekIncreamentalValue = increament.get(currSize - 1);

        return peek + peekIncreamentalValue;
    }

    public void increament(int index, int increamentBy) {
        increament.set(index, increamentBy);
    }

    public boolean isEmpty() {
        return stack.isEmpty();
    }

    public void print() {
        System.out.println(stack + " <- PEEK");
        System.out.println(increament + " <- PEEK");
    }

}
