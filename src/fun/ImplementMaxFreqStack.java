/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package fun;

import java.util.HashMap;
import java.util.Map;
import java.util.Stack;

/**
 *
 * @author sangeetdas
 */
public class ImplementMaxFreqStack {
    //https://leetcode.com/problems/maximum-frequency-stack/
    //https://leetcode.com/problems/maximum-frequency-stack/solution/
    
    Map<Integer, Integer> valFreq;
    Map<Integer, Stack<Integer>> freqStack;
    int maxFreq;

    public ImplementMaxFreqStack() {
        valFreq = new HashMap<>();
        freqStack = new HashMap<>();
        maxFreq = 0;
    }

    public void push(int val) {
        valFreq.put(val, valFreq.getOrDefault(val, 0) + 1);
        int currValFreq = valFreq.get(val);
        maxFreq = Math.max(maxFreq, currValFreq);
        freqStack.putIfAbsent(currValFreq, new Stack<>());
        freqStack.get(currValFreq).push(val);
    }

    public int pop() {
        int val = freqStack.get(maxFreq).pop();
        valFreq.put(val, valFreq.getOrDefault(val, 0) - 1);
        if (freqStack.get(maxFreq).isEmpty()) {
            maxFreq--;
        }
        return val;
    }
}
