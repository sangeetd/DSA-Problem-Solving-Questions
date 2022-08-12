/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.dsaproblemsolvingquestions.project.fun;

import java.util.Stack;

/**
 *
 * @author sangeetdas
 */
public class BinarySearchTreeIterator<T> {
    //..............T: O(H), at most using addNodes() we are just travelling
    //tree upto the height of tree and not the complete N nodes of tree
    //https://leetcode.com/problems/binary-search-tree-iterator
    Stack<TreeNode<T>> stack;

    public BinarySearchTreeIterator(TreeNode<T> root) {
        this.stack = new Stack<>();
        addNodes(root);
    }

    public boolean hasNext() {
        return !stack.isEmpty();
    }

    public T next() {
        TreeNode<T> currNode = stack.pop();
        addNodes(currNode.getRight());
        return currNode.getData();
    }

    private void addNodes(TreeNode<T> root) {
        TreeNode<T> curr = root;
        while (curr != null) {
            stack.push(curr);
            curr = curr.getLeft();
        }
    }
}
