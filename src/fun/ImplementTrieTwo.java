/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package fun;

import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author sangeetdas
 */
public class ImplementTrieTwo {

    //https://leetcode.com/problems/implement-trie-ii-prefix-tree/description/
    class TrieNode {

        Map<Character, TrieNode> nodes;
        int end;
        int countPrefix;

        public TrieNode() {
            nodes = new HashMap<>();
            end = 0;
            countPrefix = 0;
        }
    }

    TrieNode ROOT;

    public ImplementTrieTwo() {
        ROOT = new TrieNode();
    }

    public void insert(String word) {
        addToTrie(word);
    }

    public int countWordsEqualTo(String word) {
        return countExactWord(word);
    }

    public int countWordsStartingWith(String prefix) {
        return countPrefixMatchedWord(prefix);
    }

    public void erase(String word) {
        eraseWord(word);
    }

    private void addToTrie(String word) {
        TrieNode currRoot = ROOT;
        for (char chr : word.toCharArray()) {

            if (currRoot.nodes.containsKey(chr)) {
                currRoot = currRoot.nodes.get(chr);
                currRoot.countPrefix++;
            } else {
                TrieNode node = new TrieNode();
                node.countPrefix = 1;
                currRoot.nodes.put(chr, node);
                currRoot = node;
            }
        }
        currRoot.end++;
    }

    private int countExactWord(String word) {

        TrieNode currRoot = ROOT;

        for (char chr : word.toCharArray()) {

            if (currRoot.nodes.containsKey(chr)) {
                currRoot = currRoot.nodes.get(chr);
            } else {
                return 0;
            }
        }
        return currRoot.end;
    }

    private int countPrefixMatchedWord(String word) {

        TrieNode currRoot = ROOT;

        for (char chr : word.toCharArray()) {

            if (currRoot.nodes.containsKey(chr)) {
                currRoot = currRoot.nodes.get(chr);
            } else {
                return 0;
            }
        }
        return currRoot.countPrefix;
    }

    private void eraseWord(String word) {

        TrieNode currRoot = ROOT;

        for (char chr : word.toCharArray()) {

            if (currRoot.nodes.containsKey(chr)) {
                currRoot = currRoot.nodes.get(chr);
                currRoot.countPrefix--;
            } else {
                return;
            }
        }

        currRoot.end--;
    }

}
