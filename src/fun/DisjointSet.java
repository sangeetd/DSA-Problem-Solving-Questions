/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

/**
 *
 * @author sangeetdas
 */
public class DisjointSet {

    //https://www.geeksforgeeks.org/disjoint-set-data-structures/
    int[] parent;
    int[] rank;
    int n;

    public DisjointSet(int n) {
        this.n = n;
        parent = new int[n];
        rank = new int[n];
    }

    void makeSet() {
        for (int i = 0; i < n; i++) {
            // Initially, all elements are in
            // their own set.
            parent[i] = i;
        }
    }

    int find(int u) {
        // Finds the representative of the set
        // that x is an element of
        if (parent[u] != u) {
            // if x is not the parent of itself
            // Then x is not the representative of
            // his set,
            parent[u] = find(parent[u]);

            // so we recursively call Find on its parent
            // and move i's node directly under the
            // representative of this set
        }

        return parent[u];
    }

    // Unites the set that includes x and the set
    // that includes x
    void union(int u, int v) {
        // Find representatives of two sets
        int xRoot = find(u);
        int yRoot = find(v);

        // Elements are in the same set, no need
        // to unite anything.
        if (xRoot == yRoot) {
            return;
        }

        // If x's rank is less than y's rank
        if (rank[xRoot] < rank[yRoot]) {
            // Then move x under y  so that depth
            // of tree remains less
            parent[xRoot] = yRoot;
        } // Else if y's rank is less than x's rank
        else if (rank[yRoot] < rank[xRoot]) {
            // Then move y under x so that depth of
            // tree remains less
            parent[yRoot] = xRoot;
        } else {
            // Then move y under x (doesn't matter
            // which one goes where)
            parent[yRoot] = xRoot;

            // And increment the result tree's
            // rank by 1
            rank[xRoot] = rank[xRoot] + 1;
        }
    }
}
