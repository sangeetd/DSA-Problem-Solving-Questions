package fun;

import java.util.Objects;

/**
 *
 * @author sangeetdas
 */
public class Pair<T, V> {
    T key;
    V value;

    public Pair(T key, V value) {
        this.key = key;
        this.value = value;
    }

    public T getKey() {
        return key;
    }

    public V getValue() {
        return value;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true; // Check if both references point to the same object
        if (obj == null || getClass() != obj.getClass()) return false; // Ensure class type matches

        Pair<?, ?> pair = (Pair<?, ?>) obj; // Safe cast to Pair

        // Compare key and value using Objects.equals() to handle nulls
        return Objects.equals(key, pair.key) && Objects.equals(value, pair.value);
    }

    @Override
    public int hashCode() {
        // Generate hash code using Objects.hash() to combine key and value
        return Objects.hash(key, value);
    }
}
