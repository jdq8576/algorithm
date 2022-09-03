package nowcoder;

import java.util.HashMap;
import java.util.Map;

public class LRU {

    private class Node {
        int key, value;
        Node prev, next;

        public Node(int key, int value) {
            this.key = key;
            this.value = value;
            this.prev = null;
            this.next = null;
        }
    }

    private int k;
    private Map<Integer, Node> map;

    private Node head, tail;

    public LRU(int capacity) {
        this.k = capacity;
        this.map = new HashMap<>();
        this.head = new Node(-1, -1);
        this.tail = new Node(-1, -1);
        this.head.next = this.tail;
        this.tail.prev = this.head;
    }

    public int get(int key) {
        if (!map.containsKey(key)) {
            return -1;
        } else {
            Node node = map.get(key);
            removeNode(node);
            // 头插法
            headInsert(node);
            return node.value;
        }
    }

    public void set(int key, int value) {
        if (map.containsKey(key)) {
            Node node = map.get(key);
            node.value = value;
            removeNode(node);
            headInsert(node);
        } else {
            if (this.map.size() == k) {
                Node node = this.tail.prev;
                map.remove(node.key);
                removeNode(node);
            }
            Node node = new Node(key, value);
            headInsert(node);
            map.put(key, node);
        }
    }

    private void removeNode(Node node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }

    private void headInsert(Node node) {
        node.next = this.head.next;
        this.head.next = node;
        node.prev = this.head;
        node.next.prev = node;
    }
}