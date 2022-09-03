package nowcoder;

class TireTree {
    class TireNode {
        TireNode[] next;
        int prefixNumber;
        boolean end;

        public TireNode() {
            this.next = new TireNode[26];
            this.prefixNumber = 0;
            this.end = false;
        }
    }

    private TireNode root;

    public TireTree() {
        this.root = new TireNode();
    }

    void insert(String word) {
        TireNode node = this.root;
        for (char c : word.toCharArray()) {
            if (node.next[c - 'a'] == null) {
                node.next[c - 'a'] = new TireNode();
            }
            node = node.next[c - 'a'];
            node.prefixNumber++;
        }
        node.end = true;
    }

    void delete(String word) {
        TireNode node = this.root;
        for (char c : word.toCharArray()) {
            if (node.next[c - 'a'] == null) {
                return;
            }
            node = node.next[c - 'a'];
            node.prefixNumber--;
        }
        if (node.prefixNumber == 0) {
            node.end = false;
        }
    }

    boolean search(String word) {
        TireNode node = this.root;
        for (char c : word.toCharArray()) {
            if (node.next[c - 'a'] == null) {
                return false;
            }
            node = node.next[c - 'a'];
        }
        return node.end;
    }

    int prefixNumber(String word) {
        TireNode node = this.root;
        for (char c : word.toCharArray()) {
            if (node.next[c - 'a'] == null) {
                return 0;
            } else {
                node = node.next[c - 'a'];
            }
        }
        return node.prefixNumber;
    }

}
