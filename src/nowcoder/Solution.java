package nowcoder;


import java.util.*;

public class Solution {

    public int lowestCommonAncestor1(TreeNode root, int p, int q) {
        if (p > q) {
            int c = p;
            p = q;
            q = c;
        }
        if (root.val >= p && root.val <= q) {
            return root.val;
        } else if (root.val >= q) {
            return lowestCommonAncestor1(root.left, p, q);
        } else {
            return lowestCommonAncestor1(root.right, p, q);
        }
    }

    public int lowestCommonAncestor(TreeNode root, int o1, int o2) {
        final TreeNode treeNode = lowestCommonAncestorHelper(root, o1, o2);
        return treeNode.val;
    }

    private TreeNode lowestCommonAncestorHelper(TreeNode root, int o1, int o2) {
        if (root == null || root.val == o1 || root.val == o2) {
            return root;
        } else {
            TreeNode p1 = lowestCommonAncestorHelper(root.left, o1, o2);
            TreeNode p2 = lowestCommonAncestorHelper(root.right, o1, o2);
            if (p1 != null && p2 != null) {
                return root;
            } else if (p1 != null && p2 == null) {
                return p1;
            } else if (p1 == null && p2 != null) {
                return p2;
            } else {
                return null;
            }
        }
    }

    private TreeNode preNode = null;

    public TreeNode Convert(TreeNode pRootOfTree) {
        if (pRootOfTree == null) {
            return null;
        }
        TreeNode node = getLeft(pRootOfTree);
        inOrder(pRootOfTree);
        return null;
    }

    private void inOrder(TreeNode pRootOfTree) {
        if (pRootOfTree.left != null) {
            inOrder(pRootOfTree.left);
        }
        pRootOfTree.left = preNode;
        if (preNode != null) {
            preNode.right = pRootOfTree;
        }
        preNode = pRootOfTree;
        if (pRootOfTree.right != null) {
            inOrder(pRootOfTree.right);
        }
    }

    private TreeNode getLeft(TreeNode pRootOfTree) {
        TreeNode node = pRootOfTree;
        while (node.left != null) {
            node = node.left;
        }
        return node;
    }

    public int getLongestPalindrome(String A) {
        int len = 0;
        final int length = A.length();
        final char[] chars = A.toCharArray();
        boolean[][] dp = new boolean[A.length()][A.length()];
        for (int i = length - 1; i >= 0; i--) {
            for (int l = 0; i + l < length; l++) {
                int j = i + l;
                if (l == 0) {
                    dp[i][j] = true;
                } else if (l == 1) {
                    dp[i][j] = (chars[i] == chars[j]);
                } else {
                    dp[i][j] = (chars[i] == chars[j]) && dp[i + 1][j - 1];
                }
                if (dp[i][j] && j - i + 1 > len) {
                    len = j - i + 1;
                }
            }
        }
        return len;
    }

    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
        if (pHead1 == null || pHead2 == null) {
            return null;
        }
        ListNode f1 = pHead1;
        ListNode f2 = pHead2;
        boolean t1 = false, t2 = false;
        while (f1 != f2) {
            f1 = f1.next;
            if (f1 == null) {
                if (!t1) {
                    t1 = true;
                    f1 = pHead2;
                } else {
                    return null;
                }
            }
            f2 = f2.next;
            if (f2 == null) {
                if (!t2) {
                    t2 = true;
                    f2 = pHead1;
                } else {
                    return null;
                }
            }
        }
        return f1;
    }

    public ListNode EntryNodeOfLoop(ListNode pHead) {
        if (pHead.next == null) {
            return null;
        }
        ListNode slow = pHead, fast = pHead;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (slow == fast) {
                break;
            }
        }
        if (fast == null) {
            return null;
        }
        fast = pHead;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }

    public int maxProfit1(int[] prices) {
        int res = 0;
        int minVal = prices[0];
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] < minVal) {
                minVal = prices[i];
            }
            if (res < prices[i] - minVal) {
                res = prices[i] - minVal;
            }
        }
        return res;
    }

    public int StrToInt(String s) {
        boolean flag = true;
        final char[] chars = s.toCharArray();
        int index = 0;
        while (index < s.length() && chars[index] == ' ') {
            index++;
        }
        if (index == s.length()) {
            return 0;
        }
        // 第一个非空格
        if (chars[index] == '+') {
            flag = true;
            index++;
        } else if (chars[index] == '-') {
            flag = false;
            index++;
        } else if (!(chars[index] >= '0' && chars[index] <= '9')) {
            return 0;
        }
        long sum = 0;
        while (index < s.length() && (chars[index] >= '0' && chars[index] <= '9')) {
            sum = sum * 10 + chars[index] - '0';
            index++;
            if (flag && sum >= Integer.MAX_VALUE) {
                return Integer.MAX_VALUE;
            } else if (!flag && (sum * -1) < Integer.MIN_VALUE) {
                return Integer.MIN_VALUE;
            }
        }
        if (flag) {
            return (int) (sum);
        } else {
            return (int) (sum * -1);
        }
    }

    public long maxWater(int[] arr) {
        long sum = 0;
        int len = arr.length - 1;
        int[] left = new int[arr.length];
        int[] right = new int[arr.length];
        left[0] = arr[0];
        right[len] = arr[len];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] < left[i - 1]) {
                left[i] = left[i - 1];
            } else {
                left[i] = arr[i];
            }
        }
        for (int i = len - 1; i >= 0; i--) {
            if (arr[i] < right[i + 1]) {
                right[i] = right[i + 1];
            } else {
                right[i] = arr[i];
            }
        }
        for (int i = 0; i <= len; i++) {
            sum = sum + Math.min(left[i], right[i]) - arr[i];
        }
        return sum;
    }

    public String LCS(String s1, String s2) {
        int[][] dp = new int[2001][2001];
        dp[0][0] = 0;
        // 行 s1 列 s2
        for (int i = 1; i <= s1.length(); i++) {
            dp[i][0] = 0;
        }
        for (int i = 1; i <= s2.length(); i++) {
            dp[0][i] = 0;
        }
        for (int i = 1; i <= s1.length(); i++) {
            for (int j = 1; j <= s2.length(); j++) {
                if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        if (dp[s1.length()][s2.length()] == 0) {
            return "-1";
        }
        char[] arr = new char[dp[s1.length()][s2.length()]];
        int index = dp[s1.length()][s2.length()] - 1;
        int j = s2.length(), i = s1.length();
        while (dp[i][j] >= 1) {
            if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                arr[index--] = s1.charAt(i - 1);
                j--;
                i--;
            } else if (dp[i][j - 1] > dp[i - 1][j]) {
                j--;
            } else {
                i--;
            }
        }
        return new String(arr);
    }

    public int maxProfit2(int[] prices) {
        int ans = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > prices[i - 1]) {
                ans = prices[i] + prices[i - 1];
            }
        }
        return ans;
    }

    public int maxProfit3(int[] prices) {
        int fstBuy = Integer.MIN_VALUE, fstSell = 0;
        int secBuy = Integer.MIN_VALUE, secSell = 0;
        for (int price : prices) {
            fstBuy = Math.max(fstBuy, -1 * price);
            fstSell = Math.max(fstSell, fstBuy + price);
            secBuy = Math.max(secBuy, fstSell - price);
            secSell = Math.max(secSell, price - secBuy);
        }
        return secSell;
    }

    public int maxProfit4(int[] prices) {
        int minVal = prices[0];
        int ans = 0;
        for (int i = 1; i < prices.length; i++) {
            if (ans < prices[i] - minVal) {
                ans = prices[i] - minVal;
            }
            if (prices[i] < minVal) {
                minVal = prices[i];
            }
        }
        return ans;
    }

    public int maxProfit5(int[] prices) {
        int ans = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > prices[i - 1]) {
                ans = ans + prices[i] - prices[i - 1];
            }
        }
        return ans;
    }

    public int maxProfit6(int[] prices, int k) {
        int n = prices.length;
        if (n <= 1) {
            return 0;
        }
        k = Math.min(k, prices.length / 2);
        int[][][] dp = new int[prices.length][k + 1][2];
        for (int i = 1; i <= k; i++) {
            dp[0][i][0] = 0;
            dp[0][i][1] = -prices[0];
        }
        for (int i = 1; i < prices.length; i++) {
            for (int j = 1; j <= k; j++) {
                // 0 表示不拥有股票 1 表示拥有股票
                dp[i][j][0] = Math.max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i]);
                dp[i][j][1] = Math.max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i]);
            }
        }
        return dp[n - 1][k][0];
    }

    public int maxProfit7(int[] prices) {
        if (prices.length <= 1) {
            return 0;
        }
        int k = 2;
        if (prices.length < 4) {
            k = 1;
        }
        int[][][] dp = new int[prices.length][k + 1][2];
        for (int i = 1; i <= k; i++) {
            dp[0][i][0] = 0;
            dp[0][i][1] = -prices[0];
        }
        for (int i = 1; i < prices.length; i++) {
            for (int j = 1; j <= k; j++) {
                dp[i][j][0] = Math.max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i]);
                dp[i][j][1] = Math.max(dp[i - 1][j - 1][0] - prices[i], dp[i - 1][j][1]);
            }
        }
        return dp[prices.length - 1][k][0];
    }

    public int maxProfit8(int[] prices, int k) {
        int len = prices.length;
        if (len < 2) {
            return 0;
        }
        int[][][] dp = new int[len][k + 1][2];
        for (int i = 1; i <= k; i++) {
            dp[0][i][0] = 0;
            dp[0][i][1] = -prices[0];
        }
        for (int i = 1; i < len; i++) {
            for (int j = 1; j <= k; j++) {
                dp[i][j][0] = Math.max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i]);
                dp[i][j][1] = Math.max(dp[i - 1][j - 1][0] - prices[i], dp[i - 1][j][1]);
            }
        }
        return dp[len - 1][k][0];
    }

    public String[] trieU(String[][] operators) {
        List<String> list = new ArrayList<>();
        TireTree tireTree = new TireTree();
        for (String[] operator : operators) {
            String oper = operator[0];
            String word = operator[1];
            if (oper.equals("1")) {
                tireTree.insert(word);
            } else if (oper.equals("2")) {
                tireTree.delete(word);
            } else if (oper.equals("3")) {
                final boolean search = tireTree.search(word);
                if (search) {
                    list.add("YES");
                } else {
                    list.add("NO");
                }
            } else if (oper.equals("4")) {
                final int num = tireTree.prefixNumber(word);
                list.add(String.valueOf(num));
            }
        }
        String[] strings = new String[list.size()];
        for (int i = 0; i < list.size(); i++) {
            strings[i] = list.get(i);
        }
        return strings;
    }

    public int GetUglyNumber_Solution(int index) {
        if (index <= 1) {
            return index;
        }
        ArrayList<Integer> list = new ArrayList<>();
        list.add(1);
        int index2 = 0, index3 = 0, index5 = 0;
        int turn = 2;
        while (turn <= index) {
            turn++;
            int val = list.get(index2) * 2;
            val = Math.min(val, list.get(index3) * 3);
            val = Math.min(val, list.get(index5) * 5);
            if (val == list.get(index2) * 2) {
                index2++;
            }
            if (val == list.get(index3) * 3) {
                index3++;
            }
            if (val == list.get(index5) * 5) {
                index5++;
            }
            list.add(val);
        }
        return list.get(list.size() - 1);
    }

    public ArrayList<ArrayList<Integer>> threeSum(int[] num) {
        Arrays.sort(num);
        ArrayList<ArrayList<Integer>> list = new ArrayList<>();
        for (int i = 0; i < num.length; i++) {
            // 去重
            if (i != 0 && num[i] == num[i - 1]) {
                continue;
            }
            int l = i + 1;
            int r = num.length - 1;
            int sum = 0 - num[i];
            while (l < r) {
                if (num[l] + num[r] == sum) {
                    ArrayList<Integer> list1 = new ArrayList<>();
                    list1.add(num[i]);
                    list1.add(num[l]);
                    list1.add(num[r]);
                    list.add(list1);
                    while (l + 1 < num.length && num[l] == num[l + 1]) {
                        l++;
                    }
                    while (r - 1 >= 0 && num[r] == num[r - 1]) {
                        r--;
                    }
                    l++;
                    r--;
                } else if (num[l] + num[r] > sum) {
                    r--;
                } else {
                    l++;
                }
            }
        }
        return list;
    }

    public int InversePairs(int[] array) {
        return mergeSort(array, 0, array.length - 1);
    }

    private int mergeSort(int[] array, int l, int r) {
        if (l < r) {
            int mid = l + (r - l) / 2;
            final int i1 = mergeSort(array, l, mid);
            final int i2 = mergeSort(array, mid + 1, r);
            int sum = (i1 + i2) % 1000000007;
            int[] arr = new int[r - l + 1];
            int index = 0;
            int l1 = l, r1 = mid, l2 = mid + 1, r2 = r;
            while (l1 <= r1 || l2 <= r2) {
                if (l1 > r1) {
                    arr[index++] = array[l2++];
                } else if (l2 > r2) {
                    sum = sum + r2 - (mid + 1) + 1;
                    arr[index++] = array[l1++];
                } else if (array[l1] <= array[l2]) {
                    sum = sum + l2 - (mid + 1);
                    arr[index++] = array[l1++];
                } else {
                    arr[index++] = array[l2++];
                }
            }
            for (int i = 0; i < index; i++) {
                array[l + i] = arr[i];
            }
            return sum % 1000000007;
        } else {
            return 0;
        }
    }

    public int maxProduct(int[] nums) {
        int ans = nums[0];
        int[] minDp = new int[nums.length];
        int[] maxDp = new int[nums.length];
        minDp[0] = nums[0];
        maxDp[0] = nums[0];
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] >= 0) {
                maxDp[i] = Math.max(maxDp[i - 1] * nums[i], nums[i]);
                minDp[i] = Math.min(minDp[i - 1] * nums[i], nums[i]);
            } else {
                maxDp[i] = Math.max(minDp[i - 1] * nums[i], nums[i]);
                minDp[i] = Math.min(maxDp[i - 1] * nums[i], nums[i]);
            }
            ans = Math.max(ans, maxDp[i]);
        }
        return ans;
    }

    public Interval[] insertInterval(Interval[] intervals, Interval newInterval) {
        ArrayList<Interval> list = new ArrayList<>();
        for (Interval interval : intervals) {
            list.add(interval);
        }
        list.add(newInterval);
        Collections.sort(list, new Comparator<Interval>() {
            @Override
            public int compare(Interval o1, Interval o2) {
                if (o1.start != o2.start) {
                    return o1.start - o2.start;
                } else {
                    return o1.end - o2.end;
                }
            }
        });
        ArrayList<Interval> list1 = new ArrayList<>();
        list1.add(list.get(0));
        int preS = list.get(0).start;
        int preE = list.get(0).end;
        for (int i = 1; i < list.size(); i++) {
            int s = list.get(i).start;
            int e = list.get(i).end;
            if (preE < s) {
                preE = e;
                preS = s;
                list1.add(list.get(i));
            } else {
                if ((s >= preS && e >= preE)) {
                    list1.remove(list1.size() - 1);
                    preE = e;
                    list1.add(new Interval(preS, preE));
                }
            }
        }
        Interval[] intervals1 = new Interval[list1.size()];
        for (int i = 0; i < intervals1.length; i++) {
            intervals1[i] = list1.get(i);
        }
        return intervals1;
    }

    public int maxValue(int[][] grid) {
        int[][] dp = new int[grid.length][grid[0].length];
        dp[0][0] = grid[0][0];
        for (int i = 1; i < grid.length; i++) {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }
        for (int i = 1; i < grid[0].length; i++) {
            dp[0][i] = dp[0][i - 1] + grid[0][i];
        }
        for (int i = 1; i < grid.length; i++) {
            for (int j = 1; j < grid[i].length; j++) {
                dp[i][j] = grid[i][j] + Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
        return dp[grid.length - 1][grid[0].length - 1];
    }

    public int cuttingRope(int n) {
        if (n == 2) {
            return 1;
        } else if (n == 3) {
            return 2;
        } else {
            long ans = 1;
            if (n % 3 == 1) {
                for (int i = 1; i < n / 3; i++) {
                    ans = ans * 3;
                    ans = ans % 1000000007;
                }
                ans = ans * 4;
                ans = ans % 1000000007;
            } else if (n % 3 == 2) {
                for (int i = 1; i <= n / 3; i++) {
                    ans = ans * 3;
                    ans = ans % 1000000007;
                }
                ans = ans * 2;
                ans = ans % 1000000007;
            } else {
                for (int i = 1; i <= n / 3; i++) {
                    ans = ans * 3;
                    ans = ans % 1000000007;
                }
            }
            return (int) ans;
        }
    }

    public int[] getLeastNumbers(int[] arr, int k) {
        for (int i = (k - 2) / 2; i >= 0; i--) {
            buildHeap(arr, i, k - 1);
        }
        for (int i = k; i < arr.length; i++) {
            if (arr[i] < arr[0]) {
                arr[0] = arr[i];
                buildHeap(arr, 0, k - 1);
            }
        }
        int[] res = new int[k];
        for (int i = 0; i < k; i++) {
            res[i] = arr[i];
        }
        return res;
    }

    private void buildHeap(int[] arr, int index, int len) {
        while (index * 2 + 1 <= len) {
            int i = index * 2 + 1;
            if (i + 1 <= len && arr[i + 1] > arr[i]) {
                i++;
            }
            if (arr[i] > arr[index]) {
                int t = arr[i];
                arr[i] = arr[index];
                arr[index] = t;
                index = i;
            } else {
                break;
            }
        }
    }

    public int singleNumber1(int[] nums) {
        long ans = 0;
        for (int num : nums) {
            ans = ans ^ num;
        }
        return (int) ans;
    }

    public int singleNumber(int[] nums) {
        int[] t = new int[32];
        int ans = 0;
        for (int i = 0; i < 32; i++) {
            t[i] = 0;
            for (int num : nums) {
                if (((num >> i) & 1) == 0) {
                    t[i]++;
                }
            }
        }
        for (int i = 0; i < 32; i++) {
            if (t[i] % 3 == 1) {
                ans = ans + (1 << i);
            }
        }
        return ans;
    }

    public boolean isSubsequence(String T, String S) {
        int i = 0, j = 0;
        while (i < S.length()) {
            if (S.charAt(i) == T.charAt(j)) {
                i++;
                j++;
            } else {
                i++;
            }
            if (j == T.length()) {
                return true;
            }
        }
        return false;
    }

    public String[] repeatedDNA(String DNA) {
        ArrayList<String> list = new ArrayList<>();
        Map<String, Integer> map = new LinkedHashMap<>();
        for (int i = 0; i <= DNA.length() - 10; i++) {
            String s = DNA.substring(i, i + 10);
            map.put(s, map.getOrDefault(s, 0) + 1);
        }
        for (String s : map.keySet()) {
            if (map.get(s) >= 2) {
                list.add(s);
            }
        }
        return list.toArray(new String[]{});
    }

    public int minNumberDisappeared1(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] <= 0) {
                nums[i] = n + 1;
            }
        }
        for (int i = 0; i < nums.length; i++) {
            if (Math.abs(nums[i]) <= n) {
                nums[Math.abs(nums[i]) - 1] = -1 * Math.abs(nums[Math.abs(nums[i]) - 1]);
            }
        }
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > 0) {
                return i + 1;
            }
        }
        return n + 1;
    }

    public int minNumberDisappeared(int[] nums) {
        HashSet<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }
        int j = 1;
        while (set.contains(j)) {
            j++;
        }
        return j;
    }

    public int maxLength(int[] arr) {
        HashMap<Integer, Integer> map = new HashMap<>();
        int ans = 1;
        map.put(arr[0], 1);
        int l = 0, r = 1;
        while (r < arr.length) {
            while (map.containsKey(arr[r])) {
                map.remove(arr[l++]);
            }
            map.put(arr[r], 1);
            r++;
            if (r - l + 1 > ans) {
                ans = r - l + 1;
            }
        }
        return ans;
    }

    public ListNode mergeKLists(ArrayList<ListNode> lists) {
        if (lists.size() == 0) {
            return null;
        }
        return mergeKListsHelper(lists, 0, lists.size() - 1);
    }

    private ListNode mergeKListsHelper(ArrayList<ListNode> lists, int l, int r) {
        if (l == r) {
            return lists.get(l);
        } else {
            int mid = l + (r - l) / 2;
            return mergeTwoLists(mergeKListsHelper(lists, l, mid), mergeKListsHelper(lists, mid + 1, r));
        }
    }

    private ListNode mergeTwoLists(ListNode node1, ListNode node2) {
        ListNode l1 = node1, l2 = node2;
        ListNode head = new ListNode(0);
        ListNode tail = head;
        while (l1 != null || l2 != null) {
            if (l1 == null) {
                ListNode t = l2.next;
                tail.next = l2;
                tail = tail.next;
                l2 = t;
            } else if (l2 == null) {
                ListNode t = l1.next;
                tail.next = l1;
                tail = tail.next;
                l1 = t;
            } else if (l1.val <= l2.val) {
                ListNode t = l1.next;
                tail.next = l1;
                tail = tail.next;
                l1 = t;
            } else {
                ListNode t = l2.next;
                tail.next = l2;
                tail = tail.next;
                l2 = t;
            }
        }
        return head.next;
    }

    public int minTrace(int[][] triangle) {
        for (int i = triangle.length - 2; i >= 0; i--) {
            for (int j = 0; j <= i; j++) {
                triangle[i][j] += Math.min(triangle[i + 1][j], triangle[i + 1][j + 1]);
            }
        }
        return triangle[0][0];
    }

    public ArrayList<ArrayList<Integer>> subsets1(int[] nums) {
        Arrays.sort(nums);
        Set<ArrayList<Integer>> set = new HashSet<>();
        ArrayList<ArrayList<Integer>> list = new ArrayList<>();
        list.add(new ArrayList<>());
        set.add(new ArrayList<>());
        for (int num : nums) {
            final int size = list.size();
            for (int i = 0; i < size; i++) {
                ArrayList<Integer> temp = new ArrayList<>(list.get(i));
                temp.add(num);
                list.add(temp);
                set.add(temp);
            }
        }
        list.clear();
        for (ArrayList<Integer> integers : set) {
            list.add(integers);
        }
        Collections.sort(list, new Comparator<ArrayList<Integer>>() {
            @Override
            public int compare(ArrayList<Integer> o1, ArrayList<Integer> o2) {
                int i = 0, j = 0;
                while (i < o1.size() && j < o2.size()) {
                    if (o1.get(i) == o2.get(j)) {
                        i++;
                        j++;
                    } else {
                        return o1.get(i) - o2.get(j);
                    }
                }
                if (i == o1.size()) {
                    return -1;
                } else {
                    return 1;
                }
            }
        });
        return list;
    }

    public ArrayList<ArrayList<Integer>> subsets(int[] nums) {
        ArrayList<ArrayList<Integer>> list = new ArrayList<>();
        ArrayList<Integer> subset = new ArrayList<>();
        if (nums.length == 0) {
            return new ArrayList<>();
        }
        Arrays.sort(nums);
        int[] vis = new int[nums.length];
        backTrace(nums, 0, list, subset, vis);
        return list;
    }

    private void backTrace(int[] nums, int start, ArrayList<ArrayList<Integer>> list, ArrayList<Integer> subset, int[] vis) {
        list.add(new ArrayList<>(subset));
        for (int i = start; i < nums.length; i++) {
            if (i > 0 && nums[i] == nums[i - 1] && vis[i - 1] == 0) {
                continue;
            }
            vis[i] = 1;
            subset.add(nums[i]);
            backTrace(nums, i + 1, list, subset, vis);
            subset.remove(subset.size() - 1);
            vis[i] = 0;
        }
    }

    public int gasStation(int[] gas, int[] cost) {
        int l = 0, r = 1;
        int last = gas[0] - cost[0];
        while (true) {
            while (last < 0 && l != r) {
                last = last - gas[l] + cost[l];
                l = (l + 1) % cost.length;
                if (l == 0) {
                    return -1;
                }
            }
            last = last + gas[r] - cost[r];
            r = (r + 1) % cost.length;
            if (r == l) {
                return l;
            }
        }
    }

    public ArrayList<ArrayList<Integer>> combinationCount(int target, int[] nums) {
        ArrayList<ArrayList<Integer>> ans = new ArrayList<>();
        ArrayList<Integer> path = new ArrayList<>();
        Arrays.sort(nums);
        if (nums.length == 0) {
            return ans;
        }
        helper(path, nums, target, 0, 0, ans);
        return ans;
    }

    private void helper(ArrayList<Integer> path, int[] nums, int target, int start, int sum, ArrayList<ArrayList<Integer>> ans) {
        if (sum == target) {
            ans.add(new ArrayList<>(path));
        } else {
            for (int i = start; i < nums.length; i++) {
                if (sum + nums[i] > target) {
                    break;
                }
                sum += nums[i];
                path.add(nums[i]);
                helper(path, nums, target, i, sum, ans);
                sum -= nums[i];
                path.remove(path.size() - 1);
            }
        }
    }

    public int rob1(int[] nums) {
        if (nums.length == 1) {
            return nums[0];
        } else if (nums.length == 2) {
            return Math.max(nums[0], nums[1]);
        } else {
            int[] dp = new int[nums.length];
            dp[0] = nums[0];
            dp[1] = Math.max(nums[0], nums[1]);
            for (int i = 2; i < nums.length; i++) {
                dp[i] = Math.max(dp[i - 1], nums[i] + dp[i - 2]);
            }
            return dp[nums.length - 1];
        }
    }

    public int[] MySort(int[] arr) {
        quickSort(arr, 0, arr.length - 1);
        return arr;
    }

    private void quickSort(int[] arr, int l, int r) {
        if (l < r) {
            int index = quickSortHelper(arr, l, r);
            quickSort(arr, l, index - 1);
            quickSort(arr, index + 1, r);
        }
    }

    private int quickSortHelper(int[] arr, int l, int r) {
        int pivot = arr[l];
        while (l < r) {
            while (l < r && arr[r] >= pivot) {
                r--;
            }
            if (l < r) {
                arr[l] = arr[r];
            }
            while (l < r && arr[l] < pivot) {
                l++;
            }
            if (l < r) {
                arr[r] = arr[l];
            }
        }
        arr[l] = pivot;
        return l;
    }

    public int[] FindGreatestSumOfSubArray(int[] array) {
        // l是起点  r是终点
        int ll = 0, rr = 0;
        int ss = array[0];
        int l = 0, r = 0, s = array[0];
        for (int i = 1; i < array.length; i++) {
            if (s < 0) {
                s = array[i];
                l = i;
            } else {
                s += array[i];
            }
            r = i;
            if (s > ss || ((r - l) > (rr - ll) && s == ss)) {
                ss = s;
                ll = l;
                rr = r;
            }
        }
        return Arrays.copyOfRange(array, ll, rr + 1);
    }

    public int rob(int[] nums) {
        int ans = 0;
        int[] dp = new int[nums.length];
        // 不选第一个
        dp[0] = 0;
        for (int i = 1; i < nums.length; i++) {
            if (i == 1) {
                dp[i] = nums[i];
            } else {
                dp[i] = Math.max(nums[i] + dp[i - 2], dp[i - 1]);
            }
            if (ans < dp[i]) {
                ans = dp[i];
            }
        }
        dp[0] = nums[0];
        // 不选最后一个
        for (int i = 1; i < nums.length - 1; i++) {
            if (i == 1) {
                dp[i] = Math.max(dp[i - 1], nums[i]);
            } else {
                dp[i] = Math.max(nums[i] + dp[i - 2], dp[i - 1]);
            }
            if (ans < dp[i]) {
                ans = dp[i];
            }
        }
        return ans;
    }

    public int[] finalPrices(int[] prices) {
        for (int i = 0; i < prices.length; i++) {
            for (int j = i + 1; j < prices.length; j++) {
                if (prices[j] < prices[i]) {
                    prices[i] -= prices[j];
                    break;
                }
            }
        }
        return prices;
    }

    boolean isSymmetricalHelper(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        } else if (left == null || right == null) {
            return false;
        } else {
            if (left.val != right.val) {
                return false;
            }
            return isSymmetricalHelper(left.right, right.left) && isSymmetricalHelper(left.left, right.right);
        }
    }

    boolean isSymmetrical(TreeNode pRoot) {
        if (pRoot == null) {
            return true;
        }
        return isSymmetricalHelper(pRoot.left, pRoot.right);
    }

    private ArrayList<ArrayList<Integer>> res_1106 = new ArrayList<>();

    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root, int expectNumber) {
        ArrayList<Integer> list = new ArrayList<>();
        if (root != null) {
            list.add(root.val);
            int sum = root.val;
            FindPathHelper(root, expectNumber, list, sum);
        }
        return res_1106;
    }

    private void FindPathHelper(TreeNode root, int expectNumber, ArrayList<Integer> list, int sum) {
        if (root.left == null && root.right == null) {
            if (sum == expectNumber) {
                res_1106.add(list);
            }
        } else {
            if (root.right != null) {
                final ArrayList<Integer> list1 = new ArrayList<>(list);
                list1.add(root.right.val);
                FindPathHelper(root.right, expectNumber, list1, sum + root.right.val);
            }
            if (root.left != null) {
                final ArrayList<Integer> list2 = new ArrayList<>(list);
                list2.add(root.left.val);
                FindPathHelper(root.left, expectNumber, list2, sum + root.left.val);
            }
        }
    }

    private int getNodes(TreeNode node) {
        return node == null ? 0 : 1 + getNodes(node.left) + getNodes(node.right);
    }

    public int KthNode(TreeNode root, int k) {
        if (root == null) {
            return -1;
        }
        int val = getNodes(root.left);
        if (val >= k) {
            return KthNode(root.left, k);
        } else if (val + 1 == k) {
            return root.val;
        } else {
            k = k - val - 1;
            return KthNode(root.right, k);
        }
    }

    public TreeNode buildTree(int[] inorder, int[] postorder) {
        if (postorder.length == 0) {
            return null;
        }
        final int last = postorder.length - 1;
        TreeNode root = new TreeNode(postorder[last]);
        int index = -1;
        int val = postorder[last];
        for (int i = 0; i < inorder.length; i++) {
            if (inorder[i] == val) {
                index = i;
                break;
            }
        }
        if (index != 0) {
            // 有左子树
            root.left = buildTree(Arrays.copyOfRange(inorder, 0, index), Arrays.copyOfRange(postorder, 0, index));
        }
        if (index + 1 != inorder.length) {
            // 有右子树
            root.right = buildTree(Arrays.copyOfRange(inorder, index + 1, inorder.length), Arrays.copyOfRange(postorder, index, last));
        }
        return root;
    }

    public int[][] levelOrderBottom(TreeNode root) {
        Queue<TreeNode> queue = new ArrayDeque<>();
        Stack<ArrayList<Integer>> stack = new Stack<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            ArrayList<Integer> list = new ArrayList<>();
            final int size = queue.size();
            for (int i = 0; i < size; i++) {
                final TreeNode node = queue.poll();
                list.add(node.val);
                if (node.left != null) {
                    queue.add(node.left);
                }
                if (node.right != null) {
                    queue.add(node.right);
                }
            }
            stack.add(list);
        }
        int[][] res = new int[stack.size()][];
        int index = 0;
        while (!stack.isEmpty()) {
            final ArrayList<Integer> list = stack.pop();
            res[index] = new int[list.size()];
            for (int i = 0; i < list.size(); i++) {
                res[index][i] = list.get(i);
            }
            index++;
        }
        return res;
    }

    public int countBitDiff(int m, int n) {
        int c = m ^ n;
        int count = 0;
        while (c != 0) {
            count = count + c & 1;
            c = c >> 1;
        }
        return count;
    }

    public boolean isUnique(String str) {
        Set<Character> set = new HashSet<>();
        for (char c : str.toCharArray()) {
            if (set.contains(c)) {
                return false;
            } else {
                set.add(c);
            }
        }
        return true;
    }

    public boolean NimGame(int n) {
        return n % 4 != 0;
    }

    private int[][] dir_1108 = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    public char[][] surroundedArea(char[][] board) {
        int r = board.length - 1;
        int c = board[0].length - 1;

        // change A
        for (int i = 0; i <= c; i++) {
            if (board[0][i] == 'O') {
                changeA(i, 0, board);
            }
            if (board[r][i] == 'O') {
                changeA(i, r, board);
            }
        }
        for (int i = 0; i <= r; i++) {
            if (board[i][0] == 'O') {
                changeA(0, i, board);
            }
            if (board[i][c] == 'O') {
                changeA(c, i, board);
            }
        }

        for (int i = 0; i <= r; i++) {
            for (int j = 0; j <= c; j++) {
                if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                }
                if (board[i][j] == 'A') {
                    board[i][j] = 'O';
                }
            }
        }

        return board;
    }

    private void changeA(int c, int r, char[][] board) {
        board[r][c] = 'A';
        for (int i = 0; i < dir_1108.length; i++) {
            int cc = c + dir_1108[i][0];
            int rr = r + dir_1108[i][1];
            if (cc < 0 || cc >= board[0].length) {
                continue;
            }
            if (rr < 0 || rr >= board.length) {
                continue;
            }
            if (board[rr][cc] == 'A' || board[rr][cc] == 'X') {
                continue;
            }
            changeA(cc, rr, board);
        }
    }

    public int run(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int l = run(root.left);
        int r = run(root.right);
        if (l == 0 && r == 0) {
            return 1;
        } else if (l == 0) {
            return r + 1;
        } else if (r == 0) {
            return l + 1;
        } else {
            return l < r ? l + 1 : r + 1;
        }
    }

    private int res_1109 = 0;

    public int combination(int[] nums, int target) {
        process(nums, target);
        return res_1109;
    }

    private void process(int[] nums, int reminder) {
        if (reminder == 0) {
            res_1109++;
            return;
        }
        if (reminder < 0) {
            return;
        }
        for (int num : nums) {
            reminder -= num;
            process(nums, reminder);
            reminder += num;
        }
    }

    private ArrayList<ArrayList> list_1109 = new ArrayList<>();

    private void dfs(int[] arr, ArrayList<Integer> list, int size, int target, int sum, int index) {
        if (list.size() == size) {
            if (sum == target) {
                list_1109.add(new ArrayList(list));
            }
            return;
        }
        for (int i = index; i < arr.length; i++) {
            list.add(arr[i]);
            sum = sum + arr[i];
            dfs(arr, list, size, target, sum, i + 1);
            list.remove(list.size() - 1);
            sum = sum - arr[i];
        }
    }

    public int[][] combination(int k, int n) {
        int[] arr = new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
        ArrayList<Integer> list = new ArrayList<>();
        dfs(arr, list, k, n, 0, 0);
        int[][] res = new int[list_1109.size()][];
        for (int i = 0; i < res.length; i++) {
            final ArrayList arrayList = list_1109.get(i);
            res[i] = new int[arrayList.size()];
            for (int j = 0; j < arrayList.size(); j++) {
                res[i][j] = (int) arrayList.get(j);
            }
        }
        return res;
    }

    public int getDis(int[] arr, int n) {
        int minV = arr[0];
        int res = 0;
        for (int i = 1; i < arr.length; i++) {
            final int gap = arr[i] - minV;
            if (gap > 0) {
                res = res > gap ? res : gap;
            } else {
                minV = arr[i];
            }
        }
        return res;
    }

    public int[][] flipChess(int[][] arr, int[][] f) {
        int[][] dir = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        int r = arr.length;
        int c = arr[0].length;
        int[][] turns = new int[r][c];
        for (int i = 0; i < f.length; i++)
            for (int j = 0; j < dir.length; j++) {
                final int i1 = f[i][0] - 1 + dir[j][0];
                final int i2 = f[i][1] - 1 + dir[j][1];
                if (i1 < 0 || i1 >= r) {
                    continue;
                }
                if (i2 < 0 || i2 >= c) {
                    continue;
                }
                turns[i1][i2]++;
            }
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                if (turns[i][j] % 2 != 0) {
                    if (arr[i][j] == 1) {
                        arr[i][j] = 0;
                    } else {
                        arr[i][j] = 1;
                    }
                }
            }
        }
        return arr;
    }

    public int maximalRectangle(int[][] matrix) {
        int n = matrix.length;
        int m = matrix[0].length;
        for (int i = 0; i < n; i++) {
            for (int j = m - 2; j >= 0; j--) {
                matrix[i][j] = matrix[i][j] == 1 ? matrix[i][j] + matrix[i][j + 1] : 0;
            }
        }
        int maxArea = 0;
        int j = 0;
        while (j < m) {
            int maxHeight = 0;
            int[] heights = new int[n];
            for (int k = 0; k < n; k++) {
                heights[k] = matrix[k][j];
                maxHeight = Math.max(maxHeight, heights[k]);
            }
            if (maxHeight == 0) {
                j++;
                continue;
            }
            maxArea = Math.max(maxArea, largestRectangleArea(heights));
            j++;
        }
        return maxArea;
    }

    private int largestRectangleArea(int[] heights) {
        Stack<Integer> stack = new Stack<>();
        int maxArea = 0, n = heights.length;
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && heights[i] < heights[stack.peek()]) {
                int h = heights[stack.pop()];
                int L = stack.isEmpty() ? 0 : stack.peek() + 1;
                maxArea = Math.max(maxArea, h * (i - L));
            }
            stack.push(i);
        }
        while (!stack.isEmpty()) {
            int h = heights[stack.pop()];
            int L = stack.isEmpty() ? 0 : stack.peek() + 1;
            maxArea = Math.max(maxArea, h * (n - L));
        }
        return maxArea;
    }

    private int res_1111 = 0;

    public int FindPath1111(TreeNode root, int sum) {
        ArrayList<Integer> list = new ArrayList<>();
        if (root == null) {
            return 0;
        }
        list.add(0);
        findPathHelper(root, sum, list, 0);
        return res_1111;
    }

    private void findPathHelper(TreeNode root, int target, List<Integer> list, int sum) {
        sum += root.val;
        for (int i = 0; i < list.size(); i++) {
            if (sum - list.get(i) == target) {
                res_1111++;
            }
        }
        ArrayList<Integer> list1 = new ArrayList<>(list);
        list1.add(sum);
        if (root.left != null) {
            findPathHelper(root.left, target, list1, sum);
        }
        if (root.right != null) {
            findPathHelper(root.right, target, new ArrayList<>(list1), sum);
        }
    }


    private int res_1111_v2 = 0;

    private void dfs(TreeNode root, int sum) {
        if (root == null) {
            return;
        }
        if (sum == root.val) {
            res_1111_v2++;
        }
        dfs(root.left, sum - root.val);
        dfs(root.right, sum - root.val);
    }

    public int FindPathV2(TreeNode root, int sum) {
        if (root == null) {
            return 0;
        }
        dfs(root, sum);
        FindPathV2(root.left, sum);
        FindPathV2(root.right, sum);
        return res_1111_v2;
    }

    private int binarySearch(List<Integer> list, int target) {
        // 返回第一个小于等于target的值
        int ret = -1;
        int l = 0, r = list.size() - 1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (list.get(mid) >= target) {
                ret = mid;
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        return ret;
    }

    public int LIS(int[] arr) {
        if (arr.length == 0) {
            return 0;
        }
        List<Integer> list = new ArrayList<>();
        list.add(arr[0]);
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > list.get(list.size() - 1)) {
                list.add(arr[i]);
            }
            int index = binarySearch(list, arr[i]);
            list.set(index, arr[i]);
        }
        return list.size();
    }

    public int LCSV2(String s1, String s2) {
        int m = s1.length() + 1;// 行
        int n = s2.length() + 1;// 列
        int[][] dp = new int[m][n];
        for (int i = 0; i < n; i++) {
            dp[0][i] = 0;
        }
        for (int i = 0; i < m; i++) {
            dp[i][0] = 0;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i][j - 1], dp[i - 1][j]);
                }
            }
        }
        return dp[m - 1][n - 1];
    }

    public int maxArea(int[] height) {
        if (height.length == 0) {
            return 0;
        }
        int res = 0;
        int l = 0, r = height.length - 1;
        int leftH = height[l];
        int rightH = height[r];
        while (l < r) {
            int val = Math.min(leftH, rightH);
            int temp = val * (r - l);
            if (temp > res) {
                res = temp;
            }
            if (leftH < rightH) {
                leftH = height[++l];
            } else {
                rightH = height[--r];
            }
        }
        return res;
    }

    private boolean judgeLeave(TreeNode node) {
        if (node == null) {
            return false;
        }
        return (node.left == null && node.right == null);
    }

    public TreeNode pruneLeaves(TreeNode root) {
        if (root == null) {
            return null;
        }
        if (judgeLeave(root.left) || judgeLeave(root.right)) {
            return null;
        }
        root.left = pruneLeaves(root.left);
        root.right = pruneLeaves(root.right);
        return root;
    }

    public String compressString(String param) {
        StringBuilder sb = new StringBuilder();
        final int len = param.length();
        int index = 0;
        while (index < len) {
            char c = param.charAt(index);
            index++;
            int sum = 1;
            while (index < len) {
                if (param.charAt(index) == c) {
                    sum++;
                } else {
                    break;
                }
                index++;
            }
            sb.append(c);
            sb.append(sum);
        }
        return sb.toString();
    }

    public int lengthOfLongestSubstring(String s) {
        int res = 1;
        int[] vis = new int[128];
        int l = 0, r = 1;
        vis[s.charAt(l)] = 1;
        while (r < s.length()) {
            while (r < s.length() && vis[s.charAt(r)] == 1) {
                vis[s.charAt(l++)] = 0;
            }
            vis[s.charAt(r)] = 1;
            if (r - l + 1 > res) {
                res = r - l + 1;
            }
            r++;
        }
        return res;
    }

    public int largestRectangleAreaV2(int[] heights) {
        Stack<Integer> stack = new Stack<>();
        int maxArea = 0, n = heights.length;
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && heights[i] < heights[stack.peek()]) {
                int h = heights[stack.pop()];
                int L = stack.isEmpty() ? 0 : stack.peek() + 1;
                maxArea = Math.max(maxArea, h * (n - L));
            }
            stack.push(i);
        }
        while (!stack.isEmpty()) {
            int h = heights[stack.pop()];
            int L = stack.isEmpty() ? 0 : stack.peek() + 1;
            maxArea = Math.max(maxArea, h * (n - L));
        }
        return maxArea;
    }

    private ArrayList<String> helper(String start, String des, String mid, int target) {
        ArrayList<String> list = new ArrayList<>();
        if (target == 1) {
            list.add(String.format("move from %s to %s", start, des));
        } else if (target == 2) {
            list.add(String.format("move from %s to %s", start, mid));
            list.add(String.format("move from %s to %s", start, des));
            list.add(String.format("move from %s to %s", mid, des));
        } else {
            ArrayList<String> stringArrayList = helper(start, mid, des, target - 1);
            stringArrayList.add(String.format("move from %s to %s", start, des));
            ArrayList<String> stringArrayList1 = helper(mid, des, start, target - 1);
            stringArrayList1.stream().forEach(str -> stringArrayList.add(str));
            return stringArrayList;
        }
        return list;
    }

    public ArrayList<String> getSolution(int n) {
        ArrayList<String> list = helper("left", "mid", "right", n);
        return list;
    }

    public String maxLexicographical(String num) {
        int index = -1;
        int index2 = -1;
        for (int i = 0; i < num.length(); i++) {
            if (num.charAt(i) == '0') {
                index = i;
                for (int j = index; j < num.length(); j++) {
                    if (num.charAt(j) == '0') {
                        index2 = j;
                    } else {
                        break;
                    }
                }
                break;
            }
        }
        if (index == -1) {
            return num;
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < num.length(); i++) {
            if (i < index || i > index2) {
                sb.append(num.charAt(i));
            } else {
                sb.append('1');
            }
        }
        return sb.toString();
    }

    private Map<TreeNode, Integer> map = new HashMap<>();

    private List<List<TreeNode>> levelOrder(TreeNode root) {
        List<List<TreeNode>> ans = new ArrayList<>();
        if (root == null) {
            return ans;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (queue.size() != 0) {
            List<TreeNode> tmp = new ArrayList<>();
            int index = 0;
            for (int i = queue.size(); i > 0; i--) {
                TreeNode cur = queue.poll();
                index++;
                if (cur.val == -1) {
                    continue;
                }
                tmp.add(cur);
                map.put(cur, index - 1);
                queue.offer(cur.left == null ? new TreeNode(-1) : cur.left);
                queue.offer(cur.right == null ? new TreeNode(-1) : cur.right);
                cur.left = null;
                cur.right = null;
            }
            ans.add(tmp);
        }
        return ans;
    }

    public TreeNode cyclicShiftTree(TreeNode root, int k) {
        List<List<TreeNode>> lists = levelOrder(root);
        for (int i = lists.size() - 1; i > 0; i--) {
            List<TreeNode> child = lists.get(i);
            List<TreeNode> parent = lists.get(i - 1);
            for (int j = 0; j < child.size(); j++) {
                TreeNode chi = child.get(j);
                TreeNode par = parent.get((map.get(chi) + k) % (parent.size() * 2) / 2);
                if ((map.get(chi) + k) % (parent.size() * 2) % 2 == 0) {
                    par.left = chi;
                } else {
                    par.right = chi;
                }
            }
        }
        return root;
    }

    public int maxValue(String s, int k) {
        k = Math.min(s.length(), k);
        List<String> list = new ArrayList<>();
        for (int i = 0; i <= s.length() - k; i++) {
            list.add(s.substring(i, i + k));
        }
        Collections.sort(list, new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                for (int i = 0; i < o1.length(); i++) {
                    if (o1.charAt(i) > o2.charAt(i)) {
                        return -1;
                    } else if (o1.charAt(i) < o2.charAt(i)) {
                        return 1;
                    }
                }
                return 0;
            }
        });
        return Integer.valueOf(list.get(0));
    }

    private int calculateHelper(String s, int start, int des) {
        Stack<Integer> stack = new Stack<>();
        boolean flag = true;
        while (start <= des) {
            if (s.charAt(start) == '-') {
                flag = false;
                start++;
            } else if (s.charAt(start) == '+') {
                flag = true;
                start++;
            } else if (s.charAt(start) == '(') {
                int sum = 1;
                int begin = start + 1;
                int end = -1;
                start++;
                while (start <= des) {
                    if (s.charAt(start) == '(') {
                        sum++;
                    } else if (s.charAt(start) == ')') {
                        sum--;
                    }
                    if (sum == 0) {
                        end = start - 1;
                        break;
                    }
                    start++;
                }
                int val = calculateHelper(s, begin, end);
                stack.push(val * (flag ? 1 : -1));
                start++;
            } else if (s.charAt(start) >= '0' && s.charAt(start) <= '9') {
                int sum = 0;
                while (start <= des) {
                    if (s.charAt(start) >= '0' && s.charAt(start) <= '9') {
                        sum = sum * 10 + s.charAt(start) - 48;
                    } else {
                        break;
                    }
                    start++;
                }
                stack.push(sum * (flag ? 1 : -1));
            }
        }
        int res = 0;
        while (!stack.isEmpty()) {
            res = res + stack.pop();
        }
        return res;
    }

    public int calculate(String s) {
        return calculateHelper(s, 0, s.length() - 1);
    }

    public int FillArray(int[] a, int k) {
        int len = a.length;
        long[][] dp = new long[len + 1][k + 1];
        final int MOD = 1000000007;
        for (int j = 1; j <= k; j++) {
            dp[1][j] = j;
        }
        for (int i = 2; i <= len; i++) {
            for (int j = 1; j <= k; j++) {
                dp[i][j] = (dp[i][j - 1] + dp[i - 1][j]) % MOD;
            }
        }
        int i = 0;
        int ans = 1;
        while (i < len) {
            while (i < len && a[i] != 0) {
                i++;
            }
            if (i == len) {
                break;
            }
            int start = i;
            int low = (i > 0 ? a[i - 1] : 1);
            while (i < len && a[i] == 0) {
                i++;
            }
            int end = i;
            int high = (i < len ? a[i] : k);
            ans = (int) ((ans * dp[end - start][high - low + 1]) % MOD);
        }
        return ans;
    }

    public int calculateSimple(String s) {
        /**
         * 1 +
         * 2 -
         * 3 *
         * 4 /
         */
        int flag = 1;
        Stack<Integer> stack = new Stack<>();
        int index = 0;
        while (index < s.length()) {
            if (s.charAt(index) == '+') {
                flag = 1;
                index++;
            } else if (s.charAt(index) == '-') {
                flag = 2;
                index++;
            } else if (s.charAt(index) == '*') {
                flag = 3;
                index++;
            } else if (s.charAt(index) == '/') {
                flag = 4;
                index++;
            } else {
                int sum = 0;
                while (index < s.length()) {
                    if (s.charAt(index) >= '0' && s.charAt(index) <= '9') {
                        sum = sum * 10 + s.charAt(index) - 48;
                        index++;
                    } else {
                        break;
                    }
                }
                switch (flag) {
                    case 1:
                        stack.push(sum);
                        break;
                    case 2:
                        stack.push(sum * -1);
                        break;
                    case 3:
                        int val = stack.pop();
                        stack.push(val * sum);
                        break;
                    case 4:
                        val = stack.pop();
                        stack.push(val / sum);
                        break;
                }
            }
        }
        int res = 0;
        while (!stack.isEmpty()) {
            res = res + stack.pop();
        }
        return res;
    }

    private boolean dfs(String[] board, String word, int[][] vis, int index, int r, int c) {
        int[][] dirs = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        if (index == word.length()) {
            return true;
        }
        for (int i = 0; i < dirs.length; i++) {
            int rr = r + dirs[i][0];
            int cc = c + dirs[i][1];
            if (rr < 0 || rr >= board.length || cc < 0 || cc >= board[0].length()) {
                continue;
            }
            if (vis[rr][cc] == 0 && board[rr].charAt(cc) == word.charAt(index)) {
                vis[rr][cc] = 1;
                if (dfs(board, word, vis, index + 1, rr, cc))
                    return true;
                vis[rr][cc] = 0;
            }
        }
        return false;
    }

    public boolean exist(String[] board, String word) {
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length(); j++) {
                if (board[i].charAt(j) == word.charAt(0)) {
                    int[][] vis = new int[board.length][board[0].length()];
                    vis[i][j] = 1;
                    if (dfs(board, word, vis, 1, i, j)) {
                        return true;
                    }
                }

            }
        }
        return false;
    }

    public boolean isValidString(String s) {
        LinkedList<Integer> left = new LinkedList<>();
        LinkedList<Integer> star = new LinkedList<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(') {
                left.push(i);
            } else if (c == '*') {
                star.push(i);
            } else {
                if (!left.isEmpty()) {
                    left.pop();
                } else if (!star.isEmpty()) {
                    star.pop();
                } else {
                    return false;
                }
            }
        }
        while (!left.isEmpty() && !star.isEmpty()) {
            int top1 = left.pop();
            int top2 = star.pop();
            if (top1 > top2) {
                return false;
            }
        }
        return left.isEmpty();
    }

    public ListNode insertionSortList(ListNode head) {
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode next = head.next;
        head.next = null;
        while (next != null) {
            ListNode temp = next.next;
            next.next = null;
            ListNode node = dummy;
            while (node.next != null) {
                if (node.next.val >= next.val) {
                    // 插入值
                    next.next = node.next;
                    node.next = next;
                    break;
                }
                node = node.next;
            }
            if (node.next == null) {
                node.next = next;
            }
            next = temp;
        }
        return dummy.next;
    }

    public int findTargetSumWays(int[] nums, int target) {
        if (nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            if (target == nums[0] && target == (nums[0] * -1)) {
                return 2;
            } else if (target == nums[0] || target == (nums[0] * -1)) {
                return 1;
            } else {
                return 0;
            }
        }
        int len = nums.length;
        return findTargetSumWays(Arrays.copyOfRange(nums, 1, len), target + nums[0]) + findTargetSumWays(Arrays.copyOfRange(nums, 1, len), target - nums[0]);
    }

    public int findTargetSumWaysV2(int[] nums, int target) {
        final int n = nums.length;
        if (n == 0) {
            return 0;
        }
        int sum = 0;
        sum = Arrays.stream(nums).sum();
        int v = (sum + target) / 2;
        if ((sum + target) % 2 == 1) {
            return 0;
        }
        int[][] dp = new int[n + 1][v + 1];
        dp[0][0] = 1;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= v; j++) {
                dp[i + 1][j] = dp[i][j];
                if (j >= nums[i]) {
                    dp[i + 1][j] = dp[i + 1][j] + dp[i][j - nums[i]];
                }
            }
        }
        return dp[n][v];
    }

    HashMap<TreeNode, Integer> mp1_20221118 = new HashMap();
    HashMap<TreeNode, Integer> mp2_20221118 = new HashMap();

    public int rob(TreeNode root) {
        mp1_20221118.put(null, 0);
        mp2_20221118.put(null, 0);
        f(root);//递归
        return Math.max(mp1_20221118.get(root), mp2_20221118.get(root));//返回最大值
    }

    void f(TreeNode root) {//递归
        if (root == null)
            return;
        f(root.left);//左递归
        f(root.right);//右递归
        mp1_20221118.put(root, root.val + mp2_20221118.get(root.left) + mp2_20221118.get(root.right));
        mp2_20221118.put(root, Math.max(mp1_20221118.get(root.left), mp2_20221118.get(root.left)) + Math.max(mp1_20221118.get(root.right), mp2_20221118.get(root.right)));//当前节点不偷
    }

    public int[][] generate(int num) {
        if (num == 1) {
            return new int[][]{{1}};
        } else if (num == 2) {
            return new int[][]{{1}, {1, 1}};
        } else {
            int[][] res = new int[num][];
            res[0] = new int[]{1};
            res[1] = new int[]{1, 1};
            for (int i = 2; i < num; i++) {
                res[i] = new int[i + 1];
                res[i][0] = 1;
                res[i][i] = 1;
                for (int j = 1; j < i; j++) {
                    res[i][j] = res[i - 1][j] + res[i - 1][j - 1];
                }
            }
            return res;
        }
    }

    public int[] plusOne(int[] nums) {
        int len = nums.length;
        int[] res = new int[len + 1];
        int plus = 1;
        for (int i = len; i >= 1; i--) {
            res[i] = nums[i - 1] + plus;
            plus = (res[i] / 10);
            res[i] = res[i] % 10;
        }
        res[0] = plus;
        if (res[0] != 0) {
            return res;
        } else {
            return Arrays.copyOfRange(res, 1, res.length);
        }
    }

    public int numKLenSubstrRepeats(String s, int k) {
        int[] count = new int[128];
        if (k >= s.length()) {
            for (int i = 0; i < s.length(); i++) {
                if (count[s.charAt(i)] == 1) {
                    return 1;
                }
                count[s.charAt(i)]++;
            }
            return 0;
        } else {
            int ans = 0;
            for (int i = 0; i < k; i++) {
                count[s.charAt(i)]++;
                if (count[s.charAt(i)] == 2 && ans == 0) {
                    ans++;
                }
            }
            for (int i = k; i < s.length(); i++) {
                count[s.charAt(i - k)]--;
                count[s.charAt(i)]++;
                for (int j = 'a'; j < 'z'; j++) {
                    if (count[j] >= 2) {
                        ans++;
                        break;
                    }
                }
            }
            return ans;
        }
    }

    public int longestCommonSubarry(int[] A, int[] B) {
        int res = 0;
        int n = A.length, m = B.length;
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < B.length; j++) {
                if (A[i] == B[j]) {
                    dp[i + 1][j + 1] = dp[i][j] + 1;
                } else {
                    dp[i + 1][j + 1] = 0;
                }
                res = Math.max(res, dp[i + 1][j + 1]);
            }
        }
        return res;
    }

    public boolean wordDiv(String s, String[] dic) {
        HashSet<String> wordDic = new HashSet<>();
        Arrays.stream(dic).forEach(str -> wordDic.add(str));
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int end = 1; end <= s.length(); end++) {
            for (int start = 0; start < end; start++) {
                if (dp[start] && wordDic.contains(s.substring(start, end))) {
                    dp[end] = true;
                }
            }
        }
        return dp[s.length()];
    }

    private void dfs(String s, int index, String[] dic, ArrayList<ArrayList<String>> arrayLists, ArrayList<String> list) {
        if (index == s.length()) {
            arrayLists.add(new ArrayList<>(list));
        } else {
            for (String s1 : dic) {
                if (s.substring(index).startsWith(s1)) {
                    list.add(s1);
                    dfs(s, index + s1.length(), dic, arrayLists, list);
                    list.remove(list.size() - 1);
                }
            }
        }
    }

    public String[] wordDiv2(String s, String[] dic) {
        ArrayList<ArrayList<String>> arrayLists = new ArrayList<>();
        ArrayList<String> list = new ArrayList<>();
        HashSet<String> set = new HashSet<>();
        int index = 0;
        dfs(s, 0, dic, arrayLists, list);
        for (ArrayList<String> arrayList : arrayLists) {
            StringBuilder sb = new StringBuilder();
            arrayList.stream().forEach(str -> sb.append(str + " "));
            sb.deleteCharAt(sb.length() - 1);
            set.add(sb.toString());
        }
        String[] strings = new String[set.size()];
        for (String s1 : set) {
            strings[index++] = s1;
        }
        return strings;
    }

    public static void main(String[] args) {
        System.out.println(new Solution().wordDiv2("nowcoder", new String[]{"now", "coder", "no", "wcoder"}));
    }
}
