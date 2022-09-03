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
}
