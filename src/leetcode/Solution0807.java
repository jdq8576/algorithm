package leetcode;

import java.net.ServerSocket;
import java.util.*;

public class Solution0807 {
    public int arithmeticTriplets(int[] nums, int diff) {
        int cnt = 0;
        for(int i = 0;i < nums.length;i++){
            for(int j = i + 1;j < nums.length;j++){
                if (nums[j] - nums[i] == diff){
                    int target = nums[j] + diff;
                    cnt += find(nums,j+1,nums.length-1,target);
                }
            }
        }
        return cnt;
    }

    private int find(int[] nums, int l, int r,int target) {
        while (l <= r){
            int mid = l + (r - l)/2;
            if (nums[mid] == target){
                return 1;
            }else if (nums[mid] > target){
                r = mid-1;
            }else{
                l = mid + 1;
            }
        }
        return 0;
    }
    public static int reachableNodes(int n, int[][] edges, int[] restricted) {
        boolean[] vis = new boolean[n];
        Set<Integer> set = new HashSet<>();
        for (int i : restricted) {
            set.add(i);
        }
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int[] edge : edges) {
            int f1 = edge[0];
            int f2 = edge[1];
            if (map.containsKey(f1)){
                map.get(f1).add(f2);
            }else{
                List<Integer> list = new LinkedList<>();
                list.add(f2);
                map.put(f1,list);
            }
            if (map.containsKey(f2)){
                map.get(f2).add(f1);
            }else{
                List<Integer> list = new LinkedList<>();
                list.add(f1);
                map.put(f2,list);
            }
        }
        Queue<Integer> queue = new ArrayDeque<>();
        int cnt = 0;
        queue.add(0);
        vis[0] = true;
        while (!queue.isEmpty()){
            int size = queue.size();
            for(int i = 0;i < size;i++){
                final Integer poll = queue.poll();
                vis[poll] = true;
                cnt++;
                final List<Integer> list = map.get(poll);
                if (list!= null){
                    for (Integer integer : list) {
                        if (!set.contains(integer) && vis[integer] == false){
                            queue.add(integer);
                        }
                    }
                }
            }
        }
        return cnt;
    }

    public boolean validPartition(int[] nums) {
        if (nums.length == 1){
            return false;
        }
        boolean[] dp = new boolean[nums.length];
        dp[0] = false;
        for(int i = 1;i < nums.length;i++){
            if (nums[i] == nums[i-1]){
                if (i == 1){
                    dp[i] = true;
                }else{
                    dp[i] = dp[i-2];
                }
            }
            if (dp[i] == true){
                continue;
            }
            if (i >= 2){
                if (nums[i] - nums[i-1] == 1 && nums[i-1] - nums[i-2] == 1){
                    if (i == 2){
                        dp[i] = true;
                    }else{
                        dp[i] = dp[i-3];
                    }
                }
                if (dp[i] == true){
                    continue;
                }
                if (nums[i] == nums[i-1] && nums[i-1] == nums[i-2]){
                    if (i == 2){
                        dp[i] = true;
                    }else{
                        dp[i] = dp[i-3];
                    }
                }
            }
        }
        return dp[nums.length-1];
    }

    public int longestIdealString(String s, int k) {
        if (s.length() == 1){
            return 1;
        }
        int[] dp = new int[26];
        int max = 1;
        for(int i = 0;i < s.length();i++){
            int c = s.charAt(i) - 'a';
            dp[c]++;
            for(int j = 0;j < 26;j++){
                if (j != c && Math.abs(j - c) <= k && dp[c] < dp[j] + 1){
                    dp[c] = dp[j] + 1;
                }
            }
            if (dp[c] > max){
                max = dp[c];
            }
        }
        return max;
    }
}
