package writtenexamination;

import java.util.HashSet;
import java.util.Set;

public class Solution {
    /**
     * @param nums 数组
     * @return 最大排序上升子段长度
     */
    public int longestConsecutive(int[] nums) {
        Set<Integer> num_set = new HashSet<>();
        for (int num : nums) {
            num_set.add(num);
        }
        int longestStreak = 0;
        for (int num : num_set) {
            if (!num_set.contains(num - 1)) {
                int currentNum = num;
                int currentStreak = 1;
                while (num_set.contains(currentNum + 1)) {
                    currentStreak++;
                    currentNum++;
                }
                longestStreak = Math.max(longestStreak, currentStreak);
            }
        }
        return longestStreak;
    }

    /**
     * 最小更改次数
     */
    public int minModify(int[] nums, int gap) {
        int cnt = 0;
        int minv = nums[0], maxv = nums[0];
        for (int i = 0; i < nums.length; i++) {
            minv = Math.min(minv, nums[i]);
            maxv = Math.max(maxv, nums[i]);
            if (maxv - minv > 2 * gap) {
                cnt++;
                maxv = minv = nums[i];
            }
        }
        return cnt;
    }
}
