package nowcoder;

import java.util.Comparator;
import java.util.PriorityQueue;

public class TheMidNumber {


    private PriorityQueue<Integer> minHeap = new PriorityQueue<>(new Comparator<Integer>() {
        @Override
        public int compare(Integer o1, Integer o2) {
            return o1.intValue() - o2.intValue();
        }
    });

    private PriorityQueue<Integer> maxHeap = new PriorityQueue<>(new Comparator<Integer>() {
        @Override
        public int compare(Integer o1, Integer o2) {
            return o2.intValue() - o1.intValue();
        }
    });


    public void Insert(Integer num) {
        /*
            1. 让小顶堆保留大的一半 大顶堆保留小的一半
            2. 让小顶堆的个数等于大顶堆的个数+1或者等于 大顶堆的个数
         */
        if (minHeap.isEmpty() || minHeap.peek() < num) {
            minHeap.add(num);
        } else {
            maxHeap.add(num);
        }

        if (maxHeap.size() < minHeap.size() - 1) {
            maxHeap.add(minHeap.poll());
        } else if (maxHeap.size() > minHeap.size()) {
            minHeap.add(maxHeap.poll());
        }
    }

    public Double GetMedian() {
        if (minHeap.size() == maxHeap.size() + 1) {
            return Double.valueOf(minHeap.peek());
        } else {
            System.out.printf("%d %d\n", minHeap.size(), maxHeap.size());
            return Double.valueOf((minHeap.peek() + maxHeap.peek()) / 2.0);
        }
    }
    
}