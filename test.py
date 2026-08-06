from typing import List


def two_sum(nums: List[int], target: int) -> List[int]:
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i
    return []


if __name__ == "__main__":

    print(two_sum([2, 7, 11, 15], 9))  # expected [0, 1]
    print(two_sum([3, 2, 4], 6))  # expected [1, 2]
    print(two_sum([], 5))  # expected []
