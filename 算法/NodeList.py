# 总结：ListNode是一种很简单的数据结构，如果简单题，可以考虑直接遍历储存解决，非常直观，但是效率低下。另外比较普遍的是
# 递归（比如深度搜索），双指针，可以极大提高效率
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

    def generateList(l: list):
        prenode = ListNode(0)
        lastnode = prenode
        for val in l:
            lastnode.next = ListNode(val)
            lastnode = lastnode.next
        return prenode.next

class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random


# 优秀答案
class Solution:
    # 15 翻转链表（递归）
    def reverseList(self, head: ListNode) -> ListNode:
        if not (head.next):
            return head

        next_node = head.next
        res = self.reverseList(head.next)
        next_node.next = head
        head.next = None
        return res

    # 35 复杂链表，深度搜索
    def copyRandomList(self, head: 'Node') -> 'Node':
        def dfs(head):
            if head == None:
                return head
            if head in visit:
                return visit[head]
            copy = Node(head.val, None, None)
            visit[head] = copy
            copy.next = dfs(head.next)
            copy.random = dfs(head.random)
            return copy

        visit = {}
        return dfs(head)

    # 递归
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head:
            return head
        cur = head
        while cur:
            new_node = Node(cur.val, None, None)
            new_node.next = cur.next
            cur.next = new_node
            cur = cur.next.next

        cur = head
        while cur:
            cur.next.random = cur.random.next if cur.random else None
            cur = cur.next.next

        new_head = head.next

        old_list = head
        new_list = head.next
        while old_list:
            old_list.next = old_list.next.next
            new_list.next = new_list.next.next if new_list.next else None
            old_list = old_list.next
            new_list = new_list.next

        return new_head

    # 52 链表第一个公共节点（双指针）
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        node1, node2 = headA, headB

        while node1 != node2:
            node1 = node1.next if node1 else headB
            node2 = node2.next if node2 else headA

        return node1




# 自己作品，感觉链表大量简单题都可以用暴力法将所有node放在列表或中解决
class Solution:
    # 15 反转链表
    def reverseList(self, head: ListNode) -> ListNode:
        nodes = []
        while head != None:
            nodes.append(head)
            head = head.next
        nodes = nodes[::-1]
        for i in range(len(nodes)-1):
            nodes[i].next = nodes[i+1]
            nodes[i+1].next = None
        return nodes[0]

    # 25 合并2个有序链表
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 == None:
            return l2
        if l2 == None:
            return l1
        nodes_item = {}
        while l1 != None:
            nodes_item[l1] = l1.val
            l1 = l1.next
        while l2 != None:
            nodes_item[l2] = l2.val
            l2 = l2.next
        nodes = sorted(nodes_item.items(), key=lambda item: item[1])
        for i in range(len(nodes) - 1):
            nodes[i][0].next = nodes[i + 1][0]
            nodes[i + 1][0].next = None
        return nodes[0][0]

    # 52 链表第一个公共节点（效率极低）
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if not headA or not headB:
            return None
        list_A = []
        list_B = []
        while headA:
            list_A.append(headA)
            headA = headA.next
        while headB:
            list_B.append(headB)
            headB = headB.next
        for nodeB in list_B:
            for nodeA in list_A:
                if nodeA == nodeB:
                    return nodeB
        return None

    #02.01移除重复节点
    def removeDuplicateNodes(self, head: ListNode) -> ListNode:
        if not head:
            return head
        res = ListNode(0)
        res.next = head
        nodes = set()
        head = res
        while head and head.next:
            if head.next.val in nodes:
                head.next = head.next.next
            else:
                nodes.add(head.next.val)
                head = head.next

        return res.next

def show(node):
    if not node:
        return
    else:
        print(node.val)
        show(node.next)
l = list(range(1,6))
head = ListNode.generateList(l)
head1 = ListNode.generateList(l)
s = Solution()
# node = s.reverseList(head)
# show(node)

node = s.mergeTwoLists(head,head1)
show(node)
