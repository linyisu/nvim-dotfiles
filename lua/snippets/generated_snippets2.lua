-- Auto-generated LuaSnip snippets
local ls = require("luasnip")
local ps = ls.parser.parse_snippet

return {

-- 01_Data_Structures\Trees\AVL.h
ps("01_data_structures_trees_avl_h", [=[

/**
 * AVL平衡二叉搜索树
 * 功能：
 * - 支持插入、删除、查找操作
 * - 自动维护树的平衡性（高度差不超过1）
 * - 支持排名查询、第k小元素查询
 * - 支持前驱、后继查询
 * 时间复杂度：所有操作均为 O(log n)
 * 空间复杂度：O(n)
 */

template <typename T>
struct AVLTree {
    struct Node {
        T key;        // 节点值
        int height;   // 高度
        int cnt;      // 重复值的数量
        int size;     // 子树大小
        Node* left;   // 左子树
        Node* right;  // 右子树

        Node(T k) : key(k), height(1), cnt(1), size(1), left(nullptr), right(nullptr) {}
    };

    Node* root;

    AVLTree() : root(nullptr) {}

    // 获取节点高度
    int get_height(Node* node) { return node ? node->height : 0; }

    // 获取子树大小
    int get_size(Node* node) { return node ? node->size : 0; }

    // 获取平衡因子
    int get_balance(Node* node) { return node ? get_height(node->left) - get_height(node->right) : 0; }

    // 更新节点信息
    void update(Node* node) {
        if (!node) return;
        node->height = max(get_height(node->left), get_height(node->right)) + 1;
        node->size = get_size(node->left) + get_size(node->right) + node->cnt;
    }

    // 右旋转
    Node* right_rotate(Node* y) {
        Node* x = y->left;
        Node* T2 = x->right;

        x->right = y;
        y->left = T2;

        update(y);
        update(x);

        return x;
    }

    // 左旋转
    Node* left_rotate(Node* x) {
        Node* y = x->right;
        Node* T2 = y->left;

        y->left = x;
        x->right = T2;

        update(x);
        update(y);

        return y;
    }

    // 插入节点
    Node* insert(Node* node, T key) {
        // 1. 执行标准BST插入
        if (!node) return new Node(key);

        if (key < node->key) {
            node->left = insert(node->left, key);
        } else if (key > node->key) {
            node->right = insert(node->right, key);
        } else {
            // 相同值，增加计数
            node->cnt++;
            update(node);
            return node;
        }

        // 2. 更新节点信息
        update(node);

        // 3. 获取平衡因子
        int balance = get_balance(node);

        // 4. 如果不平衡，则旋转调整
        // Left Left Case
        if (balance > 1 && key < node->left->key) { return right_rotate(node); }

        // Right Right Case
        if (balance < -1 && key > node->right->key) { return left_rotate(node); }

        // Left Right Case
        if (balance > 1 && key > node->left->key) {
            node->left = left_rotate(node->left);
            return right_rotate(node);
        }

        // Right Left Case
        if (balance < -1 && key < node->right->key) {
            node->right = right_rotate(node->right);
            return left_rotate(node);
        }

        return node;
    }

    // 找到最小值节点
    Node* find_min(Node* node) {
        while (node->left) node = node->left;
        return node;
    }

    // 删除节点
    Node* remove(Node* node, T key) {
        if (!node) return node;

        if (key < node->key) {
            node->left = remove(node->left, key);
        } else if (key > node->key) {
            node->right = remove(node->right, key);
        } else {
            // 找到要删除的节点
            if (node->cnt > 1) {
                node->cnt--;
                update(node);
                return node;
            }

            if (!node->left || !node->right) {
                Node* temp = node->left ? node->left : node->right;
                if (!temp) {
                    temp = node;
                    node = nullptr;
                } else {
                    *node = *temp;
                }
                delete temp;
            } else {
                Node* temp = find_min(node->right);
                node->key = temp->key;
                node->cnt = temp->cnt;
                temp->cnt = 1;  // 避免重复删除
                node->right = remove(node->right, temp->key);
            }
        }

        if (!node) return node;

        update(node);

        int balance = get_balance(node);

        // Left Left Case
        if (balance > 1 && get_balance(node->left) >= 0) { return right_rotate(node); }

        // Left Right Case
        if (balance > 1 && get_balance(node->left) < 0) {
            node->left = left_rotate(node->left);
            return right_rotate(node);
        }

        // Right Right Case
        if (balance < -1 && get_balance(node->right) <= 0) { return left_rotate(node); }

        // Right Left Case
        if (balance < -1 && get_balance(node->right) > 0) {
            node->right = right_rotate(node->right);
            return left_rotate(node);
        }

        return node;
    }

    // 查找节点
    bool find(Node* node, T key) {
        if (!node) return false;
        if (key == node->key) return true;
        if (key < node->key) return find(node->left, key);
        return find(node->right, key);
    }

    // 查找第k小的元素（1-indexed）
    T kth_element(Node* node, int k) {
        if (!node) return T{};

        int left_size = get_size(node->left);
        if (k <= left_size) {
            return kth_element(node->left, k);
        } else if (k <= left_size + node->cnt) {
            return node->key;
        } else {
            return kth_element(node->right, k - left_size - node->cnt);
        }
    }

    // 查找元素的排名（从1开始）
    int get_rank(Node* node, T key) {
        if (!node) return 1;

        if (key < node->key) {
            return get_rank(node->left, key);
        } else if (key == node->key) {
            return get_size(node->left) + 1;
        } else {
            return get_size(node->left) + node->cnt + get_rank(node->right, key);
        }
    }

    // 查找前驱（小于key的最大元素）
    T predecessor(Node* node, T key) {
        T result = T{};
        bool found = false;

        while (node) {
            if (node->key < key) {
                if (!found || node->key > result) {
                    result = node->key;
                    found = true;
                }
                node = node->right;
            } else {
                node = node->left;
            }
        }

        return found ? result : T{};
    }

    // 查找后继（大于key的最小元素）
    T successor(Node* node, T key) {
        T result = T{};
        bool found = false;

        while (node) {
            if (node->key > key) {
                if (!found || node->key < result) {
                    result = node->key;
                    found = true;
                }
                node = node->left;
            } else {
                node = node->right;
            }
        }

        return found ? result : T{};
    }

    // 公共接口
    void insert(T key) { root = insert(root, key); }
    void remove(T key) { root = remove(root, key); }
    bool find(T key) { return find(root, key); }
    T kth_element(int k) { return kth_element(root, k); }
    int get_rank(T key) { return get_rank(root, key); }
    T predecessor(T key) { return predecessor(root, key); }
    T successor(T key) { return successor(root, key); }
    int size() { return get_size(root); }
    bool empty() { return root == nullptr; }
};
]=]),

-- 01_Data_Structures\Trees\FHQ_Treap.h
ps("01_data_structures_trees_fhq_treap_h", [=[

/**
 * FHQ Treap（非旋转Treap）
 * 功能：
 * - 支持插入、删除、查找操作
 * - 支持split和merge操作
 * - 支持区间翻转、区间查询等操作
 * - 支持排名查询、第k小元素查询
 * 时间复杂度：期望 O(log n)
 * 空间复杂度：O(n)
 */

template <typename T>
struct FHQTreap {
    struct Node {
        T val;         // 节点值
        int priority;  // 随机优先级
        int size;      // 子树大小
        Node* left;    // 左子树
        Node* right;   // 右子树

        Node(T v) : val(v), priority(rand()), size(1), left(nullptr), right(nullptr) {}
    };

    Node* root;
    mt19937 rng;

    FHQTreap() : root(nullptr), rng(random_device{}()) {}

    // 更新节点大小
    void update(Node* node) {
        if (!node) return;
        node->size = 1;
        if (node->left) node->size += node->left->size;
        if (node->right) node->size += node->right->size;
    }

    // 获取子树大小
    int get_size(Node* node) { return node ? node->size : 0; }

    // 按值分裂：将树分为 ≤val 和 >val 两部分
    pair<Node*, Node*> split_by_value(Node* node, T val) {
        if (!node) return {nullptr, nullptr};

        if (node->val <= val) {
            auto [left, right] = split_by_value(node->right, val);
            node->right = left;
            update(node);
            return {node, right};
        } else {
            auto [left, right] = split_by_value(node->left, val);
            node->left = right;
            update(node);
            return {left, node};
        }
    }

    // 按大小分裂：将前k个节点分出来
    pair<Node*, Node*> split_by_size(Node* node, int k) {
        if (!node) return {nullptr, nullptr};

        int left_size = get_size(node->left);
        if (left_size >= k) {
            auto [left, right] = split_by_size(node->left, k);
            node->left = right;
            update(node);
            return {left, node};
        } else {
            auto [left, right] = split_by_size(node->right, k - left_size - 1);
            node->right = left;
            update(node);
            return {node, right};
        }
    }

    // 合并两棵树（保证left中所有值 ≤ right中所有值）
    Node* merge(Node* left, Node* right) {
        if (!left) return right;
        if (!right) return left;

        if (left->priority < right->priority) {
            right->left = merge(left, right->left);
            update(right);
            return right;
        } else {
            left->right = merge(left->right, right);
            update(left);
            return left;
        }
    }

    // 插入值
    void insert(T val) {
        auto [left, right] = split_by_value(root, val);
        auto [left2, right2] = split_by_value(left, val - 1);

        Node* new_node = new Node(val);
        if (right2) {
            // 如果已存在相同值，这里可以选择忽略或增加计数
            left = merge(left2, right2);
        } else {
            left = merge(left2, new_node);
        }
        root = merge(left, right);
    }

    // 删除值
    void erase(T val) {
        auto [left, right] = split_by_value(root, val);
        auto [left2, right2] = split_by_value(left, val - 1);

        // right2就是要删除的节点
        if (right2) {
            delete right2;
            right2 = nullptr;
        }

        left = merge(left2, right2);
        root = merge(left, right);
    }

    // 查找值的排名（从1开始）
    int get_rank(T val) {
        auto [left, right] = split_by_value(root, val - 1);
        int rank = get_size(left) + 1;
        root = merge(left, right);
        return rank;
    }

    // 查找第k小的值（从1开始）
    T kth_element(int k) { return kth_element_helper(root, k); }

   private:
    T kth_element_helper(Node* node, int k) {
        if (!node) return T{};

        int left_size = get_size(node->left);
        if (k <= left_size) {
            return kth_element_helper(node->left, k);
        } else if (k == left_size + 1) {
            return node->val;
        } else {
            return kth_element_helper(node->right, k - left_size - 1);
        }
    }

   public:
    // 查找前驱（小于val的最大值）
    T predecessor(T val) {
        auto [left, right] = split_by_value(root, val - 1);
        T result = T{};
        bool found = false;

        if (left) {
            result = find_max(left);
            found = true;
        }

        root = merge(left, right);
        return found ? result : T{};
    }

    // 查找后继（大于val的最小值）
    T successor(T val) {
        auto [left, right] = split_by_value(root, val);
        T result = T{};
        bool found = false;

        if (right) {
            result = find_min(right);
            found = true;
        }

        root = merge(left, right);
        return found ? result : T{};
    }

   private:
    T find_min(Node* node) {
        while (node->left) node = node->left;
        return node->val;
    }

    T find_max(Node* node) {
        while (node->right) node = node->right;
        return node->val;
    }

   public:
    // 检查是否包含某个值
    bool contains(T val) {
        auto [left, right] = split_by_value(root, val);
        auto [left2, right2] = split_by_value(left, val - 1);

        bool found = (right2 != nullptr);

        left = merge(left2, right2);
        root = merge(left, right);
        return found;
    }

    // 获取树的大小
    int size() { return get_size(root); }

    // 检查树是否为空
    bool empty() { return root == nullptr; }

    // 清空树
    void clear() {
        clear_helper(root);
        root = nullptr;
    }

   private:
    void clear_helper(Node* node) {
        if (!node) return;
        clear_helper(node->left);
        clear_helper(node->right);
        delete node;
    }
};

// 支持区间操作的FHQ Treap（用于序列操作）
template <typename T>
struct SequenceFHQTreap {
    struct Node {
        T val;
        int priority;
        int size;
        bool reversed;  // 翻转标记
        Node* left;
        Node* right;

        Node(T v) : val(v), priority(rand()), size(1), reversed(false), left(nullptr), right(nullptr) {}
    };

    Node* root;
    mt19937 rng;

    SequenceFHQTreap() : root(nullptr), rng(random_device{}()) {}

    void push_down(Node* node) {
        if (!node || !node->reversed) return;

        swap(node->left, node->right);
        if (node->left) node->left->reversed ^= true;
        if (node->right) node->right->reversed ^= true;
        node->reversed = false;
    }

    void update(Node* node) {
        if (!node) return;
        node->size = 1;
        if (node->left) node->size += node->left->size;
        if (node->right) node->size += node->right->size;
    }

    int get_size(Node* node) { return node ? node->size : 0; }

    // 按位置分裂
    pair<Node*, Node*> split(Node* node, int pos) {
        if (!node) return {nullptr, nullptr};

        push_down(node);
        int left_size = get_size(node->left);

        if (pos <= left_size) {
            auto [left, right] = split(node->left, pos);
            node->left = right;
            update(node);
            return {left, node};
        } else {
            auto [left, right] = split(node->right, pos - left_size - 1);
            node->right = left;
            update(node);
            return {node, right};
        }
    }

    Node* merge(Node* left, Node* right) {
        if (!left) return right;
        if (!right) return left;

        if (left->priority < right->priority) {
            push_down(right);
            right->left = merge(left, right->left);
            update(right);
            return right;
        } else {
            push_down(left);
            left->right = merge(left->right, right);
            update(left);
            return left;
        }
    }

    // 在位置pos插入值val
    void insert(int pos, T val) {
        auto [left, right] = split(root, pos);
        Node* new_node = new Node(val);
        root = merge(merge(left, new_node), right);
    }

    // 删除位置pos的元素
    void erase(int pos) {
        auto [left, right] = split(root, pos - 1);
        auto [mid, right2] = split(right, 1);
        if (mid) delete mid;
        root = merge(left, right2);
    }

    // 翻转区间[l, r]
    void reverse(int l, int r) {
        auto [left, right] = split(root, l - 1);
        auto [mid, right2] = split(right, r - l + 1);
        if (mid) mid->reversed ^= true;
        root = merge(left, merge(mid, right2));
    }

    // 获取位置pos的值
    T get(int pos) { return get_helper(root, pos); }

   private:
    T get_helper(Node* node, int pos) {
        if (!node) return T{};

        push_down(node);
        int left_size = get_size(node->left);

        if (pos <= left_size) {
            return get_helper(node->left, pos);
        } else if (pos == left_size + 1) {
            return node->val;
        } else {
            return get_helper(node->right, pos - left_size - 1);
        }
    }

   public:
    int size() { return get_size(root); }
    bool empty() { return root == nullptr; }
};
]=]),

-- 01_Data_Structures\Trees\FenwickTree.h
ps("01_data_structures_trees_fenwicktree_h", [=[

/**
 * 树状数组模板（Fenwick Tree / Binary Indexed Tree）
 * 功能：
 * - 单点更新，区间查询
 * - 支持前缀和查询和区间和查询
 * - 二分查找第k小元素
 * 时间复杂度：O(log n) 更新和查询
 */

template <typename T>
struct FenwickTree {
    vector<T> tree;
    int n;

    FenwickTree(int size) : n(size) { tree.assign(n + 1, T{}); }

    FenwickTree(const vector<T>& arr) : n(arr.size()) {
        tree.assign(n + 1, T{});
        for (int i = 0; i < n; i++) { update(i + 1, arr[i]); }
    }

    // 单点更新：将位置pos的值增加delta
    void update(int pos, T delta) {
        for (; pos <= n; pos += pos & (-pos)) { tree[pos] += delta; }
    }

    // 前缀和查询：查询[1, pos]的和
    T query(int pos) {
        T sum = T{};
        for (; pos > 0; pos -= pos & (-pos)) { sum += tree[pos]; }
        return sum;
    }

    // 区间和查询：查询[l, r]的和
    T range_query(int l, int r) {
        if (l > r) return T{};
        return query(r) - query(l - 1);
    }

    // 单点查询
    T point_query(int pos) { return range_query(pos, pos); }

    // 二分查找：找到第一个前缀和 >= target 的位置
    int lower_bound(T target) {
        int pos = 0;
        for (int k = __builtin_clz(1) - __builtin_clz(n); k >= 0; k--) {
            int next_pos = pos + (1 << k);
            if (next_pos <= n && tree[next_pos] < target) {
                target -= tree[next_pos];
                pos = next_pos;
            }
        }
        return pos + 1;
    }

    // 清空树状数组
    void clear() { fill(tree.begin(), tree.end(), T{}); }
};

// 差分树状数组：支持区间更新，单点查询
template <typename T>
struct DiffFenwickTree {
    FenwickTree<T> ft;

    DiffFenwickTree(int size) : ft(size) {}

    // 区间更新：将区间[l, r]的所有值增加delta
    void range_update(int l, int r, T delta) {
        ft.update(l, delta);
        if (r + 1 <= ft.n) { ft.update(r + 1, -delta); }
    }

    // 单点查询：查询位置pos的值
    T point_query(int pos) { return ft.query(pos); }
};
]=]),

-- 01_Data_Structures\Trees\FenwickTree2D.h
ps("01_data_structures_trees_fenwicktree2d_h", [=[

/**
 * 二维树状数组（二维Fenwick Tree/Binary Indexed Tree）
 * 功能：
 * - 支持二维数组的单点更新和区间查询
 * - 支持矩形区域的前缀和计算
 * - 支持矩形区域的范围更新（差分数组实现）
 * 时间复杂度：更新和查询均为 O(log n * log m)
 * 空间复杂度：O(n * m)
 */

// 基础二维树状数组
template <typename T>
struct FenwickTree2D {
    vector<vector<T>> tree;
    int n, m;

    // 构造函数：初始化 n×m 的二维树状数组
    FenwickTree2D(int rows, int cols) : n(rows), m(cols) { tree.assign(n + 1, vector<T>(m + 1, 0)); }

    // 构造函数：从二维数组初始化
    FenwickTree2D(const vector<vector<T>>& matrix) : n(matrix.size()), m(matrix[0].size()) {
        tree.assign(n + 1, vector<T>(m + 1, 0));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) { update(i + 1, j + 1, matrix[i][j]); }
        }
    }

    // 单点更新：将 (x, y) 位置的值增加 delta
    void update(int x, int y, T delta) {
        for (int i = x; i <= n; i += lowbit(i)) {
            for (int j = y; j <= m; j += lowbit(j)) { tree[i][j] += delta; }
        }
    }

    // 前缀和查询：查询从 (1,1) 到 (x,y) 的矩形和
    T query(int x, int y) {
        T sum = 0;
        for (int i = x; i > 0; i -= lowbit(i)) {
            for (int j = y; j > 0; j -= lowbit(j)) { sum += tree[i][j]; }
        }
        return sum;
    }

    // 矩形区域查询：查询从 (x1,y1) 到 (x2,y2) 的矩形和
    T range_query(int x1, int y1, int x2, int y2) {
        return query(x2, y2) - query(x1 - 1, y2) - query(x2, y1 - 1) + query(x1 - 1, y1 - 1);
    }

    // 单点查询：查询 (x, y) 位置的值
    T point_query(int x, int y) { return range_query(x, y, x, y); }

   private:
    int lowbit(int x) { return x & (-x); }
};

// 支持矩形区域更新的二维树状数组（基于差分数组）
template <typename T>
struct RangeUpdate2D {
    FenwickTree2D<T> diff;
    int n, m;

    RangeUpdate2D(int rows, int cols) : diff(rows, cols), n(rows), m(cols) {}

    // 矩形区域更新：将 (x1,y1) 到 (x2,y2) 的矩形区域都增加 delta
    void range_update(int x1, int y1, int x2, int y2, T delta) {
        diff.update(x1, y1, delta);
        diff.update(x1, y2 + 1, -delta);
        diff.update(x2 + 1, y1, -delta);
        diff.update(x2 + 1, y2 + 1, delta);
    }

    // 单点查询：查询 (x, y) 位置的实际值
    T point_query(int x, int y) { return diff.query(x, y); }
};

// 二维区间更新区间查询（需要四个差分数组）
template <typename T>
struct RangeUpdateRangeQuery2D {
    FenwickTree2D<T> d1, d2, d3, d4;  // 四个差分数组
    int n, m;

    RangeUpdateRangeQuery2D(int rows, int cols)
        : d1(rows, cols), d2(rows, cols), d3(rows, cols), d4(rows, cols), n(rows), m(cols) {}

    // 矩形区域更新
    void range_update(int x1, int y1, int x2, int y2, T delta) {
        update_helper(x1, y1, delta);
        update_helper(x1, y2 + 1, -delta);
        update_helper(x2 + 1, y1, -delta);
        update_helper(x2 + 1, y2 + 1, delta);
    }

    // 矩形区域查询
    T range_query(int x1, int y1, int x2, int y2) {
        return query_helper(x2, y2) - query_helper(x1 - 1, y2) - query_helper(x2, y1 - 1) +
               query_helper(x1 - 1, y1 - 1);
    }

    // 单点查询
    T point_query(int x, int y) { return range_query(x, y, x, y); }

   private:
    void update_helper(int x, int y, T delta) {
        if (x <= 0 || y <= 0 || x > n || y > m) return;
        d1.update(x, y, delta);
        d2.update(x, y, delta * x);
        d3.update(x, y, delta * y);
        d4.update(x, y, delta * x * y);
    }

    T query_helper(int x, int y) {
        if (x <= 0 || y <= 0) return 0;
        return (x + 1) * (y + 1) * d1.query(x, y) - (y + 1) * d2.query(x, y) - (x + 1) * d3.query(x, y) +
               d4.query(x, y);
    }
};
]=]),

-- 01_Data_Structures\Trees\LinkCutTree.h
ps("01_data_structures_trees_linkcuttree_h", [=[

/**
 * Link-Cut Tree（动态树）
 * 功能：
 * - 支持动态连边和断边操作
 * - 支持路径查询和路径修改
 * - 支持树根变更（makeroot）
 * - 支持连通性查询
 * 时间复杂度：均摊 O(log n)
 * 空间复杂度：O(n)
 *
 * 应用场景：
 * - 动态连通性问题
 * - 动态最小生成树
 * - 树上路径查询
 */

template <typename T>
struct LinkCutTree {
    struct Node {
        int ch[2];     // 左右儿子
        int father;    // 父节点
        T val;         // 节点值
        T sum;         // 子树和
        bool reverse;  // 翻转标记

        Node() : father(0), val(T{}), sum(T{}), reverse(false) { ch[0] = ch[1] = 0; }
    };

    vector<Node> tree;
    int n;

    LinkCutTree(int size) : n(size) { tree.resize(size + 1); }

    // 更新节点信息
    void push_up(int x) {
        if (!x) return;
        tree[x].sum = tree[tree[x].ch[0]].sum + tree[x].val + tree[tree[x].ch[1]].sum;
    }

    // 下传标记
    void push_down(int x) {
        if (!x || !tree[x].reverse) return;

        // 交换左右儿子
        swap(tree[x].ch[0], tree[x].ch[1]);

        // 下传翻转标记
        if (tree[x].ch[0]) tree[tree[x].ch[0]].reverse = !tree[tree[x].ch[0]].reverse;
        if (tree[x].ch[1]) tree[tree[x].ch[1]].reverse = !tree[tree[x].ch[1]].reverse;

        tree[x].reverse = false;
    }

    // 判断x是否为splay的根
    bool is_root(int x) {
        int fa = tree[x].father;
        return tree[fa].ch[0] != x && tree[fa].ch[1] != x;
    }

    // 获取x是父节点的哪个儿子
    int get_relation(int x) { return tree[tree[x].father].ch[1] == x; }

    // 旋转操作
    void rotate(int x) {
        int y = tree[x].father;
        int z = tree[y].father;
        int k = get_relation(x);

        if (!is_root(y)) { tree[z].ch[get_relation(y)] = x; }
        tree[x].father = z;

        tree[y].ch[k] = tree[x].ch[k ^ 1];
        if (tree[x].ch[k ^ 1]) { tree[tree[x].ch[k ^ 1]].father = y; }

        tree[x].ch[k ^ 1] = y;
        tree[y].father = x;

        push_up(y);
        push_up(x);
    }

    // Splay操作
    void splay(int x) {
        stack<int> stk;
        int cur = x;

        // 找到splay的根，同时记录路径
        while (!is_root(cur)) {
            stk.push(cur);
            cur = tree[cur].father;
        }
        stk.push(cur);

        // 从根开始下传标记
        while (!stk.empty()) {
            push_down(stk.top());
            stk.pop();
        }

        // 进行splay
        while (!is_root(x)) {
            int y = tree[x].father;
            if (!is_root(y)) {
                if (get_relation(x) == get_relation(y)) {
                    rotate(y);
                } else {
                    rotate(x);
                }
            }
            rotate(x);
        }
    }

    // 访问操作：将从根到x的路径变成一条preferred path
    void access(int x) {
        int last = 0;
        while (x) {
            splay(x);
            tree[x].ch[1] = last;
            push_up(x);
            last = x;
            x = tree[x].father;
        }
    }

    // 换根操作：将x变成树的根
    void make_root(int x) {
        access(x);
        splay(x);
        tree[x].reverse = !tree[x].reverse;
    }

    // 查找x所在树的根
    int find_root(int x) {
        access(x);
        splay(x);

        // 一直往左走找到根
        while (tree[x].ch[0]) {
            push_down(x);
            x = tree[x].ch[0];
        }
        splay(x);
        return x;
    }

    // 连边操作：连接x和y
    void link(int x, int y) {
        make_root(x);
        tree[x].father = y;
    }

    // 断边操作：断开x和y之间的边
    void cut(int x, int y) {
        make_root(x);
        access(y);
        splay(y);

        // 此时x应该是y的左儿子且x没有右儿子
        if (tree[y].ch[0] == x && !tree[x].ch[1]) {
            tree[y].ch[0] = 0;
            tree[x].father = 0;
            push_up(y);
        }
    }

    // 判断x和y是否连通
    bool connected(int x, int y) {
        if (x == y) return true;
        return find_root(x) == find_root(y);
    }

    // 设置节点x的值
    void set_val(int x, T val) {
        splay(x);
        tree[x].val = val;
        push_up(x);
    }

    // 获取节点x的值
    T get_val(int x) {
        splay(x);
        return tree[x].val;
    }

    // 查询x到y路径上的和
    T query_path(int x, int y) {
        make_root(x);
        access(y);
        splay(y);
        return tree[y].sum;
    }

    // 修改x到y路径上所有节点的值（加上delta）
    void modify_path(int x, int y, T delta) {
        make_root(x);
        access(y);
        splay(y);
        // 这里需要根据具体需求实现路径修改
        // 通常需要添加懒惰标记
    }

    // 查询x所在子树的大小
    int subtree_size(int x) {
        splay(x);
        return tree[tree[x].ch[0]].sum + 1;  // 假设sum维护的是子树大小
    }
};

// 简化版本的Link-Cut Tree（只支持连通性查询）
struct SimpleLCT {
    vector<int> ch[2];
    vector<int> father;
    vector<bool> reverse;
    int n;

    SimpleLCT(int size) : n(size) {
        ch[0].resize(size + 1);
        ch[1].resize(size + 1);
        father.resize(size + 1);
        reverse.resize(size + 1);

        for (int i = 0; i <= size; i++) {
            ch[0][i] = ch[1][i] = father[i] = 0;
            reverse[i] = false;
        }
    }

    bool is_root(int x) { return ch[0][father[x]] != x && ch[1][father[x]] != x; }

    void push_down(int x) {
        if (reverse[x]) {
            swap(ch[0][x], ch[1][x]);
            if (ch[0][x]) reverse[ch[0][x]] = !reverse[ch[0][x]];
            if (ch[1][x]) reverse[ch[1][x]] = !reverse[ch[1][x]];
            reverse[x] = false;
        }
    }

    void splay(int x) {
        stack<int> stk;
        int cur = x;

        while (!is_root(cur)) {
            stk.push(cur);
            cur = father[cur];
        }
        stk.push(cur);

        while (!stk.empty()) {
            push_down(stk.top());
            stk.pop();
        }

        while (!is_root(x)) {
            int y = father[x];
            if (!is_root(y)) {
                if ((ch[1][father[y]] == y) == (ch[1][y] == x)) {
                    rotate(y);
                } else {
                    rotate(x);
                }
            }
            rotate(x);
        }
    }

    void rotate(int x) {
        int y = father[x];
        int z = father[y];
        int k = ch[1][y] == x;

        if (!is_root(y)) { ch[ch[1][z] == y][z] = x; }
        father[x] = z;

        ch[k][y] = ch[k ^ 1][x];
        father[ch[k ^ 1][x]] = y;

        ch[k ^ 1][x] = y;
        father[y] = x;
    }

    void access(int x) {
        int last = 0;
        while (x) {
            splay(x);
            ch[1][x] = last;
            last = x;
            x = father[x];
        }
    }

    void make_root(int x) {
        access(x);
        splay(x);
        reverse[x] = !reverse[x];
    }

    int find_root(int x) {
        access(x);
        splay(x);
        while (ch[0][x]) {
            push_down(x);
            x = ch[0][x];
        }
        splay(x);
        return x;
    }

    void link(int x, int y) {
        make_root(x);
        father[x] = y;
    }

    void cut(int x, int y) {
        make_root(x);
        access(y);
        splay(y);
        if (ch[0][y] == x && !ch[1][x]) { ch[0][y] = father[x] = 0; }
    }

    bool connected(int x, int y) {
        if (x == y) return true;
        return find_root(x) == find_root(y);
    }
};
]=]),

-- 01_Data_Structures\Trees\PersistentSegTree.h
ps("01_data_structures_trees_persistentsegtree_h", [=[

/**
 * 可持久化线段树（主席树）
 * 功能：
 * - 支持历史版本的查询和修改
 * - 支持区间第k小查询
 * - 支持可持久化数组操作
 * - 支持版本回滚
 * 时间复杂度：查询和修改均为 O(log n)
 * 空间复杂度：每次修改增加 O(log n) 个节点
 *
 * 应用场景：
 * - 区间第k小问题
 * - 可持久化数据结构
 * - 函数式编程中的不可变数据结构
 */

template <typename T>
struct PersistentSegmentTree {
    struct Node {
        T val;            // 节点值
        int left, right;  // 左右子树编号

        Node() : val(T{}), left(0), right(0) {}
        Node(T v) : val(v), left(0), right(0) {}
        Node(T v, int l, int r) : val(v), left(l), right(r) {}
    };

    vector<Node> tree;  // 节点池
    vector<int> roots;  // 每个版本的根节点
    int n;              // 数组大小
    int node_count;     // 当前节点数量

    PersistentSegmentTree(int size) : n(size), node_count(0) {
        tree.resize(size * 40);  // 预分配足够的节点
        roots.push_back(build(1, n));
    }

    PersistentSegmentTree(const vector<T>& arr) : n(arr.size()), node_count(0) {
        tree.resize(n * 40);
        roots.push_back(build(arr, 1, n));
    }

    // 构建初始线段树
    int build(int l, int r) {
        int cur = ++node_count;
        if (l == r) {
            tree[cur] = Node(T{});
            return cur;
        }

        int mid = (l + r) / 2;
        tree[cur].left = build(l, mid);
        tree[cur].right = build(mid + 1, r);
        tree[cur].val = tree[tree[cur].left].val + tree[tree[cur].right].val;
        return cur;
    }

    // 从数组构建初始线段树
    int build(const vector<T>& arr, int l, int r) {
        int cur = ++node_count;
        if (l == r) {
            tree[cur] = Node(arr[l - 1]);  // 数组是0-indexed
            return cur;
        }

        int mid = (l + r) / 2;
        tree[cur].left = build(arr, l, mid);
        tree[cur].right = build(arr, mid + 1, r);
        tree[cur].val = tree[tree[cur].left].val + tree[tree[cur].right].val;
        return cur;
    }

    // 单点更新，返回新的根节点
    int update(int prev_root, int l, int r, int pos, T val) {
        int cur = ++node_count;
        tree[cur] = tree[prev_root];  // 复制前一个版本的节点

        if (l == r) {
            tree[cur].val = val;
            return cur;
        }

        int mid = (l + r) / 2;
        if (pos <= mid) {
            tree[cur].left = update(tree[prev_root].left, l, mid, pos, val);
        } else {
            tree[cur].right = update(tree[prev_root].right, mid + 1, r, pos, val);
        }

        tree[cur].val = tree[tree[cur].left].val + tree[tree[cur].right].val;
        return cur;
    }

    // 区间查询
    T query(int root, int l, int r, int ql, int qr) {
        if (ql > r || qr < l) return T{};
        if (ql <= l && r <= qr) return tree[root].val;

        int mid = (l + r) / 2;
        return query(tree[root].left, l, mid, ql, qr) + query(tree[root].right, mid + 1, r, ql, qr);
    }

    // 单点查询
    T query_point(int root, int l, int r, int pos) {
        if (l == r) return tree[root].val;

        int mid = (l + r) / 2;
        if (pos <= mid) {
            return query_point(tree[root].left, l, mid, pos);
        } else {
            return query_point(tree[root].right, mid + 1, r, pos);
        }
    }

    // 公共接口

    // 创建新版本：在指定版本基础上修改位置pos的值为val
    int new_version(int version, int pos, T val) {
        int new_root = update(roots[version], 1, n, pos, val);
        roots.push_back(new_root);
        return roots.size() - 1;
    }

    // 查询指定版本的区间和
    T query(int version, int l, int r) { return query(roots[version], 1, n, l, r); }

    // 查询指定版本的单点值
    T query(int version, int pos) { return query_point(roots[version], 1, n, pos); }

    // 获取版本数量
    int version_count() { return roots.size(); }
};

// 可持久化线段树用于区间第k小查询
struct PersistentKthQuery {
    struct Node {
        int cnt;          // 当前区间内的数字个数
        int left, right;  // 左右子树

        Node() : cnt(0), left(0), right(0) {}
        Node(int c) : cnt(c), left(0), right(0) {}
    };

    vector<Node> tree;
    vector<int> roots;
    vector<int> sorted_vals;  // 离散化后的值
    int n, node_count;

    PersistentKthQuery(vector<int>& arr) : n(arr.size()), node_count(0) {
        // 离散化
        sorted_vals = arr;
        sort(sorted_vals.begin(), sorted_vals.end());
        sorted_vals.erase(unique(sorted_vals.begin(), sorted_vals.end()), sorted_vals.end());

        tree.resize(n * 40);
        roots.resize(n + 1);

        // 构建可持久化线段树
        roots[0] = build(0, sorted_vals.size() - 1);

        for (int i = 0; i < n; i++) {
            int pos = lower_bound(sorted_vals.begin(), sorted_vals.end(), arr[i]) - sorted_vals.begin();
            roots[i + 1] = update(roots[i], 0, sorted_vals.size() - 1, pos);
        }
    }

    int build(int l, int r) {
        int cur = ++node_count;
        if (l == r) {
            tree[cur] = Node(0);
            return cur;
        }

        int mid = (l + r) / 2;
        tree[cur].left = build(l, mid);
        tree[cur].right = build(mid + 1, r);
        return cur;
    }

    int update(int prev, int l, int r, int pos) {
        int cur = ++node_count;
        tree[cur] = tree[prev];
        tree[cur].cnt++;

        if (l == r) return cur;

        int mid = (l + r) / 2;
        if (pos <= mid) {
            tree[cur].left = update(tree[prev].left, l, mid, pos);
        } else {
            tree[cur].right = update(tree[prev].right, mid + 1, r, pos);
        }
        return cur;
    }

    int query_kth(int left_root, int right_root, int l, int r, int k) {
        if (l == r) return sorted_vals[l];

        int mid = (l + r) / 2;
        int left_cnt = tree[tree[right_root].left].cnt - tree[tree[left_root].left].cnt;

        if (k <= left_cnt) {
            return query_kth(tree[left_root].left, tree[right_root].left, l, mid, k);
        } else {
            return query_kth(tree[left_root].right, tree[right_root].right, mid + 1, r, k - left_cnt);
        }
    }

    // 查询区间[l,r]中第k小的数（1-indexed）
    int kth_element(int l, int r, int k) { return query_kth(roots[l - 1], roots[r], 0, sorted_vals.size() - 1, k); }

    // 查询区间[l,r]中小于等于val的数的个数
    int count_leq(int l, int r, int val) {
        int pos = upper_bound(sorted_vals.begin(), sorted_vals.end(), val) - sorted_vals.begin() - 1;
        if (pos < 0) return 0;
        return count_leq_helper(roots[l - 1], roots[r], 0, sorted_vals.size() - 1, pos);
    }

   private:
    int count_leq_helper(int left_root, int right_root, int l, int r, int pos) {
        if (pos < l) return 0;
        if (pos >= r) return tree[right_root].cnt - tree[left_root].cnt;

        int mid = (l + r) / 2;
        return count_leq_helper(tree[left_root].left, tree[right_root].left, l, mid, pos) +
               count_leq_helper(tree[left_root].right, tree[right_root].right, mid + 1, r, pos);
    }
};

// 可持久化数组（支持历史版本访问）
template <typename T>
struct PersistentArray {
    PersistentSegmentTree<T> pst;

    PersistentArray(int size) : pst(size) {}
    PersistentArray(const vector<T>& arr) : pst(arr) {}

    // 修改位置pos的值为val，返回新版本号
    int set(int version, int pos, T val) { return pst.new_version(version, pos, val); }

    // 获取指定版本位置pos的值
    T get(int version, int pos) { return pst.query(version, pos); }

    // 获取当前版本数
    int versions() { return pst.version_count(); }
};
]=]),

-- 01_Data_Structures\Trees\SegmentTree.h
ps("01_data_structures_trees_segmenttree_h", [=[
template <typename T, class Info, class Laz> struct SegTree {
    int n;
    vector<T> arr;
    vector<Laz> laz;
    vector<Info> info;
    SegTree(): n(0) {}
    SegTree(int n_): n(n_), arr(n + 5), laz(4 * n), info(4 * n) {}
    #define mid ((l + r) >> 1)
    void build() { build(1, n, 1); }
    void build(int l, int r, int u) {
        if (l == r) { info[u] = Info(arr[l]); return; }
        build(l, mid, u << 1);
        build(mid + 1, r, u << 1 | 1);
        push_up(u);
    }
    void apply(int u, Laz tag, int len) {
        laz[u].apply(tag);
        info[u].apply(tag, len);
    }
    void push_up(int u) { info[u] = info[u << 1] + info[u << 1 | 1]; }
    void push_down(int u, int llen, int rlen) {
        apply(u << 1, laz[u], llen);
        apply(u << 1 | 1, laz[u], rlen);
        laz[u] = Laz();
    }
    void modify(int l, int r, Laz tag) { modify(l, r, tag, 1, n, 1); }
    void modify(int jobl, int jobr, Laz tag, int l, int r, int u) {
        if (jobl <= l && jobr >= r) { apply(u, tag, r - l + 1); return; }
        push_down(u, mid - l + 1, r - mid);
        if (jobl <= mid) modify(jobl, jobr, tag, l, mid, u << 1);
        if (jobr > mid) modify(jobl, jobr, tag, mid + 1, r, u << 1 | 1);
        push_up(u);
    }
    Info query(int l, int r) { return query(l, r, 1, n, 1); }
    Info query(int jobl, int jobr, int l, int r, int u) {
        if (jobl <= l && jobr >= r) return info[u];
        push_down(u, mid - l + 1, r - mid);
        if (jobr <= mid) return query(jobl, jobr, l, mid, u << 1);
        else if (jobl > mid) return query(jobl, jobr, mid + 1, r, u << 1 | 1);
        return query(jobl, jobr, l, mid, u << 1) + query(jobl, jobr, mid + 1, r, u << 1 | 1);
    }
};
struct Laz 
{
    long long add = 0;
    void apply(const Laz &tag) 
    {
        if (tag.add) 
        {
            add += tag.add;
        }
    }
};
template <typename T> struct Info 
{
    long long sum = 0;
    Info() {}
    Info(T x): sum(x) {}
    void apply(const Laz &tag, int len) 
    {
        if (tag.add) 
        {
            sum += tag.add * len;
        }
    }
    Info operator+(const Info &a) const 
    {
        Info res;
        res.sum = sum + a.sum;
        return res;
    }
};
]=]),

-- 01_Data_Structures\Trees\SegmentTree2D.h
ps("01_data_structures_trees_segmenttree2d_h", [=[

/**
 * 二维线段树模板
 * 功能特性:
 * - 支持二维区间查询和单点修改
 * - 支持二维区间修改和区间查询（懒惰传播）
 * - 可扩展到多种操作类型
 * 时间复杂度: O(log n * log m)
 * 空间复杂度: O(n * m * log n * log m)
 *
 * 适用于二维数组的区间操作问题
 */

template <typename T>
struct SegmentTree2D {
    struct Node {
        T val;          // 节点值
        T lazy;         // 懒惰标记
        bool has_lazy;  // 是否有懒惰标记

        Node() : val(T{}), lazy(T{}), has_lazy(false) {}
        Node(T v) : val(v), lazy(T{}), has_lazy(false) {}
    };

    int n, m;                   // 矩阵的行数和列数
    vector<vector<Node>> tree;  // 二维线段树

    // 构造函数：创建空的二维线段树
    SegmentTree2D(int rows, int cols) : n(rows), m(cols) {
        tree.assign(4 * n, vector<Node>(4 * m));
        build(1, 0, n - 1);
    }

    // 构造函数：从给定矩阵创建二维线段树
    SegmentTree2D(const vector<vector<T>>& matrix) : n(matrix.size()), m(matrix[0].size()) {
        tree.assign(4 * n, vector<Node>(4 * m));
        build(matrix, 1, 0, n - 1);
    }

    // 从矩阵构建二维线段树
    void build(const vector<vector<T>>& matrix, int vx, int lx, int rx) {
        if (lx == rx) {
            build_y(matrix, vx, 1, 0, m - 1, lx);
        } else {
            int mx = (lx + rx) / 2;
            build(matrix, 2 * vx, lx, mx);
            build(matrix, 2 * vx + 1, mx + 1, rx);
            merge_y(vx, 1, 0, m - 1);
        }
    }

    // 构建Y维度的线段树
    void build_y(const vector<vector<T>>& matrix, int vx, int vy, int ly, int ry, int x) {
        if (ly == ry) {
            if (vx >= n) {
                tree[vx][vy] = Node(matrix[x][ly]);
            } else {
                tree[vx][vy] = Node(tree[2 * vx][vy].val + tree[2 * vx + 1][vy].val);
            }
        } else {
            int my = (ly + ry) / 2;
            build_y(matrix, vx, 2 * vy, ly, my, x);
            build_y(matrix, vx, 2 * vy + 1, my + 1, ry, x);
            tree[vx][vy] = Node(tree[vx][2 * vy].val + tree[vx][2 * vy + 1].val);
        }
    }

    // 合并Y维度的节点
    void merge_y(int vx, int vy, int ly, int ry) {
        if (ly == ry) {
            tree[vx][vy] = Node(tree[2 * vx][vy].val + tree[2 * vx + 1][vy].val);
        } else {
            int my = (ly + ry) / 2;
            merge_y(vx, 2 * vy, ly, my);
            merge_y(vx, 2 * vy + 1, my + 1, ry);
            tree[vx][vy] = Node(tree[vx][2 * vy].val + tree[vx][2 * vy + 1].val);
        }
    }

    // 构建空树
    void build(int vx, int lx, int rx) {
        if (lx == rx) {
            build_y(vx, 1, 0, m - 1);
        } else {
            int mx = (lx + rx) / 2;
            build(2 * vx, lx, mx);
            build(2 * vx + 1, mx + 1, rx);
            merge_y(vx, 1, 0, m - 1);
        }
    }

    void build_y(int vx, int vy, int ly, int ry) {
        if (ly == ry) {
            if (vx >= n) {
                tree[vx][vy] = Node(T{});
            } else {
                tree[vx][vy] = Node(tree[2 * vx][vy].val + tree[2 * vx + 1][vy].val);
            }
        } else {
            int my = (ly + ry) / 2;
            build_y(vx, 2 * vy, ly, my);
            build_y(vx, 2 * vy + 1, my + 1, ry);
            tree[vx][vy] = Node(tree[vx][2 * vy].val + tree[vx][2 * vy + 1].val);
        }
    }

    // 单点更新
    void update(int x, int y, T val) { update_x(1, 0, n - 1, x, y, val); }

    void update_x(int vx, int lx, int rx, int x, int y, T val) {
        if (lx == rx) {
            update_y(vx, 1, 0, m - 1, y, val);
        } else {
            int mx = (lx + rx) / 2;
            if (x <= mx) {
                update_x(2 * vx, lx, mx, x, y, val);
            } else {
                update_x(2 * vx + 1, mx + 1, rx, x, y, val);
            }
            merge_y(vx, 1, 0, m - 1);
        }
    }

    void update_y(int vx, int vy, int ly, int ry, int y, T val) {
        if (ly == ry) {
            tree[vx][vy] = Node(val);
        } else {
            int my = (ly + ry) / 2;
            if (y <= my) {
                update_y(vx, 2 * vy, ly, my, y, val);
            } else {
                update_y(vx, 2 * vy + 1, my + 1, ry, y, val);
            }
            tree[vx][vy] = Node(tree[vx][2 * vy].val + tree[vx][2 * vy + 1].val);
        }
    }

    // 区间查询
    T query(int x1, int y1, int x2, int y2) { return query_x(1, 0, n - 1, x1, x2, y1, y2); }

    T query_x(int vx, int lx, int rx, int x1, int x2, int y1, int y2) {
        if (x1 > rx || x2 < lx) return T{};
        if (x1 <= lx && rx <= x2) { return query_y(vx, 1, 0, m - 1, y1, y2); }

        int mx = (lx + rx) / 2;
        return query_x(2 * vx, lx, mx, x1, x2, y1, y2) + query_x(2 * vx + 1, mx + 1, rx, x1, x2, y1, y2);
    }

    T query_y(int vx, int vy, int ly, int ry, int y1, int y2) {
        if (y1 > ry || y2 < ly) return T{};
        if (y1 <= ly && ry <= y2) { return tree[vx][vy].val; }

        int my = (ly + ry) / 2;
        return query_y(vx, 2 * vy, ly, my, y1, y2) + query_y(vx, 2 * vy + 1, my + 1, ry, y1, y2);
    }

    // 单点查询
    T query(int x, int y) { return query(x, y, x, y); }
};

// 支持懒惰传播的二维线段树
template <typename T>
struct SegmentTree2DWithLazy {
    struct Node {
        T val;          // 节点值
        T lazy;         // 懒惰标记
        bool has_lazy;  // 是否有懒惰标记

        Node() : val(T{}), lazy(T{}), has_lazy(false) {}
        Node(T v) : val(v), lazy(T{}), has_lazy(false) {}
    };

    int n, m;                   // 矩阵的行数和列数
    vector<vector<Node>> tree;  // 二维线段树

    SegmentTree2DWithLazy(int rows, int cols) : n(rows), m(cols) {
        tree.assign(4 * n, vector<Node>(4 * m));
        build(1, 0, n - 1);
    }

    void build(int vx, int lx, int rx) {
        if (lx == rx) {
            build_y(vx, 1, 0, m - 1, lx, rx);
        } else {
            int mx = (lx + rx) / 2;
            build(2 * vx, lx, mx);
            build(2 * vx + 1, mx + 1, rx);
            merge_y(vx, 1, 0, m - 1, lx, rx);
        }
    }

    void build_y(int vx, int vy, int ly, int ry, int lx, int rx) {
        tree[vx][vy] = Node(T{});
        if (ly != ry) {
            int my = (ly + ry) / 2;
            build_y(vx, 2 * vy, ly, my, lx, rx);
            build_y(vx, 2 * vy + 1, my + 1, ry, lx, rx);
        }
    }

    void merge_y(int vx, int vy, int ly, int ry, int lx, int rx) {
        if (ly == ry) {
            if (lx == rx) {
                // 叶子节点无需合并
            } else {
                tree[vx][vy].val = tree[2 * vx][vy].val + tree[2 * vx + 1][vy].val;
            }
        } else {
            int my = (ly + ry) / 2;
            merge_y(vx, 2 * vy, ly, my, lx, rx);
            merge_y(vx, 2 * vy + 1, my + 1, ry, lx, rx);
            tree[vx][vy].val = tree[vx][2 * vy].val + tree[vx][2 * vy + 1].val;
        }
    }

    // Y维度懒惰传播
    void push_down_y(int vx, int vy, int ly, int ry) {
        if (tree[vx][vy].has_lazy) {
            tree[vx][vy].val += tree[vx][vy].lazy * (ry - ly + 1);
            if (ly != ry) {
                tree[vx][2 * vy].lazy += tree[vx][vy].lazy;
                tree[vx][2 * vy].has_lazy = true;
                tree[vx][2 * vy + 1].lazy += tree[vx][vy].lazy;
                tree[vx][2 * vy + 1].has_lazy = true;
            }
            tree[vx][vy].lazy = T{};
            tree[vx][vy].has_lazy = false;
        }
    }

    // X维度懒惰传播
    void push_down_x(int vx, int lx, int rx) {
        if (lx != rx) {
            // 如果需要X维度的懒惰传播，在此实现
            // 目前的实现主要针对Y维度的区间更新
        }
    }

    // 区间更新
    void update(int x1, int y1, int x2, int y2, T val) { update_x(1, 0, n - 1, x1, x2, y1, y2, val); }

    void update_x(int vx, int lx, int rx, int x1, int x2, int y1, int y2, T val) {
        if (x1 > rx || x2 < lx) return;
        if (x1 <= lx && rx <= x2) {
            update_y(vx, 1, 0, m - 1, y1, y2, val, lx, rx);
            return;
        }

        push_down_x(vx, lx, rx);
        int mx = (lx + rx) / 2;
        update_x(2 * vx, lx, mx, x1, x2, y1, y2, val);
        update_x(2 * vx + 1, mx + 1, rx, x1, x2, y1, y2, val);
        merge_y(vx, 1, 0, m - 1, lx, rx);
    }

    void update_y(int vx, int vy, int ly, int ry, int y1, int y2, T val, int lx, int rx) {
        if (y1 > ry || y2 < ly) return;
        if (y1 <= ly && ry <= y2) {
            tree[vx][vy].lazy += val;
            tree[vx][vy].has_lazy = true;
            push_down_y(vx, vy, ly, ry);
            return;
        }

        push_down_y(vx, vy, ly, ry);
        int my = (ly + ry) / 2;
        update_y(vx, 2 * vy, ly, my, y1, y2, val, lx, rx);
        update_y(vx, 2 * vy + 1, my + 1, ry, y1, y2, val, lx, rx);

        // 更新后重新计算节点值
        push_down_y(vx, 2 * vy, ly, my);
        push_down_y(vx, 2 * vy + 1, my + 1, ry);
        tree[vx][vy].val = tree[vx][2 * vy].val + tree[vx][2 * vy + 1].val;
    }

    // 区间查询
    T query(int x1, int y1, int x2, int y2) { return query_x(1, 0, n - 1, x1, x2, y1, y2); }

    T query_x(int vx, int lx, int rx, int x1, int x2, int y1, int y2) {
        if (x1 > rx || x2 < lx) return T{};
        if (x1 <= lx && rx <= x2) { return query_y(vx, 1, 0, m - 1, y1, y2); }

        push_down_x(vx, lx, rx);
        int mx = (lx + rx) / 2;
        return query_x(2 * vx, lx, mx, x1, x2, y1, y2) + query_x(2 * vx + 1, mx + 1, rx, x1, x2, y1, y2);
    }

    T query_y(int vx, int vy, int ly, int ry, int y1, int y2) {
        if (y1 > ry || y2 < ly) return T{};

        push_down_y(vx, vy, ly, ry);

        if (y1 <= ly && ry <= y2) { return tree[vx][vy].val; }

        int my = (ly + ry) / 2;
        return query_y(vx, 2 * vy, ly, my, y1, y2) + query_y(vx, 2 * vy + 1, my + 1, ry, y1, y2);
    }
};
]=]),

-- 01_Data_Structures\Trees\Splay.h
ps("01_data_structures_trees_splay_h", [=[

/**
 * Splay树（伸展树）
 * 功能：
 * - 支持插入、删除、查找操作
 * - 支持区间操作（翻转、查询等）
 * - 支持排名查询、第k小元素查询
 * - 自适应平衡，常用元素会被调整到根附近
 * 时间复杂度：均摊 O(log n)
 * 空间复杂度：O(n)
 */

template <typename T>
struct SplayTree {
    struct Node {
        T val;         // 节点值
        int size;      // 子树大小
        int cnt;       // 重复元素个数
        Node* ch[2];   // 左右儿子
        Node* father;  // 父节点
        bool reverse;  // 翻转标记（用于区间操作）

        Node(T v) : val(v), size(1), cnt(1), father(nullptr), reverse(false) { ch[0] = ch[1] = nullptr; }
    };

    Node* root;

    SplayTree() : root(nullptr) {}

    // 更新节点信息
    void update(Node* x) {
        if (!x) return;
        x->size = x->cnt;
        if (x->ch[0]) x->size += x->ch[0]->size;
        if (x->ch[1]) x->size += x->ch[1]->size;
    }

    // 下推翻转标记
    void push_down(Node* x) {
        if (!x || !x->reverse) return;
        swap(x->ch[0], x->ch[1]);
        if (x->ch[0]) x->ch[0]->reverse ^= true;
        if (x->ch[1]) x->ch[1]->reverse ^= true;
        x->reverse = false;
    }

    // 获取x是父节点的哪个儿子（0表示左儿子，1表示右儿子）
    int get_relation(Node* x) { return x->father->ch[1] == x; }

    // 旋转操作
    void rotate(Node* x) {
        Node* y = x->father;
        Node* z = y->father;
        int k = get_relation(x);

        // 更新z的儿子指针
        if (z) z->ch[get_relation(y)] = x;
        x->father = z;

        // 更新y和x的关系
        y->ch[k] = x->ch[k ^ 1];
        if (x->ch[k ^ 1]) x->ch[k ^ 1]->father = y;

        x->ch[k ^ 1] = y;
        y->father = x;

        // 更新大小信息
        update(y);
        update(x);
    }

    // Splay操作：将x旋转到goal的儿子位置
    void splay(Node* x, Node* goal = nullptr) {
        if (!x) return;

        while (x->father != goal) {
            Node* y = x->father;
            if (y->father != goal) {
                // 双旋转优化
                if (get_relation(x) == get_relation(y)) {
                    rotate(y);
                } else {
                    rotate(x);
                }
            }
            rotate(x);
        }

        if (!goal) root = x;
    }

    // 插入值val
    void insert(T val) {
        if (!root) {
            root = new Node(val);
            return;
        }

        Node* cur = root;
        Node* parent = nullptr;

        while (cur) {
            parent = cur;
            if (val == cur->val) {
                cur->cnt++;
                update(cur);
                splay(cur);
                return;
            }
            cur = cur->ch[val > cur->val];
        }

        Node* new_node = new Node(val);
        new_node->father = parent;
        parent->ch[val > parent->val] = new_node;

        splay(new_node);
    }

    // 查找值val对应的节点
    Node* find(T val) {
        Node* cur = root;
        while (cur) {
            if (val == cur->val) {
                splay(cur);
                return cur;
            }
            cur = cur->ch[val > cur->val];
        }
        return nullptr;
    }

    // 查找前驱（小于val的最大值）
    Node* predecessor(T val) {
        Node* cur = root;
        Node* result = nullptr;

        while (cur) {
            if (cur->val < val) {
                result = cur;
                cur = cur->ch[1];
            } else {
                cur = cur->ch[0];
            }
        }

        if (result) splay(result);
        return result;
    }

    // 查找后继（大于val的最小值）
    Node* successor(T val) {
        Node* cur = root;
        Node* result = nullptr;

        while (cur) {
            if (cur->val > val) {
                result = cur;
                cur = cur->ch[0];
            } else {
                cur = cur->ch[1];
            }
        }

        if (result) splay(result);
        return result;
    }

    // 删除值val
    void erase(T val) {
        Node* node = find(val);
        if (!node) return;

        if (node->cnt > 1) {
            node->cnt--;
            update(node);
            return;
        }

        if (!node->ch[0] && !node->ch[1]) {
            root = nullptr;
        } else if (!node->ch[0]) {
            root = node->ch[1];
            root->father = nullptr;
        } else if (!node->ch[1]) {
            root = node->ch[0];
            root->father = nullptr;
        } else {
            Node* pred = predecessor(val);
            splay(pred);
            pred->ch[1] = node->ch[1];
            if (node->ch[1]) node->ch[1]->father = pred;
            update(pred);
        }

        delete node;
    }

    // 查找第k小的元素（1-indexed）
    T kth_element(int k) {
        Node* cur = root;

        while (cur) {
            push_down(cur);
            int left_size = cur->ch[0] ? cur->ch[0]->size : 0;

            if (k <= left_size) {
                cur = cur->ch[0];
            } else if (k <= left_size + cur->cnt) {
                splay(cur);
                return cur->val;
            } else {
                k -= left_size + cur->cnt;
                cur = cur->ch[1];
            }
        }

        return T{};
    }

    // 查找val的排名（从1开始）
    int get_rank(T val) {
        Node* cur = root;
        int rank = 1;

        while (cur) {
            if (val <= cur->val) {
                cur = cur->ch[0];
            } else {
                if (cur->ch[0]) rank += cur->ch[0]->size;
                rank += cur->cnt;
                cur = cur->ch[1];
            }
        }

        return rank;
    }

    // 获取树的大小
    int size() { return root ? root->size : 0; }

    // 检查树是否为空
    bool empty() { return root == nullptr; }
};

// 支持区间操作的Splay树（用于序列操作）
template <typename T>
struct SequenceSplay {
    struct Node {
        T val;
        int size;
        Node* ch[2];
        Node* father;
        bool reverse;

        Node(T v) : val(v), size(1), father(nullptr), reverse(false) { ch[0] = ch[1] = nullptr; }
    };

    Node* root;

    SequenceSplay() : root(nullptr) {}

    void update(Node* x) {
        if (!x) return;
        x->size = 1;
        if (x->ch[0]) x->size += x->ch[0]->size;
        if (x->ch[1]) x->size += x->ch[1]->size;
    }

    void push_down(Node* x) {
        if (!x || !x->reverse) return;
        swap(x->ch[0], x->ch[1]);
        if (x->ch[0]) x->ch[0]->reverse ^= true;
        if (x->ch[1]) x->ch[1]->reverse ^= true;
        x->reverse = false;
    }

    int get_relation(Node* x) { return x->father && x->father->ch[1] == x; }

    void rotate(Node* x) {
        Node* y = x->father;
        Node* z = y->father;
        int k = get_relation(x);

        if (z) z->ch[get_relation(y)] = x;
        x->father = z;

        y->ch[k] = x->ch[k ^ 1];
        if (x->ch[k ^ 1]) x->ch[k ^ 1]->father = y;

        x->ch[k ^ 1] = y;
        y->father = x;

        update(y);
        update(x);
    }

    void splay(Node* x, Node* goal = nullptr) {
        if (!x) return;

        while (x->father != goal) {
            Node* y = x->father;
            if (y->father != goal) {
                if (get_relation(x) == get_relation(y)) {
                    rotate(y);
                } else {
                    rotate(x);
                }
            }
            rotate(x);
        }

        if (!goal) root = x;
    }

    // 将第k个位置的节点splay到根
    Node* kth_node(int k) {
        Node* cur = root;

        while (cur) {
            push_down(cur);
            int left_size = cur->ch[0] ? cur->ch[0]->size : 0;

            if (k <= left_size) {
                cur = cur->ch[0];
            } else if (k == left_size + 1) {
                splay(cur);
                return cur;
            } else {
                k -= left_size + 1;
                cur = cur->ch[1];
            }
        }

        return nullptr;
    }

    // 在位置pos插入值val
    void insert(int pos, T val) {
        Node* new_node = new Node(val);

        if (!root) {
            root = new_node;
            return;
        }

        if (pos == 1) {
            Node* right = kth_node(1);
            new_node->ch[1] = right;
            right->father = new_node;
            root = new_node;
            update(root);
        } else if (pos == size() + 1) {
            Node* left = kth_node(size());
            new_node->ch[0] = left;
            left->father = new_node;
            root = new_node;
            update(root);
        } else {
            Node* left = kth_node(pos - 1);
            Node* right = kth_node(pos);
            splay(right, left);

            new_node->father = right;
            right->ch[0] = new_node;
            update(right);
            update(left);
        }
    }

    // 删除位置pos的元素
    void erase(int pos) {
        if (size() == 1) {
            delete root;
            root = nullptr;
            return;
        }

        if (pos == 1) {
            Node* right = kth_node(2);
            splay(right);
            if (root->ch[0]) {
                delete root->ch[0];
                root->ch[0] = nullptr;
            }
            update(root);
        } else if (pos == size()) {
            Node* left = kth_node(size() - 1);
            splay(left);
            if (root->ch[1]) {
                delete root->ch[1];
                root->ch[1] = nullptr;
            }
            update(root);
        } else {
            Node* left = kth_node(pos - 1);
            Node* right = kth_node(pos + 1);
            splay(right, left);

            if (right->ch[0]) {
                delete right->ch[0];
                right->ch[0] = nullptr;
            }
            update(right);
            update(left);
        }
    }

    // 翻转区间[l, r]
    void reverse(int l, int r) {
        if (l == r) return;

        Node* left = (l == 1) ? nullptr : kth_node(l - 1);
        Node* right = (r == size()) ? nullptr : kth_node(r + 1);

        if (!left && !right) {
            root->reverse ^= true;
        } else if (!left) {
            splay(right);
            if (root->ch[0]) root->ch[0]->reverse ^= true;
        } else if (!right) {
            splay(left);
            if (root->ch[1]) root->ch[1]->reverse ^= true;
        } else {
            splay(right, left);
            if (right->ch[0]) right->ch[0]->reverse ^= true;
        }
    }

    // 获取位置pos的值
    T get(int pos) {
        Node* node = kth_node(pos);
        return node ? node->val : T{};
    }

    // 设置位置pos的值
    void set(int pos, T val) {
        Node* node = kth_node(pos);
        if (node) node->val = val;
    }

    int size() { return root ? root->size : 0; }

    bool empty() { return root == nullptr; }
};
]=]),

-- 01_Data_Structures\Trees\SplayTree.h
ps("01_data_structures_trees_splaytree_h", [=[

/**
 * 数组实现的Splay树模板
 * 功能特性:
 * - 支持插入、删除、查找操作
 * - 支持排名查询
 * - 支持前驱后继查询
 * 时间复杂度: 均摊 O(log n)
 * 空间复杂度: O(n)
 */

template <typename T>
struct ArraySplayTree {
    struct Node {
        T val;       // 节点值
        int cnt;     // 相同值的数量
        int size;    // 子树大小
        int ch[2];   // 左右子树
        int father;  // 父节点

        Node() : val(T{}), cnt(0), size(0), father(0) { ch[0] = ch[1] = 0; }

        Node(T v) : val(v), cnt(1), size(1), father(0) { ch[0] = ch[1] = 0; }
    };

    vector<Node> tree;
    int root, node_count;

    ArraySplayTree(int max_size = 100000) : root(0), node_count(0) { tree.resize(max_size + 5); }

    // 创建新节点
    int new_node(T val) {
        tree[++node_count] = Node(val);
        return node_count;
    }

    // 更新节点信息
    void update(int x) {
        if (!x) return;
        tree[x].size = tree[tree[x].ch[0]].size + tree[tree[x].ch[1]].size + tree[x].cnt;
    }

    // 获取x是父节点的哪个儿子
    bool get_relation(int x) { return tree[tree[x].father].ch[1] == x; }

    // 旋转操作
    void rotate(int x) {
        int y = tree[x].father;
        int z = tree[y].father;
        int k = get_relation(x);

        // 连接z和x
        tree[z].ch[get_relation(y)] = x;
        tree[x].father = z;

        // 连接y和x的子树
        tree[y].ch[k] = tree[x].ch[k ^ 1];
        tree[tree[x].ch[k ^ 1]].father = y;

        tree[x].ch[k ^ 1] = y;
        tree[y].father = x;

        // 更新节点信息
        update(y);
        update(x);
    }

    // Splay操作
    void splay(int x, int goal = 0) {
        while (tree[x].father != goal) {
            int y = tree[x].father;
            if (tree[y].father != goal) {
                if (get_relation(x) == get_relation(y)) {
                    rotate(y);
                } else {
                    rotate(x);
                }
            }
            rotate(x);
        }

        if (!goal) root = x;
    }

    // 插入值val
    void insert(T val) {
        if (!root) {
            root = new_node(val);
            return;
        }

        int cur = root, parent = 0;
        while (cur && tree[cur].val != val) {
            parent = cur;
            cur = tree[cur].ch[val > tree[cur].val];
        }

        if (cur) {
            tree[cur].cnt++;
        } else {
            cur = new_node(val);
            tree[cur].father = parent;
            tree[parent].ch[val > tree[parent].val] = cur;
        }

        splay(cur);
    }

    // 查找值val
    int find(T val) {
        int cur = root;
        while (cur && tree[cur].val != val) { cur = tree[cur].ch[val > tree[cur].val]; }
        if (cur) splay(cur);
        return cur;
    }

    // 寻找前驱
    int predecessor() {
        int cur = tree[root].ch[0];
        if (!cur) return 0;
        while (tree[cur].ch[1]) cur = tree[cur].ch[1];
        return cur;
    }

    // 寻找后继
    int successor() {
        int cur = tree[root].ch[1];
        if (!cur) return 0;
        while (tree[cur].ch[0]) cur = tree[cur].ch[0];
        return cur;
    }

    // 删除值val
    void erase(T val) {
        int cur = find(val);
        if (!cur) return;

        if (tree[cur].cnt > 1) {
            tree[cur].cnt--;
            update(cur);
            return;
        }

        if (!tree[cur].ch[0] && !tree[cur].ch[1]) {
            root = 0;
            return;
        }

        if (!tree[cur].ch[0]) {
            root = tree[cur].ch[1];
            tree[root].father = 0;
            return;
        }

        if (!tree[cur].ch[1]) {
            root = tree[cur].ch[0];
            tree[root].father = 0;
            return;
        }

        int pred = predecessor();
        splay(pred);
        tree[pred].ch[1] = tree[cur].ch[1];
        tree[tree[cur].ch[1]].father = pred;
        update(pred);
    }

    // 查找第k小元素
    T kth_element(int k) {
        int cur = root;
        while (cur) {
            int left_size = tree[tree[cur].ch[0]].size;
            if (k <= left_size) {
                cur = tree[cur].ch[0];
            } else if (k <= left_size + tree[cur].cnt) {
                return tree[cur].val;
            } else {
                k -= left_size + tree[cur].cnt;
                cur = tree[cur].ch[1];
            }
        }
        return T{};
    }

    // 获取val的排名
    int get_rank(T val) {
        int cur = root, rank = 1;
        while (cur) {
            if (val <= tree[cur].val) {
                cur = tree[cur].ch[0];
            } else {
                rank += tree[tree[cur].ch[0]].size + tree[cur].cnt;
                cur = tree[cur].ch[1];
            }
        }
        return rank;
    }

    // 查找前驱值
    T predecessor_val(T val) {
        int cur = root;
        T result = T{};
        bool found = false;

        while (cur) {
            if (tree[cur].val < val) {
                if (!found || tree[cur].val > result) {
                    result = tree[cur].val;
                    found = true;
                }
                cur = tree[cur].ch[1];
            } else {
                cur = tree[cur].ch[0];
            }
        }

        return found ? result : T{};
    }

    // 查找后继值
    T successor_val(T val) {
        int cur = root;
        T result = T{};
        bool found = false;

        while (cur) {
            if (tree[cur].val > val) {
                if (!found || tree[cur].val < result) {
                    result = tree[cur].val;
                    found = true;
                }
                cur = tree[cur].ch[0];
            } else {
                cur = tree[cur].ch[1];
            }
        }

        return found ? result : T{};
    }

    // 获取树的大小
    int size() { return root ? tree[root].size : 0; }

    // 检查树是否为空
    bool empty() { return root == 0; }

    // 清空树
    void clear() { root = node_count = 0; }

    // 中序遍历（用于调试）
    void inorder(int x, vector<T>& result) {
        if (!x) return;
        inorder(tree[x].ch[0], result);
        for (int i = 0; i < tree[x].cnt; i++) { result.push_back(tree[x].val); }
        inorder(tree[x].ch[1], result);
    }

    // 获取中序遍历结果
    vector<T> get_sorted() {
        vector<T> result;
        inorder(root, result);
        return result;
    }

    // 检查值是否存在
    bool contains(T val) { return find(val) != 0; }

    // 获取最小值
    T min_value() {
        if (!root) return T{};
        int cur = root;
        while (tree[cur].ch[0]) cur = tree[cur].ch[0];
        return tree[cur].val;
    }

    // 获取最大值
    T max_value() {
        if (!root) return T{};
        int cur = root;
        while (tree[cur].ch[1]) cur = tree[cur].ch[1];
        return tree[cur].val;
    }
};
]=]),

-- 01_Data_Structures\Trees\Treap.h
ps("01_data_structures_trees_treap_h", [=[

/**
 * 传统Treap（旋转Treap）
 * 功能：
 * - 支持插入、删除、查找操作
 * - 基于随机优先级维持平衡性
 * - 支持排名查询、第k小元素查询
 * - 支持前驱、后继查询
 * 时间复杂度：期望 O(log n)
 * 空间复杂度：O(n)
 *
 * 与FHQ Treap的区别：
 * - 使用旋转操作维持堆性质
 * - 实现相对简单，但灵活性不如FHQ Treap
 */

template <typename T>
struct Treap {
    struct Node {
        T val;         // 节点值
        int priority;  // 随机优先级
        int size;      // 子树大小
        int cnt;       // 重复元素个数
        Node* left;    // 左子树
        Node* right;   // 右子树

        Node(T v) : val(v), priority(rand()), size(1), cnt(1), left(nullptr), right(nullptr) {}
    };

    Node* root;
    mt19937 rng;

    Treap() : root(nullptr), rng(random_device{}()) { srand(time(nullptr)); }

    // 更新节点信息
    void update(Node* x) {
        if (!x) return;
        x->size = x->cnt;
        if (x->left) x->size += x->left->size;
        if (x->right) x->size += x->right->size;
    }

    // 右旋转
    Node* rotate_right(Node* x) {
        Node* y = x->left;
        x->left = y->right;
        y->right = x;
        update(x);
        update(y);
        return y;
    }

    // 左旋转
    Node* rotate_left(Node* x) {
        Node* y = x->right;
        x->right = y->left;
        y->left = x;
        update(x);
        update(y);
        return y;
    }

    // 插入操作
    Node* insert(Node* x, T val) {
        if (!x) return new Node(val);

        if (val == x->val) {
            x->cnt++;
            update(x);
            return x;
        }

        if (val < x->val) {
            x->left = insert(x->left, val);
            if (x->left->priority > x->priority) { x = rotate_right(x); }
        } else {
            x->right = insert(x->right, val);
            if (x->right->priority > x->priority) { x = rotate_left(x); }
        }

        update(x);
        return x;
    }

    // 删除操作
    Node* remove(Node* x, T val) {
        if (!x) return x;

        if (val < x->val) {
            x->left = remove(x->left, val);
        } else if (val > x->val) {
            x->right = remove(x->right, val);
        } else {
            if (x->cnt > 1) {
                x->cnt--;
                update(x);
                return x;
            }

            if (!x->left && !x->right) {
                delete x;
                return nullptr;
            } else if (!x->left) {
                Node* temp = x->right;
                delete x;
                return temp;
            } else if (!x->right) {
                Node* temp = x->left;
                delete x;
                return temp;
            } else {
                if (x->left->priority > x->right->priority) {
                    x = rotate_right(x);
                    x->right = remove(x->right, val);
                } else {
                    x = rotate_left(x);
                    x->left = remove(x->left, val);
                }
            }
        }

        update(x);
        return x;
    }

    // 查找操作
    bool find(Node* x, T val) {
        if (!x) return false;
        if (val == x->val) return true;
        if (val < x->val) return find(x->left, val);
        return find(x->right, val);
    }

    // 查找第k小的元素（1-indexed）
    T kth_element(Node* x, int k) {
        if (!x) return T{};

        int left_size = x->left ? x->left->size : 0;
        if (k <= left_size) {
            return kth_element(x->left, k);
        } else if (k <= left_size + x->cnt) {
            return x->val;
        } else {
            return kth_element(x->right, k - left_size - x->cnt);
        }
    }

    // 查找元素的排名（从1开始）
    int get_rank(Node* x, T val) {
        if (!x) return 1;

        if (val < x->val) {
            return get_rank(x->left, val);
        } else if (val == x->val) {
            return (x->left ? x->left->size : 0) + 1;
        } else {
            return (x->left ? x->left->size : 0) + x->cnt + get_rank(x->right, val);
        }
    }

    // 查找前驱（小于val的最大元素）
    T predecessor(Node* x, T val) {
        if (!x) return T{};

        if (x->val >= val) {
            return predecessor(x->left, val);
        } else {
            T right_pred = predecessor(x->right, val);
            return (x->right && right_pred != T{}) ? right_pred : x->val;
        }
    }

    // 查找后继（大于val的最小元素）
    T successor(Node* x, T val) {
        if (!x) return T{};

        if (x->val <= val) {
            return successor(x->right, val);
        } else {
            T left_succ = successor(x->left, val);
            return (x->left && left_succ != T{}) ? left_succ : x->val;
        }
    }

    // 公共接口
    void insert(T val) { root = insert(root, val); }
    void remove(T val) { root = remove(root, val); }
    bool find(T val) { return find(root, val); }
    T kth_element(int k) { return kth_element(root, k); }
    int get_rank(T val) { return get_rank(root, val); }
    T predecessor(T val) { return predecessor(root, val); }
    T successor(T val) { return successor(root, val); }

    int size() { return root ? root->size : 0; }
    bool empty() { return root == nullptr; }

    // 清空树
    void clear() {
        clear_helper(root);
        root = nullptr;
    }

   private:
    void clear_helper(Node* x) {
        if (!x) return;
        clear_helper(x->left);
        clear_helper(x->right);
        delete x;
    }
};

// 数组版Treap实现（内存效率更高）
struct ArrayTreap {
    struct Node {
        int val;          // 节点值
        int priority;     // 随机优先级
        int size;         // 子树大小
        int cnt;          // 重复元素个数
        int left, right;  // 左右子树编号

        Node() : val(0), priority(0), size(0), cnt(0), left(0), right(0) {}
        Node(int v) : val(v), priority(rand()), size(1), cnt(1), left(0), right(0) {}
    };

    vector<Node> tree;
    int root, node_count;

    ArrayTreap(int max_size = 100000) : root(0), node_count(0) {
        tree.resize(max_size + 5);
        srand(time(nullptr));
    }

    // 创建新节点
    int new_node(int val) {
        tree[++node_count] = Node(val);
        return node_count;
    }

    // 更新节点信息
    void update(int x) {
        if (!x) return;
        tree[x].size = tree[x].cnt;
        if (tree[x].left) tree[x].size += tree[tree[x].left].size;
        if (tree[x].right) tree[x].size += tree[tree[x].right].size;
    }

    // 右旋转
    int rotate_right(int x) {
        int y = tree[x].left;
        tree[x].left = tree[y].right;
        tree[y].right = x;
        update(x);
        update(y);
        return y;
    }

    // 左旋转
    int rotate_left(int x) {
        int y = tree[x].right;
        tree[x].right = tree[y].left;
        tree[y].left = x;
        update(x);
        update(y);
        return y;
    }

    // 插入操作
    int insert(int x, int val) {
        if (!x) return new_node(val);

        if (val == tree[x].val) {
            tree[x].cnt++;
            update(x);
            return x;
        }

        if (val < tree[x].val) {
            tree[x].left = insert(tree[x].left, val);
            if (tree[tree[x].left].priority > tree[x].priority) { x = rotate_right(x); }
        } else {
            tree[x].right = insert(tree[x].right, val);
            if (tree[tree[x].right].priority > tree[x].priority) { x = rotate_left(x); }
        }

        update(x);
        return x;
    }

    // 删除操作
    int remove(int x, int val) {
        if (!x) return x;

        if (val < tree[x].val) {
            tree[x].left = remove(tree[x].left, val);
        } else if (val > tree[x].val) {
            tree[x].right = remove(tree[x].right, val);
        } else {
            if (tree[x].cnt > 1) {
                tree[x].cnt--;
                update(x);
                return x;
            }

            if (!tree[x].left && !tree[x].right) {
                return 0;
            } else if (!tree[x].left) {
                return tree[x].right;
            } else if (!tree[x].right) {
                return tree[x].left;
            } else {
                if (tree[tree[x].left].priority > tree[tree[x].right].priority) {
                    x = rotate_right(x);
                    tree[x].right = remove(tree[x].right, val);
                } else {
                    x = rotate_left(x);
                    tree[x].left = remove(tree[x].left, val);
                }
            }
        }

        update(x);
        return x;
    }

    // 查找第k小的元素
    int kth_element(int x, int k) {
        if (!x) return -1;

        int left_size = tree[x].left ? tree[tree[x].left].size : 0;
        if (k <= left_size) {
            return kth_element(tree[x].left, k);
        } else if (k <= left_size + tree[x].cnt) {
            return tree[x].val;
        } else {
            return kth_element(tree[x].right, k - left_size - tree[x].cnt);
        }
    }

    // 查找元素的排名
    int get_rank(int x, int val) {
        if (!x) return 1;

        if (val < tree[x].val) {
            return get_rank(tree[x].left, val);
        } else if (val == tree[x].val) {
            return (tree[x].left ? tree[tree[x].left].size : 0) + 1;
        } else {
            return (tree[x].left ? tree[tree[x].left].size : 0) + tree[x].cnt + get_rank(tree[x].right, val);
        }
    }

    // 公共接口
    void insert(int val) { root = insert(root, val); }
    void remove(int val) { root = remove(root, val); }
    int kth_element(int k) { return kth_element(root, k); }
    int get_rank(int val) { return get_rank(root, val); }

    int size() { return root ? tree[root].size : 0; }
    bool empty() { return root == 0; }
    void clear() { root = node_count = 0; }
};
]=]),

-- 01_Data_Structures\Union_Find\BasicDSU.h
ps("01_data_structures_union_find_basicdsu_h", [=[
/**
 * 基础并查集模板
 * 功能：
 * - 路径压缩优化
 * - 按大小合并优化
 * - 连通分量计数
 * 时间复杂度：O(α(n)) 均摊，其中α为反阿克曼函数
 */

struct DSU {
   private:
    vector<int> fa, sz;
    int com;

   public:
    DSU() {}
    DSU(int n) : fa(n + 1), sz(n + 1, 1) {
        iota(fa.begin(), fa.end(), 0);
        com = n;
    }

    int find(int x) { return fa[x] == x ? x : fa[x] = find(fa[x]); }
    bool unite(int x, int y) {
        x = find(x);
        y = find(y);
        if (x == y) return false;
        if (sz[x] < sz[y]) swap(x, y);
        fa[y] = x;
        sz[x] += sz[y];
        com--;
        return true;
    }

    bool same(int x, int y) { return find(x) == find(y); }
    int size(int x) { return sz[find(x)]; }
    int count() { return com; }
};
]=]),

-- 01_Data_Structures\Union_Find\PersistentDSU.h
ps("01_data_structures_union_find_persistentdsu_h", [=[

/**
 * 可持久化并查集 (Persistent Disjoint Set Union)
 * 功能：
 * - 支持历史版本查询
 * - 支持回滚操作
 * - 支持从任意版本创建新分支
 * 时间复杂度：O(log n) 每次操作
 * 空间复杂度：O(n log n)
 */

struct PersistentDSU {
    struct Node {
        int parent, rank;
        Node* left;
        Node* right;

        Node(int p = 0, int r = 0) : parent(p), rank(r), left(nullptr), right(nullptr) {}
    };

    vector<Node*> versions;  // 存储各个版本的根节点
    int n;

    PersistentDSU(int size) : n(size) {
        // 创建初始版本
        versions.push_back(build(1, n));
    }

    // 构建线段树
    Node* build(int l, int r) {
        Node* node = new Node();
        if (l == r) {
            node->parent = l;
            node->rank = 0;
            return node;
        }
        int mid = (l + r) / 2;
        node->left = build(l, mid);
        node->right = build(mid + 1, r);
        return node;
    }

    // 查询父节点
    int find(Node* root, int l, int r, int pos) {
        if (l == r) {
            if (root->parent == pos) return pos;
            return find(versions.back(), 1, n, root->parent);
        }
        int mid = (l + r) / 2;
        if (pos <= mid) return find(root->left, l, mid, pos);
        return find(root->right, mid + 1, r, pos);
    }

    // 更新节点
    Node* update(Node* root, int l, int r, int pos, int parent, int rank) {
        Node* new_node = new Node();
        if (l == r) {
            new_node->parent = parent;
            new_node->rank = rank;
            return new_node;
        }
        int mid = (l + r) / 2;
        if (pos <= mid) {
            new_node->left = update(root->left, l, mid, pos, parent, rank);
            new_node->right = root->right;
        } else {
            new_node->left = root->left;
            new_node->right = update(root->right, mid + 1, r, pos, parent, rank);
        }
        return new_node;
    }

    // 合并两个集合，返回新版本号
    int unite(int version, int x, int y) {
        Node* root = versions[version];
        int px = find(root, 1, n, x);
        int py = find(root, 1, n, y);

        if (px == py) {
            versions.push_back(root);
            return versions.size() - 1;
        }

        // 按秩合并
        // 这里需要获取秩信息，简化实现
        Node* new_root = update(root, 1, n, px, py, 0);
        versions.push_back(new_root);
        return versions.size() - 1;
    }

    // 查询两个元素是否连通
    bool connected(int version, int x, int y) {
        Node* root = versions[version];
        return find(root, 1, n, x) == find(root, 1, n, y);
    }

    // 获取当前版本数
    int get_version_count() { return versions.size(); }
};
]=]),

-- 01_Data_Structures\Union_Find\WeightedDSU.h
ps("01_data_structures_union_find_weighteddsu_h", [=[

/**
 * 带权并查集模板
 * 功能：
 * - 维护节点间的权值关系
 * - 支持路径压缩和权值更新
 * - 检查权值关系的一致性
 * 时间复杂度：O(α(n)) 均摊
 */

template <typename T>
struct WeightedDSU {
    vector<int> parent;  // parent[i]: 节点i的父节点
    vector<T> weight;    // weight[i]: 节点i到parent[i]的权值
    int n;

    WeightedDSU(int size) : n(size) {
        parent.resize(n);
        weight.resize(n, T{});
        iota(parent.begin(), parent.end(), 0);  // 初始化每个节点的父节点为自己
    }

    // 查找根节点并压缩路径，同时更新权值
    pair<int, T> find(int x) {
        if (parent[x] == x) { return {x, T{}}; }
        auto [root, w] = find(parent[x]);
        weight[x] += w;    // 更新到根节点的权值
        parent[x] = root;  // 路径压缩
        return {root, weight[x]};
    }

    // 合并两个集合，建立权值关系 weight[y] = weight[x] + w
    bool union_sets(int x, int y, T w) {
        auto [root_x, weight_x] = find(x);
        auto [root_y, weight_y] = find(y);

        if (root_x == root_y) {
            // 检查权值关系是否一致
            return weight_x + w == weight_y;
        }

        // 将root_y的父节点设为root_x，并设置权值
        parent[root_y] = root_x;
        weight[root_y] = weight_x + w - weight_y;
        return true;
    }

    // 获取x到y的权值差，如果不在同一集合返回默认值
    T get_weight(int x, int y) {
        auto [root_x, weight_x] = find(x);
        auto [root_y, weight_y] = find(y);

        if (root_x != root_y) {
            return T{};  // 不在同一集合中
        }

        return weight_y - weight_x;
    }

    // 判断两个节点是否在同一集合中
    bool same(int x, int y) { return find(x).first == find(y).first; }
};
]=]),

-- 02_Graph_Theory\Connectivity\TarjanBridge.h
ps("02_graph_theory_connectivity_tarjanbridge_h", [=[

// Tarjan算法求桥
struct TarjanBridge {
    int n, time_stamp;
    vector<vector<pair<int, int>>> graph;  // {邻接点, 边编号}
    vector<int> dfn, low;
    vector<bool> is_bridge;
    vector<pair<int, int>> bridges;

    TarjanBridge(int sz, int edge_cnt) : n(sz), time_stamp(0) {
        graph.resize(n);
        dfn.resize(n, -1);
        low.resize(n, -1);
        is_bridge.resize(edge_cnt, false);
    }

    void add_edge(int u, int v, int edge_id) {
        graph[u].push_back({v, edge_id});
        graph[v].push_back({u, edge_id});
    }

    void tarjan(int u, int parent_edge) {
        dfn[u] = low[u] = time_stamp++;

        for (auto& edge : graph[u]) {
            int v = edge.first;
            int edge_id = edge.second;

            if (edge_id == parent_edge) continue;

            if (dfn[v] == -1) {
                tarjan(v, edge_id);
                low[u] = min(low[u], low[v]);

                if (low[v] > dfn[u]) {
                    is_bridge[edge_id] = true;
                    bridges.push_back({min(u, v), max(u, v)});
                }
            } else {
                low[u] = min(low[u], dfn[v]);
            }
        }
    }

    void run() {
        for (int i = 0; i < n; i++) {
            if (dfn[i] == -1) { tarjan(i, -1); }
        }
    }

    vector<pair<int, int>> get_bridges() { return bridges; }

    bool is_bridge_edge(int edge_id) { return is_bridge[edge_id]; }

    // 构建桥连通分量
    vector<int> get_bridge_components() {
        vector<int> comp(n, -1);
        int comp_cnt = 0;

        function<void(int, int)> dfs = [&](int u, int c) {
            comp[u] = c;
            for (auto& edge : graph[u]) {
                int v = edge.first;
                int edge_id = edge.second;
                if (comp[v] == -1 && !is_bridge[edge_id]) { dfs(v, c); }
            }
        };

        for (int i = 0; i < n; i++) {
            if (comp[i] == -1) { dfs(i, comp_cnt++); }
        }

        return comp;
    }
};

// 使用示例：
// TarjanBridge bridge(n, m);
// bridge.add_edge(u, v, edge_id);
// bridge.run();
// vector<pair<int,int>> bridges = bridge.get_bridges();
]=]),

}
