-- Auto-generated LuaSnip snippets
local ls = require("luasnip")
local ps = ls.parser.parse_snippet

return {

-- 05_Geometry\Utils\Precision.h
ps("05_geometry_utils_precision_h", [=[

/*
 * 几何精度处理工具
 * 时间复杂度: O(1)
 * 空间复杂度: O(1)
 * 适用场景: 几何计算中的精度控制和浮点数比较
 */

const double EPS = 1e-9;
const double PI = acos(-1.0);

// 浮点数比较函数
int dcmp(double x) {
    if (fabs(x) < EPS) return 0;
    return x < 0 ? -1 : 1;
}

bool equal(double a, double b) { return fabs(a - b) < EPS; }
bool less(double a, double b) { return a < b - EPS; }
bool less_equal(double a, double b) { return a < b + EPS; }
bool greater(double a, double b) { return a > b + EPS; }
bool greater_equal(double a, double b) { return a > b - EPS; }

// 角度标准化
double normalize_angle(double angle) {
    while (angle < 0) angle += 2 * PI;
    while (angle >= 2 * PI) angle -= 2 * PI;
    return angle;
}

double normalize_angle_pm_pi(double angle) {
    while (angle < -PI) angle += 2 * PI;
    while (angle > PI) angle -= 2 * PI;
    return angle;
}

// 安全的数学函数
double safe_acos(double x) { return acos(max(-1.0, min(1.0, x))); }
double safe_asin(double x) { return asin(max(-1.0, min(1.0, x))); }
double safe_sqrt(double x) { return sqrt(max(0.0, x)); }

// 数值稳定的二次方程求解
pair<double, double> solve_quadratic(double a, double b, double c) {
    if (fabs(a) < EPS) {
        if (fabs(b) < EPS) return {NAN, NAN};
        return {-c / b, NAN};
    }

    double discriminant = b * b - 4 * a * c;
    if (discriminant < -EPS) return {NAN, NAN};

    discriminant = max(0.0, discriminant);
    double sqrt_d = sqrt(discriminant);

    double x1, x2;
    if (b >= 0) {
        x1 = (-b - sqrt_d) / (2 * a);
        x2 = c / (a * x1);
    } else {
        x2 = (-b + sqrt_d) / (2 * a);
        x1 = c / (a * x2);
    }

    if (x1 > x2) swap(x1, x2);
    return {x1, x2};
}

// 高精度距离计算
double distance2_stable(double x1, double y1, double x2, double y2) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    return dx * dx + dy * dy;
}

// 高精度三角形面积计算（海伦公式的数值稳定版本）
double triangle_area_stable(double x1, double y1, double x2, double y2, double x3, double y3) {
    double a = safe_sqrt(distance2_stable(x2, y2, x3, y3));
    double b = safe_sqrt(distance2_stable(x1, y1, x3, y3));
    double c = safe_sqrt(distance2_stable(x1, y1, x2, y2));

    if (a > b) swap(a, b);
    if (b > c) swap(b, c);
    if (a > b) swap(a, b);

    double s = (a + b + c) / 2;
    double area_sq = s * (s - a) * (s - b) * (s - c);
    return safe_sqrt(area_sq);
}

// 自适应精度处理
class AdaptivePrecision {
   private:
    double base_eps;
    double scale_factor;

   public:
    AdaptivePrecision(double eps = EPS, double scale = 1e6) : base_eps(eps), scale_factor(scale) {}

    double get_eps(double magnitude) const { return base_eps * max(1.0, magnitude / scale_factor); }

    bool equal(double a, double b) const {
        double mag = max(fabs(a), fabs(b));
        return fabs(a - b) < get_eps(mag);
    }

    int sign(double x) const {
        double eps = get_eps(fabs(x));
        if (fabs(x) < eps) return 0;
        return x < 0 ? -1 : 1;
    }
};

// 有理数近似
struct Fraction {
    long long num, den;

    Fraction(long long n = 0, long long d = 1) : num(n), den(d) {
        if (den < 0) {
            num = -num;
            den = -den;
        }
        long long g = __gcd(abs(num), abs(den));
        num /= g;
        den /= g;
    }

    Fraction(double x, long long max_den = 1000000) {
        num = 0;
        den = 1;
        long long a = (long long)floor(x);
        long long p0 = 1, p1 = a, q0 = 0, q1 = 1;

        double r = x - a;
        while (fabs(r) > EPS && q1 <= max_den) {
            r = 1.0 / r;
            a = (long long)floor(r);
            long long p2 = a * p1 + p0;
            long long q2 = a * q1 + q0;

            if (q2 > max_den) break;

            p0 = p1;
            p1 = p2;
            q0 = q1;
            q1 = q2;
            r = r - a;
        }

        num = p1;
        den = q1;
        if (den < 0) {
            num = -num;
            den = -den;
        }
        long long g = __gcd(abs(num), abs(den));
        num /= g;
        den /= g;
    }

    double to_double() const { return (double)num / den; }

    Fraction operator+(const Fraction& f) const { return Fraction(num * f.den + f.num * den, den * f.den); }
    Fraction operator-(const Fraction& f) const { return Fraction(num * f.den - f.num * den, den * f.den); }
    Fraction operator*(const Fraction& f) const { return Fraction(num * f.num, den * f.den); }
    Fraction operator/(const Fraction& f) const { return Fraction(num * f.den, den * f.num); }

    bool operator==(const Fraction& f) const { return num * f.den == f.num * den; }
    bool operator<(const Fraction& f) const { return num * f.den < f.num * den; }
};
]=]),

-- 05_Geometry\Utils\Transformation.h
ps("05_geometry_utils_transformation_h", [=[

/*
 * 几何坐标变换工具
 * 时间复杂度: O(1)
 * 空间复杂度: O(1)
 * 适用场景: 二维几何中的坐标变换操作
 */

const double PI = acos(-1.0);

// 角度弧度转换
double to_radian(double degree) { return degree * PI / 180.0; }
double to_degree(double radian) { return radian * 180.0 / PI; }

// 极坐标结构
struct Polar {
    double r, theta;
    Polar(double r = 0, double theta = 0) : r(r), theta(theta) {}
};

// 直角坐标与极坐标转换
Polar to_polar(double x, double y) { return Polar(sqrt(x * x + y * y), atan2(y, x)); }

pair<double, double> to_cartesian(const Polar& p) { return make_pair(p.r * cos(p.theta), p.r * sin(p.theta)); }

// 二维变换矩阵
struct Transform2D {
    double a, b, c, d, tx, ty;  // 变换矩阵 [a c tx; b d ty; 0 0 1]

    Transform2D() : a(1), b(0), c(0), d(1), tx(0), ty(0) {}

    // 平移变换
    static Transform2D translate(double dx, double dy) {
        Transform2D t;
        t.tx = dx;
        t.ty = dy;
        return t;
    }

    // 旋转变换（绕原点，弧度）
    static Transform2D rotate(double theta) {
        Transform2D t;
        t.a = cos(theta);
        t.c = -sin(theta);
        t.b = sin(theta);
        t.d = cos(theta);
        return t;
    }

    // 缩放变换
    static Transform2D scale(double sx, double sy) {
        Transform2D t;
        t.a = sx;
        t.d = sy;
        return t;
    }

    // 反射变换（沿x轴）
    static Transform2D reflect_x() {
        Transform2D t;
        t.d = -1;
        return t;
    }

    // 反射变换（沿y轴）
    static Transform2D reflect_y() {
        Transform2D t;
        t.a = -1;
        return t;
    }

    // 剪切变换
    static Transform2D shear(double sx, double sy) {
        Transform2D t;
        t.c = sx;
        t.b = sy;
        return t;
    }

    // 变换合成
    Transform2D operator*(const Transform2D& other) const {
        Transform2D result;
        result.a = a * other.a + c * other.b;
        result.b = b * other.a + d * other.b;
        result.c = a * other.c + c * other.d;
        result.d = b * other.c + d * other.d;
        result.tx = a * other.tx + c * other.ty + tx;
        result.ty = b * other.tx + d * other.ty + ty;
        return result;
    }

    // 应用变换
    pair<double, double> apply(double x, double y) const { return make_pair(a * x + c * y + tx, b * x + d * y + ty); }

    // 逆变换
    Transform2D inverse() const {
        double det = a * d - b * c;
        if (fabs(det) < 1e-9) return Transform2D();  // 奇异矩阵

        Transform2D inv;
        inv.a = d / det;
        inv.b = -b / det;
        inv.c = -c / det;
        inv.d = a / det;
        inv.tx = (c * ty - d * tx) / det;
        inv.ty = (b * tx - a * ty) / det;
        return inv;
    }
};

// 点绕某点旋转
pair<double, double> rotate_point(double x, double y, double cx, double cy, double theta) {
    double costh = cos(theta), sinth = sin(theta);
    double dx = x - cx, dy = y - cy;
    return make_pair(cx + dx * costh - dy * sinth, cy + dx * sinth + dy * costh);
}

// 点关于直线对称
pair<double, double> reflect_point(double x, double y, double a, double b, double c) {
    // 直线方程: ax + by + c = 0
    double norm2 = a * a + b * b;
    double t = -(a * x + b * y + c) / norm2;
    return make_pair(x + 2 * a * t, y + 2 * b * t);
}

// 齐次坐标变换
struct HomogeneousTransform {
    double matrix[3][3];

    HomogeneousTransform() {
        memset(matrix, 0, sizeof(matrix));
        matrix[0][0] = matrix[1][1] = matrix[2][2] = 1;
    }

    // 应用变换
    pair<double, double> apply(double x, double y) const {
        double w = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2];
        return make_pair((matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / w,
                         (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / w);
    }

    // 透视变换
    static HomogeneousTransform perspective(double px, double py) {
        HomogeneousTransform t;
        t.matrix[2][0] = px;
        t.matrix[2][1] = py;
        return t;
    }
};

// 仿射变换参数计算
Transform2D compute_affine_transform(const vector<pair<double, double>>& src, const vector<pair<double, double>>& dst) {
    if (src.size() != dst.size() || src.size() < 3) {
        return Transform2D();  // 至少需要3个点
    }

    // 使用最小二乘法计算变换参数
    // 这里实现简化版本，实际应用中可能需要更复杂的算法
    Transform2D t;

    // 简单的平移计算
    double sx = 0, sy = 0, dx = 0, dy = 0;
    for (int i = 0; i < src.size(); i++) {
        sx += src[i].first;
        sy += src[i].second;
        dx += dst[i].first;
        dy += dst[i].second;
    }

    t.tx = dx / src.size() - sx / src.size();
    t.ty = dy / src.size() - sy / src.size();

    return t;
}
]=]),

-- 06_String_Algorithms\Advanced\Booth.h
ps("06_string_algorithms_advanced_booth_h", [=[

/**
 * Booth算法
 * 功能：线性时间内找到字符串的字典序最小循环移位位置
 * 时间复杂度：O(n)
 * 空间复杂度：O(n)
 * 适用场景：字符串最小表示、循环字符串比较、字符串规范化
 */
struct BoothAlgorithm {
    // Booth算法主函数
    static int minimum_rotation(const string& s) {
        string ss = s + s;
        int n = s.length();
        int i = 0, j = 1, k = 0;

        while (i < n && j < n) {
            k = 0;
            while (k < n && ss[i + k] == ss[j + k]) { k++; }

            if (k == n) break;

            if (ss[i + k] > ss[j + k]) {
                i = max(i + k + 1, j + 1);
                if (i == j) i++;
            } else {
                j = max(j + k + 1, i + 1);
                if (i == j) j++;
            }
        }

        return min(i, j);
    }

    // 获取字符串的最小表示
    static string get_minimum_representation(const string& s) {
        int pos = minimum_rotation(s);
        return s.substr(pos) + s.substr(0, pos);
    }

    // 获取字符串的最大表示
    static string get_maximum_representation(const string& s) {
        string ss = s + s;
        int n = s.length();
        int i = 0, j = 1, k = 0;

        while (i < n && j < n) {
            k = 0;
            while (k < n && ss[i + k] == ss[j + k]) { k++; }

            if (k == n) break;

            if (ss[i + k] < ss[j + k]) {  // 注意这里改为小于号
                i = max(i + k + 1, j + 1);
                if (i == j) i++;
            } else {
                j = max(j + k + 1, i + 1);
                if (i == j) j++;
            }
        }

        int pos = min(i, j);
        return s.substr(pos) + s.substr(0, pos);
    }

    // 比较两个字符串的循环表示
    static int compare_cyclic(const string& s1, const string& s2) {
        string min1 = get_minimum_representation(s1);
        string min2 = get_minimum_representation(s2);

        if (min1 < min2) return -1;
        if (min1 > min2) return 1;
        return 0;
    }

    // 检查两个字符串是否循环同构
    static bool are_cyclic_isomorphic(const string& s1, const string& s2) {
        if (s1.length() != s2.length()) return false;
        return get_minimum_representation(s1) == get_minimum_representation(s2);
    }

    // 计算字符串的所有循环移位并按字典序排序
    static vector<string> all_rotations_sorted(const string& s) {
        vector<string> rotations;
        int n = s.length();

        for (int i = 0; i < n; i++) { rotations.push_back(s.substr(i) + s.substr(0, i)); }

        sort(rotations.begin(), rotations.end());
        return rotations;
    }

    // 使用Booth算法的字符串周期检测
    static vector<int> find_all_periods(const string& s) {
        vector<int> periods;
        int n = s.length();

        for (int period = 1; period <= n; period++) {
            if (n % period == 0) {
                bool is_period = true;
                string pattern = s.substr(0, period);

                for (int i = period; i < n; i += period) {
                    if (s.substr(i, period) != pattern) {
                        is_period = false;
                        break;
                    }
                }

                if (is_period) { periods.push_back(period); }
            }
        }

        return periods;
    }

    // 计算最小周期
    static int minimum_period(const string& s) {
        auto periods = find_all_periods(s);
        return periods.empty() ? s.length() : periods[0];
    }

    // 使用Booth算法优化的字符串匹配
    static vector<int> cyclic_string_matching(const string& text, const string& pattern) {
        vector<int> matches;
        string min_pattern = get_minimum_representation(pattern);
        int n = text.length();
        int m = pattern.length();

        for (int i = 0; i <= n - m; i++) {
            string substr = text.substr(i, m);
            if (get_minimum_representation(substr) == min_pattern) { matches.push_back(i); }
        }

        return matches;
    }

    // Lyndon词判断（基于Booth算法）
    static bool is_lyndon_word(const string& s) {
        if (s.empty()) return false;

        int min_pos = minimum_rotation(s);
        if (min_pos != 0) return false;

        // 检查是否为primitive（不是其他字符串的幂）
        int period = minimum_period(s);
        return period == s.length();
    }

    // 构造字符串的Booth标准形
    static string booth_canonical_form(const string& s) {
        string min_repr = get_minimum_representation(s);

        // 添加额外的标准化步骤
        // 例如，如果需要处理大小写不敏感，可以转换为小写
        transform(min_repr.begin(), min_repr.end(), min_repr.begin(), ::tolower);

        return min_repr;
    }

    // 使用Booth算法的高效循环字符串哈希
    static unsigned long long cyclic_hash(const string& s) {
        string canonical = get_minimum_representation(s);
        unsigned long long hash_val = 0;
        unsigned long long base = 31;
        unsigned long long mod = 1e9 + 7;

        for (char c : canonical) { hash_val = (hash_val * base + (c - 'a' + 1)) % mod; }

        return hash_val;
    }

   private:
    // 辅助函数：比较两个字符串的字典序（考虑循环）
    static int cyclic_compare(const string& s, int pos1, int pos2, int len) {
        for (int i = 0; i < len; i++) {
            char c1 = s[(pos1 + i) % s.length()];
            char c2 = s[(pos2 + i) % s.length()];
            if (c1 < c2) return -1;
            if (c1 > c2) return 1;
        }
        return 0;
    }

   public:
    // 扩展Booth算法：处理多字符串的最小表示
    static vector<string> minimum_representations(const vector<string>& strings) {
        vector<string> results;
        results.reserve(strings.size());

        for (const string& s : strings) { results.push_back(get_minimum_representation(s)); }

        return results;
    }

    // 计算循环等价类的代表元素
    static string equivalence_class_representative(const string& s) { return get_minimum_representation(s); }

    // 判断字符串是否是原始的（primitive）
    static bool is_primitive(const string& s) { return minimum_period(s) == s.length(); }
};
]=]),

-- 06_String_Algorithms\Advanced\Lyndon.h
ps("06_string_algorithms_advanced_lyndon_h", [=[

/**
 * Lyndon词分解算法
 * 功能：将字符串分解为Lyndon词的连接，支持各种Lyndon词相关操作
 * 时间复杂度：Duval算法 O(n)，标准分解 O(n²)
 * 空间复杂度：O(n)
 * 适用场景：字符串周期性分析、最小循环移位、字符串比较
 */
struct LyndonDecomposition {
    // Duval算法 - 线性时间Lyndon分解
    static vector<string> duval_algorithm(const string& s) {
        vector<string> factorization;
        int n = s.length();
        int i = 0;

        while (i < n) {
            int j = i + 1, k = i;

            while (j < n && s[k] <= s[j]) {
                if (s[k] < s[j]) {
                    k = i;
                } else {
                    k++;
                }
                j++;
            }

            while (i <= k) {
                factorization.push_back(s.substr(i, j - k));
                i += j - k;
            }
        }

        return factorization;
    }

    // 检查字符串是否为Lyndon词
    static bool is_lyndon_word(const string& s) {
        int n = s.length();
        for (int i = 1; i < n; i++) {
            if (s.substr(i) + s.substr(0, i) <= s) { return false; }
        }
        return true;
    }

    // 生成长度为n，字母表大小为k的所有Lyndon词
    static vector<string> generate_lyndon_words(int n, int k) {
        vector<string> result;
        string current(n, 'a');

        function<void(int)> generate = [&](int pos) {
            if (pos == n) {
                if (is_lyndon_word(current)) { result.push_back(current); }
                return;
            }

            for (int c = 0; c < k; c++) {
                current[pos] = 'a' + c;
                generate(pos + 1);
            }
        };

        generate(0);
        return result;
    }

    // 计算字符串的Lyndon分解并返回详细信息
    static vector<pair<string, pair<int, int>>> detailed_decomposition(const string& s) {
        vector<pair<string, pair<int, int>>> result;
        int n = s.length();
        int i = 0;

        while (i < n) {
            int j = i + 1, k = i;

            while (j < n && s[k] <= s[j]) {
                if (s[k] < s[j]) {
                    k = i;
                } else {
                    k++;
                }
                j++;
            }

            while (i <= k) {
                int len = j - k;
                result.push_back({s.substr(i, len), {i, i + len - 1}});
                i += len;
            }
        }

        return result;
    }

    // 使用Lyndon分解求最小后缀
    static int minimum_suffix(const string& s) {
        int n = s.length();
        int i = 0, j = 1, k = 0;

        while (i < n && j < n) {
            if (s[(i + k) % n] == s[(j + k) % n]) {
                k++;
            } else if (s[(i + k) % n] > s[(j + k) % n]) {
                i = i + k + 1;
                if (i <= j) i = j + 1;
                k = 0;
            } else {
                j = j + k + 1;
                if (j <= i) j = i + 1;
                k = 0;
            }
        }

        return min(i, j);
    }

    // 使用Lyndon分解求最大后缀
    static int maximum_suffix(const string& s) {
        int n = s.length();
        int i = 0, j = 1, k = 0;

        while (i < n && j < n) {
            if (s[(i + k) % n] == s[(j + k) % n]) {
                k++;
            } else if (s[(i + k) % n] < s[(j + k) % n]) {
                i = i + k + 1;
                if (i <= j) i = j + 1;
                k = 0;
            } else {
                j = j + k + 1;
                if (j <= i) j = i + 1;
                k = 0;
            }
        }

        return min(i, j);
    }

    // 计算字符串的字典序最小循环移位
    static string minimum_rotation(const string& s) {
        string doubled = s + s;
        int pos = minimum_suffix(doubled.substr(0, 2 * s.length() - 1));
        return doubled.substr(pos, s.length());
    }

    // 计算字符串的字典序最大循环移位
    static string maximum_rotation(const string& s) {
        string doubled = s + s;
        int pos = maximum_suffix(doubled.substr(0, 2 * s.length() - 1));
        return doubled.substr(pos, s.length());
    }

    // 检查两个字符串是否循环等价
    static bool are_cyclic_equivalent(const string& s1, const string& s2) {
        if (s1.length() != s2.length()) return false;
        return minimum_rotation(s1) == minimum_rotation(s2);
    }

    // 使用Lyndon分解进行字符串比较
    static int lyndon_compare(const string& s1, const string& s2) {
        auto decomp1 = duval_algorithm(s1);
        auto decomp2 = duval_algorithm(s2);

        int i = 0;
        while (i < decomp1.size() && i < decomp2.size()) {
            if (decomp1[i] < decomp2[i]) return -1;
            if (decomp1[i] > decomp2[i]) return 1;
            i++;
        }

        if (decomp1.size() < decomp2.size()) return -1;
        if (decomp1.size() > decomp2.size()) return 1;
        return 0;
    }

    // 计算Lyndon词的标准分解
    static pair<string, string> standard_factorization(const string& s) {
        if (!is_lyndon_word(s)) return {"", ""};

        int n = s.length();
        if (n == 1) return {s, ""};

        // 找到最大的真前缀，使得它是一个Lyndon词
        for (int i = n - 1; i >= 1; i--) {
            string prefix = s.substr(0, i);
            string suffix = s.substr(i);

            if (is_lyndon_word(prefix) && prefix < suffix) { return {prefix, suffix}; }
        }

        return {s.substr(0, 1), s.substr(1)};
    }

    // 构造Lyndon词的Christoffel词表示
    static string christoffel_word(int p, int q) {
        // 构造斜率为p/q的Christoffel词
        string result;
        int a = 0, b = 0;

        while (a < q || b < p) {
            if ((a + 1) * q < (b + 1) * p) {
                result += 'a';
                a++;
            } else {
                result += 'b';
                b++;
            }
        }

        return result;
    }
};
]=]),

-- 06_String_Algorithms\Advanced\MinimalRotation.h
ps("06_string_algorithms_advanced_minimalrotation_h", [=[

/**
 * 最小表示法算法集合
 * 功能：求字符串的最小循环移位、字典序最小的旋转等
 * 时间复杂度：O(n)
 * 空间复杂度：O(n)
 * 适用场景：字符串规范化、循环字符串比较、旋转字符串处理
 */
struct MinimalRotation {
    // 标准的最小表示法算法
    static int find_minimal_rotation(const string& s) {
        int n = s.length();
        string doubled = s + s;
        int i = 0, j = 1, k = 0;

        while (i < n && j < n) {
            k = 0;
            while (k < n && doubled[i + k] == doubled[j + k]) { k++; }

            if (k == n) break;

            if (doubled[i + k] > doubled[j + k]) {
                i = i + k + 1;
                if (i <= j) i = j + 1;
            } else {
                j = j + k + 1;
                if (j <= i) j = i + 1;
            }
        }

        return min(i, j);
    }

    // 获取最小循环表示
    static string get_minimal_string(const string& s) {
        int pos = find_minimal_rotation(s);
        return s.substr(pos) + s.substr(0, pos);
    }

    // 最大表示法
    static int find_maximal_rotation(const string& s) {
        int n = s.length();
        string doubled = s + s;
        int i = 0, j = 1, k = 0;

        while (i < n && j < n) {
            k = 0;
            while (k < n && doubled[i + k] == doubled[j + k]) { k++; }

            if (k == n) break;

            if (doubled[i + k] < doubled[j + k]) {
                i = i + k + 1;
                if (i <= j) i = j + 1;
            } else {
                j = j + k + 1;
                if (j <= i) j = i + 1;
            }
        }

        return min(i, j);
    }

    // 获取最大循环表示
    static string get_maximal_string(const string& s) {
        int pos = find_maximal_rotation(s);
        return s.substr(pos) + s.substr(0, pos);
    }

    // 数值版本的最小表示法（处理数字序列）
    static int find_minimal_rotation_numeric(const vector<int>& arr) {
        int n = arr.size();
        vector<int> doubled(arr.begin(), arr.end());
        doubled.insert(doubled.end(), arr.begin(), arr.end());

        int i = 0, j = 1, k = 0;

        while (i < n && j < n) {
            k = 0;
            while (k < n && doubled[i + k] == doubled[j + k]) { k++; }

            if (k == n) break;

            if (doubled[i + k] > doubled[j + k]) {
                i = i + k + 1;
                if (i <= j) i = j + 1;
            } else {
                j = j + k + 1;
                if (j <= i) j = i + 1;
            }
        }

        return min(i, j);
    }

    // 获取数值序列的最小循环表示
    static vector<int> get_minimal_array(const vector<int>& arr) {
        int pos = find_minimal_rotation_numeric(arr);
        vector<int> result;
        int n = arr.size();

        for (int i = 0; i < n; i++) { result.push_back(arr[(pos + i) % n]); }

        return result;
    }

    // 比较两个字符串的循环字典序
    static int compare_cyclic(const string& s1, const string& s2) {
        string min1 = get_minimal_string(s1);
        string min2 = get_minimal_string(s2);

        if (min1 < min2) return -1;
        if (min1 > min2) return 1;
        return 0;
    }

    // 检查两个字符串是否循环等价
    static bool are_cyclic_equivalent(const string& s1, const string& s2) {
        if (s1.length() != s2.length()) return false;
        return get_minimal_string(s1) == get_minimal_string(s2);
    }

    // 计算字符串的所有不同循环移位
    static vector<string> get_unique_rotations(const string& s) {
        set<string> unique_set;
        int n = s.length();

        for (int i = 0; i < n; i++) { unique_set.insert(s.substr(i) + s.substr(0, i)); }

        return vector<string>(unique_set.begin(), unique_set.end());
    }

    // 使用最小表示法的字符串标准化
    static string normalize(const string& s) { return get_minimal_string(s); }

    // 计算循环字符串的哈希值
    static unsigned long long cyclic_hash(const string& s) {
        string canonical = get_minimal_string(s);
        unsigned long long hash_val = 0;
        unsigned long long base = 131;

        for (char c : canonical) { hash_val = hash_val * base + c; }

        return hash_val;
    }

    // 找到所有具有相同最小表示的字符串的起始位置
    static vector<int> find_all_minimal_positions(const string& s) {
        string minimal = get_minimal_string(s);
        vector<int> positions;
        int n = s.length();

        for (int i = 0; i < n; i++) {
            string rotation = s.substr(i) + s.substr(0, i);
            if (rotation == minimal) { positions.push_back(i); }
        }

        return positions;
    }

    // 判断字符串是否为其最小表示
    static bool is_minimal_representation(const string& s) { return find_minimal_rotation(s) == 0; }

    // 计算到最小表示的距离（需要的循环移位次数）
    static int distance_to_minimal(const string& s) { return find_minimal_rotation(s); }

    // 字符串的周期性分析
    static vector<int> find_periods_using_minimal_rotation(const string& s) {
        vector<int> periods;
        int n = s.length();

        for (int p = 1; p <= n; p++) {
            if (n % p == 0) {
                string pattern = s.substr(0, p);
                bool is_period = true;

                for (int i = p; i < n; i += p) {
                    if (s.substr(i, p) != pattern) {
                        is_period = false;
                        break;
                    }
                }

                if (is_period) { periods.push_back(p); }
            }
        }

        return periods;
    }

    // 使用最小表示法进行字符串分组
    static map<string, vector<string>> group_by_cyclic_equivalence(const vector<string>& strings) {
        map<string, vector<string>> groups;

        for (const string& s : strings) {
            string canonical = get_minimal_string(s);
            groups[canonical].push_back(s);
        }

        return groups;
    }

    // 扩展到处理权重字符串（每个字符有权重）
    static int find_minimal_rotation_weighted(const vector<pair<char, int>>& weighted_string) {
        int n = weighted_string.size();
        vector<pair<char, int>> doubled = weighted_string;
        doubled.insert(doubled.end(), weighted_string.begin(), weighted_string.end());

        int i = 0, j = 1, k = 0;

        while (i < n && j < n) {
            k = 0;
            while (k < n && doubled[i + k] == doubled[j + k]) { k++; }

            if (k == n) break;

            auto cmp = [](const pair<char, int>& a, const pair<char, int>& b) {
                if (a.first != b.first) return a.first < b.first;
                return a.second < b.second;
            };

            if (cmp(doubled[j + k], doubled[i + k])) {
                i = i + k + 1;
                if (i <= j) i = j + 1;
            } else {
                j = j + k + 1;
                if (j <= i) j = i + 1;
            }
        }

        return min(i, j);
    }
};
]=]),

-- 06_String_Algorithms\Automaton\AhoCorasick.h
ps("06_string_algorithms_automaton_ahocorasick_h", [=[

/*
 * AC自动机算法
 * 时间复杂度: 构建O(∑|模式串|), 匹配O(n+输出数量)
 * 空间复杂度: O(∑|模式串|×字符集大小)
 * 适用场景: 多模式串匹配、敏感词过滤、文本检索
 */
struct AhoCorasick {
    struct Node {
        map<char, int> children;
        int fail;
        vector<int> output;
        int depth;

        Node() : fail(0), depth(0) {}
    };

    vector<Node> trie;
    vector<string> patterns;
    int node_count;

    AhoCorasick() {
        node_count = 0;
        trie.push_back(Node());
    }

    // 添加模式串
    void add_pattern(const string& pattern, int pattern_id = -1) {
        if (pattern_id == -1) { pattern_id = patterns.size(); }
        patterns.push_back(pattern);

        int current = 0;
        for (char c : pattern) {
            if (trie[current].children.find(c) == trie[current].children.end()) {
                trie[current].children[c] = ++node_count;
                trie.push_back(Node());
                trie[node_count].depth = trie[current].depth + 1;
            }
            current = trie[current].children[c];
        }
        trie[current].output.push_back(pattern_id);
    }

    // 构建失配指针
    void build_failure_links() {
        queue<int> q;

        // 第一层节点的失配指针指向根节点
        for (auto& [c, child] : trie[0].children) {
            trie[child].fail = 0;
            q.push(child);
        }

        while (!q.empty()) {
            int current = q.front();
            q.pop();

            for (auto& [c, child] : trie[current].children) {
                q.push(child);

                // 寻找失配指针
                int fail_node = trie[current].fail;
                while (fail_node != 0 && trie[fail_node].children.find(c) == trie[fail_node].children.end()) {
                    fail_node = trie[fail_node].fail;
                }

                if (trie[fail_node].children.find(c) != trie[fail_node].children.end() &&
                    trie[fail_node].children[c] != child) {
                    trie[child].fail = trie[fail_node].children[c];
                } else {
                    trie[child].fail = 0;
                }

                // 合并输出集合
                for (int pattern_id : trie[trie[child].fail].output) { trie[child].output.push_back(pattern_id); }
            }
        }
    }

    // 在文本中搜索所有模式串
    vector<pair<int, int>> search(const string& text) {
        vector<pair<int, int>> matches;  // (位置, 模式串ID)
        int current = 0;

        for (int i = 0; i < text.length(); i++) {
            char c = text[i];

            // 寻找匹配的转移
            while (current != 0 && trie[current].children.find(c) == trie[current].children.end()) {
                current = trie[current].fail;
            }

            if (trie[current].children.find(c) != trie[current].children.end()) { current = trie[current].children[c]; }

            // 输出所有匹配的模式串
            for (int pattern_id : trie[current].output) {
                int pattern_len = patterns[pattern_id].length();
                matches.push_back({i - pattern_len + 1, pattern_id});
            }
        }

        return matches;
    }

    // 统计每个模式串的出现次数
    vector<int> count_occurrences(const string& text) {
        vector<int> count(patterns.size(), 0);
        vector<pair<int, int>> matches = search(text);

        for (auto& [pos, pattern_id] : matches) { count[pattern_id]++; }

        return count;
    }

    // 检查文本中是否包含任何模式串
    bool contains_any_pattern(const string& text) {
        int current = 0;

        for (char c : text) {
            while (current != 0 && trie[current].children.find(c) == trie[current].children.end()) {
                current = trie[current].fail;
            }

            if (trie[current].children.find(c) != trie[current].children.end()) { current = trie[current].children[c]; }

            if (!trie[current].output.empty()) { return true; }
        }

        return false;
    }

    // 获取第一个匹配的模式串信息
    pair<int, int> find_first_match(const string& text) {
        int current = 0;

        for (int i = 0; i < text.length(); i++) {
            char c = text[i];

            while (current != 0 && trie[current].children.find(c) == trie[current].children.end()) {
                current = trie[current].fail;
            }

            if (trie[current].children.find(c) != trie[current].children.end()) { current = trie[current].children[c]; }

            if (!trie[current].output.empty()) {
                int pattern_id = trie[current].output[0];
                int pattern_len = patterns[pattern_id].length();
                return {i - pattern_len + 1, pattern_id};
            }
        }

        return {-1, -1};  // 未找到
    }

    // 替换文本中的所有模式串
    string replace_all(const string& text, const string& replacement) {
        vector<pair<int, int>> matches = search(text);
        if (matches.empty()) return text;

        // 按位置排序
        sort(matches.begin(), matches.end());

        string result;
        int last_end = 0;

        for (auto& [pos, pattern_id] : matches) {
            int pattern_len = patterns[pattern_id].length();

            // 跳过重叠的匹配
            if (pos < last_end) continue;

            // 添加中间的文本
            result += text.substr(last_end, pos - last_end);
            // 添加替换文本
            result += replacement;

            last_end = pos + pattern_len;
        }

        // 添加剩余的文本
        result += text.substr(last_end);

        return result;
    }

    // 计算模式串集合的总匹配长度
    int total_match_length(const string& text) {
        vector<pair<int, int>> matches = search(text);
        vector<bool> covered(text.length(), false);

        for (auto& [pos, pattern_id] : matches) {
            int pattern_len = patterns[pattern_id].length();
            for (int i = pos; i < pos + pattern_len; i++) { covered[i] = true; }
        }

        return count(covered.begin(), covered.end(), true);
    }

    // 获取Trie树的统计信息
    struct TrieStats {
        int total_nodes;
        int total_patterns;
        int max_depth;
        double avg_depth;
    };

    TrieStats get_stats() {
        TrieStats stats;
        stats.total_nodes = node_count + 1;
        stats.total_patterns = patterns.size();
        stats.max_depth = 0;

        int total_depth = 0;
        int pattern_count = 0;

        for (int i = 0; i <= node_count; i++) {
            if (!trie[i].output.empty()) {
                pattern_count += trie[i].output.size();
                total_depth += trie[i].depth * trie[i].output.size();
                stats.max_depth = max(stats.max_depth, trie[i].depth);
            }
        }

        stats.avg_depth = pattern_count > 0 ? (double)total_depth / pattern_count : 0;

        return stats;
    }

    // 打印Trie树结构（调试用）
    void print_trie() {
        function<void(int, string)> dfs = [&](int node, string prefix) {
            if (!trie[node].output.empty()) {
                cout << prefix << " -> patterns: ";
                for (int id : trie[node].output) { cout << patterns[id] << " "; }
                cout << endl;
            }

            for (auto& [c, child] : trie[node].children) { dfs(child, prefix + c); }
        };

        cout << "Trie structure:" << endl;
        dfs(0, "");
    }

    // 清空自动机
    void clear() {
        trie.clear();
        patterns.clear();
        node_count = 0;
        trie.push_back(Node());
    }
};

// 使用示例
/*
AhoCorasick ac;

// 添加模式串
ac.add_pattern("he");
ac.add_pattern("she");
ac.add_pattern("his");
ac.add_pattern("hers");

// 构建AC自动机
ac.build_failure_links();

// 在文本中搜索
string text = "ushers";
vector<pair<int, int>> matches = ac.search(text);

for (auto& [pos, pattern_id] : matches) {
    cout << "Found pattern '" << ac.patterns[pattern_id]
         << "' at position " << pos << endl;
}

// 统计出现次数
vector<int> counts = ac.count_occurrences(text);
for (int i = 0; i < counts.size(); i++) {
    cout << "Pattern '" << ac.patterns[i]
         << "' appears " << counts[i] << " times" << endl;
}
*/
]=]),

-- 06_String_Algorithms\Automaton\PalindromicTree.h
ps("06_string_algorithms_automaton_palindromictree_h", [=[

/**
 * 回文树（Palindromic Tree / Eertree）
 * 功能：构建字符串的回文自动机，可以快速处理回文子串相关问题
 * 时间复杂度：构建 O(n)，查询 O(1)
 * 空间复杂度：O(n)
 * 适用场景：回文子串计数、查找所有本质不同的回文子串、动态添加字符
 */
struct PalindromicTree {
    struct Node {
        map<char, int> next;  // 边
        int len;              // 回文串长度
        int fail;             // 失配指针
        int cnt;              // 出现次数

        Node() : len(0), fail(0), cnt(0) {}
    };

    string s;
    vector<Node> tree;
    int node_cnt, last;

    PalindromicTree() { init(); }

    void init() {
        tree.clear();
        tree.resize(2);
        node_cnt = 1;
        last = 0;

        // 奇根 len = -1
        tree[0].len = -1;
        tree[0].fail = 0;

        // 偶根 len = 0
        tree[1].len = 0;
        tree[1].fail = 0;
    }

    int get_fail(int pos, int x) {
        while (pos - tree[x].len - 1 < 0 || s[pos - tree[x].len - 1] != s[pos]) { x = tree[x].fail; }
        return x;
    }

    void extend(char c) {
        s.push_back(c);
        int pos = s.length() - 1;

        int cur = get_fail(pos, last);

        if (tree[cur].next.find(c) == tree[cur].next.end()) {
            // 新建节点
            int new_node = ++node_cnt;
            tree.resize(node_cnt + 1);

            tree[new_node].len = tree[cur].len + 2;

            if (tree[new_node].len == 1) {
                tree[new_node].fail = 1;  // 偶根
            } else {
                int fail_node = get_fail(pos, tree[cur].fail);
                tree[new_node].fail = tree[fail_node].next[c];
            }

            tree[cur].next[c] = new_node;
        }

        last = tree[cur].next[c];
        tree[last].cnt++;
    }

    void build(const string& str) {
        init();
        for (char c : str) { extend(c); }
    }

    // 计算每个回文串的实际出现次数
    void count() {
        for (int i = node_cnt; i >= 0; i--) { tree[tree[i].fail].cnt += tree[i].cnt; }
    }

    // 获取所有本质不同的回文子串
    vector<string> get_all_palindromes() {
        vector<string> result;

        function<void(int, string)> dfs = [&](int node, string current) {
            if (tree[node].len > 0) { result.push_back(current); }

            for (auto& edge : tree[node].next) {
                char c = edge.first;
                int next_node = edge.second;
                string next_str = c + current + c;
                dfs(next_node, next_str);
            }
        };

        // 从奇根和偶根开始遍历
        dfs(0, "");
        dfs(1, "");

        return result;
    }

    // 获取回文子串的数量
    int get_palindrome_count() {
        return node_cnt - 1;  // 不包括两个根节点
    }

    // 获取最长回文子串的长度
    int get_max_length() {
        int max_len = 0;
        for (int i = 2; i <= node_cnt; i++) { max_len = max(max_len, tree[i].len); }
        return max_len;
    }

    // 检查从位置pos开始的回文串
    vector<int> get_palindromes_at_position(int pos) {
        vector<int> lengths;

        // 重新构建到指定位置
        string temp_s = s;
        init();
        for (int i = 0; i <= pos; i++) { extend(temp_s[i]); }

        int cur = last;
        while (cur > 1) {
            lengths.push_back(tree[cur].len);
            cur = tree[cur].fail;
        }

        return lengths;
    }

    // 计算回文子串总数
    long long total_palindrome_count() {
        count();
        long long total = 0;
        for (int i = 2; i <= node_cnt; i++) { total += tree[i].cnt; }
        return total;
    }

    void print_tree() {
        for (int i = 0; i <= node_cnt; i++) {
            cout << "Node " << i << ": len=" << tree[i].len << " fail=" << tree[i].fail << " cnt=" << tree[i].cnt
                 << "\n";
            for (auto& edge : tree[i].next) { cout << "  -> " << edge.first << " to " << edge.second << "\n"; }
        }
    }
};
]=]),

-- 06_String_Algorithms\Automaton\SuffixAutomaton.h
ps("06_string_algorithms_automaton_suffixautomaton_h", [=[

/*
 * 后缀自动机算法
 * 时间复杂度: 构建O(n), 查询O(m)
 * 空间复杂度: O(n×字符集大小)
 * 适用场景: 子串查询、最长公共子串、字典序第k小子串
 */
struct SuffixAutomaton {
    struct Node {
        map<char, int> children;
        int link;        // 后缀链接
        int length;      // 最长字符串长度
        int first_pos;   // 第一次出现位置
        int cnt;         // 出现次数
        bool is_cloned;  // 是否为克隆节点

        Node() : link(-1), length(0), first_pos(-1), cnt(0), is_cloned(false) {}
    };

    vector<Node> nodes;
    int last;  // 当前状态
    int size;  // 节点数量

    SuffixAutomaton() {
        nodes.resize(1);
        last = 0;
        size = 1;
        nodes[0].length = 0;
        nodes[0].link = -1;
    }

    // 扩展字符c
    void extend(char c) {
        int cur = size++;
        nodes.resize(size);
        nodes[cur].length = nodes[last].length + 1;
        nodes[cur].first_pos = nodes[cur].length - 1;

        int p = last;
        while (p != -1 && nodes[p].children.find(c) == nodes[p].children.end()) {
            nodes[p].children[c] = cur;
            p = nodes[p].link;
        }

        if (p == -1) {
            nodes[cur].link = 0;
        } else {
            int q = nodes[p].children[c];
            if (nodes[p].length + 1 == nodes[q].length) {
                nodes[cur].link = q;
            } else {
                int clone = size++;
                nodes.resize(size);
                nodes[clone] = nodes[q];
                nodes[clone].length = nodes[p].length + 1;
                nodes[clone].is_cloned = true;
                nodes[clone].first_pos = -1;

                while (p != -1 && nodes[p].children[c] == q) {
                    nodes[p].children[c] = clone;
                    p = nodes[p].link;
                }

                nodes[q].link = nodes[cur].link = clone;
            }
        }

        last = cur;
    }

    // 构建后缀自动机
    void build(const string& s) {
        for (char c : s) { extend(c); }
    }

    // 检查子串是否存在
    bool contains(const string& pattern) {
        int curr = 0;
        for (char c : pattern) {
            if (nodes[curr].children.find(c) == nodes[curr].children.end()) { return false; }
            curr = nodes[curr].children[c];
        }
        return true;
    }

    // 计算子串第一次出现位置
    int first_occurrence(const string& pattern) {
        int curr = 0;
        for (char c : pattern) {
            if (nodes[curr].children.find(c) == nodes[curr].children.end()) { return -1; }
            curr = nodes[curr].children[c];
        }
        return nodes[curr].first_pos - pattern.length() + 1;
    }

    // 计算不同子串数量
    long long count_distinct_substrings() {
        long long result = 0;
        for (int i = 1; i < size; i++) {
            if (!nodes[i].is_cloned) { result += nodes[i].length - (nodes[nodes[i].link].length); }
        }
        return result;
    }

    // 计算每个状态的出现次数
    void calculate_occurrences() {
        vector<int> order(size);
        iota(order.begin(), order.end(), 0);

        // 按长度排序
        sort(order.begin(), order.end(), [&](int a, int b) { return nodes[a].length > nodes[b].length; });

        // 初始化终止状态
        for (int i = 0; i < size; i++) {
            if (!nodes[i].is_cloned) { nodes[i].cnt = 1; }
        }

        // 自底向上计算
        for (int v : order) {
            if (nodes[v].link != -1) { nodes[nodes[v].link].cnt += nodes[v].cnt; }
        }
    }

    // 获取子串出现次数
    int get_occurrences(const string& pattern) {
        int curr = 0;
        for (char c : pattern) {
            if (nodes[curr].children.find(c) == nodes[curr].children.end()) { return 0; }
            curr = nodes[curr].children[c];
        }
        return nodes[curr].cnt;
    }

    // 找到第k小的子串
    string kth_substring(long long k) {
        vector<long long> dp(size);
        function<long long(int)> count_paths = [&](int v) -> long long {
            if (dp[v] != 0) return dp[v];
            dp[v] = 1;  // 空字符串
            for (auto& [c, u] : nodes[v].children) { dp[v] += count_paths(u); }
            return dp[v];
        };

        count_paths(0);

        if (k > dp[0]) return "";  // k太大

        string result;
        int curr = 0;
        k--;  // 跳过空字符串

        while (k > 0) {
            for (auto& [c, next] : nodes[curr].children) {
                if (k <= dp[next]) {
                    result += c;
                    curr = next;
                    k--;
                    break;
                } else {
                    k -= dp[next];
                }
            }
        }

        return result;
    }

    // 计算最长公共子串
    int longest_common_substring(const string& s1, const string& s2) {
        // 构建s1的后缀自动机
        clear();
        build(s1);

        int result = 0;
        int curr = 0;
        int length = 0;

        for (char c : s2) {
            while (curr != 0 && nodes[curr].children.find(c) == nodes[curr].children.end()) {
                curr = nodes[curr].link;
                length = nodes[curr].length;
            }

            if (nodes[curr].children.find(c) != nodes[curr].children.end()) {
                curr = nodes[curr].children[c];
                length++;
                result = max(result, length);
            } else {
                curr = 0;
                length = 0;
            }
        }

        return result;
    }

    // 获取所有不同子串
    vector<string> get_all_substrings() {
        vector<string> result;

        function<void(int, string)> dfs = [&](int v, string current) {
            if (v != 0) result.push_back(current);

            for (auto& [c, u] : nodes[v].children) { dfs(u, current + c); }
        };

        dfs(0, "");
        return result;
    }

    // 计算字典序第k小的循环移位
    string kth_cyclic_shift(const string& s, long long k) {
        string doubled = s + s;
        clear();
        build(doubled);

        int curr = 0;
        string result;

        for (int i = 0; i < s.length(); i++) {
            vector<char> candidates;
            for (auto& [c, next] : nodes[curr].children) { candidates.push_back(c); }
            sort(candidates.begin(), candidates.end());

            for (char c : candidates) {
                int next = nodes[curr].children[c];
                // 这里需要更复杂的计算来确定第k小
                // 简化实现
                result += c;
                curr = next;
                break;
            }
        }

        return result;
    }

    // 多字符串的最长公共子串
    int longest_common_substring_multi(const vector<string>& strings) {
        if (strings.empty()) return 0;

        clear();
        build(strings[0]);

        vector<vector<int>> dp(strings.size(), vector<int>(size, 0));

        for (int i = 1; i < strings.size(); i++) {
            int curr = 0;
            int length = 0;

            for (char c : strings[i]) {
                while (curr != 0 && nodes[curr].children.find(c) == nodes[curr].children.end()) {
                    curr = nodes[curr].link;
                    length = nodes[curr].length;
                }

                if (nodes[curr].children.find(c) != nodes[curr].children.end()) {
                    curr = nodes[curr].children[c];
                    length++;
                } else {
                    curr = 0;
                    length = 0;
                }

                dp[i][curr] = max(dp[i][curr], length);
            }

            // 向上传播
            vector<int> order(size);
            iota(order.begin(), order.end(), 0);
            sort(order.begin(), order.end(), [&](int a, int b) { return nodes[a].length > nodes[b].length; });

            for (int v : order) {
                if (nodes[v].link != -1) {
                    dp[i][nodes[v].link] = max(dp[i][nodes[v].link], min(dp[i][v], nodes[nodes[v].link].length));
                }
            }
        }

        int result = 0;
        for (int v = 0; v < size; v++) {
            int min_len = nodes[v].length;
            for (int i = 1; i < strings.size(); i++) { min_len = min(min_len, dp[i][v]); }
            result = max(result, min_len);
        }

        return result;
    }

    // 清空自动机
    void clear() {
        nodes.clear();
        nodes.resize(1);
        last = 0;
        size = 1;
        nodes[0].length = 0;
        nodes[0].link = -1;
    }

    // 获取状态数
    int get_size() const { return size; }

    // 调试输出
    void debug_print() {
        cout << "Suffix Automaton with " << size << " states:" << endl;
        for (int i = 0; i < size; i++) {
            cout << "State " << i << ": len=" << nodes[i].length << ", link=" << nodes[i].link
                 << ", pos=" << nodes[i].first_pos << ", cnt=" << nodes[i].cnt << endl;
            cout << "  Children: ";
            for (auto& [c, next] : nodes[i].children) { cout << c << "->" << next << " "; }
            cout << endl;
        }
    }

    // 构建后缀树（从后缀自动机）
    vector<vector<pair<int, pair<int, int>>>> build_suffix_tree() {
        vector<vector<pair<int, pair<int, int>>>> tree(size);

        for (int v = 1; v < size; v++) {
            int parent = nodes[v].link;
            int start = nodes[parent].length;
            int end = nodes[v].length - 1;
            tree[parent].push_back({v, {start, end}});
        }

        return tree;
    }
};

// 使用示例：
// SuffixAutomaton sam;
// sam.build("abcbc");
// bool exists = sam.contains("bc");
// int distinct = sam.count_distinct_substrings();
// sam.calculate_occurrences();
// int count = sam.get_occurrences("bc");
]=]),

-- 06_String_Algorithms\Palindromes\Manacher.h
ps("06_string_algorithms_palindromes_manacher_h", [=[

/*
 * Manacher算法 - 回文串处理
 * 时间复杂度: O(n)
 * 空间复杂度: O(n)
 * 功能: 找出字符串中所有回文子串的半径
 */
vector<int> manacher(string s) {
    string t = "#";
    for (auto c : s) {
        t += c;
        t += "#";
    }
    int n = t.size();
    vector<int> r(n);
    for (int i = 0, j = 0; i < n; ++i) {
        if (2 * j - i >= 0 && j + r[j] > i) { r[i] = min(r[2 * j - i], j + r[j] - i); }
        while (i - r[i] >= 0 && i + r[i] < n && t[i - r[i]] == t[i + r[i]]) { r[i] += 1; }
        if (i + r[i] > j + r[j]) { j = i; }
    }
    return r;
}

// 获取r[i]位置的回文子串
pair<int, string> get(const string &s, const vector<int> &r, int i) {
    int len = r[i] - 1;
    if (len <= 0) return {-1, ""};
    int st = (i - len) / 2;
    return {st, s.substr(st, len)};
}
]=]),

-- 06_String_Algorithms\Palindromes\PalindromePartition.h
ps("06_string_algorithms_palindromes_palindromepartition_h", [=[

/*
 * 回文分割算法
 * 时间复杂度: 预处理O(n²), 查询根据具体算法而定
 * 空间复杂度: O(n²)
 * 适用场景: 回文分割、回文子串计数、动态规划优化
 */
struct PalindromePartition {
    string s;
    int n;
    vector<vector<bool>> is_palindrome;

    PalindromePartition(const string& str) : s(str), n(str.length()) { precompute_palindromes(); }

   private:
    // 预计算所有子串是否为回文
    void precompute_palindromes() {
        is_palindrome.assign(n, vector<bool>(n, false));

        // 长度为1的子串都是回文
        for (int i = 0; i < n; i++) { is_palindrome[i][i] = true; }

        // 长度为2的子串
        for (int i = 0; i < n - 1; i++) {
            if (s[i] == s[i + 1]) { is_palindrome[i][i + 1] = true; }
        }

        // 长度大于2的子串
        for (int len = 3; len <= n; len++) {
            for (int i = 0; i <= n - len; i++) {
                int j = i + len - 1;
                if (s[i] == s[j] && is_palindrome[i + 1][j - 1]) { is_palindrome[i][j] = true; }
            }
        }
    }

   public:
    // 检查子串s[i..j]是否为回文
    bool check_palindrome(int i, int j) const { return is_palindrome[i][j]; }

    // 最小分割次数（DP）
    int min_cuts() const {
        vector<int> dp(n + 1, INT_MAX);
        dp[0] = 0;

        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                if (is_palindrome[j][i - 1] && dp[j] != INT_MAX) { dp[i] = min(dp[i], dp[j] + 1); }
            }
        }

        return dp[n] - 1;  // 减1因为分割次数比段数少1
    }

    // 返回一种最小分割方案
    vector<string> min_cut_partition() const {
        vector<int> dp(n + 1, INT_MAX);
        vector<int> parent(n + 1, -1);
        dp[0] = 0;

        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                if (is_palindrome[j][i - 1] && dp[j] != INT_MAX) {
                    if (dp[j] + 1 < dp[i]) {
                        dp[i] = dp[j] + 1;
                        parent[i] = j;
                    }
                }
            }
        }

        // 重构分割方案
        vector<string> result;
        int pos = n;
        while (pos > 0) {
            int prev = parent[pos];
            result.push_back(s.substr(prev, pos - prev));
            pos = prev;
        }

        reverse(result.begin(), result.end());
        return result;
    }

    // 所有可能的回文分割方案
    vector<vector<string>> all_partitions() const {
        vector<vector<string>> result;
        vector<string> current;
        backtrack(0, current, result);
        return result;
    }

   private:
    void backtrack(int start, vector<string>& current, vector<vector<string>>& result) const {
        if (start == n) {
            result.push_back(current);
            return;
        }

        for (int end = start; end < n; end++) {
            if (is_palindrome[start][end]) {
                current.push_back(s.substr(start, end - start + 1));
                backtrack(end + 1, current, result);
                current.pop_back();
            }
        }
    }

   public:
    // 计算分割方案数量
    long long count_partitions() const {
        vector<long long> dp(n + 1, 0);
        dp[0] = 1;

        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                if (is_palindrome[j][i - 1]) { dp[i] += dp[j]; }
            }
        }

        return dp[n];
    }

    // 最长回文子序列长度
    int longest_palindromic_subsequence() const {
        vector<vector<int>> dp(n, vector<int>(n, 0));

        // 长度为1的子序列
        for (int i = 0; i < n; i++) { dp[i][i] = 1; }

        // 长度为2的子序列
        for (int i = 0; i < n - 1; i++) {
            if (s[i] == s[i + 1]) {
                dp[i][i + 1] = 2;
            } else {
                dp[i][i + 1] = 1;
            }
        }

        // 长度大于2的子序列
        for (int len = 3; len <= n; len++) {
            for (int i = 0; i <= n - len; i++) {
                int j = i + len - 1;
                if (s[i] == s[j]) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }

        return dp[0][n - 1];
    }

    // 最短回文串（在开头添加字符使整个串成为回文）
    string shortest_palindrome() const {
        string rev_s = s;
        reverse(rev_s.begin(), rev_s.end());
        string combined = s + "#" + rev_s;

        // 计算KMP的next数组
        vector<int> next(combined.length(), 0);
        for (int i = 1; i < combined.length(); i++) {
            int j = next[i - 1];
            while (j > 0 && combined[i] != combined[j]) { j = next[j - 1]; }
            if (combined[i] == combined[j]) { j++; }
            next[i] = j;
        }

        int overlap = next[combined.length() - 1];
        return rev_s.substr(0, n - overlap) + s;
    }

    // 验证是否可以通过最多k次删除字符得到回文
    bool can_be_palindrome_k_deletions(int k) const {
        int lps = longest_palindromic_subsequence();
        return n - lps <= k;
    }

    // 通过插入字符使字符串变成回文的最少插入次数
    int min_insertions_for_palindrome() const { return n - longest_palindromic_subsequence(); }

    // 找到所有回文子串
    vector<pair<int, int>> all_palindromic_substrings() const {
        vector<pair<int, int>> palindromes;

        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                if (is_palindrome[i][j]) { palindromes.push_back({i, j}); }
            }
        }

        return palindromes;
    }

    // 最长回文子串
    pair<int, int> longest_palindromic_substring() const {
        int max_len = 0;
        pair<int, int> result = {0, 0};

        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                if (is_palindrome[i][j] && j - i + 1 > max_len) {
                    max_len = j - i + 1;
                    result = {i, j};
                }
            }
        }

        return result;
    }
};
]=]),

-- 06_String_Algorithms\Pattern_Matching\BoyerMoore.h
ps("06_string_algorithms_pattern_matching_boyermoore_h", [=[

/*
 * Boyer-Moore字符串匹配算法
 * 时间复杂度: 预处理O(m+σ), 匹配O(n)平均, O(nm)最坏
 * 空间复杂度: O(m+σ)
 * 适用场景: 长模式串匹配、文本搜索、字符集较大的匹配
 */
struct BoyerMoore {
    string pattern;
    vector<int> bad_char, good_suffix;

    BoyerMoore(const string& p) : pattern(p) { preprocess(); }

    void preprocess() {
        int m = pattern.length();
        bad_char.assign(256, -1);
        good_suffix.assign(m, 0);

        // 预处理坏字符表
        for (int i = 0; i < m; i++) { bad_char[(int)pattern[i]] = i; }

        // 预处理好后缀表
        vector<int> suffix(m, 0);
        compute_suffix(suffix);

        // 情况1：模式串的后缀在模式串中的其他位置出现
        for (int i = 0; i < m; i++) { good_suffix[i] = m; }

        for (int i = m - 1; i >= 0; i--) {
            if (suffix[i] == i + 1) {
                for (int j = 0; j < m - 1 - i; j++) {
                    if (good_suffix[j] == m) { good_suffix[j] = m - 1 - i; }
                }
            }
        }

        // 情况2：模式串的后缀的一部分匹配模式串的前缀
        for (int i = 0; i <= m - 2; i++) { good_suffix[m - 1 - suffix[i]] = m - 1 - i; }
    }

    void compute_suffix(vector<int>& suffix) {
        int m = pattern.length();
        suffix[m - 1] = m;
        int g = m - 1, f = 0;

        for (int i = m - 2; i >= 0; i--) {
            if (i > g && suffix[i + m - 1 - f] < i - g) {
                suffix[i] = suffix[i + m - 1 - f];
            } else {
                if (i < g) g = i;
                f = i;
                while (g >= 0 && pattern[g] == pattern[g + m - 1 - f]) { g--; }
                suffix[i] = f - g;
            }
        }
    }

    vector<int> search(const string& text) {
        vector<int> matches;
        int n = text.length();
        int m = pattern.length();

        int s = 0;  // shift of the pattern
        while (s <= n - m) {
            int j = m - 1;

            // 从右到左比较
            while (j >= 0 && pattern[j] == text[s + j]) { j--; }

            if (j < 0) {
                matches.push_back(s);
                s += good_suffix[0];
            } else {
                int bad_char_shift = j - bad_char[(int)text[s + j]];
                int good_suffix_shift = good_suffix[j];
                s += max(bad_char_shift, good_suffix_shift);
            }
        }

        return matches;
    }

    int count_matches(const string& text) { return search(text).size(); }

    bool contains(const string& text) { return !search(text).empty(); }
};

// Rabin-Karp字符串匹配算法
struct RabinKarp {
    static const int BASE = 257;
    static const int MOD = 1e9 + 7;

    string pattern;
    long long pattern_hash;
    long long base_pow;

    RabinKarp(const string& p) : pattern(p) {
        pattern_hash = compute_hash(pattern);
        base_pow = 1;
        for (int i = 0; i < pattern.length() - 1; i++) { base_pow = base_pow * BASE % MOD; }
    }

    long long compute_hash(const string& s) {
        long long hash_val = 0;
        for (char c : s) { hash_val = (hash_val * BASE + c) % MOD; }
        return hash_val;
    }

    vector<int> search(const string& text) {
        vector<int> matches;
        int n = text.length();
        int m = pattern.length();

        if (m > n) return matches;

        // 计算第一个窗口的哈希值
        long long window_hash = compute_hash(text.substr(0, m));

        for (int i = 0; i <= n - m; i++) {
            // 检查哈希值是否匹配
            if (window_hash == pattern_hash) {
                // 哈希值匹配，进行字符串比较确认
                if (text.substr(i, m) == pattern) { matches.push_back(i); }
            }

            // 滚动哈希：移除最左边的字符，添加新字符
            if (i < n - m) {
                window_hash = (window_hash - (text[i] * base_pow % MOD) + MOD) % MOD;
                window_hash = (window_hash * BASE + text[i + m]) % MOD;
            }
        }

        return matches;
    }

    int count_matches(const string& text) { return search(text).size(); }

    bool contains(const string& text) { return !search(text).empty(); }
};

// Z算法（Z-function）
struct ZAlgorithm {
    static vector<int> z_function(const string& s) {
        int n = s.length();
        vector<int> z(n);

        for (int i = 1, l = 0, r = 0; i < n; i++) {
            if (i <= r) { z[i] = min(r - i + 1, z[i - l]); }
            while (i + z[i] < n && s[z[i]] == s[i + z[i]]) { z[i]++; }
            if (i + z[i] - 1 > r) {
                l = i;
                r = i + z[i] - 1;
            }
        }

        return z;
    }

    static vector<int> search(const string& pattern, const string& text) {
        string combined = pattern + "#" + text;
        vector<int> z = z_function(combined);
        vector<int> matches;

        int pattern_len = pattern.length();
        for (int i = pattern_len + 1; i < combined.length(); i++) {
            if (z[i] == pattern_len) { matches.push_back(i - pattern_len - 1); }
        }

        return matches;
    }

    static int count_matches(const string& pattern, const string& text) { return search(pattern, text).size(); }

    static bool contains(const string& pattern, const string& text) { return !search(pattern, text).empty(); }
};
]=]),

-- 06_String_Algorithms\Pattern_Matching\KMP.h
ps("06_string_algorithms_pattern_matching_kmp_h", [=[

/*
 * KMP字符串匹配算法
 * 时间复杂度: 预处理O(m), 匹配O(n+m)
 * 空间复杂度: O(m)
 * 适用场景: 字符串模式匹配、周期性检测、前缀函数相关问题
 */
struct KMP {
    vector<int> failure;
    string pattern;

    KMP(const string& p) : pattern(p) { build_failure_function(); }

    // 构建失配函数（前缀函数）
    void build_failure_function() {
        int m = pattern.length();
        failure.assign(m, 0);

        for (int i = 1; i < m; i++) {
            int j = failure[i - 1];
            while (j > 0 && pattern[i] != pattern[j]) {
                j = failure[j - 1];
            }
            if (pattern[i] == pattern[j]) {
                j++;
            }
            failure[i] = j;
        }
    }

    // 在文本中查找模式串的所有出现位置
    vector<int> search(const string& text) {
        vector<int> matches;
        int n = text.length(), m = pattern.length();
        if (m == 0) return matches;

        int j = 0;
        for (int i = 0; i < n; i++) {
            while (j > 0 && text[i] != pattern[j]) {
                j = failure[j - 1];
            }
            if (text[i] == pattern[j]) {
                j++;
            }
            if (j == m) {
                matches.push_back(i - m + 1);
                j = failure[j - 1];
            }
        }

        return matches;
    }

    // 计算字符串的周期
    vector<int> get_periods(const string& s) {
        KMP kmp(s);
        vector<int> periods;
        int n = s.length();
        int len = n;

        while (len > 0) {
            len = kmp.failure[len - 1];
            if (len > 0 && n % (n - len) == 0) {
                periods.push_back(n - len);
            }
        }

        reverse(periods.begin(), periods.end());
        return periods;
    }

    // 计算字符串的border（真前缀且为后缀）
    vector<int> get_borders(const string& s) {
        KMP kmp(s);
        vector<int> borders;
        int n = s.length();
        int len = kmp.failure[n - 1];

        while (len > 0) {
            borders.push_back(len);
            len = kmp.failure[len - 1];
        }

        reverse(borders.begin(), borders.end());
        return borders;
    }

    // 统计模式串在文本中的出现次数
    int count_occurrences(const string& text) { return search(text).size(); }

    // 检查字符串是否为另一个字符串的子串
    bool is_substring(const string& text) { return !search(text).empty(); }

    // 计算两个字符串的最长公共前缀长度数组
    static vector<int> lcp_array(const string& s, const string& t) {
        string combined = s + "#" + t;
        KMP kmp(combined);

        vector<int> lcp;
        int n = s.length(), m = t.length();

        for (int i = 0; i <= m; i++) {
            if (n + 1 + i < combined.length()) {
                int pos = n + 1 + i;
                int len = 0;
                while (len < min(n, m - i) && s[len] == t[i + len]) {
                    len++;
                }
                lcp.push_back(len);
            }
        }

        return lcp;
    }

    // 计算字符串自身的Z函数
    static vector<int> z_function(const string& s) {
        int n = s.length();
        vector<int> z(n);

        for (int i = 1, l = 0, r = 0; i < n; i++) {
            if (i <= r) {
                z[i] = min(r - i + 1, z[i - l]);
            }
            while (i + z[i] < n && s[z[i]] == s[i + z[i]]) {
                z[i]++;
            }
            if (i + z[i] - 1 > r) {
                l = i;
                r = i + z[i] - 1;
            }
        }

        return z;
    }

    // 使用KMP求解字符串匹配的变种问题

    // 求最长相等前后缀长度
    int longest_prefix_suffix() {
        if (pattern.empty()) return 0;
        return failure[pattern.length() - 1];
    }

    // 计算字符串的最小周期
    int minimum_period() {
        int n = pattern.length();
        int lps = longest_prefix_suffix();
        int period = n - lps;

        // 如果n能被period整除，则period是最小周期
        if (n % period == 0) {
            return period;
        }
        return n;  // 整个字符串就是一个周期
    }

    // 检查字符串是否具有周期性
    bool is_periodic() { return minimum_period() < pattern.length(); }

    // 计算添加最少字符使字符串变成回文
    int min_chars_to_palindrome() {
        string rev = pattern;
        reverse(rev.begin(), rev.end());

        KMP kmp(rev);
        vector<int> matches = kmp.search(pattern);

        if (!matches.empty()) {
            int overlap = rev.length() - matches[0];
            return pattern.length() - overlap;
        }

        return pattern.length();
    }

    // 在文本串中查找第一个匹配位置
    int find_first(const string& text) {
        vector<int> matches = search(text);
        return matches.empty() ? -1 : matches[0];
    }

    // 在文本串中查找最后一个匹配位置
    int find_last(const string& text) {
        vector<int> matches = search(text);
        return matches.empty() ? -1 : matches.back();
    }

    // 模式串预处理信息
    void print_failure_function() {
        cout << "Pattern: " << pattern << endl;
        cout << "Failure function: ";
        for (int x : failure) {
            cout << x << " ";
        }
        cout << endl;
    }
};

// 扩展KMP算法
struct ExtendedKMP {
    vector<int> z;
    string s;

    ExtendedKMP(const string& str) : s(str) { compute_z_array(); }

    void compute_z_array() {
        int n = s.length();
        z.assign(n, 0);

        for (int i = 1, l = 0, r = 0; i < n; i++) {
            if (i <= r) {
                z[i] = min(r - i + 1, z[i - l]);
            }
            while (i + z[i] < n && s[z[i]] == s[i + z[i]]) {
                z[i]++;
            }
            if (i + z[i] - 1 > r) {
                l = i;
                r = i + z[i] - 1;
            }
        }
    }

    // 在文本中查找模式串
    vector<int> search(const string& text, const string& pattern) {
        string combined = pattern + "#" + text;
        ExtendedKMP ext_kmp(combined);

        vector<int> matches;
        int p_len = pattern.length();

        for (int i = p_len + 1; i < combined.length(); i++) {
            if (ext_kmp.z[i] == p_len) {
                matches.push_back(i - p_len - 1);
            }
        }

        return matches;
    }
};

// 使用示例
/*
// 基本KMP使用
KMP kmp("ABABCABABA");
string text = "ABABABABCABABAABABCABABA";
vector<int> matches = kmp.search(text);

for (int pos : matches) {
    cout << "Found at position: " << pos << endl;
}

// 计算字符串周期
vector<int> periods = kmp.get_periods("ABCABCABC");
cout << "Periods: ";
for (int p : periods) {
    cout << p << " ";
}
cout << endl;
*/
]=]),

-- 06_String_Algorithms\Pattern_Matching\RabinKarp.h
ps("06_string_algorithms_pattern_matching_rabinkarp_h", [=[

/*
 * Rabin-Karp字符串匹配算法
 * 时间复杂度: 预处理O(m), 匹配O(n+m)平均, O(nm)最坏
 * 空间复杂度: O(1)
 * 适用场景: 多模式串匹配、滚动哈希匹配、概率性匹配
 */
struct RabinKarpAlgorithm {
    static const long long BASE1 = 131;
    static const long long BASE2 = 137;
    static const long long MOD1 = 1e9 + 7;
    static const long long MOD2 = 1e9 + 9;

    string pattern;
    pair<long long, long long> pattern_hash;
    pair<long long, long long> base_pow;

    RabinKarpAlgorithm(const string& p) : pattern(p) {
        pattern_hash = compute_double_hash(pattern);
        base_pow = {1, 1};

        for (int i = 0; i < pattern.length() - 1; i++) {
            base_pow.first = base_pow.first * BASE1 % MOD1;
            base_pow.second = base_pow.second * BASE2 % MOD2;
        }
    }

    pair<long long, long long> compute_double_hash(const string& s) {
        long long hash1 = 0, hash2 = 0;

        for (char c : s) {
            hash1 = (hash1 * BASE1 + c) % MOD1;
            hash2 = (hash2 * BASE2 + c) % MOD2;
        }

        return {hash1, hash2};
    }

    vector<int> search(const string& text) {
        vector<int> matches;
        int n = text.length();
        int m = pattern.length();

        if (m > n) return matches;

        // 计算第一个窗口的双重哈希值
        pair<long long, long long> window_hash = compute_double_hash(text.substr(0, m));

        for (int i = 0; i <= n - m; i++) {
            // 检查双重哈希值是否匹配
            if (window_hash == pattern_hash) {
                // 哈希值匹配，进行字符串比较确认（避免误报）
                if (text.substr(i, m) == pattern) { matches.push_back(i); }
            }

            // 滚动哈希：移除最左边的字符，添加新字符
            if (i < n - m) {
                // 第一个哈希函数
                window_hash.first = (window_hash.first - (text[i] * base_pow.first % MOD1) + MOD1) % MOD1;
                window_hash.first = (window_hash.first * BASE1 + text[i + m]) % MOD1;

                // 第二个哈希函数
                window_hash.second = (window_hash.second - (text[i] * base_pow.second % MOD2) + MOD2) % MOD2;
                window_hash.second = (window_hash.second * BASE2 + text[i + m]) % MOD2;
            }
        }

        return matches;
    }

    bool contains(const string& text) { return !search(text).empty(); }

    int count_occurrences(const string& text) { return search(text).size(); }
};

// Z算法的独立实现
struct ZFunction {
    static vector<int> compute_z(const string& s) {
        int n = s.length();
        vector<int> z(n, 0);

        for (int i = 1, l = 0, r = 0; i < n; i++) {
            if (i <= r) { z[i] = min(r - i + 1, z[i - l]); }

            while (i + z[i] < n && s[z[i]] == s[i + z[i]]) { z[i]++; }

            if (i + z[i] - 1 > r) {
                l = i;
                r = i + z[i] - 1;
            }
        }

        return z;
    }

    static vector<int> pattern_match(const string& pattern, const string& text) {
        string combined = pattern + "$" + text;
        vector<int> z = compute_z(combined);
        vector<int> matches;

        int pattern_len = pattern.length();
        for (int i = pattern_len + 1; i < combined.length(); i++) {
            if (z[i] == pattern_len) { matches.push_back(i - pattern_len - 1); }
        }

        return matches;
    }

    // 计算字符串的周期
    static vector<int> compute_periods(const string& s) {
        vector<int> z = compute_z(s);
        vector<int> periods;
        int n = s.length();

        for (int i = 1; i < n; i++) {
            if (i + z[i] == n) { periods.push_back(i); }
        }

        return periods;
    }

    // 查找最小周期
    static int minimal_period(const string& s) {
        vector<int> periods = compute_periods(s);
        return periods.empty() ? s.length() : periods[0];
    }

    // 判断字符串是否是周期性的
    static bool is_periodic(const string& s) { return minimal_period(s) < s.length(); }

    // 计算最长公共前缀数组
    static vector<int> longest_common_prefix(const string& s, const vector<string>& strings) {
        vector<int> lcp;

        for (const string& t : strings) {
            string combined = s + "$" + t;
            vector<int> z = compute_z(combined);
            int max_lcp = 0;

            for (int i = s.length() + 1; i < combined.length(); i++) { max_lcp = max(max_lcp, z[i]); }

            lcp.push_back(max_lcp);
        }

        return lcp;
    }
};
]=]),

-- 06_String_Algorithms\Pattern_Matching\Z_Algorithm.h
ps("06_string_algorithms_pattern_matching_z_algorithm_h", [=[

/*
 * Z算法 - 线性时间字符串匹配
 * 时间复杂度: O(n)
 * 空间复杂度: O(n)
 * 适用场景: 前缀匹配、字符串匹配、周期性检测
 */
struct ZAlgorithm {
    vector<int> z;
    string s;

    ZAlgorithm(const string& str) : s(str) {
        int n = s.length();
        z.resize(n);
        compute_z();
    }

    // 计算Z数组
    void compute_z() {
        int n = s.length();
        if (n == 0) return;

        z[0] = n;
        int l = 0, r = 0;

        for (int i = 1; i < n; i++) {
            if (i <= r) { z[i] = min(r - i + 1, z[i - l]); }

            while (i + z[i] < n && s[z[i]] == s[i + z[i]]) { z[i]++; }

            if (i + z[i] - 1 > r) {
                l = i;
                r = i + z[i] - 1;
            }
        }
    }

    // 在文本中查找模式串的所有出现位置
    vector<int> find_occurrences(const string& pattern, const string& text) {
        string combined = pattern + "$" + text;
        ZAlgorithm za(combined);

        vector<int> occurrences;
        int p_len = pattern.length();

        for (int i = p_len + 1; i < combined.length(); i++) {
            if (za.z[i] == p_len) { occurrences.push_back(i - p_len - 1); }
        }

        return occurrences;
    }

    // 获取Z数组
    vector<int> get_z_array() const { return z; }

    // 计算字符串的最长公共前缀后缀长度
    int longest_prefix_suffix() {
        int n = s.length();
        for (int i = 1; i < n; i++) {
            if (z[i] == n - i) { return z[i]; }
        }
        return 0;
    }

    // 检查字符串是否为周期字符串
    bool is_periodic() {
        int n = s.length();
        for (int period = 1; period <= n / 2; period++) {
            if (n % period == 0) {
                bool valid = true;
                for (int i = period; i < n && valid; i++) {
                    if (z[i] < n - i) { valid = false; }
                }
                if (valid) return true;
            }
        }
        return false;
    }

    // 获取最小周期长度
    int minimum_period() {
        int n = s.length();
        for (int period = 1; period <= n; period++) {
            if (n % period == 0) {
                bool valid = true;
                for (int i = period; i < n && valid; i++) {
                    if (z[i] < n - i) { valid = false; }
                }
                if (valid) return period;
            }
        }
        return n;
    }
};

// 扩展Z算法 - 处理两个不同字符串
struct ExtendedZ {
    // 计算字符串s关于字符串t的扩展Z数组
    static vector<int> extended_z(const string& s, const string& t) {
        int n = s.length(), m = t.length();
        vector<int> z(n);

        if (n == 0) return z;

        // 首先计算z[0]
        int k = 0;
        while (k < n && k < m && s[k] == t[k]) k++;
        z[0] = k;

        int l = 0, r = 0;
        if (z[0] > 0) {
            l = 0;
            r = z[0] - 1;
        }

        for (int i = 1; i < n; i++) {
            if (i > r) {
                // 从头开始匹配
                k = 0;
                while (i + k < n && k < m && s[i + k] == t[k]) k++;
                z[i] = k;
                if (k > 0) {
                    l = i;
                    r = i + k - 1;
                }
            } else {
                // 利用已知信息
                int beta = r - i + 1;
                if (z[i - l] < beta) {
                    z[i] = z[i - l];
                } else {
                    k = beta;
                    while (i + k < n && k < m && s[i + k] == t[k]) k++;
                    z[i] = k;
                    l = i;
                    r = i + k - 1;
                }
            }
        }

        return z;
    }

    // 查找字符串t在字符串s中的所有出现位置
    static vector<int> find_all(const string& s, const string& t) {
        vector<int> z = extended_z(s, t);
        vector<int> positions;

        for (int i = 0; i < s.length(); i++) {
            if (z[i] == t.length()) { positions.push_back(i); }
        }

        return positions;
    }
};
]=]),

-- 06_String_Algorithms\Rolling_Hash\DoubleHash.h
ps("06_string_algorithms_rolling_hash_doublehash_h", [=[

/*
 * 双哈希滚动哈希
 * 时间复杂度: 预处理O(n), 查询O(1)
 * 空间复杂度: O(n)
 * 适用场景: 字符串哈希、降低冲突概率、高精度字符串比较
 */
struct DoubleHash {
    static const int MOD1 = 1e9 + 7;
    static const int MOD2 = 1e9 + 9;
    static const int BASE1 = 31;
    static const int BASE2 = 37;

    vector<long long> hash1, hash2, power1, power2;
    string s;
    int n;

    DoubleHash(const string& str) : s(str), n(str.length()) {
        hash1.resize(n + 1);
        hash2.resize(n + 1);
        power1.resize(n + 1);
        power2.resize(n + 1);
        build();
    }

   private:
    void build() {
        power1[0] = power2[0] = 1;
        hash1[0] = hash2[0] = 0;

        for (int i = 0; i < n; i++) {
            int char_val = s[i] - 'a' + 1;

            power1[i + 1] = (power1[i] * BASE1) % MOD1;
            power2[i + 1] = (power2[i] * BASE2) % MOD2;

            hash1[i + 1] = (hash1[i] * BASE1 + char_val) % MOD1;
            hash2[i + 1] = (hash2[i] * BASE2 + char_val) % MOD2;
        }
    }

   public:
    // 获取子串s[l..r]的双哈希值
    pair<long long, long long> get_hash(int l, int r) const {
        long long h1 = (hash1[r + 1] - (hash1[l] * power1[r - l + 1]) % MOD1 + MOD1) % MOD1;
        long long h2 = (hash2[r + 1] - (hash2[l] * power2[r - l + 1]) % MOD2 + MOD2) % MOD2;
        return {h1, h2};
    }

    // 比较两个子串是否相等
    bool equal(int l1, int r1, int l2, int r2) const { return get_hash(l1, r1) == get_hash(l2, r2); }

    // 计算字符串的双哈希值（静态方法）
    static pair<long long, long long> compute_hash(const string& str) {
        long long h1 = 0, h2 = 0;
        long long pow1 = 1, pow2 = 1;

        for (char c : str) {
            int char_val = c - 'a' + 1;
            h1 = (h1 + char_val * pow1) % MOD1;
            h2 = (h2 + char_val * pow2) % MOD2;
            pow1 = (pow1 * BASE1) % MOD1;
            pow2 = (pow2 * BASE2) % MOD2;
        }

        return {h1, h2};
    }

    // 在文本中查找模式串
    vector<int> find_pattern(const string& pattern) const {
        vector<int> positions;
        if (pattern.length() > n) return positions;

        auto pattern_hash = compute_hash(pattern);
        int len = pattern.length();

        for (int i = 0; i <= n - len; i++) {
            if (get_hash(i, i + len - 1) == pattern_hash) { positions.push_back(i); }
        }

        return positions;
    }

    // 最长公共前缀
    int lcp(int i, int j) const {
        int left = 0, right = min(n - i, n - j);
        int result = 0;

        while (left <= right) {
            int mid = (left + right) / 2;
            if (equal(i, i + mid - 1, j, j + mid - 1)) {
                result = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return result;
    }

    // 计算所有长度为k的不同子串数量
    int count_distinct_substrings(int k) const {
        if (k > n) return 0;

        set<pair<long long, long long>> unique_hashes;

        for (int i = 0; i <= n - k; i++) { unique_hashes.insert(get_hash(i, i + k - 1)); }

        return unique_hashes.size();
    }

    // 找到最长重复子串
    pair<int, int> longest_repeated_substring() const {
        int left = 1, right = n;
        int max_len = 0;
        int position = -1;

        while (left <= right) {
            int mid = (left + right) / 2;

            map<pair<long long, long long>, int> hash_pos;
            bool found = false;

            for (int i = 0; i <= n - mid; i++) {
                auto h = get_hash(i, i + mid - 1);
                if (hash_pos.count(h)) {
                    max_len = mid;
                    position = hash_pos[h];
                    found = true;
                    break;
                } else {
                    hash_pos[h] = i;
                }
            }

            if (found) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return {max_len, position};
    }

    // 查找所有长度为k的重复子串
    vector<pair<int, vector<int>>> find_repeated_substrings(int k) const {
        if (k > n) return {};

        map<pair<long long, long long>, vector<int>> hash_positions;

        for (int i = 0; i <= n - k; i++) {
            auto h = get_hash(i, i + k - 1);
            hash_positions[h].push_back(i);
        }

        vector<pair<int, vector<int>>> result;
        for (const auto& [hash_val, positions] : hash_positions) {
            if (positions.size() > 1) { result.push_back({k, positions}); }
        }

        return result;
    }

    // 检查字符串是否有长度为k的重复子串
    bool has_repeated_substring(int k) const {
        if (k > n) return false;

        set<pair<long long, long long>> seen_hashes;

        for (int i = 0; i <= n - k; i++) {
            auto h = get_hash(i, i + k - 1);
            if (seen_hashes.count(h)) { return true; }
            seen_hashes.insert(h);
        }

        return false;
    }

    // 计算两个字符串的最长公共子串
    static pair<int, int> longest_common_substring(const string& s1, const string& s2) {
        DoubleHash h1(s1), h2(s2);
        int n1 = s1.length(), n2 = s2.length();

        int left = 0, right = min(n1, n2);
        int max_len = 0;
        int pos1 = -1, pos2 = -1;

        while (left <= right) {
            int mid = (left + right) / 2;

            set<pair<long long, long long>> hashes1;
            for (int i = 0; i <= n1 - mid; i++) { hashes1.insert(h1.get_hash(i, i + mid - 1)); }

            bool found = false;
            for (int i = 0; i <= n2 - mid; i++) {
                auto h = h2.get_hash(i, i + mid - 1);
                if (hashes1.count(h)) {
                    max_len = mid;
                    pos2 = i;
                    // 找到s1中对应的位置
                    for (int j = 0; j <= n1 - mid; j++) {
                        if (h1.get_hash(j, j + mid - 1) == h) {
                            pos1 = j;
                            break;
                        }
                    }
                    found = true;
                    break;
                }
            }

            if (found) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return {max_len, pos1};  // 返回长度和在s1中的位置
    }

    // 字符串匹配的KMP风格算法
    vector<int> string_matching_optimized(const string& pattern) const {
        vector<int> result;
        if (pattern.length() > n) return result;

        DoubleHash pattern_hash(pattern);
        auto target_hash = pattern_hash.get_hash(0, pattern.length() - 1);

        for (int i = 0; i <= n - (int)pattern.length(); i++) {
            if (get_hash(i, i + pattern.length() - 1) == target_hash) { result.push_back(i); }
        }

        return result;
    }
};
]=]),

}
