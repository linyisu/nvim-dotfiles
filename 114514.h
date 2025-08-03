#pragma once
#include <bitset>
#include <iostream>
#include <map>
#include <queue>
#include <ranges>
#include <set>
#include <stack>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

template <typename T>
concept StringLike = std::same_as<std::decay_t<T>, std::string> || std::same_as<std::decay_t<T>, std::string_view>;

template <typename T>
concept Pair = requires(T a) {
    { a.first };
    { a.second };
} && !StringLike<T>;

template <typename T>
concept Tuple = requires { typename std::tuple_size<std::remove_cvref_t<T>>::type; } &&
                std::tuple_size_v<std::remove_cvref_t<T>> >= 2 && !Pair<T> && !StringLike<T>;

template <typename T>
concept Queue = requires(T t) {
    typename T::value_type;
    typename T::container_type;
    t.front();
    t.back();
    t.size();
    t.empty();
    t.push(std::declval<typename T::value_type>());
    t.pop();
} && std::same_as<std::decay_t<T>, std::queue<typename T::value_type, typename T::container_type>>;

template <typename T>
concept Stack = requires(T t) {
    typename T::value_type;
    typename T::container_type;
    t.top();
    t.size();
    t.empty();
    t.push(std::declval<typename T::value_type>());
    t.pop();
} && std::same_as<std::decay_t<T>, std::stack<typename T::value_type, typename T::container_type>>;

template <typename T>
concept PriorityQueue = requires(T t) {
    typename T::value_type;
    typename T::container_type;
    t.top();
    t.size();
    t.empty();
    t.push(std::declval<typename T::value_type>());
    t.pop();
} && std::same_as<std::decay_t<T>, std::priority_queue<typename T::value_type, typename T::container_type>>;

template <typename T>
concept AssociativeContainer =
    requires(T t) {
        typename T::key_type;
        typename T::value_type;
        t.begin();
        t.end();
        t.size();
        t.empty();
    } && (std::same_as<std::decay_t<T>, std::set<typename T::key_type>> ||
          std::same_as<std::decay_t<T>, std::multiset<typename T::key_type>> ||
          std::same_as<std::decay_t<T>, std::map<typename T::key_type, typename T::mapped_type>> ||
          std::same_as<std::decay_t<T>, std::multimap<typename T::key_type, typename T::mapped_type>> ||
          std::same_as<std::decay_t<T>, std::unordered_set<typename T::key_type>> ||
          std::same_as<std::decay_t<T>, std::unordered_multiset<typename T::key_type>> ||
          std::same_as<std::decay_t<T>, std::unordered_map<typename T::key_type, typename T::mapped_type>> ||
          std::same_as<std::decay_t<T>, std::unordered_multimap<typename T::key_type, typename T::mapped_type>>);

template <typename T>
concept VectorBitset = requires(T t) {
    typename T::value_type;
    requires std::same_as<std::decay_t<T>, std::vector<typename T::value_type>>;
    requires requires(typename T::value_type v) {
        { v.size() } -> std::same_as<std::size_t>;
        { v.to_string() } -> std::same_as<std::string>;
        { v[std::size_t{}] } -> std::same_as<typename std::decay_t<typename T::value_type>::reference>;
    };
};

template <typename T>
concept Range =
    requires(T a) {
        std::ranges::begin(a);
        std::ranges::end(a);
    } && !StringLike<T> && !Pair<T> && !Tuple<T> && !Queue<T> && !Stack<T> && !PriorityQueue<T> &&
    !AssociativeContainer<T> && !VectorBitset<T>;

template <typename T>
concept Range2D = Range<T> && requires(T t) { typename T::value_type; } && Range<typename T::value_type>;

void _print_one(const auto& x, int indent = 0)
    requires(!Queue<std::remove_cvref_t<decltype(x)>> && !Stack<std::remove_cvref_t<decltype(x)>> &&
             !PriorityQueue<std::remove_cvref_t<decltype(x)>> &&
             !AssociativeContainer<std::remove_cvref_t<decltype(x)>> && !Pair<std::remove_cvref_t<decltype(x)>> &&
             !Tuple<std::remove_cvref_t<decltype(x)>> && !Range<std::remove_cvref_t<decltype(x)>> &&
             !VectorBitset<std::remove_cvref_t<decltype(x)>>)
{
    if constexpr (StringLike<decltype(x)>) {
        std::cerr << '"' << x << '"';
    } else {
        std::cerr << x;
    }
}

template <Pair P>
void _print_one(const P& p, int indent) {
    std::cerr << '(';
    _print_one(p.first, indent);
    std::cerr << ", ";
    _print_one(p.second, indent);
    std::cerr << ')';
}

template <Tuple T>
void _print_one(const T& t, int indent) {
    std::cerr << '(';
    bool first = true;
    std::apply(
        [&](auto&&... elems) {
            ((std::cerr << (first ? "" : ", "), first = false, _print_one(elems, indent)), ...);
        },
        t);
    std::cerr << ')';
}

template <Range2D R>
void _print_one(const R& r, int indent) {
    std::cerr << "{\n";
    int idx = 0;
    for (auto&& row : r) {
        for (int i = 0; i < indent + 2; ++i) std::cerr << ' ';
        std::cerr << "[" << idx++ << "]: {";

        bool first = true;
        for (auto&& e : row) {
            if (!first) std::cerr << ", ";
            first = false;
            _print_one(e, indent + 2);
        }
        std::cerr << "}\n";
    }
    for (int i = 0; i < indent; ++i) std::cerr << ' ';
    std::cerr << "}";
}

template <VectorBitset VB>
void _print_one(const VB& vb, int indent) {
    if (vb.empty()) {
        std::cerr << "{}";
        return;
    }

    // Find the maximum bitset size for alignment
    std::size_t max_size = 0;
    for (const auto& bitset : vb) {
        max_size = std::max(max_size, bitset.size());
    }

    std::cerr << "{\n";
    int idx = 0;
    for (const auto& bitset : vb) {
        for (int i = 0; i < indent + 2; ++i) std::cerr << ' ';
        std::cerr << "[" << idx++ << "]: ";

        // Print leading zeros to align
        std::string bitstr = bitset.to_string();
        std::size_t leading_zeros = max_size - bitset.size();
        for (std::size_t i = 0; i < leading_zeros; ++i) {
            std::cerr << '0';
        }
        std::cerr << bitstr << '\n';
    }
    for (int i = 0; i < indent; ++i) std::cerr << ' ';
    std::cerr << "}";
}

template <Range R>
void _print_one(const R& r, int indent)
    requires(!Range2D<R>)
{
    std::cerr << '{';
    bool first = true;
    for (auto&& e : r) {
        if (!first) std::cerr << ", ";
        first = false;
        _print_one(e, indent);
    }
    std::cerr << '}';
}

template <Queue Q>
void _print_one(Q q, int indent = 0) {
    std::cerr << "queue{";
    bool first = true;
    while (!q.empty()) {
        if (!first) std::cerr << ", ";
        first = false;
        _print_one(q.front(), indent);
        q.pop();
    }
    std::cerr << '}';
}

template <Stack S>
void _print_one(S s, int indent = 0) {
    std::cerr << "stack{";
    std::vector<typename S::value_type> elements;
    while (!s.empty()) {
        elements.push_back(s.top());
        s.pop();
    }

    bool first = true;
    for (auto it = elements.rbegin(); it != elements.rend(); ++it) {
        if (!first) std::cerr << ", ";
        first = false;
        _print_one(*it, indent);
    }
    std::cerr << '}';
}

template <PriorityQueue PQ>
void _print_one(PQ pq, int indent = 0) {
    std::cerr << "priority_queue{";
    std::vector<typename PQ::value_type> elements;
    while (!pq.empty()) {
        elements.push_back(pq.top());
        pq.pop();
    }

    bool first = true;
    for (const auto& elem : elements) {
        if (!first) std::cerr << ", ";
        first = false;
        _print_one(elem, indent);
    }
    std::cerr << '}';
}

template <AssociativeContainer AC>
void _print_one(const AC& ac, int indent = 0) {
    if constexpr (requires { typename AC::mapped_type; }) {
        // map-like containers
        std::cerr << "map{";
    } else {
        // set-like containers
        std::cerr << "set{";
    }

    bool first = true;
    for (const auto& elem : ac) {
        if (!first) std::cerr << ", ";
        first = false;
        _print_one(elem, indent);
    }
    std::cerr << '}';
}

template <typename T>
void _debug_print_args(const char* names, const T& arg) {
    std::cerr << names << " = ";
    _print_one(arg, 0);
    std::cerr << '\n';
}

template <typename T, typename... Args>
void _debug_print_args(const char* names, const T& arg, const Args&... args) {
    const char* comma = names;
    while (*comma && *comma != ',') comma++;

    std::cerr << std::string(names, comma) << " = ";
    _print_one(arg, 0);
    std::cerr << '\n';

    if (*comma == ',') {
        comma++;
        while (*comma == ' ') comma++;
        _debug_print_args(comma, args...);
    }
}

#define dbg(...)                                                      \
    do {                                                                \
        std::cerr << "=== DEBUG [Line " << __LINE__ << "] ===" << '\n'; \
        _debug_print_args(#__VA_ARGS__, __VA_ARGS__);                   \
        std::cerr << "=============" << '\n';                           \
    } while (0)
