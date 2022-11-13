#include "Utils/General.h"

template <>
struct std::hash<std::pair<uint, uint>>
{
    std::size_t operator()(std::pair<uint, uint> const &idxPair) const noexcept
    {
        size_t h = (size_t(idxPair.first) << 32) + size_t(idxPair.second);
        h *= 1231231557ull; // "random" uneven integer
        h ^= (h >> 32);
        return h;
    }
};

std::pair<uint, uint> makeOrderedIdxPair(uint idx1, uint idx2)
{
    return std::pair<uint, uint>(std::max(idx1, idx2), std::min(idx1, idx2));
}

static int g_tabIndent = -1;

TabIndentContext::TabIndentContext()
{
    g_tabIndent += 1;
}

TabIndentContext::~TabIndentContext()
{
    g_tabIndent -= 1;
}

std::ostream & operator<<(
    std::ostream &os, const TabIndentContext &ctx)
{
    for (auto currTabIndent = 0; currTabIndent < g_tabIndent * 4; currTabIndent++)
        os << " ";
    return os;
}