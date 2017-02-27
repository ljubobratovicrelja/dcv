/**
Module contains core types and structures used in library.

Copyright: Copyright Relja Ljubobratovic 2016.

Authors: Relja Ljubobratovic

License: $(LINK3 http://www.boost.org/LICENSE_1_0.txt, Boost Software License - Version 1.0).
*/

module dcv.core.types;

import std.typecons : Tuple;
import std.traits : isIntegral, isNumeric;

import mir.ndslice;

alias Point(T) = Tuple!(T, "x", T, "y");
alias Size(T) = Tuple!(T, "width", T, "height");

struct Rectangle(T)
if (isNumeric!T)
{
    T x = T(0), y = T(0), width = T(0), height = T(0);

    T xmax() const { return x + width; }
    T ymax() const { return y + height; }

    T area() const { return width * height; }

    Size!T shape() const { return Size!T(width, height); }
    bool empty() const { return area == T(0); }

    auto rows() const { return iota!T([height], y); }
    auto cols() const { return iota!T([width], x); }
}

Rectangle!T region(T)(T x, T y, T width, T height)
{
    return Rectangle!T(x, y, width, height);
}

Rectangle!T region(T)(Size!T size)
{
    return Rectangle!T(T(0), T(0), size.width, size.height);
}

auto sliced(SliceKind kind, Iterator, T)
(
    Slice!(kind, [2], Iterator) matrix,
    const ref Rectangle!T region
) if (isIntegral!T)
in
{
    assert(!region.empty, "Given region must not be empty.");
}
body
{
    return matrix[region.y .. region.ymax, region.x .. region.xmax];
}
