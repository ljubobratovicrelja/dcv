/**
Value interpolation module.

Copyright: Copyright Relja Ljubobratovic 2016.

Authors: Relja Ljubobratovic

License: $(LINK3 http://www.boost.org/LICENSE_1_0.txt, Boost Software License - Version 1.0).
*/
module dcv.imgproc.interpolate;

import std.traits : isNumeric, isScalarType, isIntegral, allSameType, allSatisfy, ReturnType,
    isFloatingPoint;

import mir.ndslice.internal : fastmath;

import mir.ndslice.slice;


/**
Test if given function is proper form for interpolation.
*/
static bool isInterpolationFunc(alias F)()
{
    auto s = [0., 1.].sliced(2);
    return (__traits(compiles, F(s, 0))); // TODO: check the return type?
}

/**
Test for 1D (vector) interpolation function.
*/
static bool isInterpolationFunc1D(alias F)()
{
    return isInterpolationFunc!F;
}

/**
Test for 2D (matrix) interpolation function.
*/
static bool isInterpolationFunc2D(alias F)()
{
    auto s = [0, 1, 2, 3].sliced(2, 2);
    return (__traits(compiles, F(s, 3, 3)));
}

unittest
{
    static assert(isInterpolationFunc!linear);
    static assert(isInterpolationFunc1D!linear);
    static assert(isInterpolationFunc2D!linear);
}

/**
Linear interpolation.

Params:
    slice = Input slice which values are interpolated.
    pos = Position on which slice values are interpolated.

Returns:
    Interpolated resulting value.
*/
pure auto linear(SliceKind kind, size_t[] packs, Iterator, P, size_t N)(Slice!(kind, packs, Iterator) slice, P[N] pos...)
{
    static assert(packs.length == 1, "Packed slices are not supported.");
    static assert(packs[0] == N, "Invalid indexing dimensionality.");

    static if (N == 1)
    {
        return linearImpl_1(slice, pos[0]);
    }
    else static if (N == 2)
    {
        return linearImpl_2(slice, pos[0], pos[1]);
    }
    else static if (N == 3)
    {
        return linearImpl_3(slice, pos[0], pos[1], pos[2]);
    }
    else
    {
        static assert(0, "Unsupported slice dimension for linear interpolation.");
    }
}

unittest
{
    auto arr1 = [0., 1.].sliced(2);
    assert(linear(arr1, 0.) == 0.);
    assert(linear(arr1, 1.) == 1.);
    assert(linear(arr1, 0.1) == 0.1);
    assert(linear(arr1, 0.5) == 0.5);
    assert(linear(arr1, 0.9) == 0.9);

    auto arr1_integral = [0, 10].sliced(2);
    assert(linear(arr1_integral, 0.) == 0);
    assert(linear(arr1_integral, 1.) == 10);
    assert(linear(arr1_integral, 0.1) == 1);
    assert(linear(arr1_integral, 0.5) == 5);
    assert(linear(arr1_integral, 0.9) == 9);

    auto arr2 = [0., 0., 0., 1.].sliced(2, 2);
    assert(arr2.linear(0.5, 0.5) == 0.25);
    assert(arr2.linear(0., 0.) == 0.);
    assert(arr2.linear(1., 1.) == 1.);
    assert(arr2.linear(1., 0.) == 0.);
}

private:

pure @fastmath auto linearImpl_1(Iterator)(Slice!(Contiguous, [1], Iterator) range, float pos)
{
    import mir.math.internal;

    assert(pos < range.length);
    alias T = DeepElementType!(typeof(range));

    if (pos == range.length - 1)
    {
        return range[$ - 1];
    }

    size_t round = cast(size_t)pos.floor;
    float weight = pos - cast(float)round;

    static if (isIntegral!T)
    {
        // TODO: is this branch really necessary?
        auto v1 = cast(float)range[round];
        auto v2 = cast(float)range[round + 1];
    }
    else
    {
        auto v1 = range[round];
        auto v2 = range[round + 1];
    }
    return cast(T)(v1 * (1. - weight) + v2 * (weight));
}

pure @fastmath auto linearImpl_2(SliceKind kind, size_t[] packs, Iterator)(Slice!(kind, packs, Iterator) range, float pos_x, float pos_y)
{
    import mir.math.internal : floor;
    import std.traits : Unqual;

    alias T = Unqual!(DeepElementType!(typeof(range)));

    assert(pos_x < range.length!0 && pos_y < range.length!1);

    size_t rx = cast(size_t)pos_x.floor;
    size_t ry = cast(size_t)pos_y.floor;
    float wx = pos_x - cast(float)rx;
    float wy = pos_y - cast(float)ry;

    auto w00 = (1. - wx) * (1. - wy);
    auto w01 = (wx) * (1. - wy);
    auto w10 = (1. - wx) * (wy);
    auto w11 = (wx) * (wy);

    auto x_end = rx == range.length!0 - 1;
    auto y_end = ry == range.length!1 - 1;

    static if (isIntegral!T)
    {
        // TODO: (same as in 1D vesion) is this branch really necessary?
        float v1, v2, v3, v4;
        v1 = cast(float)range[rx, ry];
        v2 = cast(float)range[x_end ? rx : rx + 1, ry];
        v3 = cast(float)range[rx, y_end ? ry : ry + 1];
        v4 = cast(float)range[x_end ? rx : rx + 1, y_end ? ry : ry + 1];
    }
    else
    {
        auto v1 = range[rx, ry];
        auto v2 = range[x_end ? rx : rx + 1, ry];
        auto v3 = range[rx, y_end ? ry : ry + 1];
        auto v4 = range[x_end ? rx : rx + 1, y_end ? ry : ry + 1];
    }
    return cast(T)(v1 * w00 + v2 * w01 + v3 * w10 + v4 * w11);
}

pure @fastmath auto linearImpl_3(SliceKind kind, size_t[] packs, Iterator)(Slice!(kind, packs, Iterator) range, float pos_x, float pos_y, float pos_z)
{
    import mir.math.internal : floor;
    import std.traits : Unqual;

    alias T = Unqual!(DeepElementType!(typeof(range)));

    assert(pos_x < range.length!0 && pos_y < range.length!1);

    size_t channel = cast(size_t)pos_z;
    size_t rx = cast(size_t)pos_x.floor;
    size_t ry = cast(size_t)pos_y.floor;
    float wx = pos_x - cast(float)rx;
    float wy = pos_y - cast(float)ry;

    auto w00 = (1. - wx) * (1. - wy);
    auto w01 = (wx) * (1. - wy);
    auto w10 = (1. - wx) * (wy);
    auto w11 = (wx) * (wy);

    auto x_end = rx == range.length!0 - 1;
    auto y_end = ry == range.length!1 - 1;

    static if (isIntegral!T)
    {
        // TODO: (same as in 1D vesion) is this branch really necessary?
        float v1, v2, v3, v4;
        v1 = cast(float)range[rx, ry, channel];
        v2 = cast(float)range[x_end ? rx : rx + 1, ry, channel];
        v3 = cast(float)range[rx, y_end ? ry : ry + 1, channel];
        v4 = cast(float)range[x_end ? rx : rx + 1, y_end ? ry : ry + 1, channel];
    }
    else
    {
        auto v1 = range[rx, ry, channel];
        auto v2 = range[x_end ? rx : rx + 1, ry, channel];
        auto v3 = range[rx, y_end ? ry : ry + 1, channel];
        auto v4 = range[x_end ? rx : rx + 1, y_end ? ry : ry + 1, channel];
    }
    return cast(T)(v1 * w00 + v2 * w01 + v3 * w10 + v4 * w11);
}
