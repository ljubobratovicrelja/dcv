/**
Module contains $(LINK3 https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method, Lucas-Kanade) optical flow algorithm implementation.

Copyright: Copyright Relja Ljubobratovic 2016.

Authors: Relja Ljubobratovic

License: $(LINK3 http://www.boost.org/LICENSE_1_0.txt, Boost Software License - Version 1.0).
*/

module dcv.tracking.opticalflow.lucaskanade;

import std.typecons : Flag, No;
import std.traits : isFloatingPoint;

import mir.ndslice.slice : Slice, Contiguous, Canonical;
import mir.ndslice.topology : map, windows;
import mir.ndslice.allocation : slice, makeSlice;

import dcv.tracking.opticalflow.base;

enum LucasKanadeError
{
    brightness,
    eigenvalue
}

/**
Lucas-Kanade optical flow method implementation.
*/
struct LucasKanadeFlow(PixelType, CoordType)
{
    mixin SparseOpticalFlow!(PixelType, CoordType);

    private
    {
        float _sigma = 0.84f;
        size_t _iters = 10;
        int[2] _win = [41, 41];
        LucasKanadeError _error;
    }

    @property
    {
        ///
        float sigma() { return _sigma; }
        ///
        size_t iterationCount() { return _iters; }
        ///
        int[2] windowSize() const { return _win; }
        ///
        ref errorMethod() { return _error; }

        ///
        void sigma(float value)
        in
        {
            assert(value > 0, "Invalid sigma value - should be larger than 0.");
        }
        body
        {
            _sigma = value;
        }

        ///
        void iterationCount(size_t value)
        in
        {
            assert(value > 0, "Invalid iteration count - should be larger than 0.");
        }
        body
        {
            _iters = value;
        }
    }

    /// Set window size.
    void setWindowSize(int rows, int cols)
    in
    {
        assert(rows >= 3 && cols >= 3, "Minimal size for the window side is 3.");
        assert(rows % 2 != 0 && cols % 2 != 0, "Window size should be odd.");
    }
    body
    {
        _win[0] = rows;
        _win[1] = cols;
    }

    /// ditto
    void setWindowSize(int size)
    {
        setWindowSize(size, size);
    }

    nothrow @nogc
    void evaluateImpl
    (
        Slice!(Contiguous, [2], const(PixelType)*) prevFrame,
        Slice!(Contiguous, [2], const(PixelType)*) currFrame,
        Slice!(Contiguous, [2], const(CoordType)*) prevPoints,
        ref Slice!(Contiguous, [2], CoordType*) currPoints,
        Slice!(Contiguous, [1], float*) error
    )
    {
        import std.math : PI;
        import std.experimental.allocator.mallocator : Mallocator;
        import std.algorithm.iteration : sum;

        import mir.math.internal : floor, sqrt, exp;
        import mir.ndslice.topology : flattened;

        import dcv.imgproc.interpolate : linear;
        import dcv.imgproc.filter : gaussian;

        alias T = CoordType;

        static if (is(PixelType == T))
        {
            auto f1 = prevFrame;
            auto f2 = currFrame;
        }
        else
        {
            auto f1 = prevFrame.as!T;
            auto f2 = current.as!T;
        }

        immutable rows = f1.length!0;
        immutable cols = f1.length!1;
        immutable rl = cast(int)(rows - 1);
        immutable cl = cast(int)(cols - 1);
        immutable pointCount = prevPoints.length;
        immutable pixelCount = rows * cols;
        immutable calcError = !error.empty;
        immutable wwh = _win[0] / 2, whh = _win[1] / 2;

        auto gaussBuf = makeSlice!T(Mallocator.instance, _win[0], _win[1]);
        scope(exit) { Mallocator.instance.deallocate(gaussBuf.array); }

        auto gauss = gaussBuf.slice;
        gauss[] = gaussian!T([cast(size_t)_win[0], cast(size_t)_win[0]], sigma, sigma);

        // TODO: norm1 from mir?
        auto gnorm = gauss.flattened.sum;
        gauss[] /= gnorm;

        auto fxs = f1.windows(3, 3).map!sobel_x;
        auto fys = f1.windows(3, 3).map!sobel_y;

        T a1, a2, a3, b1, b2;

        foreach (ptn; 0 .. pointCount) // TODO: solve in parallel
        {
            auto p = prevPoints[ptn];

            int
                rb = cast(int)p[1] - wwh,
                re = cast(int)p[1] + wwh + 1,
                cb = cast(int)p[0] - whh,
                ce = cast(int)p[0] + whh + 1;

            rb = rb < 1 ? 1 : rb;
            re = re >= rl - 1 ? rl - 2 : re;
            cb = cb < 1 ? 1 : cb;
            ce = ce >= cl - 1 ? cl - 2 : ce;

            if (re - rb <= 0 || ce - cb <= 0)
            {
                continue;
            }

            a1 = a2 = a3 = b1 = b2 = 0.0f;

            foreach (iteration; 0 .. iterationCount)
            {
                immutable nx = currPoints[ptn, 0] - p[0];
                immutable ny = currPoints[ptn, 1] - p[1];

                size_t ii = 0, jj;
                foreach (i; rb .. re)
                {
                    jj = 0;
                    foreach (j; cb .. ce)
                    {
                        auto w = gauss[ii, jj];

                        immutable nnx = cast(T)j + nx;
                        immutable nny = cast(T)i + ny;

                        if (nnx < 0 || nnx >= cols || nny < 0 || nny >= rows)
                            continue;

                        immutable fx = fxs[i-1, j-1]; // -1 because of shift by sobel convolution kernel size
                        immutable fy = fys[i-1, j-1];
                        immutable ft = linear(f2, nny, nnx) - f1[i, j];

                        immutable fxx = fx * fx;
                        immutable fyy = fy * fy;
                        immutable fxy = fx * fy;

                        a1 += w * fxx;
                        a2 += w * fxy;
                        a3 += w * fyy;

                        b1 += w * fx * ft;
                        b2 += w * fy * ft;
                        ++jj;
                    }
                    ++ii;
                }

                auto d = (a1 * a3 - a2 * a2);

                if (d)
                {
                    d = 1.0f / d;
                    currPoints[ptn, 0] += (a2 * b2 - a3 * b1) * d;
                    currPoints[ptn, 1] += (a2 * b1 - a1 * b2) * d;
                }
            }

            if (calcError)
            {
                float n = cast(float)(_win[0]*_win[1]);
                error[ptn] = ((a1 + a3) - sqrt((a1 - a3) * (a1 - a3) + a2 * a2)) / n;
            }
        }
    }
}

nothrow @nogc static
{
    auto sobel_x(T)(Slice!(Canonical, [2], const(T)*) window)
    {
        return (window[0, 2] - window[0, 0]) +
           2 * (window[1, 2] - window[1, 0]) +
               (window[2, 2] - window[2, 0]);
    }

    auto sobel_y(T)(Slice!(Canonical, [2], const(T)*) window)
    {
        return (window[2, 0] - window[0, 0]) +
           2 * (window[2, 1] - window[0, 1]) +
               (window[2, 2] - window[0, 2]);
    }
}

// TODO: implement functional tests.
version (unittest)
{
    import std.algorithm.iteration : map;
    import std.range : iota;
    import std.array : array;
    import std.random : uniform;

    private auto createImage()
    {
        return new Image(5, 5, ImageFormat.IF_MONO, BitDepth.BD_8,
                25.iota.map!(v => cast(ubyte)uniform(0, 255)).array);
    }
}

unittest
{
    LucasKanadeFlow flow = new LucasKanadeFlow;
    auto f1 = createImage();
    auto f2 = createImage();
    auto p = 10.iota.map!(v => cast(float[2])[cast(float)uniform(0, 2), cast(float)uniform(0, 2)]).array;
    auto r = 10.iota.map!(v => cast(float[2])[3.0f, 3.0f]).array;
    auto f = flow.evaluate(f1, f2, p, r);
    assert(f.length == p.length);
    assert(flow.cornerResponse.length == p.length);
}

unittest
{
    LucasKanadeFlow flow = new LucasKanadeFlow;
    auto f1 = createImage();
    auto f2 = createImage();
    auto p = 10.iota.map!(v => cast(float[2])[cast(float)uniform(0, 2), cast(float)uniform(0, 2)]).array;
    auto f = 10.iota.map!(v => cast(float[2])[cast(float)uniform(0, 2), cast(float)uniform(0, 2)]).array;
    auto r = 10.iota.map!(v => cast(float[2])[3.0f, 3.0f]).array;
    auto fe = flow.evaluate(f1, f2, p, r, f);
    assert(f.length == fe.length);
    assert(f.ptr == fe.ptr);
    assert(flow.cornerResponse.length == p.length);
}
