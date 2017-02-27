/**
Module contains $(LINK3 https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method, Lucas-Kanade) optical flow algorithm implementation.

Copyright: Copyright Relja Ljubobratovic 2016.

Authors: Relja Ljubobratovic

License: $(LINK3 http://www.boost.org/LICENSE_1_0.txt, Boost Software License - Version 1.0).
*/

module dcv.tracking.opticalflow.lucaskanade;

import std.typecons : Flag, No;
import std.traits : isFloatingPoint;

import mir.ndslice.slice : Slice, Contiguous, Canonical, SliceKind;
import mir.ndslice.topology : map, windows, zip, pack;
import mir.ndslice.allocation : slice, makeSlice;

import dcv.tracking.opticalflow.base;
import dcv.core.types;

enum LucasKanadeError
{
    none,
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
        Size!size_t _win = [41, 41];
        LucasKanadeError _error = LucasKanadeError.none;
    }

    @property
    {
        ///
        float sigma() { return _sigma; }
        ///
        size_t iterationCount() { return _iters; }
        ///
        Size!size_t windowSize() const { return _win; }
        ///
        ref errorMeasure() { return _error; }

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
    void setWindowSize(size_t rows, size_t cols)
    in
    {
        assert(rows >= 3 && cols >= 3, "Minimal size for the window side is 3.");
        assert(rows % 2 != 0 && cols % 2 != 0, "Window size should be odd.");
    }
    body
    {
        _win.height = rows;
        _win.width = cols;
    }

    /// ditto
    void setWindowSize(size_t size)
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
    in
    {
        if (errorMeasure != LucasKanadeError.none)
            assert(error.length == prevPoints.length, "If error measure is selected, error buffer has to be pre-allocated.");
    }
    body
    {
        import std.experimental.allocator.mallocator : Mallocator;
        import std.algorithm.iteration : sum;

        import mir.math.internal : sqrt, exp;
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
        immutable pointCount = prevPoints.length;
        immutable wpxc = cast(float)(_win[0]*_win[1]);

        auto gaussBuf = makeSlice!T(Mallocator.instance, _win[0], _win[1]);
        scope(exit) { Mallocator.instance.deallocate(gaussBuf.array); }

        auto gauss = gaussBuf.slice;
        gauss[] = gaussian!T([_win[1], _win[0]], sigma, sigma);

        // TODO: norm1 from mir?
        auto gnorm = gauss.flattened.sum;
        gauss[] /= gnorm;

        auto fxs = f1.windows(3, 3).map!sobel_x;
        auto fys = f1.windows(3, 3).map!sobel_y;
        auto f1s = f1[1 .. $ -1, 1 .. $ - 1];
        auto f2s = f2[1 .. $ -1, 1 .. $ - 1];

        T a1, a2, a3, b1, b2;

        foreach (ptn; 0..pointCount)
        {
            auto p = prevPoints[ptn];
            auto c = currPoints[ptn];

            immutable pReg = getRegion!T(p, _win, f1.shape);

            if (pReg.empty)
                continue;

            foreach (_; 0..iterationCount)
            {
                immutable nx = c[0] - p[0];
                immutable ny = c[1] - p[1];

                size_t ii = 0, jj;
                a1 = a2 = a3 = b1 = b2 = 0.0f;

                foreach (i; pReg.rows)
                {
                    jj = 0;
                    foreach (j; pReg.cols)
                    {
                        immutable nnx = cast(T)j + nx;
                        immutable nny = cast(T)i + ny;

                        if (nnx < 0 || nnx >= cols - 2 || nny < 0 || nny >= rows - 2)
                            continue;

                        immutable w = gauss[ii, jj];
                        immutable fx = fxs[i, j]; // -1 because of shift by sobel convolution kernel size
                        immutable fy = fys[i, j];
                        immutable ft = linear(f2s, nny, nnx) - f1s[i, j];

                        a1 += w * fx * fx;
                        a2 += w * fx * fy;
                        a3 += w * fy * fy;
                        b1 += w * fx * ft;
                        b2 += w * fy * ft;
                        ++jj;
                    }
                    ++ii;
                }

                a1 /= wpxc;
                a2 /= wpxc;
                a3 /= wpxc;
                b1 /= wpxc;
                b2 /= wpxc;

                auto d = (a1 * a3 - a2 * a2);

                if (d)
                {
                    d = 1.0f / d;
                    c[0] += (a2 * b2 - a3 * b1) * d;
                    c[1] += (a2 * b1 - a1 * b2) * d;
                }
            }

            if (errorMeasure == LucasKanadeError.brightness)
            {
                import mir.ndslice.algorithm : reduce;

                immutable cReg = getRegion!T(currPoints[ptn], _win, f1.shape);

                if (!cReg.empty)
                {
                    auto f1w = f1s.sliced(pReg);
                    auto f2w = f2s.sliced(cReg);

                    if (f1w.shape == f2w.shape)
                        error[ptn] = reduce!(calcBrightnessError!T)(T(0), f1w, f2w) / wpxc;
                }
            }
            else if (errorMeasure == LucasKanadeError.eigenvalue)
            {
                error[ptn] = calcEigenvalueError(a1, a2, a3, b1, b2) / wpxc;
            }
        }
    }
}

nothrow @nogc static
{

    T calcBrightnessError(T)(T e, T p, T c)
    {
        import mir.math.internal : fabs;
        return e + (p - c).fabs;
    }

    T calcEigenvalueError(T)(T a1, T a2, T a3, T b1, T b2)
    {
        import mir.math.internal : sqrt;
        return ((a1 + a3) - sqrt((a1 - a3) * (a1 - a3) + a2 * a2));
    }

    private Rectangle!size_t getRegion(T)(Slice!(Contiguous, [1], const(T)*) xy, Size!size_t win, size_t[2] shape)
    {
        import std.math : round;

        Rectangle!size_t reg;

        immutable x = cast(int)round(xy[0]) - 1;
        immutable y = cast(int)round(xy[1]) - 1;
        immutable w2 = cast(int)win.width / 2;
        immutable h2 = cast(int)win.height / 2;

        int rb = y - h2;
        int re = y + h2 + 1;
        int cb = x - w2;
        int ce = x + w2 + 1;

        rb = rb < 1 ? 1 : rb;
        re = re >= cast(int)shape[0] - 1 ? cast(int)shape[0] - 2 : re;
        cb = cb < 1 ? 1 : cb;
        ce = ce >= cast(int)shape[1] - 1 ? cast(int)shape[1] - 2 : ce;

        auto h = re - rb;
        auto w = ce - cb;

        if (w > 1 && h > 1)
        {
            reg = region(cast(size_t)cb, cast(size_t)rb, cast(size_t)w, cast(size_t)h);
        }

        return reg;
    }
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
