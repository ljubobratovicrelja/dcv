/**
Module contains $(LINK3 https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method, Lucas-Kanade) optical flow algorithm implementation.

Copyright: Copyright Relja Ljubobratovic 2016.

Authors: Relja Ljubobratovic

License: $(LINK3 http://www.boost.org/LICENSE_1_0.txt, Boost Software License - Version 1.0).
*/

module dcv.tracking.opticalflow.lucaskanade;

import std.typecons : Flag, No;
import std.math : PI, floor;

import mir.ndslice.slice : Slice, Contiguous, Canonical;
import mir.ndslice.topology : map, windows;
import mir.ndslice.allocation : slice;

import dcv.tracking.opticalflow.base;

/**
Lucas-Kanade optical flow method implementation.
*/
struct LucasKanadeFlow(PixelType, CoordType)
{
    mixin SparseOpticalFlow!(PixelType, CoordType);

    float sigma = 0.84f;
    float[] cornerResponse;
    size_t iterationCount = 10;
    size_t[2] windowSize = [41, 41];

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
        import dcv.imgproc.interpolate : linear;

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

        T gaussMul = 1.0f / (2.0f * PI * sigma);
        T gaussDel = 2.0f * (sigma ^^ 2);

        auto fxs = f1.windows(3, 3).map!sobel_x;
        auto fys = f1.windows(3, 3).map!sobel_y;

        foreach (ptn; 0 .. pointCount)
        {
            import std.math : sqrt, exp;

            auto p = prevPoints[ptn];

            auto rb = cast(int)(p[1] - windowSize[0] / 2.0f);
            auto re = cast(int)(p[1] + windowSize[0] / 2.0f);
            auto cb = cast(int)(p[0] - windowSize[1] / 2.0f);
            auto ce = cast(int)(p[0] + windowSize[1] / 2.0f);

            rb = rb < 1 ? 1 : rb;
            re = re > rl ? rl : re;
            cb = cb < 1 ? 1 : cb;
            ce = ce > cl ? cl : ce;

            if (re - rb <= 0 || ce - cb <= 0)
            {
                continue;
            }

            T a1, a2, a3;
            T b1, b2;

            a1 = 0.0f;
            a2 = 0.0f;
            a3 = 0.0f;
            b1 = 0.0f;
            b2 = 0.0f;

            const auto rm = floor(cast(T)re - (windowSize[0] / 2.0f));
            const auto cm = floor(cast(T)ce - (windowSize[1] / 2.0f));

            foreach (iteration; 0 .. iterationCount)
            {
                immutable nx = currPoints[ptn, 0] - p[0];
                immutable ny = currPoints[ptn, 1] - p[1];

                foreach (i; rb .. re)
                    foreach (j; cb .. ce)
                    {
                        // TODO: gaussian weighting produces errors - examine
                        auto dx = cm - cast(int)j;
                        auto dy = rm - cast(int)i;

                        auto w = exp(-( (dx*dx + dy*dy) / gaussDel ));

                        immutable nnx = cast(T)j + nx;
                        immutable nny = cast(T)i + ny;

                        if (nnx < 0 || nnx >= cols || nny < 0 || nny >= rows)
                            continue;

                        // TODO: consider subpixel precision for gradient sampling.
                        immutable fx = fxs[i-1, j-1];
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
                    }

                // TODO: consider resp normalization...
                //cornerResponse[ptn] = ((a1 + a3) - sqrt((a1 - a3) * (a1 - a3) + a2 * a2));

                auto d = (a1 * a3 - a2 * a2);

                if (d)
                {
                    d = 1.0f / d;
                    currPoints[ptn, 0] += (a2 * b2 - a3 * b1) * d;
                    currPoints[ptn, 1] += (a2 * b1 - a1 * b2) * d;
                }
            }
        }
    }

    static nothrow @nogc
    {
        auto sobel_x(Slice!(Canonical, [2], const(CoordType)*) window)
        {
            return     (window[0, 2] - window[0, 0]) +
                   2 * (window[1, 2] - window[1, 0]) +
                       (window[2, 2] - window[2, 0]);
        }

        auto sobel_y(Slice!(Canonical, [2], const(CoordType)*) window)
        {
            return     (window[2, 0] - window[0, 0]) +
                   2 * (window[2, 1] - window[0, 1]) +
                       (window[2, 2] - window[0, 2]);
        }
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
