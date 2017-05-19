/**
Module implements utilities for constructing image pyramids, to aid algorithms which rely on pyramidal (coarse-to-fine) processing.

Copyright: Copyright Relja Ljubobratovic 2017.

Authors: Relja Ljubobratovic

License: $(LINK3 http://www.boost.org/LICENSE_1_0.txt, Boost Software License - Version 1.0).
*/
module dcv.imgproc.pyramid;

import std.experimental.allocator.mallocator : Mallocator;
import std.traits : isFloatingPoint;
import mir.ndslice;

import dcv.core.types;


/**
Create gaussian pyramid of an image.

Example:
----
auto pyramid = imread("some/image.jpg")
    .sliced
    .as!float
    .gaussianPyramid(3, 0.5, [3, 3], 3f, 3f);

foreach(l; pyramid)
{
    l.imshow;
    waitKey();
}
----

Params:
    image = Source image.
    levels = Number of pyramid levels.
    scale = Downscaling factor between pyramid levels. Should be between 0 and 1.
    size = Size of the gaussian kernel. [rows, colums]
    sigmax = Gaussian function variance in x axis.
    sigmay = Gaussian function variance in y axis.

Returns:
    Gaussian pyramid levels contained in a range, starting with lowest level, going towards the original (input) image.
*/
nothrow @nogc
auto gaussianPyramid(SliceKind kind, size_t[] packs, Iterator)
(
    Slice!(kind, packs, Iterator) image,
    size_t levels,
    float scale,
    size_t[2] size = [3, 3],
    float sigmax = 1f,
    float sigmay = 1f
)
in
{
    assert(!image.empty, "Input image must not be empty.");
    assert(scale > 0 && scale < 1, "Scale must be between 0 and 1.");
    assert(size[0] > 2 && size[1] > 2, "Gaussian kernel size must be at least 3x3.");
    assert(sigmax > 0f, "Sigma values must be larger than 0.");
    assert(sigmay > 0f, "Sigma values must be larger than 0.");
}
body
{
    import std.typecons : Unqual;

    alias T = Unqual!(DeepElementType!(typeof(image)));

    auto scaler = GaussianScaler!(T, packs)(size[0], size[1], sigmax, sigmay);
    return Pyramid!(typeof(scaler), kind, packs, Iterator)(image, scaler, levels, scale);
}

/// Gaussian pyramid built with 3 levels resized with 0.5 scaling factor, and 3x3 gaussian smoothing.
nothrow @nogc unittest
{
    int[128*128] buffer;
    auto pyramid = buffer.sliced(128, 128).gaussianPyramid(3, 0.5, [3, 3], 1f, 1f);

    assert(pyramid.front.shape == [32, 32]); pyramid.popFront;
    assert(pyramid.front.shape == [64, 64]); pyramid.popFront;
    assert(pyramid.front.shape == [128, 128]); pyramid.popFront;
    assert(pyramid.empty);
}


/**
Image pyramid range construction.

This structure is used as base range for pyramidal image scaling. It takes custom scaling functor type as template
argument, and parameters of the Slice structure used to define the input image. Scaling functor (Scaler) must
implement following method:
----
void evaluate(Slice!(Contiguous, packs, T*) lower, Slice!(Contiguous, packs, T*) upper);
----
... Where packs correspond to the same parameter value from Pyramid structure, and T corresponds to
the element type of the pyramid's Iterator. Lower and upper slices are pre-allocated by the Pyramid's internals,
and Scaler's function is to perform the downscaling from lower's values and store them in upper buffer.
In it's implementation, structure does not use the GC. If scaler functor's evaluate method is defined as 'nothorw @nogc',
whole pyramid can be considered as such.

Params:
    Scaler      = Scaling functor type used to define pyramid upper levels from the original, input image.
    kind        = Kind of the input image slice.
    packs       = Packs fo the input image slice.
    Iterator    = Iterator type of the input image slice.

See also gaussianPyramid.
*/
struct Pyramid(Scaler, SliceKind kind, size_t[] packs, Iterator)
{
    import std.typecons : RefCounted;
    import std.math : round;

    alias T = typeof(Iterator.init[size_t.init]);
    alias Level = Slice!(Contiguous, packs, T*);
    enum N = packs[0];

    static assert(packs.length == 1, "Packed slices are not allowed.");
    static assert(N == 2 || N == 3, "Invalid dimensionality - only 2 and 3 dimensional images allowed.");

    private
    {
        RefCounted!(T[]) _buf = void; // memory buffer for all of the pyramid.
        T* _ptr; // pointer to the beginning of the current pyramid level.
        float _rows, _cols; // size of the current pyramid level.
        float _scale; // scaling factor used to construct the pyramid.
        size_t _channels; // channel count of the pyramid input image.
        ptrdiff_t _currentLevel; // current level of the pyramid.
    }

    @disable this();

    /**
    Constructor of the pyramid.

    Params:
        image = Input image.
        scaler = Instance of the pyramid scaler functor.
        levels = Number of levels this pyramid should be constructed of. Must be greater than 0.
        scale = Scaling factor used to scale each next upper level in pyramid. Must be between 0.0 and 1.0.
    */
    this(Slice!(kind, packs, Iterator) image, Scaler scaler, size_t levels, float scale = 0.5f)
    in
    {
        assert(levels > 0, "Level count must be larger than 0.");
        assert(!image.empty, "Invalid image given - should not be empty.");
        assert(scale > 0f && scale < 1f, "Invalid pyramid scaling - should be 0 < scale < 1.");
    }
    body
    {
        static assert(is(DeepElementType!(typeof(image)) == T),
            "Invalid slice iterator value type - must be the same with pyramid value type.");

        _scale = scale;
        _currentLevel = levels;

        float rowCount = cast(float)image.length!0;
        float colCount = cast(float)image.length!1;
        size_t sumPixelCount = image.length!0 * image.length!1;

        _channels = 1;

        static if (N == 3)
            _channels = image.length!2;

        foreach(l; 1 .. levels)
        {
            rowCount *= scale;
            colCount *= scale;
            sumPixelCount += cast(size_t)round(rowCount) * cast(size_t)round(colCount);
        }

        _buf = cast(T[])Mallocator.instance.allocate(T.sizeof * sumPixelCount * _channels);

        auto tPtr = _buf.ptr;
        auto tSlice = tPtr.sliced(image.shape);
        tSlice[] = image[];

        rowCount = cast(float)image.length!0;
        colCount = cast(float)image.length!1;

        foreach(l; 1 .. levels)
        {
            tPtr += cast(size_t)round(rowCount) * cast(size_t)round(colCount) * _channels;
            rowCount *= scale;
            colCount *= scale;

            static if (N == 2)
                auto cSlice = tPtr.sliced(cast(size_t)round(rowCount), cast(size_t)round(colCount));
            else
                auto cSlice = tPtr.sliced(cast(size_t)round(rowCount), cast(size_t)round(colCount), _channels);

            scaler.evaluate(tSlice, cSlice);

            tSlice = cSlice;
        }

        _rows = rowCount;
        _cols = colCount;
        _ptr = tPtr;
    }

    ~this()
    {
        if (_buf.refCountedStore.refCount == 1)
            Mallocator.instance.deallocate(_buf);
    }

    /// Check if each level of the pyramid has been visited.
    bool empty() const
    {
        return _currentLevel <= 0;
    }

    /// Get current pyramid level image.
    Slice!(Contiguous, packs, T*) front()
    {
        static if (N == 2)
            return _ptr.sliced(cast(size_t)round(_rows), cast(size_t)round(_cols));
        else
            return _ptr.sliced(cast(size_t)round(_rows), cast(size_t)round(_cols), _channels);
    }

    /// Go to the next level of the pyramid.
    void popFront()
    in
    {
        assert(_currentLevel > 0);
    }
    body
    {
        _rows /= _scale;
        _cols /= _scale;
        _ptr -= cast(size_t)round(_rows) * cast(size_t)round(_cols) * _channels;
        --_currentLevel;
    }
}

package static struct GaussianScaler(T, size_t[] packs) if (isFloatingPoint!T)
{
@nogc nothrow:
    size_t width, height;
    T sigmax, sigmay;

    void evaluate(Slice!(Contiguous, packs, T*) lower, Slice!(Contiguous, packs, T*) upper)
    {
        import std.algorithm.iteration : sum;
        import std.experimental.allocator.building_blocks.scoped_allocator;

        import dcv.imgproc.imgmanip : resize;
        import dcv.imgproc.convolution : conv;
        import dcv.imgproc.filter : gaussian;

        ScopedAllocator!Mallocator alloc;

        auto gBuf = makeSlice!T(alloc, [height, width]);
        auto blurBuf = makeSlice!T(alloc, lower.shape);

        auto g = gBuf.slice;
        auto blur = blurBuf.slice;

        g[] = gaussian!T([height, width], sigmax, sigmay);
        g[] /= g.flattened.sum;

        conv(lower, blur, g);
        resize(blur, upper);
    }
}
