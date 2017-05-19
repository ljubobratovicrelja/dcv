/**
Module introduces $(LINK3 https://en.wikipedia.org/wiki/Kernel_(image_processing)#Convolution, image convolution) function.

Following example loads famous image of Lena Söderberg and performs gaussian blurring by convolving the image with gaussian kernel.

----
import dcv.io.image : imread, ReadParams;
import dcv.core.image : Image, asType;
import dcv.imgproc.convolution : conv;

Image lenaImage = imread("../data/lena.png", ReadParams(ImageFormat.IF_MONO, BitDepth.BD_8));
auto slice = lenaImage.sliced!ubyte;
----

... this loads the following image:<br>
$(IMAGE https://github.com/libmir/dcv/blob/master/examples/data/lena.png?raw=true)

----
blurred = slice
             .asType!float // convert ubyte data to float.
             .conv(gaussian!float(0.84f, 5, 5)); // convolve image with gaussian kernel

----

... which give the resulting image:<br>
$(IMAGE https://github.com/libmir/dcv/blob/master/examples/filter/result/outblur.png?raw=true)


Copyright: Copyright Relja Ljubobratovic 2016.

Authors: Relja Ljubobratovic

License: $(LINK3 http://www.boost.org/LICENSE_1_0.txt, Boost Software License - Version 1.0).
*/
module dcv.imgproc.convolution;

import std.range : ElementType;
import std.traits : isAssignable, ReturnType, Unqual;
import std.conv : to;
import std.parallelism : parallel, taskPool, TaskPool;

import mir.ndslice.internal : fastmath;

import mir.ndslice.slice;
import mir.ndslice.topology;
import mir.ndslice.algorithm : reduce;
import mir.ndslice.allocation: slice;

import dcv.core.memory;
import dcv.core.utils;

/**
Perform convolution to given tensor, using given kernel.
Convolution is supported for 1, 2, and 3 dimensional tensors.

Params:
    bc          = (Template parameter) Boundary Condition function used while indexing the image matrix.
    input       = Input tensor.
    output      = Pre-allocated buffer where convolution result will be stored. Must be the same size as the input tensor.
    kernel      = Convolution kernel tensor. For 1D input, 1D kernel is expected. For 2D input, 2D kernel is expected.
                  For 3D input, 2D or 3D kernel is expected - if 2D kernel is given, each item in kernel matrix is applied
                  to each value in corresponding 2D coordinate in the input.
    mask        = Masking input. Convolution will skip each element where mask is 0. Default value is empty slice, which
                  tells that convolution will be performed on the whole input.

Returns:
    Resulting image after convolution, of same type as input tensor.

Note:
    Input, mask and pre-allocated slices' strides must be the same.
*/
nothrow @nogc
void conv(OutputType = float, alias bc = neumann, SliceKind k1, SliceKind k2, SliceKind k3 = Contiguous, size_t[] packs,
    size_t[] kPacks, InputIterator, KernelIterator, MaskType = ubyte)
(
    Slice!(k1, packs, InputIterator) input,
    Slice!(k1, packs, OutputType*) output,
    Slice!(k2, kPacks, KernelIterator) kernel,
    Slice!(k3, kPacks, MaskType*) mask = Slice!(k3, kPacks, MaskType*).init
)
in
{
    static assert(isBoundaryCondition!bc, "Invalid boundary condition test function.");

    static assert(packs.length == 1, "Given slices must not be packed.");
    static assert(kPacks.length == 1, "Given mask and kernel slices must not be packed.");

    immutable N = packs[0];
    immutable NK = kPacks[0];

    immutable invalidKernelMsg = "Invalid kernel dimension";
    static if (N == 1)
        static assert(NK == 1, invalidKernelMsg);
    else static if (N == 2)
        static assert(NK == 2, invalidKernelMsg);
    else static if (N == 3)
        static assert(NK == 2, invalidKernelMsg);
    else
        static assert(0, "Convolution not implemented for given tensor dimension.");

    assert(input._iterator != output._iterator, "Output and input buffer cannot point to the same memory.");
    assert(output.shape == input.shape, "Output buffer size should be the same as the input");

    if (!mask.empty)
    {
        assert(mask.shape == input.shape, "Invalid mask size. Should be of same size as input tensor.");
        assert(input.strides == mask.strides, "Input input and mask need to have same strides.");
    }

    static if (k1 != Contiguous)
    {
        assert(input._strides == output._strides, "Input and output strides must be the same.");
    }
}
body
{
    convImpl!(OutputType, bc)(input, output, kernel, mask);
}

/**
Pure, lazy variant of convolution.

Constructs lazy slice, containing evaluation of convolution using the given kernel at each tensor coefficient.

Note:
    Border handing is ignored in this variant of convolution, as result giving smaller sized tensor, by the size of the
    kernel.

Params:
    input   = Input tensor.
    kernel  = Kernel used for convolution.

Result:
    Slice object containing lazy evaluated convolution.
*/
pure nothrow @nogc
auto conv(OutputType = float, SliceKind kind, size_t[] packs, InputIterator, SliceKind kKind, size_t[] kPacks, KernelIterator)
(
    Slice!(kind, packs, InputIterator) input,
    Slice!(kKind, kPacks, KernelIterator) kernel,
)
in
{
    assert(!input.empty, "Input image must not be empty.");
    assert(kernel.length!0 > 1 && kernel.length!1 > 1, "Given kernel side size must be at least 2 pixels.");
}
body
{
    import mir.ndslice.iterator : FieldIterator;
    import mir.ndslice.field : ndIotaField;
    import mir.ndslice.algorithm : reduce;

    static assert(packs.length == 1 && kPacks.length == 1, "Packed slices are not supported.");

    enum N = packs[0];
    enum M = kPacks[0];

    alias Input = typeof(input);
    alias Kernel = typeof(kernel);

    immutable r = kernel.length!0 / 2;
    immutable c = kernel.length!1 / 2;

    static struct ConvOp
    {
        Input _input;
        Kernel _kernel;
        ndIotaField!N _field;

        auto opIndex(size_t index)
        {
            immutable r = _kernel.length!0 / 2;
            immutable c = _kernel.length!1 / 2;
            auto i = _field[index];
            static if (N == 2)
                auto w = _input[i[0]-r..i[0]+r+1, i[1]-c..i[1]+c+1];
            else
                auto w = _input[i[0]-r..i[0]+r+1, i[1]-c..i[1]+c+1, i[2]];

            return reduce!kapply(OutputType(0), w, _kernel);
        }
    }

    auto convOp = ConvOp(input, kernel, ndIotaField!N(input.shape[1..$]));
    return FieldIterator!ConvOp(0, convOp).sliced(input.shape)[r..$-r, c..$-c];
}

unittest
{
    import std.math : approxEqual;
    import std.algorithm.comparison : equal;

    auto r1 = [0., 1., 2., 3., 4., 5.].sliced(6);
    auto k1 = [-1., 0., 1.].sliced(3);
    auto res1 = r1.conv(k1);
    assert(res1.equal!approxEqual([1., 2., 2., 2., 2., 1.]));
}

unittest
{
    auto image = slice!float(15, 15);
    auto kernel = slice!float(3, 3);
    auto convres = conv(image, kernel);
    assert(convres.shape == image.shape);
}

unittest
{
    auto image = slice!float(15, 15, 3);
    auto kernel = slice!float(3, 3);
    auto convres = conv(image, kernel);
    assert(convres.shape == image.shape);
}

nothrow @nogc @fastmath auto kapply(T)(const T r, const T i, const T k)
{
    return r + i * k;
}

private:

@nogc nothrow
auto convImpl(OutputType, alias bc,
SliceKind k1, SliceKind k2, SliceKind k3,
size_t[] packs, size_t[] kPacks, InputIterator, KernelIterator, MaskType)
(
    Slice!(k1, packs, InputIterator) input,
    Slice!(k1, packs, OutputType*) output,
    Slice!(k2, kPacks, KernelIterator) kernel,
    Slice!(k3, kPacks, MaskType*) mask
)
if (packs[0] == 1)
{
    alias InputType = ElementType!InputIterator;

    auto kl = kernel.length;
    auto kh = kl / 2;

    if (mask.empty)
    {
        auto packedWindows = zip!true(output, input).windows(kl);
        foreach (p; packedWindows)
        {
            p[kh].a = reduce!(kapply!InputType)(0.0f, p.unzip!'b', kernel);
        }
    }
    else
    {
        // TODO: extract masked convolution as separate function?
        auto packedWindows = zip!true(output, input, mask).windows(kl);
        foreach (p; packedWindows)
        {
            if (p[$ / 2].c)
                p[$ / 2].a = reduce!(kapply!InputType)(0.0f, p.unzip!'b', kernel);
        }
    }

    handleEdgeConv1d!bc(input, output, kernel, mask, 0, kl);
    handleEdgeConv1d!bc(input, output, kernel, mask, input.length - 1 - kh, input.length);

    return output;
}

/*
@nogc nothrow
auto convImpl(OutputType, alias bc, SliceKind k1, SliceKind k2, SliceKind k3, size_t[] packs,
    size_t[] kPacks, InputIterator, KernelIterator, MaskType)
(
    Slice!(k1, packs, InputIterator) input,
    Slice!(k1, packs, OutputType*) output,
    Slice!(k2, kPacks, KernelIterator) kernel,
    Slice!(k3, kPacks, MaskType*) mask
)
if (packs[0] == 2)
{
    auto krs = kernel.length!0; // kernel rows
    auto kcs = kernel.length!1; // kernel rows

    auto krh = krs / 2;
    auto kch = kcs / 2;

    auto useMask = !mask.empty;

    if (mask.empty)
    {
        auto packedWindows = zip!true(output, input).windows(krs, kcs);
        foreach (prow; packedWindows)
            foreach (p; prow)
            {
                p[krh, kch].a = reduce!kapply(0.0f, p.unzip!'b', kernel);
            }
    }

    foreach (r; iota(inShape[0]))
    {
        auto maskBuf = threadMask.get();
        foreach (c; 0 .. inShape[1])
        {
            innerBody[r, c] = bilateralFilterImpl(inputWindows[r, c], maskBuf, sigmaCol, sigmaSpace);
        }
    }

    foreach (border; input.shape.borders(ks)[])
    {
        auto maskBuf = threadMask.get();
        foreach (r; border.rows)
            foreach (c; border.cols)
            {
                import mir.ndslice.field;
                import mir.ndslice.iterator;

                static struct ndIotaWithShiftField
                {
                    ptrdiff_t[2] _shift;
                    ndIotaField!2 _field;
                    Slice!(kind, [2], Iterator) _input;
                    auto opIndex(ptrdiff_t index)
                    {
                        auto ret = _field[index];
                        ptrdiff_t r = _shift[0] - cast(ptrdiff_t)ret[0];
                        ptrdiff_t c = _shift[1] - cast(ptrdiff_t)ret[1];
                        return bc(_input, r, c);
                    }
                }

                auto inputWindow = FieldIterator!ndIotaWithShiftField(0, ndIotaWithShiftField([r + kh, c + kh], ndIotaField!2(ks), input)).sliced(ks, ks);
                prealloc[r, c] = bilateralFilterImpl(inputWindow, maskBuf, sigmaCol, sigmaSpace);
            }
    }

    return output;
}
*/

@nogc nothrow
auto convImpl(OutputType, alias bc, SliceKind k1, SliceKind k2, SliceKind k3, size_t[] packs,
    size_t[] kPacks, InputIterator, KernelIterator, MaskType)
(
    Slice!(k1, packs, InputIterator) input,
    Slice!(k1, packs, OutputType*) output,
    Slice!(k2, kPacks, KernelIterator) kernel,
    Slice!(k3, kPacks, MaskType*) mask
)
if (packs[0] == 2)
{
    auto krs = kernel.length!0; // kernel rows
    auto kcs = kernel.length!1; // kernel rows

    auto krh = krs / 2;
    auto kch = kcs / 2;

    auto useMask = !mask.empty;

    if (mask.empty)
    {
        auto packedWindows = zip!true(output, input).windows(krs, kcs);
        foreach (prow; packedWindows)
            foreach (p; prow)
            {
                p[krh, kch].a = reduce!kapply(0.0f, p.unzip!'b', kernel);
            }
    }
    else
    {
        auto packedWindows = zip!true(output, input, mask).windows(krs, kcs);
        foreach (prow; packedWindows)
            foreach (p; prow)
                if (p[krh, kch].c)
                    p[krh, kch].a = reduce!kapply(0.0f, p.unzip!'b', kernel);
    }

    handleEdgeConv2d!bc(input, output, kernel, mask, [0, input.length!0], [0, kch]); // upper row
    handleEdgeConv2d!bc(input, output, kernel, mask, [0, input.length!0], [input.length!1 - kch, input.length!1]); // lower row
    handleEdgeConv2d!bc(input, output, kernel, mask, [0, krh], [0, input.length!1]); // left column
    handleEdgeConv2d!bc(input, output, kernel, mask, [input.length!0 - krh, input.length!0], [0, input.length!1]); // right column

    return output;
}

nothrow @nogc
auto convImpl(OutputType, alias bc,
SliceKind k1, SliceKind k2, SliceKind k3,
size_t[] packs, size_t[] kPacks, InputIterator, KernelIterator, MaskType)
(
    Slice!(k1, packs, InputIterator) input,
    Slice!(k1, packs, OutputType*) output,
    Slice!(k2, kPacks, KernelIterator) kernel,
    Slice!(k3, kPacks, MaskType*) mask,
)
if (packs[0] == 3)
{
    foreach (i; 0 .. input.length!2)
    {
        auto i_c = input[0 .. $, 0 .. $, i];
        auto o_c = output[0 .. $, 0 .. $, i];
        convImpl!(OutputType, bc)(i_c, o_c, kernel, mask);
    }

    return output;
}

nothrow @nogc
void handleEdgeConv1d(alias bc, T, O, K, M,
    SliceKind kindi,
    SliceKind kindp,
    SliceKind kindk,
    SliceKind kindm,
    )(
    Slice!(kindi, [1], T*) input,
    Slice!(kindp, [1], O*) output,
    Slice!(kindk, [1], K*) kernel,
    Slice!(kindm, [1], M*) mask,
    size_t from, size_t to)
in
{
    assert(from < to);
}
body
{
    int kl = cast(int)kernel.length;
    int kh = kl / 2, i = cast(int)from, j;

    bool useMask = !mask.empty;

    Unqual!T t;
    foreach (ref p; output[from .. to])
    {
        if (useMask && mask[i] <= 0)
            goto loop_end;
        t = 0;
        j = -kh;
        foreach (k; kernel)
        {
            t += bc(input, i + j) * k;
            ++j;
        }
        p = t;
    loop_end:
        ++i;
    }
}

nothrow @nogc
void handleEdgeConv2d(alias bc, SliceKind kind0, SliceKind kind1, SliceKind kind2, SliceKind kind3, T, O, K, M)(
    Slice!(kind0, [2], T*) input,
    Slice!(kind1, [2], O*) output,
    Slice!(kind2, [2], K*) kernel,
    Slice!(kind3, [2], M*) mask,
    size_t[2] rowRange, size_t[2] colRange)
in
{
    assert(rowRange[0] < rowRange[1]);
    assert(colRange[0] < colRange[1]);
}
body
{
    int krl = cast(int)kernel.length!0;
    int kcl = cast(int)kernel.length!1;
    int krh = krl / 2, kch = kcl / 2;
    int r = cast(int)rowRange[0], c, i, j;

    bool useMask = !mask.empty;

    auto roi = output[rowRange[0] .. rowRange[1], colRange[0] .. colRange[1]];

    Unqual!T t;
    foreach (prow; roi)
    {
        c = cast(int)colRange[0];
        foreach (ref p; prow)
        {
            if (useMask && mask[r, c] <= 0)
                goto loop_end;
            t = 0;
            i = -krh;
            foreach (krow; kernel)
            {
                j = -kch;
                foreach (k; krow)
                {
                    t += bc(input, r + i, c + j) * k;
                    ++j;
                }
                ++i;
            }
            p = t;
        loop_end:
            ++c;
        }
        ++r;
    }
}
