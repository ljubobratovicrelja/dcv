/**
Module introduces the API that defines Optical Flow utilities in the dcv library.

Copyright: Copyright Relja Ljubobratovic 2016.

Authors: Relja Ljubobratovic

License: $(LINK3 http://www.boost.org/LICENSE_1_0.txt, Boost Software License - Version 1.0).
*/
module dcv.tracking.opticalflow.base;

import std.typecons : Flag, No;

import mir.ndslice.slice : Slice, SliceKind;

import dcv.core.image : Image;
import dcv.core.utils : emptySlice;

/**
Sparse Optical Flow algorithm interface.
*/
mixin template SparseOpticalFlow(PixelType, CoordType)
{
    import std.traits : isFloatingPoint;

    static assert(isFloatingPoint!CoordType, "Coordinate type has to be a floating point type.");

    /**
    Evaluate sparse optical flow method between two consecutive frames.

    Params:
        prevFrame       = First (previous) frame image.
        currFrame       = Second (current) frame image.
        prevPoints      = Input points, to be tracked to the current frame.
        currPoints      = Output point, tracking result from the input points of the previous to the current frame.
        error           = Error values for each point tracked.
        up              = Flag telling whether to use or not to use values present in the output point buffer as initial result.
                          If flag has negative value, input point coordinates are considered as initial for the optical flow estimate.
                          If flag is positive, buffer is cosidered to be of same size as the input point buffer.
    */
    void evaluate
    (
        Slice!(Contiguous, [2], const(PixelType)*) prevFrame,
        Slice!(Contiguous, [2], const(PixelType)*) currFrame,
        Slice!(Contiguous, [2], const(CoordType)*) prevPoints,
        ref Slice!(Contiguous, [2], CoordType*) currPoints,
        Slice!(Contiguous, [1], float*) error = Slice!(Contiguous, [1], float*).init,
        Flag!"usePrevious" up = No.usePrevious
    )
    in
    {
        assert(!prevFrame.empty, "Given image slices should not be empty.");
        assert(prevFrame.shape == currFrame.shape, "Given images should be of same size.");
        assert(!prevPoints.empty, "Given source (previous) point slice should not be empty.");
        assert(prevPoints.length!1 == 2, "Point slice malformed - should be n-by-2, for n points (x,y).");
        if (up)
            assert(currPoints.shape == prevPoints.shape, "Invalid destination frame shape - should be same as previous.");
        if(!error.empty)
            assert(error.length == prevPoints.length, "If errors are to be calculated, error slice has to be of same size as points.");
    }
    body
    {
        if (!up)
        {
            currPoints = prevPoints.slice;
        }

        evaluateImpl(prevFrame, currFrame, prevPoints, currPoints, error);
    }
}

/// Alias to a type used to define the dense optical flow field.
alias DenseFlow = Slice!(SliceKind.contiguous, [3], float*);

/**
Dense Optical Flow algorithm interface.
*/
interface DenseOpticalFlow
{
    /**
    Evaluate dense optical flow method between two consecutive frames.

    Params:
        f1          = First image, i.e. previous frame in the video.
        f2          = Second image of same size and type as $(D f1), i.e. current frame in the video.
        prealloc    = Optional pre-allocated flow buffer. If provided, has to be of same size as input images are, and with 2 channels (u, v).
        usePrevious = Should the previous flow be used. If true $(D prealloc) is treated as previous flow, and has to satisfy size requirements.

    Returns:
        Calculated flow field.
    */
    DenseFlow evaluate(inout Image f1, inout Image f2, DenseFlow prealloc = emptySlice!([3], float),
            bool usePrevious = false);
}
