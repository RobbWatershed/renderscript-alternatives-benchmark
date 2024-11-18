package com.android.example.rsmigration

import android.graphics.Bitmap
import com.awxkee.aire.Aire
import com.awxkee.aire.EdgeMode
import com.awxkee.aire.ResizeFunction
import com.awxkee.aire.ScaleColorSpace
import com.awxkee.aire.TransferFunction
import kotlin.math.roundToInt

class AireImageProcessor() : ImageProcessor {
    override val name = "Aire"
    lateinit var input: Bitmap
    var numberOfOutputImages = 1

    override fun configureInputAndOutput(
        inputImage: Bitmap,
        numberOfOutputImages: Int
    ) {
        this.input = inputImage
        this.numberOfOutputImages = numberOfOutputImages
    }

    override fun rotateHue(radian: Float, outputIndex: Int): Bitmap {
        return input
    }

    override fun blur(radius: Float, outputIndex: Int): Bitmap {
        //return Aire.gaussianBoxBlur(input, radius.toInt())
        //return Aire.linearGaussianBoxBlur(input, radius.toInt(), TransferFunction.SRGB)
        return Aire.fastGaussian2Degree(input, radius.roundToInt(), EdgeMode.CLAMP)
    }

    override fun resize(percent: Float, outputIndex: Int): Bitmap {
        return Aire.scale(
            input,
            (input.width * percent).roundToInt(),
            (input.height * percent).roundToInt(),
            ResizeFunction.Lanczos3,
            ScaleColorSpace.LINEAR
        )
    }

    override fun cleanup() {
        // Nothing
    }
}