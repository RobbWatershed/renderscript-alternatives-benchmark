package com.android.example.rsmigration

import android.content.Context
import android.graphics.Bitmap
import com.awxkee.aire.Aire
import com.awxkee.aire.ResizeFunction
import com.awxkee.aire.ScaleColorSpace
import kotlin.math.roundToInt

class AireImageProcessor(val context: Context) : ImageProcessor {
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
        return Aire.stackBlur(bitmap = input, radius.toInt())
    }

    override fun resize(percent: Float, outputIndex: Int): Bitmap {
        return Aire.scale(
            input,
            (input.width * percent).roundToInt(),
            (input.height * percent).roundToInt(),
            ResizeFunction.Lanczos3,
            ScaleColorSpace.SRGB
        )
    }

    override fun cleanup() {
        // Nothing
    }
}