package com.android.example.rsmigration

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.awxkee.aire.Aire
import com.awxkee.aire.ResizeFunction
import com.awxkee.aire.ScaleColorSpace
import com.example.simpleegl.OpenGLESBlurRenderPass
import kotlin.math.roundToInt

class OpenGLESImageProcessor(val context: Context) : ImageProcessor {
    override val name = "OpenGLES"
    lateinit var input: Bitmap
    var numberOfOutputImages = 1

    private var blurRenderPass: OpenGLESBlurRenderPass? = null

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
        if (blurRenderPass == null) {
            blurRenderPass = OpenGLESBlurRenderPass(context)
        }
        return blurRenderPass!!.render(input, radius.toInt())
    }

    override fun resize(percent: Float, outputIndex: Int): Bitmap {
        return Aire.scale(
            input,
            (input.width * percent).roundToInt(),
            (input.height * percent).roundToInt(),
            ResizeFunction.Spline36,
            ScaleColorSpace.SRGB
        )
    }

    override fun cleanup() {
        // Nothing
    }
}