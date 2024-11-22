package com.android.example.rsmigration

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.util.Size
import com.example.simpleegl.OpenGLESBlurRenderPass
import kotlin.math.roundToInt

class OpenGLESImageProcessor(val context: Context) : ImageProcessor {
    override val name = "OpenGLES"
    lateinit var input: Bitmap
    var numberOfOutputImages = 1

    private var blurRenderPass: OpenGLESBlurRenderPass? = null
    private var resizeRenderPass: OpenGLESResizeRenderPass? = null

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
        if (resizeRenderPass == null) {
            resizeRenderPass = OpenGLESResizeRenderPass(context)
        }
        return resizeRenderPass!!.resize(
            input,
            Size(
                (input.width.toFloat() * percent).roundToInt().coerceAtLeast(1),
                (input.height.toFloat() * percent).roundToInt().coerceAtLeast(1)
            ),
            SamplerMethod.LANCZOS,
            Antialiasing.NO_ANTIALIAS,
        )
    }

    override fun cleanup() {
        // Nothing
    }
}