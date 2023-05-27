/*
 * Copyright (C) 2021 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.android.example.rsmigration

import android.content.Context
import android.graphics.Bitmap
import com.android.example.rsmigration.GLES.GPUImage
import com.android.example.rsmigration.filter.GPUImageGaussianBlurFilter
import com.android.example.rsmigration.filter.GPUImageHueFilter
import com.android.example.rsmigration.filter.GPUImageResizeFilter


class GLESImageProcessor(context: Context) : ImageProcessor {
    override val name = "GL ES"

    private var gpuImage = GPUImage(context)
    private lateinit var inputXY: Pair<Int, Int>

    private lateinit var mOutputImages: Array<Bitmap>


    override fun configureInputAndOutput(inputImage: Bitmap, numberOfOutputImages: Int) {
        gpuImage.setImage(inputImage)
        inputXY = Pair(inputImage.width, inputImage.height)
        mOutputImages = Array(numberOfOutputImages) {
            Bitmap.createBitmap(inputImage.width, inputImage.height, inputImage.config)
        }
    }

    override fun rotateHue(radian: Float, outputIndex: Int): Bitmap {
        return gpuImage.getBitmapForMultipleFilters(listOf(GPUImageHueFilter(radian)))
    }

    override fun blur(radius: Float, outputIndex: Int): Bitmap {
        // NB : default radius for this implementation is 5 (see https://github.com/BradLarson/GPUImage/issues/983)
        return gpuImage.getBitmapForMultipleFilters(listOf(GPUImageGaussianBlurFilter(radius / 5f)))
    }

    override fun resize(percent: Float, outputIndex: Int): Bitmap {
        val targetXY = FloatArray(2)
        targetXY[0] = inputXY.first * percent
        targetXY[1] = inputXY.second * percent
        return gpuImage.getBitmapForMultipleFilters(listOf(GPUImageResizeFilter(targetXY)), targetXY[0].toInt(), targetXY[1].toInt())
    }

    override fun cleanup() {
        gpuImage.deleteImage()
    }
}
