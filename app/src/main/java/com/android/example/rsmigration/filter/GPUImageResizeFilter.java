/*
 * Copyright (C) 2018 CyberAgent, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.android.example.rsmigration.filter;

import android.opengl.GLES20;

public class GPUImageResizeFilter extends GPUImageFilter {

    public static final String RESIZE_VERTEX_SHADER = "void main()\n" +
            "        {\n" +
            "            gl_TexCoord[0] = gl_MultiTexCoord0;         //center\n" +
            "            gl_Position = ftransform();\n" +
            "        }\n";

    public static final String RESIZE_FRAGMENT_SHADER = "#define FIX(c) max(abs(c), 1e-5);\n" +
            "\n" +
            "        uniform sampler2D inputImageTexture;\n" +
            "        varying highp vec2 textureCoordinate;\n" +
            "        uniform mediump vec2 textureTargetSize;\n" +
            "\n" +
            "        const float PI = 3.1415926535897932384626433832795;\n" +
            "\n" +
            "        mediump vec3 weight3(mediump float x)\n" +
            "        {\n" +
            "            const mediump float radius = 3.0;\n" +
            "            mediump vec3 sample = FIX(2.0 * PI * vec3(x - 1.5, x - 0.5, x + 0.5));\n" +
            "\n" +
            "            // Lanczos3. Note: we normalize outside this function, so no point in multiplying by radius.\n" +
            "            return /*radius **/ sin(sample) * sin(sample / radius) / (sample * sample);\n" +
            "        }\n" +
            "\n" +
            "        mediump vec3 pixel(mediump float xpos, mediump float ypos)\n" +
            "        {\n" +
            "            return texture2D(inputImageTexture, vec2(xpos, ypos)).rgb;\n" +
            "        }\n" +
            "\n" +
            "        mediump vec3 line(mediump float ypos, mediump vec3 xpos1, mediump vec3 xpos2, mediump vec3 linetaps1, mediump vec3 linetaps2)\n" +
            "        {\n" +
            "            return\n" +
            "                pixel(xpos1.r, ypos) * linetaps1.r +\n" +
            "                pixel(xpos1.g, ypos) * linetaps2.r +\n" +
            "                pixel(xpos1.b, ypos) * linetaps1.g +\n" +
            "                pixel(xpos2.r, ypos) * linetaps2.g +\n" +
            "                pixel(xpos2.g, ypos) * linetaps1.b +\n" +
            "                pixel(xpos2.b, ypos) * linetaps2.b;\n" +
            "        }\n" +
            "\n" +
            "        void main()\n" +
            "        {\n" +
            "            mediump vec2 stepxy = 1.0 / textureTargetSize.xy;\n" +
            "            mediump vec2 pos = textureCoordinate.xy + stepxy * 0.5;\n" +
            "            mediump vec2 f = fract(pos / stepxy);\n" +
            "\n" +
            "            mediump vec3 linetaps1   = weight3(0.5 - f.x * 0.5);\n" +
            "            mediump vec3 linetaps2   = weight3(1.0 - f.x * 0.5);\n" +
            "            mediump vec3 columntaps1 = weight3(0.5 - f.y * 0.5);\n" +
            "            mediump vec3 columntaps2 = weight3(1.0 - f.y * 0.5);\n" +
            "\n" +
            "            // make sure all taps added together is exactly 1.0, otherwise some\n" +
            "            // (very small) distortion can occur\n" +
            "            mediump float suml = dot(linetaps1, vec3(1.0)) + dot(linetaps2, vec3(1.0));\n" +
            "            mediump float sumc = dot(columntaps1, vec3(1.0)) + dot(columntaps2, vec3(1.0));\n" +
            "            linetaps1 /= suml;\n" +
            "            linetaps2 /= suml;\n" +
            "            columntaps1 /= sumc;\n" +
            "            columntaps2 /= sumc;\n" +
            "\n" +
            "            mediump vec2 xystart = (-2.5 - f) * stepxy + pos;\n" +
            "            mediump vec3 xpos1 = vec3(xystart.x, xystart.x + stepxy.x, xystart.x + stepxy.x * 2.0);\n" +
            "            mediump vec3 xpos2 = vec3(xystart.x + stepxy.x * 3.0, xystart.x + stepxy.x * 4.0, xystart.x + stepxy.x * 5.0);\n" +
            "\n" +
            "            gl_FragColor = vec4(\n" +
            "                line(xystart.y                 , xpos1, xpos2, linetaps1, linetaps2) * columntaps1.r +\n" +
            "                line(xystart.y + stepxy.y      , xpos1, xpos2, linetaps1, linetaps2) * columntaps2.r +\n" +
            "                line(xystart.y + stepxy.y * 2.0, xpos1, xpos2, linetaps1, linetaps2) * columntaps1.g +\n" +
            "                line(xystart.y + stepxy.y * 3.0, xpos1, xpos2, linetaps1, linetaps2) * columntaps2.g +\n" +
            "                line(xystart.y + stepxy.y * 4.0, xpos1, xpos2, linetaps1, linetaps2) * columntaps1.b +\n" +
            "                line(xystart.y + stepxy.y * 5.0, xpos1, xpos2, linetaps1, linetaps2) * columntaps2.b,\n" +
            "                1.0);\n" +
            "        }\n";

    private float[] targetSizeXY;
    private int targetSizeLocation;


    public GPUImageResizeFilter(final float[] targetSizeXY) {
        super(NO_FILTER_VERTEX_SHADER, RESIZE_FRAGMENT_SHADER);
        this.targetSizeXY = targetSizeXY;
    }

    @Override
    public void onInit() {
        super.onInit();
        targetSizeLocation = GLES20.glGetUniformLocation(getProgram(), "textureTargetSize");
    }

    @Override
    public void onInitialized() {
        super.onInitialized();
        setRatio(targetSizeXY);
    }

    public void setRatio(final float[] targetSizeXY) {
        this.targetSizeXY = targetSizeXY;
        setFloatVec2(targetSizeLocation, targetSizeXY);
    }
}
