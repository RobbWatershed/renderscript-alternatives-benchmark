package com.android.example.rsmigration

import android.content.Context
import android.graphics.Bitmap
import android.opengl.EGL14
import android.opengl.EGL15
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLExt
import android.opengl.EGLSurface
import android.opengl.GLES20
import android.opengl.GLES31
import android.opengl.GLUtils
import android.os.Build
import android.util.Size
import com.example.simpleegl.BLUR_VERTEX_SHADER
import com.example.simpleegl.TEXTURE_COORDINATES
import com.example.simpleegl.checkGLError
import com.example.simpleegl.createGLES20ShaderProgram
import com.example.simpleegl.isGLES31Supported
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

enum class SamplerMethod {
    NEAREST, BILINEAR, LANCZOS
}

enum class Antialiasing(internal val value: Int) {
    ANTIALIAS(1), NO_ANTIALIAS(0)
}

var NEAREST_VERT: String = """
    #version 100
    precision mediump float;

    uniform sampler2D u_Texture;
    uniform vec2 texelSize;
    uniform ivec2 sourceSize;
    uniform ivec2 targetSize;
    varying vec2 vTexCoord;

    void main() {
        float scale = float(sourceSize.y) / float(targetSize.y);
        float centerV = min((vTexCoord.y*float(sourceSize.y) + 0.5) * scale, float(sourceSize.y) - 1.0f);
       
        float center = centerV - 0.5f;

        vec2 sourceOffset = vec2(vTexCoord.x, (centerV - 0.5) * texelSize.y);
        vec4 color = texture2D(u_Texture, sourceOffset);

        gl_FragColor = color;
    }
""".trimIndent()

val NEAREST_HORIZ: String = """
    #version 100
    precision mediump float;

    uniform sampler2D u_Texture;
    uniform vec2 texelSize;
    uniform ivec2 sourceSize;
    uniform ivec2 targetSize;
    varying vec2 vTexCoord;

    void main() {
        float scale = float(sourceSize.x) / float(targetSize.x);
        float centerV = min((vTexCoord.x*float(sourceSize.x) + 0.5) * scale, float(sourceSize.x) - 1.0f);
        float center = centerV - 0.5f;
        vec2 sourceOffset = vec2((centerV - 0.5) * texelSize.x, vTexCoord.y);
        vec4 color = texture2D(u_Texture, sourceOffset);
        gl_FragColor = color;
    }
""".trimIndent()

val BILINEAR_HORIZ: String = """
#version 100
precision mediump float;

uniform sampler2D u_Texture;
uniform vec2 texelSize;
uniform ivec2 sourceSize;
uniform ivec2 targetSize;
uniform int antialiasing;
varying vec2 vTexCoord;

float bilinear(float x) {
    x = abs(x);
    if (x < 1.0f) {
        return 1.0f - x;
    } else {
        return 0.0f;
    }
}

void main() {
    float scale = float(sourceSize.x) / float(targetSize.x);
    float centerV = min((vTexCoord.x*float(sourceSize.x) + 0.5) * scale, float(sourceSize.x) - 1.0f);
    float filterRadius = 1.0f;
    if (antialiasing == 1) {
        filterRadius = 1.0f * scale;
    }
    int start = int(max(floor(centerV - filterRadius), 0.0f));
    int end = int(min(ceil(centerV + filterRadius), float(sourceSize.x) - 1.0f));
    int length = end - start;
    if (length < 0) {
        discard;
    }
    float filterScale = 1.0f;
    if (antialiasing == 1) {
        filterScale = 1.0f / scale;
    }
    float weightSum = 0.0f;

    float center = centerV - 0.5f;

    vec4 color = vec4(0f);

    for (int i = start; i < end; i++) {
        float xPosition = float(i) * texelSize.x;
        float dx = float(i) - center;
        float weight = bilinear(dx * xPosition * filterScale);
        vec2 sourceOffset = vec2(xPosition, vTexCoord.y);
        weightSum += weight;
        vec4 wColor = texture2D(u_Texture, sourceOffset) * weight;
        color += wColor;
    }

    if (weightSum != 0.0f) {
        color = color / weightSum;
    }

    gl_FragColor = color;
}
""".trimIndent()

val BILINEAR_VERTICAL: String = """
#version 100
precision mediump float;

uniform sampler2D u_Texture;
uniform vec2 texelSize;
uniform ivec2 sourceSize;
uniform ivec2 targetSize;
uniform int antialiasing;
varying vec2 vTexCoord;

float bilinear(float x) {
    x = abs(x);
    if (x < 1.0f) {
        return 1.0f - x;
    } else {
        return 0.0f;
    }
}

void main() {
    float scale = float(sourceSize.y) / float(targetSize.y);
    float centerV = min((vTexCoord.y*float(sourceSize.y) + 0.5) * scale, float(sourceSize.y) - 1.0f);
    float filterRadius = 1.0f;
    if (antialiasing == 1) {
        filterRadius = 1.0f * scale;
    }
    int start = int(max(floor(centerV - filterRadius), 0.0f));
    int end = int(min(ceil(centerV + filterRadius), float(sourceSize.y) - 1.0f));
    int length = end - start;
    if (length < 0) {
        discard;
    }
    float filterScale = 1.0f;
    if (antialiasing == 1) {
        filterScale = 1.0f / scale;
    }
    float weightSum = 0.0f;

    float center = centerV - 0.5f;

    vec4 color = vec4(0f);

    for (int i = start; i < end; i++) {
        float yPosition = float(i) * texelSize.y;
        float dy = float(i) - center;
        float weight = bilinear(dy * yPosition * filterScale);
        vec2 sourceOffset = vec2(vTexCoord.x, yPosition);
        weightSum += weight;
        vec4 wColor = texture2D(u_Texture, sourceOffset) * weight;
        color += wColor;
    }

    if (weightSum != 0.0f) {
        color = color / weightSum;
    }

    gl_FragColor = color;
}
""".trimIndent()

val LANCZOS_HORIZ: String = """
#version 100
precision mediump float;

uniform sampler2D u_Texture;
uniform vec2 texelSize;
uniform ivec2 sourceSize;
uniform ivec2 targetSize;
uniform int antialiasing;
varying vec2 vTexCoord;

float sinc(float x) {
    if (x == 0.0f) {
        return 1.0f;
    }
    return sin(x) / x;
}

float lanczos3(float x) {
    float scale_a = 1.0f / 3.0f;
    if (abs(x) < 3.0f) {
        float d = 3.14159265358979f * x;
        return sinc(d) * sinc(d * scale_a);
    }
    return 0.0f;
}

void main() {
    float scale = float(sourceSize.x) / float(targetSize.x);
    float centerV = min((vTexCoord.x*float(sourceSize.x) + 0.5) * scale, float(sourceSize.x) - 1.0f);
    float filterRadius = 1.5f;
    if (antialiasing == 1) {
        filterRadius = 1.5f * scale;
    }
    int start = int(max(floor(centerV - filterRadius), 0.0f));
    int end = int(min(ceil(centerV + filterRadius), float(sourceSize.x) - 1.0f));
    int length = end - start;
    if (length < 0) {
        discard;
    }
    float filterScale = 1.0f;
    if (antialiasing == 1) {
        filterScale = 1.0f / scale;
    }
    float weightSum = 0.0f;

    float center = centerV - 0.5f;

    vec4 color = vec4(0f);

    for (int i = start; i < end; i++) {
        float xPosition = float(i) * texelSize.x;
        float dx = float(i) - center;
        float weight = lanczos3(dx * xPosition * filterScale);
        vec2 sourceOffset = vec2(xPosition, vTexCoord.y);
        weightSum += weight;
        vec4 wColor = texture2D(u_Texture, sourceOffset) * weight;
        color += wColor;
    }

    if (weightSum != 0.0f) {
        color = color / weightSum;
    }

    gl_FragColor = color;
}
""".trimIndent()

val LANCZOS_VERT: String = """
#version 100
precision mediump float;

uniform sampler2D u_Texture;
uniform vec2 texelSize;
uniform ivec2 sourceSize;
uniform ivec2 targetSize;
uniform int antialiasing;
varying vec2 vTexCoord;

float sinc(float x) {
    if (x == 0.0f) {
        return 1.0f;
    }
    return sin(x) / x;
}

float lanczos3(float x) {
    float scale_a = 1.0f / 3.0f;
    if (abs(x) < 3.0f) {
        float d = 3.14159265358979f * x;
        return sinc(d) * sinc(d * scale_a);
    }
    return 0.0f;
}

void main() {
    float scale = float(sourceSize.y) / float(targetSize.y);
    float centerV = min((vTexCoord.y*float(sourceSize.y) + 0.5) * scale, float(sourceSize.y) - 1.0f);
    float filterRadius = 1.5f;
    if (antialiasing == 1) {
        filterRadius = 1.5f * scale;
    }
    int start = int(max(floor(centerV - filterRadius), 0.0f));
    int end = int(min(ceil(centerV + filterRadius), float(sourceSize.y) - 1.0f));
    int length = end - start;
    if (length < 0) {
        discard;
    }
    float filterScale = 1.0f;
    if (antialiasing == 1) {
        filterScale = 1.0f / scale;
    }
    float weightSum = 0.0f;

    float center = centerV - 0.5f;

    vec4 color = vec4(0f);

    for (int i = start; i < end; i++) {
        float yPosition = float(i) * texelSize.y;
        float dy = float(i) - center;
        float weight = lanczos3(dy * yPosition * filterScale);
        vec2 sourceOffset = vec2(vTexCoord.x, yPosition);
        weightSum += weight;
        vec4 wColor = texture2D(u_Texture, sourceOffset) * weight;
        color += wColor;
    }

    if (weightSum != 0.0f) {
        color = color / weightSum;
    }

    gl_FragColor = color;
}
""".trimIndent()

class OpenGLESResizeRenderPass(private val mContext: Context) {
    private var isGLES3Used = false

    private var textureIds: IntArray
    private var frameBuffers: IntArray

    private var eglDisplay: EGLDisplay? = null
    private var eglSurface: EGLSurface? = null
    private var eglContext: EGLContext? = null

    private var nearestHorizontalProgram: Int = -1
    private var nearestVerticalProgram: Int = -1
    private var bilinearHorizontalProgram: Int = -1
    private var bilinearVerticalProgram: Int = -1
    private var lanczosHorizontalProgram: Int = -1
    private var lanczosVerticalProgram: Int = -1

    private var oldWidth: Int = -1
    private var oldHeight: Int = -1

    init {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            isGLES3Used = isGLES31Supported(mContext)
        }

        // Surface and EGL can be reused where possible
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        EGL14.eglInitialize(eglDisplay!!, null, 0, null, 0)

        val configAttribs = intArrayOf(
            EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT,
            EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
            EGL14.EGL_RED_SIZE, 8,
            EGL14.EGL_GREEN_SIZE, 8,
            EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_ALPHA_SIZE, 8,
            EGL14.EGL_NONE
        )
        val configs = arrayOfNulls<EGLConfig>(1)
        val numConfigs = IntArray(1)
        EGL14.eglChooseConfig(eglDisplay!!, configAttribs, 0, configs, 0, 1, numConfigs, 0)

        val pbufferAttribs = intArrayOf(
            EGL14.EGL_WIDTH, 1,
            EGL14.EGL_HEIGHT, 1,
            EGL14.EGL_NONE
        )
        eglSurface = EGL14.eglCreatePbufferSurface(eglDisplay!!, configs[0], pbufferAttribs, 0)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q && isGLES3Used) {
            val glAttributes = intArrayOf(
                EGL15.EGL_CONTEXT_MAJOR_VERSION, 3,
                EGL15.EGL_CONTEXT_MINOR_VERSION, 1,
                EGL15.EGL_CONTEXT_OPENGL_DEBUG, EGL14.EGL_TRUE,
                EGLExt.EGL_CONTEXT_FLAGS_KHR, EGL14.EGL_TRUE,
                EGL14.EGL_NONE
            )
            eglContext = EGL14.eglCreateContext(
                eglDisplay, configs[0], EGL14.EGL_NO_CONTEXT, glAttributes, 0
            )
        } else {
            val contextAttribs = intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE)
            eglContext = EGL14.eglCreateContext(
                eglDisplay,
                configs[0],
                EGL14.EGL_NO_CONTEXT,
                contextAttribs,
                0
            )
        }

        EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)

        bilinearHorizontalProgram =
            createGLES20ShaderProgram(BLUR_VERTEX_SHADER, BILINEAR_HORIZ)
        if (bilinearHorizontalProgram == -1) {
            throw IllegalStateException("Can't compile horizontal bilinear shader")
        }
        bilinearVerticalProgram =
            createGLES20ShaderProgram(BLUR_VERTEX_SHADER, BILINEAR_VERTICAL)
        if (bilinearVerticalProgram == -1) {
            throw IllegalStateException("Can't compile vertical bilinear shader")
        }

        lanczosHorizontalProgram =
            createGLES20ShaderProgram(BLUR_VERTEX_SHADER, LANCZOS_HORIZ)
        if (lanczosHorizontalProgram == -1) {
            throw IllegalStateException("Can't compile horizontal lanczos shader")
        }

        lanczosVerticalProgram =
            createGLES20ShaderProgram(BLUR_VERTEX_SHADER, LANCZOS_VERT)
        if (lanczosVerticalProgram == -1) {
            throw IllegalStateException("Can't compile horizontal lanczos shader")
        }

        nearestHorizontalProgram =
            createGLES20ShaderProgram(BLUR_VERTEX_SHADER, NEAREST_HORIZ)
        if (nearestHorizontalProgram == -1) {
            throw IllegalStateException("Can't compile horizontal lanczos shader")
        }

        nearestVerticalProgram =
            createGLES20ShaderProgram(BLUR_VERTEX_SHADER, NEAREST_VERT)
        if (nearestVerticalProgram == -1) {
            throw IllegalStateException("Can't compile horizontal lanczos shader")
        }


        textureIds = IntArray(3)
        GLES20.glGenTextures(3, textureIds, 0)

        frameBuffers = IntArray(2)
        GLES20.glGenFramebuffers(2, frameBuffers, 0)

        EGL14.eglMakeCurrent(
            eglDisplay,
            EGL14.EGL_NO_SURFACE,
            EGL14.EGL_NO_SURFACE,
            EGL14.EGL_NO_CONTEXT
        )
    }

    /**
     * This is undefined behaviour to call this method from multiple threads
     */
    fun resize(
        bitmap: Bitmap,
        targetSize: Size,
        samplerMethod: SamplerMethod,
        antialiasing: Antialiasing
    ): Bitmap {
        if (eglDisplay == null || eglSurface == null || eglContext == null) {
            throw IllegalStateException("Render pass has been already terminated")
        }

        EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)

        val newBitmap = if (samplerMethod == SamplerMethod.BILINEAR) {
            resizeGLES20Convolution(
                bitmap,
                targetSize,
                bilinearHorizontalProgram,
                bilinearVerticalProgram,
                antialiasing
            )
        } else if (samplerMethod == SamplerMethod.LANCZOS) {
            resizeGLES20Convolution(
                bitmap,
                targetSize,
                lanczosHorizontalProgram,
                lanczosVerticalProgram,
                antialiasing
            )
        } else {
            resizeGLES20Convolution(
                bitmap,
                targetSize,
                nearestHorizontalProgram,
                nearestVerticalProgram,
                null
            )
        }
        oldWidth = bitmap.width
        oldHeight = bitmap.height
        EGL14.eglMakeCurrent(
            eglDisplay,
            EGL14.EGL_NO_SURFACE,
            EGL14.EGL_NO_SURFACE,
            EGL14.EGL_NO_CONTEXT
        )
        return newBitmap
    }

    private fun resizeGLES20Convolution(
        bitmap: Bitmap,
        targetSize: Size,
        horizontalProgram: Int,
        verticalProgram: Int,
        antialiasing: Antialiasing?,
    ): Bitmap {
        loadBitmap(bitmap, textureIds[0])
        rebindTex(textureIds[1], Size(targetSize.width, bitmap.height))
        rebindTex(textureIds[2], targetSize)

        // Horizontal pass

        checkGLError { GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, frameBuffers[0]) }
        checkGLError {
            GLES20.glFramebufferTexture2D(
                GLES20.GL_FRAMEBUFFER,
                GLES20.GL_COLOR_ATTACHMENT0,
                GLES20.GL_TEXTURE_2D,
                textureIds[1],
                0
            )
        }
        if (GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER) != GLES20.GL_FRAMEBUFFER_COMPLETE) {
            throw RuntimeException("Framebuffer is not complete")
        }

        checkGLError { GLES20.glUseProgram(horizontalProgram) }

        val aVertexPositionLocation =
            GLES20.glGetAttribLocation(horizontalProgram, "aPosition")
        val aTexCoordLocation = GLES20.glGetAttribLocation(horizontalProgram, "aTexCoord")
        val uTextureLocation = GLES20.glGetUniformLocation(horizontalProgram, "u_Texture")
        val sourceSizeLocation =
            GLES20.glGetUniformLocation(horizontalProgram, "sourceSize")
        val targetSizeLocation =
            GLES20.glGetUniformLocation(horizontalProgram, "targetSize")
        val horTexelSizeLocation =
            GLES20.glGetUniformLocation(horizontalProgram, "texelSize")

        checkGLError { GLES20.glActiveTexture(GLES20.GL_TEXTURE0) }
        checkGLError { GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureIds[0]) }
        checkGLError { GLES20.glUniform1i(uTextureLocation, 0) }

        checkGLError { GLES20.glClearColor(0.0f, 0.0f, 0.0f, 0.0f) }
        checkGLError { GLES20.glViewport(0, 0, bitmap.width, bitmap.height) }
        checkGLError { GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT) }

        val quadVertices = floatArrayOf(
            -1.0f, 1.0f,   // Top-left corner
            -1.0f, -1.0f,  // Bottom-left corner
            1.0f, 1.0f,   // Top-right corner
            1.0f, -1.0f   // Bottom-right corner
        )

        val vertexBuffer: FloatBuffer = ByteBuffer
            .allocateDirect(quadVertices.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        vertexBuffer.put(quadVertices).position(0)

        val textureCoordinatesBuffer: FloatBuffer =
            ByteBuffer.allocateDirect(TEXTURE_COORDINATES.size * 4).run {
                order(ByteOrder.nativeOrder())
                asFloatBuffer().apply {
                    put(TEXTURE_COORDINATES)
                    position(0)
                }
            }

        GLES20.glVertexAttribPointer(
            aVertexPositionLocation,
            2,
            GLES20.GL_FLOAT,
            false,
            0,
            vertexBuffer
        )
        GLES20.glVertexAttribPointer(
            aTexCoordLocation,
            2,
            GLES20.GL_FLOAT,
            false,
            0,
            textureCoordinatesBuffer
        )

        if (antialiasing != null) {
            val antialisingLocation =
                GLES20.glGetUniformLocation(horizontalProgram, "antialiasing")
            GLES20.glUniform1i(antialisingLocation, antialiasing.value)
        }

        GLES20.glUniform2f(
            horTexelSizeLocation,
            1.0f / bitmap.width.toFloat(),
            1.0f / bitmap.height.toFloat()
        )
        GLES20.glUniform2i(
            sourceSizeLocation,
            bitmap.width,
            bitmap.height,
        )
        GLES20.glUniform2i(
            targetSizeLocation,
            targetSize.width,
            bitmap.height,
        )

        GLES20.glEnableVertexAttribArray(aVertexPositionLocation)
        GLES20.glEnableVertexAttribArray(aTexCoordLocation)

        checkGLError { GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4) }

        GLES20.glDisableVertexAttribArray(aVertexPositionLocation)
        GLES20.glDisableVertexAttribArray(aTexCoordLocation)

        // Vertical pass

        checkGLError { GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, frameBuffers[1]) }
        checkGLError {
            GLES20.glFramebufferTexture2D(
                GLES20.GL_FRAMEBUFFER,
                GLES20.GL_COLOR_ATTACHMENT0,
                GLES20.GL_TEXTURE_2D,
                textureIds[2],
                0
            )
        }
        if (GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER) != GLES20.GL_FRAMEBUFFER_COMPLETE) {
            throw RuntimeException("Framebuffer is not complete")
        }

        checkGLError { GLES20.glUseProgram(verticalProgram) }

        val vVertexPositionLocation =
            GLES20.glGetAttribLocation(verticalProgram, "aPosition")
        val vTexCoordLocation = GLES20.glGetAttribLocation(verticalProgram, "aTexCoord")
        val vuTextureLocation = GLES20.glGetUniformLocation(verticalProgram, "u_Texture")
        val vsourceSizeLocation = GLES20.glGetUniformLocation(verticalProgram, "sourceSize")
        val vtargetSizeLocation = GLES20.glGetUniformLocation(verticalProgram, "targetSize")
        val vhorTexelSizeLocation =
            GLES20.glGetUniformLocation(verticalProgram, "texelSize")

        checkGLError { GLES20.glActiveTexture(GLES20.GL_TEXTURE0) }
        checkGLError { GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureIds[1]) }
        checkGLError { GLES20.glUniform1i(vuTextureLocation, 0) }

        checkGLError { GLES20.glClearColor(0.0f, 0.0f, 0.0f, 0.0f) }
        checkGLError { GLES20.glViewport(0, 0, targetSize.width, bitmap.height) }
        checkGLError { GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT) }

        GLES20.glEnableVertexAttribArray(vVertexPositionLocation)
        GLES20.glEnableVertexAttribArray(vTexCoordLocation)

        GLES20.glVertexAttribPointer(
            vVertexPositionLocation,
            2,
            GLES20.GL_FLOAT,
            false,
            0,
            vertexBuffer
        )
        GLES20.glVertexAttribPointer(
            vTexCoordLocation,
            2,
            GLES20.GL_FLOAT,
            false,
            0,
            textureCoordinatesBuffer
        )

        if (antialiasing != null) {
            val vantialisingLocation =
                GLES20.glGetUniformLocation(verticalProgram, "antialiasing")
            GLES20.glUniform1i(vantialisingLocation, antialiasing.value)
        }

        GLES20.glUniform2f(
            vhorTexelSizeLocation,
            1.0f / targetSize.width.toFloat(),
            1.0f / bitmap.height.toFloat()
        )
        GLES20.glUniform2i(
            vsourceSizeLocation,
            targetSize.width,
            bitmap.height,
        )
        GLES20.glUniform2i(
            vtargetSizeLocation,
            targetSize.width,
            targetSize.height,
        )

        checkGLError { GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4) }

        GLES20.glFinish()

        GLES20.glDisableVertexAttribArray(vVertexPositionLocation)
        GLES20.glDisableVertexAttribArray(vTexCoordLocation)

        val buffer = ByteBuffer.allocateDirect(targetSize.width * bitmap.height * 4)
            .order(ByteOrder.nativeOrder())
        GLES31.glReadPixels(
            0,
            0,
            targetSize.width,
            targetSize.height,
            GLES31.GL_RGBA,
            GLES31.GL_UNSIGNED_BYTE,
            buffer
        )

        val outputBitmap =
            Bitmap.createBitmap(targetSize.width, targetSize.height, Bitmap.Config.ARGB_8888)
        outputBitmap.copyPixelsFromBuffer(buffer)

        return outputBitmap
    }

    private fun rebindTex(tex: Int, targetSize: Size) {
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tex)

        GLES20.glTexParameteri(
            GLES20.GL_TEXTURE_2D,
            GLES20.GL_TEXTURE_MIN_FILTER,
            GLES20.GL_LINEAR
        )
        GLES20.glTexParameteri(
            GLES20.GL_TEXTURE_2D,
            GLES20.GL_TEXTURE_MAG_FILTER,
            GLES20.GL_LINEAR
        )
        GLES20.glTexParameteri(
            GLES20.GL_TEXTURE_2D,
            GLES20.GL_TEXTURE_WRAP_S,
            GLES20.GL_CLAMP_TO_EDGE
        )
        GLES20.glTexParameteri(
            GLES20.GL_TEXTURE_2D,
            GLES20.GL_TEXTURE_WRAP_T,
            GLES20.GL_CLAMP_TO_EDGE
        )

        GLES20.glTexImage2D(
            GLES20.GL_TEXTURE_2D,
            0,
            GLES20.GL_RGBA,
            targetSize.width,
            targetSize.height,
            0,
            GLES20.GL_RGBA,
            GLES20.GL_UNSIGNED_BYTE,
            null
        )

        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0)
    }

    private fun loadBitmap(bitmap: Bitmap, tex: Int) {
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tex)
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bitmap, 0)

        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(
            GLES20.GL_TEXTURE_2D,
            GLES20.GL_TEXTURE_WRAP_S,
            GLES20.GL_CLAMP_TO_EDGE
        )
        GLES20.glTexParameteri(
            GLES20.GL_TEXTURE_2D,
            GLES20.GL_TEXTURE_WRAP_T,
            GLES20.GL_CLAMP_TO_EDGE
        )

        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0)
    }

    fun terminate() {
        if (eglDisplay != null && eglSurface != null && eglContext != null) {
            EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)
            if (textureIds.isNotEmpty()) {
                GLES20.glDeleteTextures(textureIds.size, textureIds, 0)
                textureIds = IntArray(0)
            }
            if (frameBuffers.isNotEmpty()) {
                GLES20.glDeleteFramebuffers(frameBuffers.size, frameBuffers, 0)
                frameBuffers = IntArray(0)
            }
            if (nearestHorizontalProgram != -1) {
                GLES20.glDeleteProgram(nearestHorizontalProgram)
                nearestHorizontalProgram = -1
            }
            if (nearestVerticalProgram != -1) {
                GLES20.glDeleteProgram(nearestVerticalProgram)
                nearestVerticalProgram = -1
            }
            if (bilinearHorizontalProgram != -1) {
                GLES20.glDeleteProgram(bilinearHorizontalProgram)
                bilinearHorizontalProgram = -1
            }
            if (bilinearVerticalProgram != -1) {
                GLES20.glDeleteProgram(bilinearVerticalProgram)
                bilinearVerticalProgram = -1
            }
            if (lanczosHorizontalProgram != -1) {
                GLES20.glDeleteProgram(lanczosHorizontalProgram)
                lanczosHorizontalProgram = -1
            }
            if (lanczosVerticalProgram != -1) {
                GLES20.glDeleteProgram(lanczosVerticalProgram)
                lanczosVerticalProgram = -1
            }
            val egl = eglDisplay
            if (egl != null) {
                if (eglSurface != null) {
                    EGL14.eglDestroySurface(egl, eglSurface)
                    eglSurface = null
                }
                if (eglContext != null) {
                    EGL14.eglDestroyContext(egl, eglContext)
                    eglContext = null
                }
                EGL14.eglTerminate(egl)
                eglDisplay = null
            }
        }
    }

    protected fun finalize() {
        terminate()
    }
}