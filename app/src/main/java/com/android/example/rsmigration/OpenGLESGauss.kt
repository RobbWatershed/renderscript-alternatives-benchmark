package com.example.simpleegl

import android.app.ActivityManager
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
import android.opengl.GLU
import android.opengl.GLUtils
import android.os.Build
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.sqrt


val BLUR_VERTEX_SHADER: String = """
    attribute vec4 aPosition;   
    attribute vec2 aTexCoord;  
    varying vec2 vTexCoord;       

    void main() {
        gl_Position = aPosition;
        vTexCoord = aTexCoord;  
    }
""".trimIndent()

val FRAGMENT_HORIZONTAL_BLUR_SHADER: String = """
precision mediump float;

uniform sampler2D u_Texture;   
uniform float kernel[512];
uniform int radius;
uniform vec2 texelSize;
varying vec2 vTexCoord;        

void main() {
    vec4 color = vec4(0.0);
    int vx = 0;
    for (int x = -radius; x < radius; x++) {
        vec2 offset = vec2(x, 0) * texelSize;
        vec4 lc = texture2D(u_Texture, vTexCoord + offset);
        float w = kernel[vx];
        vec4 weight = vec4(w);
        color = color + lc * weight;
        vx++;
    }
    gl_FragColor = color;
}
""".trimIndent()

val VERTICAL_FRAGMENT_SHADER: String = """
precision mediump float;

uniform sampler2D u_Texture; 
uniform float kernel[512];
uniform int radius;
uniform vec2 texelSize;
varying vec2 vTexCoord;        

void main() {
    vec4 color = vec4(0.0);
    int vx = 0;
    for (int y = -radius; y < radius; y++) {
        vec2 offset = vec2(0, y) * texelSize;
        vec4 lc = texture2D(u_Texture, vTexCoord + offset);
        float w = kernel[vx];
        vec4 weight = vec4(w);
        color = color + lc * weight;
        vx++;
    }
    gl_FragColor = color;
}
""".trimIndent()

var HORIZONTAL_GLES3_COMP = """
    #version 310 es

    layout(std430) buffer;
    layout (local_size_x = 32, local_size_y = 32) in;
    uniform layout (rgba8, binding = 0) readonly highp sampler2D u_Texture;
    uniform layout (rgba8, binding = 1) writeonly highp image2D u_outputImage;
    layout(std430, binding = 2) readonly buffer SSBO {
        float data[512];
    } ssbo;
    uniform ivec2 u_Resolution;
    uniform int radius;

    void main() {
        ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);
        if (pixelCoord.x >= u_Resolution.x || pixelCoord.y >= u_Resolution.y) {
            return;
        }

        vec4 color = vec4(0.0, 0.0, 0.0, 0.0);
        int vx = 0;
        for (int x = -radius; x < radius; ++x) {
            float w = ssbo.data[vx];
            ivec2 offset = pixelCoord + ivec2(x, 0);
            offset = clamp(offset, ivec2(0), u_Resolution - 1);
            color += texelFetch(u_Texture, offset, 0) * w;
            vx += 1;
        }
        imageStore(u_outputImage, pixelCoord, color);
    }
""".trimIndent()

var VERTICAL_GLES3_COMP = """
   #version 310 es

   layout(std430) buffer;
   layout (local_size_x = 32, local_size_y = 32) in;
   uniform layout (rgba8, binding = 0) readonly highp sampler2D u_Texture;
   uniform layout (rgba8, binding = 1) writeonly highp image2D u_outputImage;
   layout(std430, binding = 2) readonly buffer SSBO {
       float data[512];
   } ssbo;
   uniform ivec2 u_Resolution;
   uniform int radius;

   void main() {
       ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);
       if (pixelCoord.x >= u_Resolution.x || pixelCoord.y >= u_Resolution.y) {
           return;
       }

       vec4 color = vec4(0.0, 0.0, 0.0, 0.0);
       int vx = 0;
       for (int y = -radius; y < radius; ++y) {
           float w = ssbo.data[vx];
           ivec2 offset = pixelCoord + ivec2(0, y);
           offset = clamp(offset, ivec2(0), u_Resolution - 1);
           color += texelFetch(u_Texture, offset, 0) * w;
           vx += 1;
       }
       imageStore(u_outputImage, pixelCoord, color);
   }
""".trimIndent()

fun checkGlErrorImpl() {
    var error: Int
    var noError = true
    while (run { error = GLES31.glGetError(); error } != GLES31.GL_NO_ERROR) {
        val method = Thread.currentThread().stackTrace[3].methodName
        val lineNumber = Thread.currentThread().stackTrace[3].lineNumber
        Log.d(
            "GL ERROR",
            "Error: " + error + " (" + GLU.gluErrorString(error) + "): " + method + " LN:" + lineNumber
        )
        noError = false
    }
    assert(noError)
}

// a simple block that checks for GL errors after the contents of the block are executed
inline fun <R> checkGLError(block: () -> R): R {
    val v = block()
//    if (Bu) {
    checkGlErrorImpl()
//    }
    return v
}

private const val MAX_POSSIBLE_KERNEL_SIZE = 512

private val TEXTURE_COORDINATES = floatArrayOf(
    //x,    y
    0.0f, 1.0f,
    0.0f, 0.0f,
    1.0f, 1.0f,
    1.0f, 0.0f,
)

private fun compileComputeShader(shaderCode: String): Int {
    val shader = GLES31.glCreateShader(GLES31.GL_COMPUTE_SHADER)

    // Compile the shader code
    GLES31.glShaderSource(shader, shaderCode)
    GLES31.glCompileShader(shader)

    // Check for compile errors
    val compileStatus = IntArray(1)
    GLES31.glGetShaderiv(shader, GLES31.GL_COMPILE_STATUS, compileStatus, 0)
    if (compileStatus[0] == GLES31.GL_FALSE) {
        val infoLog = GLES31.glGetShaderInfoLog(shader)
        throw RuntimeException("Error compiling shader: $infoLog")
    }

    return shader
}

private fun linkComputeShaderProgram(computeShader: Int): Int {
    val program = GLES31.glCreateProgram()

    // Attach the compute shader
    GLES31.glAttachShader(program, computeShader)

    // Link the program
    GLES31.glLinkProgram(program)

    // Check for link errors
    val linkStatus = IntArray(1)
    GLES31.glGetProgramiv(program, GLES31.GL_LINK_STATUS, linkStatus, 0)
    if (linkStatus[0] == GLES31.GL_FALSE) {
        val infoLog = GLES31.glGetProgramInfoLog(program)
        throw RuntimeException("Error linking program: $infoLog")
    }

    return program
}

private fun generateGaussianKernel1D(size: Int, sigma: Float): FloatArray {
    require(size % 2 == 1) { "Kernel size must be odd" }

    val kernel = FloatArray(size)
    val mean = size / 2
    val sigma2 = 2 * sigma * sigma
    var sum = 0.0

    for (i in kernel.indices) {
        val x = i - mean
        kernel[i] = (exp(-(x * x) / sigma2) / (sqrt(PI * sigma2))).toFloat()
        sum += kernel[i]
    }

    // Normalize the kernel to ensure the sum is 1
    for (i in kernel.indices) {
        kernel[i] /= sum.toFloat()
    }

    return kernel
}

private fun getSigmaSize(kernelSize: Int): Float {
    val safeKernelSize = if (kernelSize <= 1) {
        2f
    } else {
        kernelSize.toFloat()
    }
    return 0.3f * ((safeKernelSize - 1f) * 0.5f - 1f) + 0.8f
}

private fun createGLES20ShaderProgram(vertexShaderCode: String, fragmentShaderCode: String): Int {
    val vertexShader = GLES20.glCreateShader(GLES20.GL_VERTEX_SHADER)
    GLES20.glShaderSource(vertexShader, vertexShaderCode)
    GLES20.glCompileShader(vertexShader)

    val fragmentShader = GLES20.glCreateShader(GLES20.GL_FRAGMENT_SHADER)
    GLES20.glShaderSource(fragmentShader, fragmentShaderCode)
    GLES20.glCompileShader(fragmentShader)

    val program = GLES20.glCreateProgram()
    GLES20.glAttachShader(program, vertexShader)
    GLES20.glAttachShader(program, fragmentShader)
    GLES20.glLinkProgram(program)

    return program
}

private fun isGLES31Supported(context: Context): Boolean {
    val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    val configurationInfo = activityManager.deviceConfigurationInfo
    val esVersion = configurationInfo.reqGlEsVersion
    return esVersion >= 0x30001
}

class OpenGLESBlurRenderPass(private val mContext: Context, private var allowGLES3: Boolean = false) {

    private var isGLES3Used = false

    private var textureIds: IntArray
    private var frameBuffers: IntArray
    private var kBuffers: IntArray

    private var eglDisplay: EGLDisplay? = null
    private var eglSurface: EGLSurface? = null
    private var eglContext: EGLContext? = null

    private var horizontalProgram: Int = -1
    private var verticalProgram: Int = -1

    private var oldWidth: Int = -1
    private var oldHeight: Int = -1

    init {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q && allowGLES3) {
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

            EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)

            val horizontalShaderBuilt = compileComputeShader(HORIZONTAL_GLES3_COMP)
            horizontalProgram = linkComputeShaderProgram(horizontalShaderBuilt)
            val verticalShaderBuilt = compileComputeShader(VERTICAL_GLES3_COMP)
            verticalProgram = linkComputeShaderProgram(verticalShaderBuilt)

            textureIds = IntArray(3)
            GLES20.glGenTextures(3, textureIds, 0)

            frameBuffers = IntArray(1)
            GLES20.glGenFramebuffers(1, frameBuffers, 0)

            kBuffers = IntArray(1)
            checkGLError { GLES20.glGenBuffers(1, kBuffers, 0) }
        } else {
            val contextAttribs = intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE)
            eglContext = EGL14.eglCreateContext(eglDisplay, configs[0], EGL14.EGL_NO_CONTEXT, contextAttribs, 0)

            EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)

            horizontalProgram = createGLES20ShaderProgram(BLUR_VERTEX_SHADER, FRAGMENT_HORIZONTAL_BLUR_SHADER)
            verticalProgram = createGLES20ShaderProgram(BLUR_VERTEX_SHADER, VERTICAL_FRAGMENT_SHADER)

            textureIds = IntArray(2)
            GLES20.glGenTextures(2, textureIds, 0)

            frameBuffers = IntArray(1)
            GLES20.glGenFramebuffers(1, frameBuffers, 0)

            kBuffers = IntArray(1)
            GLES20.glGenBuffers(1, kBuffers, 0)
        }

        EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
    }

    private val lock = Any()

    /**
     * This is undefined behaviour to call this method from multiple threads
     */
    fun render(bitmap: Bitmap, radius: Int) : Bitmap {
        synchronized(lock) {
            if (eglDisplay == null || eglSurface == null || eglContext == null) {
                throw IllegalStateException("Render pass has been already terminated")
            }

            EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)

            if (isGLES3Used) {
                val newBitmap = renderGLES30(bitmap, radius)
                oldWidth = newBitmap.width
                oldHeight = newBitmap.height
                EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
                return newBitmap
            } else {
                val newBitmap = renderGLES20(bitmap, radius)
                oldWidth = newBitmap.width
                oldHeight = newBitmap.height
                EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
                return newBitmap
            }
        }
    }

    private fun renderGLES30(bitmap: Bitmap, radius: Int): Bitmap {
        loadBitmap(bitmap, textureIds[0])
        rebindTex(textureIds[1], bitmap)
        rebindTex(textureIds[2], bitmap)

        val width = bitmap.width
        val height = bitmap.height

        val kernel = generateGaussianKernel1D(radius * 2 + 1, getSigmaSize(radius * 2 + 1))

        val kernelBuffer: FloatBuffer = ByteBuffer
            .allocateDirect(MAX_POSSIBLE_KERNEL_SIZE * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        kernelBuffer.put(kernel).position(0)

        GLES31.glUseProgram(horizontalProgram)

// Setup uniforms and textures
        val hresolutionLocation = GLES31.glGetUniformLocation(horizontalProgram, "u_Resolution")
        val htextureLocation = GLES31.glGetUniformLocation(horizontalProgram, "u_Texture")
        val hRadius = GLES31.glGetUniformLocation(horizontalProgram, "radius")

// Set resolution
        checkGLError { GLES31.glUniform2i(hresolutionLocation, width, height) }
        checkGLError { GLES31.glUniform1i(hRadius, radius) }

// Bind input texture to binding point 0
        checkGLError { GLES31.glActiveTexture(GLES31.GL_TEXTURE0) }
        checkGLError { GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, textureIds[0]) }
        checkGLError { GLES31.glUniform1i(htextureLocation, 0) }

        checkGLError { GLES31.glUniform1i(htextureLocation, 0) }
        checkGLError { GLES31.glActiveTexture(GLES31.GL_TEXTURE0 + 1) }
        checkGLError { GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, textureIds[1]) }

        checkGLError { GLES31.glBindImageTexture(1, textureIds[1], 0, false, 0, GLES31.GL_WRITE_ONLY, GLES31.GL_RGBA8) }

        checkGLError { GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, kBuffers[0]) }

// Allocate memory for the buffer and transfer data
        checkGLError { GLES31.glBufferData(GLES31.GL_SHADER_STORAGE_BUFFER, kernelBuffer.capacity() * 4, kernelBuffer, GLES31.GL_STREAM_READ) }
        checkGLError { GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 2, kBuffers[0]) }

        val localGroupSize = 32
        val numGroupsX = (width + localGroupSize - 1) / localGroupSize
        val numGroupsY = (height + localGroupSize - 1) / localGroupSize
        checkGLError { GLES31.glDispatchCompute(numGroupsX, numGroupsY, 1) }

        checkGLError { GLES31.glMemoryBarrier(GLES31.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT) }

        GLES31.glUseProgram(verticalProgram)

        val vresolutionLocation = GLES31.glGetUniformLocation(verticalProgram, "u_Resolution")
        val vtextureLocation = GLES31.glGetUniformLocation(verticalProgram, "u_Texture")
        val vhRadius = GLES31.glGetUniformLocation(verticalProgram, "radius")

        checkGLError { GLES31.glUniform2i(vresolutionLocation, width, height) }
        checkGLError { GLES31.glUniform1i(vhRadius, radius) }

        checkGLError { GLES31.glActiveTexture(GLES31.GL_TEXTURE0) }
        checkGLError { GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, textureIds[1]) }
        checkGLError { GLES31.glUniform1i(vtextureLocation, 0) }

        checkGLError { GLES31.glUniform1i(htextureLocation, 0) }
        checkGLError { GLES31.glActiveTexture(GLES31.GL_TEXTURE0 + 1) }
        checkGLError { GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, textureIds[2]) }

        checkGLError { GLES31.glBindImageTexture(1, textureIds[2], 0, false, 0, GLES31.GL_WRITE_ONLY, GLES31.GL_RGBA8) }

        checkGLError { GLES31.glDispatchCompute(numGroupsX, numGroupsY, 1) }

        checkGLError { GLES31.glMemoryBarrier(GLES31.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT) }

        checkGLError { GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, frameBuffers[0]) }
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

        val buffer = ByteBuffer.allocateDirect(width * height * 4).order(ByteOrder.nativeOrder())
        GLES31.glReadPixels(0, 0, width, height, GLES31.GL_RGBA, GLES31.GL_UNSIGNED_BYTE, buffer)

        val outputBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        outputBitmap.copyPixelsFromBuffer(buffer)

        return outputBitmap
    }

    private fun renderGLES20(bitmap: Bitmap, radius: Int): Bitmap {
        val kernel = generateGaussianKernel1D(radius * 2 + 1, getSigmaSize(radius * 2 + 1))

        loadBitmap(bitmap, textureIds[0])
        rebindTex(textureIds[1], bitmap)

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, frameBuffers[0])
        GLES20.glFramebufferTexture2D(
            GLES20.GL_FRAMEBUFFER,
            GLES20.GL_COLOR_ATTACHMENT0,
            GLES20.GL_TEXTURE_2D,
            textureIds[1],
            0
        )

        if (GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER) != GLES20.GL_FRAMEBUFFER_COMPLETE) {
            throw RuntimeException("Framebuffer is not complete")
        }

        GLES20.glUseProgram(horizontalProgram)

        val aVertexPositionLocation = GLES20.glGetAttribLocation(horizontalProgram, "aPosition")
        val aTexCoordLocation = GLES20.glGetAttribLocation(horizontalProgram, "aTexCoord")
        val uTexture1Location = GLES20.glGetUniformLocation(horizontalProgram, "u_Texture")
        val horizontalKernelLocation = GLES20.glGetUniformLocation(horizontalProgram, "kernel")
        val horizontalRadius = GLES20.glGetUniformLocation(horizontalProgram, "radius")
        val horTexelSize = GLES20.glGetUniformLocation(horizontalProgram, "texelSize")

        val vaVertexPositionLocation = GLES20.glGetAttribLocation(horizontalProgram, "aPosition")
        val vaTexCoordLocation = GLES20.glGetAttribLocation(horizontalProgram, "aTexCoord")
        val vuTexture1Location = GLES20.glGetUniformLocation(horizontalProgram, "u_Texture")
        val verticalKernelLocation = GLES20.glGetUniformLocation(horizontalProgram, "kernel")
        val verticalRadius = GLES20.glGetUniformLocation(horizontalProgram, "radius")
        val verticalTexelSize = GLES20.glGetUniformLocation(horizontalProgram, "texelSize")

        val kernelBuffer: FloatBuffer = ByteBuffer
            .allocateDirect(MAX_POSSIBLE_KERNEL_SIZE * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        kernelBuffer.put(kernel).position(0)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureIds[0])
        GLES20.glUniform1i(uTexture1Location, 0)

        GLES20.glClearColor(0.0f, 0.0f, 0.0f, 0.0f)
        GLES20.glViewport(0, 0, bitmap.width, bitmap.height)
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)

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

        GLES20.glVertexAttribPointer(aVertexPositionLocation, 2, GLES20.GL_FLOAT, false, 0, vertexBuffer)
        GLES20.glVertexAttribPointer(aTexCoordLocation, 2, GLES20.GL_FLOAT, false, 0, textureCoordinatesBuffer)

        GLES20.glUniform1fv(horizontalKernelLocation, MAX_POSSIBLE_KERNEL_SIZE, kernelBuffer)
        GLES20.glUniform1i(horizontalRadius, radius)
        GLES20.glUniform2f(horTexelSize, 1.0f / bitmap.width.toFloat(), 1.0f / bitmap.height.toFloat())

        GLES20.glEnableVertexAttribArray(aVertexPositionLocation)
        GLES20.glEnableVertexAttribArray(aTexCoordLocation)

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)

        // Vertical pass

        GLES20.glUseProgram(verticalProgram)

        GLES20.glFramebufferTexture2D(
            GLES20.GL_FRAMEBUFFER,
            GLES20.GL_COLOR_ATTACHMENT0,
            GLES20.GL_TEXTURE_2D,
            textureIds[0],
            0
        )

        if (GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER) != GLES20.GL_FRAMEBUFFER_COMPLETE) {
            throw RuntimeException("Framebuffer is not complete")
        }

        GLES20.glDisableVertexAttribArray(aVertexPositionLocation)
        GLES20.glDisableVertexAttribArray(aTexCoordLocation)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureIds[1])
        GLES20.glUniform1i(vuTexture1Location, 0)

        GLES20.glVertexAttribPointer(vaVertexPositionLocation, 2, GLES20.GL_FLOAT, false, 0, vertexBuffer)
        GLES20.glVertexAttribPointer(vaTexCoordLocation, 2, GLES20.GL_FLOAT, false, 0, textureCoordinatesBuffer)

        GLES20.glUniform1fv(verticalKernelLocation, MAX_POSSIBLE_KERNEL_SIZE, kernelBuffer)
        GLES20.glUniform1i(verticalRadius, radius)
        GLES20.glUniform2f(verticalTexelSize, 1.0f / bitmap.width.toFloat(), 1.0f / bitmap.height.toFloat())

        GLES20.glEnableVertexAttribArray(vaVertexPositionLocation)
        GLES20.glEnableVertexAttribArray(vaTexCoordLocation)

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)

        GLES20.glFinish()

        GLES20.glDisableVertexAttribArray(vaVertexPositionLocation)
        GLES20.glDisableVertexAttribArray(vaTexCoordLocation)

        val buffer = ByteBuffer.allocateDirect(bitmap.width * bitmap.height * 4).order(ByteOrder.nativeOrder())
        GLES20.glReadPixels(0, 0, bitmap.width, bitmap.height, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, buffer)

        // Convert buffer to Bitmap
        val outputBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        outputBitmap.copyPixelsFromBuffer(buffer)

        return outputBitmap
    }

    private var isTexReady: Boolean = false

    private fun loadBitmap(bitmap: Bitmap, tex: Int) {
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tex)
        if (isTexReady && oldWidth == bitmap.width && oldHeight == bitmap.height) {
            GLUtils.texSubImage2D(GLES20.GL_TEXTURE_2D, 0,0, 0, bitmap)
        } else {
            GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bitmap, 0)
        }

        isTexReady = true

        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0)
    }

    private fun rebindTex(tex: Int, bitmap: Bitmap) {
        if (oldWidth == bitmap.width && oldHeight == bitmap.height) {
            return
        }
        if (isGLES3Used) {
            GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, tex)

            GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_MIN_FILTER, GLES31.GL_LINEAR)
            GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_MAG_FILTER, GLES31.GL_LINEAR)
            GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_WRAP_S, GLES31.GL_CLAMP_TO_EDGE)
            GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_WRAP_T, GLES31.GL_CLAMP_TO_EDGE)

            GLES31.glTexStorage2D(
                GLES31.GL_TEXTURE_2D, 1, GLES31.GL_RGBA8,
                bitmap.width, bitmap.height
            )

            GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, 0)
        } else {
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tex)

            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

            GLES20.glTexImage2D(
                GLES20.GL_TEXTURE_2D,
                0,
                GLES20.GL_RGBA,
                bitmap.width,
                bitmap.height,
                0,
                GLES20.GL_RGBA,
                GLES20.GL_UNSIGNED_BYTE,
                null
            )

            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0)
        }
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
            if (kBuffers.isNotEmpty()) {
                GLES20.glDeleteBuffers(kBuffers.size, kBuffers, 0)
                kBuffers = IntArray(0)
            }
            if (horizontalProgram != - 1) {
                if (isGLES3Used) {
                    GLES31.glDeleteProgram(horizontalProgram)
                } else {
                    GLES20.glDeleteProgram(horizontalProgram)
                }
                horizontalProgram = -1
            }
            if (verticalProgram != -1) {
                if (isGLES3Used) {
                    GLES31.glDeleteProgram(verticalProgram)
                } else {
                    GLES20.glDeleteProgram(verticalProgram)
                }
                verticalProgram = -1
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
