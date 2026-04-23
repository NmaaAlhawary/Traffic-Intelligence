package com.gub.features.monitoring.data.network

import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.toComposeImageBitmap
import com.gub.app.Const
import kotlinx.coroutines.delay
import org.bytedeco.javacv.FFmpegFrameGrabber
import org.bytedeco.javacv.Java2DFrameConverter
import javax.imageio.ImageIO

import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.net.Socket
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.jetbrains.skia.Image.Companion.makeFromEncoded
import java.nio.ByteBuffer

suspend fun sendFrameToPythonSocket(frame: BufferedImage): Pair<ByteArray, String> = withContext(Dispatchers.IO) {
    val socket = Socket(Const.BASE_URL, 5001)
    val out = DataOutputStream(socket.getOutputStream())
    val input = DataInputStream(socket.getInputStream())

    // Encode frame
    val baos = ByteArrayOutputStream()
    ImageIO.write(frame, "jpg", baos)
    val frameBytes = baos.toByteArray()

    // Send size + frame
    out.writeInt(frameBytes.size)
    out.write(frameBytes)
    out.flush()

    // Receive size-prefixed vehicle count JSON
    val jsonLen = input.readInt()
    val jsonBytes = ByteArray(jsonLen)
    input.readFully(jsonBytes)

    // Receive annotated frame
    val jpegBytes = input.readBytes()

    socket.close()
    val count = String(jsonBytes)
    return@withContext Pair(jpegBytes, count)
}

class LiveFeedManager {

    private val converter = Java2DFrameConverter()

    suspend fun streamVideoAndProcess(
        videoPath: String,
        onFrameUpdate: (ImageBitmap, String) -> Unit
    ) {
        val grabber = FFmpegFrameGrabber(videoPath)
        grabber.start()

        while (true) {
            val frame = grabber.grab() ?: break
            val image = converter.convert(frame) ?: continue

            val (annotatedBytes, vehicleCount) = sendFrameToPythonSocket(image)
            val annotatedImage = makeFromEncoded(annotatedBytes).toComposeImageBitmap()

            onFrameUpdate(annotatedImage, vehicleCount)
//            delay(25) // Throttle or match frame rate
        }

        grabber.stop()
    }
}