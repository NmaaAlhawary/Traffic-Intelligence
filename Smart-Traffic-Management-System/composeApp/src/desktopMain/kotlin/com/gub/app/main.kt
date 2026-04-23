/**
 * Main entry point for Vehicle Detection Desktop Client
 * Author: Alims-Repo
 * Date: 2025-06-17
 */
package com.gub.app

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.WindowPosition
import androidx.compose.ui.window.application
import androidx.compose.ui.window.rememberWindowState
import com.gub.core.ui.theme.VehicleDetectionTheme
import com.gub.features.analytics.viewModel.ViewModelAnalytics
import java.awt.Toolkit



//import androidx.compose.desktop.ui.tooling.preview.Preview
//import androidx.compose.foundation.Image
//import androidx.compose.foundation.layout.*
//import androidx.compose.material3.*
//import androidx.compose.runtime.*
//import androidx.compose.ui.Modifier
//import androidx.compose.ui.graphics.ImageBitmap
//import androidx.compose.ui.graphics.asImageBitmap
//import androidx.compose.ui.graphics.toComposeImageBitmap
//import androidx.compose.ui.unit.dp
//import androidx.compose.ui.window.Window
//import androidx.compose.ui.window.application
//import java.awt.image.BufferedImage
//import java.io.*
//import java.net.Socket
//import javax.imageio.ImageIO
//import org.bytedeco.javacv.FFmpegFrameGrabber
//import org.bytedeco.javacv.Java2DFrameConverter
//import kotlinx.coroutines.*
//
//fun encodeImageToBytes(image: BufferedImage): ByteArray {
//    val output = ByteArrayOutputStream()
//    ImageIO.write(image, "jpg", output)
//    return output.toByteArray()
//}
//
//fun decodeImageFromBytes(bytes: ByteArray): BufferedImage {
//    return ImageIO.read(bytes.inputStream())
//}
//
//fun sendImageToPython(frame: BufferedImage): Pair<BufferedImage, String> {
//    val socket = Socket("127.0.0.1", 5001)
//    val out = DataOutputStream(socket.getOutputStream())
//    val input = DataInputStream(socket.getInputStream())
//
//    val imageBytes = encodeImageToBytes(frame)
//
//    out.writeInt(imageBytes.size)
//    out.write(imageBytes)
//    out.flush()
//
//    val jsonLength = input.readInt()
//    val jsonBytes = ByteArray(jsonLength)
//    input.readFully(jsonBytes)
//
//    val jpegBytes = input.readBytes()
//    socket.close()
//
//    val annotatedImage = decodeImageFromBytes(jpegBytes)
//    val detectedVehiclesJson = String(jsonBytes)
//
//    return Pair(annotatedImage, detectedVehiclesJson)
//}
//
//@Composable
//@Preview
//fun App() {
//    var imageBitmap by remember { mutableStateOf<ImageBitmap?>(null) }
//    var vehicleInfo by remember { mutableStateOf("Waiting for detection...") }
//
//    LaunchedEffect(Unit) {
//        val grabber = FFmpegFrameGrabber("detection/resources/1.mp4")
//        grabber.start()
//        val converter = Java2DFrameConverter()
//
//        while (isActive) {
//            val frame = grabber.grab() ?: break
//            val bufferedImage = converter.convert(frame) ?: continue
//
//            try {
//                val start = System.currentTimeMillis()
//                val (annotated, json) = sendImageToPython(bufferedImage)
////                println("Frame processed in ${System.currentTimeMillis() - start} ms")
//                imageBitmap = annotated.toImageBitmap()
//                vehicleInfo = json
//            } catch (e: Exception) {
//                vehicleInfo = "Error: ${e.message}"
//            }
//
//            delay(33L)
//        }
//
//        grabber.stop()
//    }
//
//    MaterialTheme {
//        Column(
//            modifier = Modifier.fillMaxSize().padding(16.dp),
//            horizontalAlignment = Alignment.CenterHorizontally
//        ) {
//            imageBitmap?.let {
//                Image(bitmap = it, contentDescription = "Annotated Frame")
//            }
//            Spacer(modifier = Modifier.height(16.dp))
//            Text(vehicleInfo)
//        }
//    }
//}
//
//fun main() = application {
//    Window(onCloseRequest = ::exitApplication, title = "Vehicle Detection (Kotlin + Python)") {
//        App()
//    }
//}
//
//fun BufferedImage.toImageBitmap(): ImageBitmap {
//    return this.toComposeImageBitmap()
//}


@OptIn(ExperimentalMaterial3Api::class)
fun main() = application {

    val (screenWidth, screenHeight) = getScreenDimensions()

    val windowWidth = screenWidth * 0.85f
    val windowHeight = screenHeight * 0.8f

    Window(
        onCloseRequest = {
            exitApplication()
        },
        title = "AI Vehicle Detection Desktop",
        state = rememberWindowState(
            width = windowWidth, height = windowHeight,
            position = WindowPosition(Alignment.Center)
        ),
        resizable = true,
    ) {
        val dark = isSystemInDarkTheme()
        val isDarkTheme = remember { mutableStateOf(dark) }
        val viewModelAnalytics: ViewModelAnalytics = remember { ViewModelAnalytics() }
        VehicleDetectionTheme(darkTheme = isDarkTheme.value) {
            TrafficManagementApp(isDarkTheme)
        }
    }
}

fun getScreenDimensions() = Toolkit.getDefaultToolkit().screenSize.let { screenSize->
    Pair(screenSize.width.dp, screenSize.height.dp)
}