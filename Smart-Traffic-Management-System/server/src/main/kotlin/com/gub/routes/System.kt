package com.gub.routes

import com.gub.models.core.LatencyPing
import com.gub.models.core.LatencyPong
import com.gub.services.StatusService
import io.ktor.server.routing.Route
import io.ktor.server.websocket.webSocket
import io.ktor.websocket.Frame
import io.ktor.websocket.readText
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.serialization.json.Json

fun Route.system() {
    val statusService = StatusService()

    webSocket("/ws/system-status") {
        println("Client connected")

        // Status sender loop
        launch(Dispatchers.IO) {
            while (true) {
                val status = statusService.getStatus()
                val json = Json.encodeToString(status)
                send(Frame.Text(json))
                delay(1000L)
            }
        }

        // Latency ping-pong listener
        for (frame in incoming) {
            if (frame is Frame.Text) {
                val text = frame.readText()
                try {
                    val ping = Json.decodeFromString<LatencyPing>(text)
                    if (ping.type == "ping") {
                        val now = System.currentTimeMillis()
                        val latency = now - ping.timestamp
                        statusService.latency = latency
                        val pong = LatencyPong(latency = latency)
                        send(Frame.Text(Json.encodeToString(pong)))
                    }
                } catch (_: Exception) {
                    // Not a latency ping, ignore
                }
            }
        }
    }
}