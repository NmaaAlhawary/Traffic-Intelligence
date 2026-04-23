package com.gub.features.analytics.data

import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.engine.cio.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.request.*
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.json
import kotlinx.coroutines.*
import kotlinx.serialization.*
import kotlinx.serialization.json.*

@Serializable
data class TrafficInput(
    val hour: Int,
    val dayofweek: Int,
    val is_weekend: Int,
    val temperature: Double,
    val humidity: Double,
    val rain: Double,
    val wind_speed: Double,
    val vehicle_count_last_5min: Int,
    val weather_condition: Int
)

@Serializable
data class PredictionResponse(
    val predicted_vehicle_count: Double
)

val client = HttpClient(CIO) {
    install(ContentNegotiation) {
        json()
    }
}

fun predictTraffic() {
    val input = TrafficInput(
        hour = 17,
        dayofweek = 1,
        is_weekend = 0,
        temperature = 31.5,
        humidity = 72.0,
        rain = 0.1,
        wind_speed = 8.0,
        vehicle_count_last_5min = 50,
        weather_condition = 0
    )

    CoroutineScope(Dispatchers.IO).launch {
        try {
            val response: PredictionResponse = client.post("http://localhost:8090/predict") {
                contentType(ContentType.Application.Json)
                setBody(input)
            }.body()

            println("✅ Predicted Vehicle Count: ${response.predicted_vehicle_count}")
        } catch (e: Exception) {
            println("❌ API Call Failed: ${e.message}")
        }
    }
}