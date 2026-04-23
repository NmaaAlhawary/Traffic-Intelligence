package com.gub.domain.model

import kotlinx.serialization.Serializable

@Serializable
data class ExportJson(
    val exportedAt: String,
    val trafficStats: List<ExportTrafficStats>,
    val signalData: List<ExportSignalData>,
    val weatherData: List<ExportWeatherData>
)

@Serializable
data class ExportTrafficStats(
    val id: String,
    val timestamp: String,
    val measureDuration: Long,
    val roadId: String,
    val vehicleCount: Int
)

@Serializable
data class ExportSignalData(
    val id: String,
    val timestamp: String,
    val roadId: String,
    val signalState: String
)

@Serializable
data class ExportWeatherData(
    val id: String,
    val timestamp: String,
    val temperature: Double,
    val humidity: Double,
    val windSpeed: Double,
    val visibility: Double
)