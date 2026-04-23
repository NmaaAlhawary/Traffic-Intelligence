package com.gub.features.analytics.data.remote

import com.gub.domain.models.analytics.ModelTrafficVolume
import io.ktor.client.HttpClient
import io.ktor.client.call.body
import io.ktor.client.request.get
import io.ktor.client.request.parameter

interface AnalyticsApis {

    suspend fun getTrafficVolume(type: ModelTrafficVolume.TrafficVolumeType): ModelTrafficVolume
}

class AnalyticsApisImpl(
    private val httpClient: HttpClient
) : AnalyticsApis {

    override suspend fun getTrafficVolume(type: ModelTrafficVolume.TrafficVolumeType): ModelTrafficVolume {
        return httpClient.get("/api/analytics/traffic-volume") {
            parameter("type", type.value) // or type.toString()
        }.body()
    }
}