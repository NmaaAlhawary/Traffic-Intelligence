package com.gub.features.monitoring.data.network

import com.gub.core.domain.Response
import com.gub.domain.models.monitoring.ModelLiveSignal
import com.gub.features.dashboard.data.remote.SystemOverviewApi
import com.gub.models.dashboard.overview.ModelSystemOverview
import io.ktor.client.HttpClient
import io.ktor.client.call.body
import io.ktor.client.request.get

interface LiveSignalApi {

    suspend fun getLiveSignal(): ModelLiveSignal
}

class LiveSignalApiImpl(
    private val httpClient: HttpClient
) : LiveSignalApi {

    override suspend fun getLiveSignal(): ModelLiveSignal {
        return httpClient.get("/api/monitoring/signal").body()
    }
}