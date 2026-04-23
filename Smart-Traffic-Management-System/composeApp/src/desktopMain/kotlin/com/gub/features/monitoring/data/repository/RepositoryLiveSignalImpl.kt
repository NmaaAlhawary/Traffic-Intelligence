package com.gub.features.monitoring.data.repository

import com.gub.core.domain.Response
import com.gub.domain.models.monitoring.ModelLiveSignal
import com.gub.features.monitoring.data.network.LiveSignalApi
import com.gub.features.monitoring.domain.repository.RepositoryLiveSignal

class RepositoryLiveSignalImpl(
    private val liveSignalApi: LiveSignalApi
) : RepositoryLiveSignal {

    override suspend fun getLiveSignal(): Response<ModelLiveSignal> {
        return try {
            Response.Success(liveSignalApi.getLiveSignal())
        } catch (e: Exception) {
            Response.Error(e.message ?: "Unknown error")
        }
    }
}