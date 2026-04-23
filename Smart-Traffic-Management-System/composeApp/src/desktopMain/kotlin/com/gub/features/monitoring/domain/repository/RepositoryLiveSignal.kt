package com.gub.features.monitoring.domain.repository

import com.gub.core.domain.Response
import com.gub.domain.models.monitoring.ModelLiveSignal

interface RepositoryLiveSignal {

    suspend fun getLiveSignal(): Response<ModelLiveSignal>
}