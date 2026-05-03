package com.gub.domain.repository

import com.gub.domain.models.monitoring.ModelLiveSignal

interface RepositoryMonitoring {

    suspend fun getLiveSignal(): ModelLiveSignal
}