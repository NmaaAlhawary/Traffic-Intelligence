package com.gub.data.repository

import com.gub.data.service.monitoring.ServiceLiveSignal
import com.gub.domain.models.monitoring.ModelLiveSignal
import com.gub.domain.repository.RepositoryMonitoring

class RepositoryMonitoringImpl(
    private val serviceLiveSignal: ServiceLiveSignal
) : RepositoryMonitoring {

    override suspend fun getLiveSignal(): ModelLiveSignal {
        return ModelLiveSignal(
            roadName = "Mirpur 10",
            east = serviceLiveSignal.eastState(),
            west = serviceLiveSignal.westState(),
            north = serviceLiveSignal.northState(),
            south = serviceLiveSignal.southState(),
            waitingTime = 2 * 60 * 1000L
        )
    }
}