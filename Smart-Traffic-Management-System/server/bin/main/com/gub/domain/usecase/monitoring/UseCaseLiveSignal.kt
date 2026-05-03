package com.gub.domain.usecase.monitoring

import com.gub.domain.models.monitoring.ModelLiveSignal
import com.gub.domain.repository.RepositoryMonitoring

class UseCaseLiveSignal(
    private val repositoryMonitoring: RepositoryMonitoring
) {

    suspend operator fun invoke(): ModelLiveSignal {
        return repositoryMonitoring.getLiveSignal()
    }
}